from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from memora_repro.answering.prompts import answer_messages
from memora_repro.data.locomo import load_locomo
from memora_repro.evaluation.metrics import bleu, locomo_f1
from memora_repro.llm.openai_client import OpenAIClient
from memora_repro.memory.builder import load_store
from memora_repro.retrieval.memora import (
    MemoraPolicyRetriever,
    MemoraSemanticRetriever,
    format_memory_context,
)
from memora_repro.utils.config import load_config
from memora_repro.utils.jsonl import append_jsonl, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/locomo.yaml", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    conversations = load_locomo("data/raw/locomo_repo/data/locomo10.json")

    out_file = Path(config["paths"]["results_dir"]) / "memora_p_predictions.jsonl"
    done_ids = {row["question_id"] for row in read_jsonl(out_file)}

    client = OpenAIClient(
        cache_dir=config["paths"]["cache_dir"],
        seed=config["openai"].get("seed"),
    )
    semantic = MemoraSemanticRetriever(
        client=client,
        embedding_model=config["openai"]["embedding_model"],
        top_k_abstractions=int(config["memora"]["semantic_top_k_abstractions"]),
        top_k_cues=int(config["memora"]["semantic_top_k_cues"]),
        max_memories=int(config["memora"]["semantic_max_memories"]),
    )
    retriever = MemoraPolicyRetriever(
        client=client,
        policy_model=config["openai"]["policy_model"],
        semantic_retriever=semantic,
        max_steps=int(config["memora"]["policy_max_steps"]),
        expand_k=int(config["memora"]["policy_expand_k"]),
    )
    answer_model = config["openai"]["answer_model"]
    qas = [qa for conversation in conversations for qa in conversation.qa]
    if args.limit is not None:
        qas = qas[: args.limit]
    store_dir = Path(config["paths"]["processed_data_dir"]) / "memora_stores"
    needed_conversation_ids = {qa.conversation_id for qa in qas}
    stores = {
        conversation_id: load_store(store_dir / f"{conversation_id}.json")
        for conversation_id in needed_conversation_ids
    }

    for qa in tqdm(qas, desc="MEMORA(P)"):
        if qa.question_id in done_ids:
            continue
        store = stores[qa.conversation_id]
        retrieved = retriever.retrieve(qa.question, store)
        context = format_memory_context(store, retrieved)
        pred = client.chat(
            model=answer_model,
            messages=answer_messages(context, qa.question),
            temperature=0.0,
        ).strip()
        row = {
            **asdict(qa),
            "method": "memora_p",
            "prediction": pred,
            "retrieved_memory_ids": [item.entry.id for item in retrieved],
            "bleu": bleu(pred, qa.answer),
            "f1": locomo_f1(pred, qa.answer, qa.category),
            "llm_judge": None,
        }
        append_jsonl(out_file, [row])


if __name__ == "__main__":
    main()
