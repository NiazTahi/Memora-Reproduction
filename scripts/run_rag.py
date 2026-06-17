from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from memora_repro.answering.prompts import answer_messages
from memora_repro.data.locomo import load_locomo
from memora_repro.evaluation.metrics import bleu, locomo_f1
from memora_repro.llm.openai_client import OpenAIClient
from memora_repro.retrieval.vector import top_k_cosine
from memora_repro.utils.config import load_config
from memora_repro.utils.jsonl import append_jsonl, read_jsonl
from memora_repro.utils.text import chunk_by_words


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/locomo.yaml", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    conversations = load_locomo("data/raw/locomo_repo/data/locomo10.json")
    by_id = {conversation.conversation_id: conversation for conversation in conversations}

    out_file = Path(config["paths"]["results_dir"]) / "rag_predictions.jsonl"
    done_ids = {row["question_id"] for row in read_jsonl(out_file)}

    client = OpenAIClient(
        cache_dir=config["paths"]["cache_dir"],
        seed=config["openai"].get("seed"),
    )
    answer_model = config["openai"]["answer_model"]
    embedding_model = config["openai"]["embedding_model"]
    chunk_size = int(config["rag"]["chunk_size_tokens"])
    top_k = int(config["rag"]["top_k"])

    qas = [qa for conversation in conversations for qa in conversation.qa]
    if args.limit is not None:
        qas = qas[: args.limit]
    needed_conversation_ids = {qa.conversation_id for qa in qas}

    rag_index = {}
    for conversation in tqdm(conversations, desc="Build RAG indices"):
        if conversation.conversation_id not in needed_conversation_ids:
            continue
        chunks = chunk_by_words(conversation.render_full_context(), chunk_size)
        embeddings = client.embed(model=embedding_model, texts=chunks)
        rag_index[conversation.conversation_id] = {
            "chunks": chunks,
            "embeddings": embeddings,
        }

    for qa in tqdm(qas, desc="RAG"):
        if qa.question_id in done_ids:
            continue
        index = rag_index[qa.conversation_id]
        query_embedding = client.embed(model=embedding_model, texts=[qa.question])[0]
        matches = top_k_cosine(query_embedding, index["embeddings"], top_k)
        retrieved_chunks = [index["chunks"][i] for i, _ in matches]
        context = "\n\n---\n\n".join(retrieved_chunks)
        pred = client.chat(
            model=answer_model,
            messages=answer_messages(context, qa.question),
            temperature=0.0,
        ).strip()
        row = {
            **asdict(qa),
            "method": "rag",
            "prediction": pred,
            "retrieved_context": retrieved_chunks,
            "bleu": bleu(pred, qa.answer),
            "f1": locomo_f1(pred, qa.answer, qa.category),
            "llm_judge": None,
        }
        append_jsonl(out_file, [row])


if __name__ == "__main__":
    main()
