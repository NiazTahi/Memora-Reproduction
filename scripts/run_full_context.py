from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from memora_repro.answering.prompts import answer_messages
from memora_repro.data.locomo import load_locomo
from memora_repro.evaluation.metrics import bleu, locomo_f1
from memora_repro.llm.openai_client import OpenAIClient
from memora_repro.utils.config import load_config
from memora_repro.utils.jsonl import append_jsonl, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/locomo.yaml", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    conversations = load_locomo("data/raw/locomo_repo/data/locomo10.json")
    by_id = {conversation.conversation_id: conversation for conversation in conversations}

    out_file = Path(config["paths"]["results_dir"]) / "full_context_predictions.jsonl"
    done_ids = {row["question_id"] for row in read_jsonl(out_file)}

    client = OpenAIClient(
        cache_dir=config["paths"]["cache_dir"],
        seed=config["openai"].get("seed"),
    )
    answer_model = config["openai"]["answer_model"]

    qas = [qa for conversation in conversations for qa in conversation.qa]
    if args.limit is not None:
        qas = qas[: args.limit]

    for qa in tqdm(qas, desc="Full Context"):
        if qa.question_id in done_ids:
            continue
        conversation = by_id[qa.conversation_id]
        context = conversation.render_full_context()
        pred = client.chat(
            model=answer_model,
            messages=answer_messages(context, qa.question),
            temperature=0.0,
        ).strip()
        row = {
            **asdict(qa),
            "method": "full_context",
            "prediction": pred,
            "bleu": bleu(pred, qa.answer),
            "f1": locomo_f1(pred, qa.answer, qa.category),
            "llm_judge": None,
        }
        append_jsonl(out_file, [row])


if __name__ == "__main__":
    main()
