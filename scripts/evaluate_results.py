from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from memora_repro.evaluation.judge import llm_judge_score
from memora_repro.evaluation.metrics import aggregate_predictions
from memora_repro.llm.openai_client import OpenAIClient
from memora_repro.utils.config import load_config
from memora_repro.utils.jsonl import read_jsonl


METHOD_FILES = {
    "full_context": "full_context_predictions.jsonl",
    "rag": "rag_predictions.jsonl",
    "memora_s": "memora_s_predictions.jsonl",
    "memora_p": "memora_p_predictions.jsonl",
}


def maybe_add_judge_scores(
    rows: list[dict],
    *,
    client: OpenAIClient,
    judge_model: str,
    limit: int | None,
    checkpoint_path: Path,
    checkpoint_every: int = 25,
) -> list[dict]:
    updated = []
    for i, row in enumerate(tqdm(rows, desc="LLM judge")):
        if limit is not None and i >= limit:
            updated.append(row)
            continue
        if row.get("llm_judge") is None:
            row = dict(row)
            row["llm_judge"] = llm_judge_score(
                client,
                model=judge_model,
                question=row["question"],
                reference=row["answer"],
                prediction=row["prediction"],
            )
        updated.append(row)
        if (i + 1) % checkpoint_every == 0:
            checkpoint_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in updated + rows[i + 1 :]) + "\n",
                encoding="utf-8",
            )
    checkpoint_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in updated) + "\n",
        encoding="utf-8",
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/locomo.yaml", type=Path)
    parser.add_argument("--with-llm-judge", action="store_true")
    parser.add_argument("--judge-limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(config["paths"]["results_dir"])
    client = None
    if args.with_llm_judge:
        client = OpenAIClient(
            cache_dir=config["paths"]["cache_dir"],
            seed=config["openai"].get("seed"),
        )

    all_summary = {}
    for method, filename in METHOD_FILES.items():
        path = results_dir / filename
        rows = read_jsonl(path)
        if not rows:
            continue
        if args.with_llm_judge and client is not None:
            rows = maybe_add_judge_scores(
                rows,
                client=client,
                judge_model=config["openai"]["judge_model"],
                limit=args.judge_limit,
                checkpoint_path=path,
            )
        summary = aggregate_predictions(rows)
        all_summary[method] = {
            category: {
                "BLEU": round(metrics.bleu, 3),
                "F1": round(metrics.f1, 3),
                "LLM": None if metrics.llm_judge is None else round(metrics.llm_judge, 3),
                "N": metrics.count,
            }
            for category, metrics in summary.items()
        }

    out_path = results_dir / "summary_table.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()
