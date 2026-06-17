from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from memora_repro.data.schemas import PAPER_CATEGORY_ORDER


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.replace(",", "")
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def locomo_f1(prediction: str, reference: str, category: int) -> float:
    if category == 1:
        predictions = [part.strip() for part in prediction.split(",")]
        references = [part.strip() for part in reference.split(",")]
        return float(np.mean([max(token_f1(pred, ref) for pred in predictions) for ref in references]))
    if category in {2, 3, 4}:
        if category == 3:
            reference = reference.split(";")[0].strip()
        return token_f1(prediction, reference)
    if category == 5:
        lower = prediction.lower()
        return float("no information available" in lower or "not mentioned" in lower)
    raise ValueError(f"Unknown category: {category}")


def bleu(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    smoothing = SmoothingFunction().method1
    return float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing))


@dataclass
class AggregateMetrics:
    bleu: float
    f1: float
    llm_judge: float | None = None
    count: int = 0


def aggregate_predictions(rows: list[dict]) -> dict[str, AggregateMetrics]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["category_name"]].append(row)
        grouped["overall"].append(row)

    output: dict[str, AggregateMetrics] = {}
    for category in [*PAPER_CATEGORY_ORDER, "overall"]:
        items = grouped.get(category, [])
        if not items:
            output[category] = AggregateMetrics(0.0, 0.0, None, 0)
            continue
        judge_scores = [
            float(item["llm_judge"])
            for item in items
            if item.get("llm_judge") is not None
        ]
        output[category] = AggregateMetrics(
            bleu=float(np.mean([item["bleu"] for item in items])),
            f1=float(np.mean([item["f1"] for item in items])),
            llm_judge=float(np.mean(judge_scores)) if judge_scores else None,
            count=len(items),
        )
    return output

