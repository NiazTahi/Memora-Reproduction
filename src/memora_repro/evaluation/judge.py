from __future__ import annotations

from memora_repro.llm.openai_client import OpenAIClient


def judge_messages(question: str, reference: str, prediction: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a strict but fair evaluator for question answering. Output strict JSON.",
        },
        {
            "role": "user",
            "content": f"""
Score whether the predicted answer is semantically correct with respect to the reference answer.

Use a score from 0 to 1:
- 1.0: fully correct
- 0.5: partially correct
- 0.0: incorrect or unsupported

Question: {question}
Reference answer: {reference}
Predicted answer: {prediction}

Output JSON:
{{"score": 0.0, "reason": "brief reason"}}
""".strip(),
        },
    ]


def llm_judge_score(
    client: OpenAIClient,
    *,
    model: str,
    question: str,
    reference: str,
    prediction: str,
) -> float:
    response = client.chat_json(
        model=model,
        messages=judge_messages(question, reference, prediction),
        temperature=0.0,
    )
    score = float(response.get("score", 0.0))
    return max(0.0, min(1.0, score))

