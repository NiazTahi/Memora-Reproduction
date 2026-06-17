from __future__ import annotations


ANSWER_SYSTEM_PROMPT = (
    "You answer questions using only the provided conversation or memory context. "
    "Be concise. If the answer is not supported, say that the information is not mentioned."
)


def answer_messages(context: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Context:\n"
                f"{context}\n\n"
                "Question:\n"
                f"{question}\n\n"
                "Answer:"
            ),
        },
    ]

