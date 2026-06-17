from __future__ import annotations

import json
from pathlib import Path

from memora_repro.data.schemas import CATEGORY_NAMES, Conversation, Message, QAItem


def load_locomo(path: str | Path, include_adversarial: bool = False) -> list[Conversation]:
    raw_samples = json.loads(Path(path).read_text(encoding="utf-8"))
    conversations: list[Conversation] = []

    for raw in raw_samples:
        conv_id = raw["sample_id"]
        raw_conv = raw["conversation"]
        messages: list[Message] = []

        for session_id in range(1, 100):
            session_key = f"session_{session_id}"
            if session_key not in raw_conv:
                continue
            timestamp = raw_conv.get(f"{session_key}_date_time")
            for turn in raw_conv[session_key]:
                messages.append(
                    Message(
                        conversation_id=conv_id,
                        session_id=session_id,
                        dia_id=turn["dia_id"],
                        speaker=turn["speaker"],
                        text=turn.get("text") or turn.get("clean_text") or "",
                        timestamp=timestamp,
                        blip_caption=turn.get("blip_caption"),
                        img_url=turn.get("img_url"),
                    )
                )

        qa_items: list[QAItem] = []
        for index, qa in enumerate(raw.get("qa", [])):
            category = int(qa["category"])
            if category == 5 and not include_adversarial:
                continue
            qa_items.append(
                QAItem(
                    question_id=f"{conv_id}:qa:{index}",
                    conversation_id=conv_id,
                    question=qa["question"],
                    answer=str(qa["answer"]),
                    category=category,
                    category_name=CATEGORY_NAMES.get(category, f"category-{category}"),
                    evidence=list(qa.get("evidence") or []),
                )
            )

        conversations.append(
            Conversation(
                conversation_id=conv_id,
                speaker_a=raw_conv.get("speaker_a", ""),
                speaker_b=raw_conv.get("speaker_b", ""),
                messages=messages,
                qa=qa_items,
            )
        )

    return conversations


def flatten_qa(conversations: list[Conversation]) -> list[QAItem]:
    return [qa for conversation in conversations for qa in conversation.qa]

