from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    conversation_id: str
    session_id: int
    dia_id: str
    speaker: str
    text: str
    timestamp: str | None = None
    blip_caption: str | None = None
    img_url: str | None = None

    def render(self) -> str:
        prefix = f"({self.timestamp}) " if self.timestamp else ""
        text = f"{prefix}{self.dia_id} {self.speaker}: {self.text}"
        if self.blip_caption:
            text += f"\n[shared image caption: {self.blip_caption}]"
        return text


@dataclass(frozen=True)
class QAItem:
    question_id: str
    conversation_id: str
    question: str
    answer: str
    category: int
    category_name: str
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    speaker_a: str
    speaker_b: str
    messages: list[Message]
    qa: list[QAItem]

    def render_full_context(self) -> str:
        return "\n".join(message.render() for message in self.messages)


CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

PAPER_CATEGORY_ORDER = ["multi-hop", "temporal", "open-domain", "single-hop"]

