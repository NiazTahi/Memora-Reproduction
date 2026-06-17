from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    id: str
    conversation_id: str
    topic: str
    message_ids: list[str]
    text: str
    timestamp: str | None = None


@dataclass
class EpisodicMemory:
    id: str
    conversation_id: str
    segment_id: str
    index: str
    value: str
    raw_segment: str


@dataclass
class MemoryEntry:
    id: str
    conversation_id: str
    primary_abstraction: str
    memory_value: str
    episodic_ids: list[str] = field(default_factory=list)
    cue_anchors: list[str] = field(default_factory=list)


@dataclass
class MemoryStore:
    conversation_id: str
    segments: list[Segment] = field(default_factory=list)
    episodic_memories: list[EpisodicMemory] = field(default_factory=list)
    entries: list[MemoryEntry] = field(default_factory=list)
    abstraction_embeddings: list[list[float]] = field(default_factory=list)
    cue_embeddings: dict[str, list[float]] = field(default_factory=dict)

