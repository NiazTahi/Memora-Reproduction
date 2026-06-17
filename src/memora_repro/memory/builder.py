from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from memora_repro.data.schemas import Conversation, Message
from memora_repro.llm.openai_client import OpenAIClient
from memora_repro.memory.prompts import (
    consolidation_prompt,
    cue_anchor_prompt,
    episodic_memory_prompt,
    factual_memory_prompt,
    segment_messages_prompt,
)
from memora_repro.memory.schemas import EpisodicMemory, MemoryEntry, MemoryStore, Segment
from memora_repro.retrieval.vector import top_k_cosine


def render_numbered_messages(messages: list[Message]) -> str:
    return "\n".join(f"{i + 1}. {message.render()}" for i, message in enumerate(messages))


def safe_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


class MemoraBuilder:
    def __init__(
        self,
        *,
        client: OpenAIClient,
        memory_model: str,
        embedding_model: str,
        consolidation_top_k: int = 5,
        consolidation_similarity_threshold: float = 0.78,
        segment_window_messages: int = 40,
    ):
        self.client = client
        self.memory_model = memory_model
        self.embedding_model = embedding_model
        self.consolidation_top_k = consolidation_top_k
        self.consolidation_similarity_threshold = consolidation_similarity_threshold
        self.segment_window_messages = segment_window_messages

    def build(
        self,
        conversation: Conversation,
        max_segments: int | None = None,
        existing_store: MemoryStore | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> MemoryStore:
        store = existing_store or MemoryStore(conversation_id=conversation.conversation_id)
        segments = self.segment_conversation(conversation, max_segments=max_segments)
        existing_segment_ids = {segment.id for segment in store.segments}
        for segment in segments:
            if segment.id not in existing_segment_ids:
                store.segments.append(segment)

        for segment in tqdm(segments, desc=f"MEMORA build {conversation.conversation_id}", leave=False):
            if any(episodic.segment_id == segment.id for episodic in store.episodic_memories):
                continue
            episodic = self.create_episodic_memory(segment)
            store.episodic_memories.append(episodic)
            factual_memories = self.extract_factual_memories(segment)
            for memory in factual_memories:
                entry = self.create_or_update_entry(
                    store=store,
                    conversation_id=conversation.conversation_id,
                    episodic_id=episodic.id,
                    primary_abstraction=memory["mem_index"],
                    memory_value=memory["mem_value"],
                )
                self.ensure_cues(store, entry)
            self.refresh_embeddings(store)
            if checkpoint_path is not None:
                save_store(store, checkpoint_path)
        self.refresh_embeddings(store)
        return store

    def segment_conversation(
        self, conversation: Conversation, max_segments: int | None = None
    ) -> list[Segment]:
        segments: list[Segment] = []
        messages = conversation.messages
        for window_start in range(0, len(messages), self.segment_window_messages):
            if max_segments is not None and len(segments) >= max_segments:
                break
            window = messages[window_start : window_start + self.segment_window_messages]
            response = self.client.chat_json(
                model=self.memory_model,
                messages=segment_messages_prompt(render_numbered_messages(window)),
                temperature=0.0,
            )
            for local_idx, episode in enumerate(response.get("episodes", [])):
                indices = [
                    int(i)
                    for i in episode.get("indices", [])
                    if isinstance(i, int) or str(i).isdigit()
                ]
                selected = [
                    window[i - 1]
                    for i in indices
                    if 1 <= i <= len(window)
                ]
                if not selected:
                    continue
                global_start = window_start + min(indices)
                segment_id = f"{conversation.conversation_id}:seg:{global_start}:{local_idx}"
                segments.append(
                    Segment(
                        id=segment_id,
                        conversation_id=conversation.conversation_id,
                        topic=str(episode.get("topic", "episode")),
                        message_ids=[message.dia_id for message in selected],
                        text="\n".join(message.render() for message in selected),
                        timestamp=selected[0].timestamp,
                    )
                )
                if max_segments is not None and len(segments) >= max_segments:
                    break
        return segments

    def create_episodic_memory(self, segment: Segment) -> EpisodicMemory:
        response = self.client.chat_json(
            model=self.memory_model,
            messages=episodic_memory_prompt(segment.text),
            temperature=0.0,
        )
        return EpisodicMemory(
            id=f"{segment.id}:episodic",
            conversation_id=segment.conversation_id,
            segment_id=segment.id,
            index=str(response.get("episodic_index") or segment.topic),
            value=str(response.get("episodic_value") or segment.text),
            raw_segment=segment.text,
        )

    def extract_factual_memories(self, segment: Segment) -> list[dict[str, str]]:
        response = self.client.chat_json(
            model=self.memory_model,
            messages=factual_memory_prompt(segment.text, segment.timestamp),
            temperature=0.0,
        )
        memories = []
        for item in response.get("memories", []):
            idx = str(item.get("mem_index", "")).strip()
            val = str(item.get("mem_value", "")).strip()
            if idx and val:
                memories.append({"mem_index": idx, "mem_value": val})
        return memories

    def create_or_update_entry(
        self,
        *,
        store: MemoryStore,
        conversation_id: str,
        episodic_id: str,
        primary_abstraction: str,
        memory_value: str,
    ) -> MemoryEntry:
        candidates = self.find_consolidation_candidates(store, primary_abstraction)
        if not candidates:
            entry = MemoryEntry(
                id=f"{conversation_id}:mem:{len(store.entries) + 1}",
                conversation_id=conversation_id,
                primary_abstraction=primary_abstraction,
                memory_value=memory_value,
                episodic_ids=[episodic_id],
            )
            store.entries.append(entry)
            store.abstraction_embeddings = []
            return entry

        response = self.client.chat_json(
            model=self.memory_model,
            messages=consolidation_prompt(
                primary_abstraction,
                memory_value,
                self.format_candidates(candidates),
            ),
            temperature=0.0,
        )
        action = str(response.get("action", "create")).lower()
        target_id = response.get("target_id")

        if action == "update" and target_id:
            for entry in store.entries:
                if entry.id == target_id:
                    entry.primary_abstraction = str(response.get("updated_index") or primary_abstraction)
                    entry.memory_value = str(response.get("updated_value") or memory_value)
                    if episodic_id not in entry.episodic_ids:
                        entry.episodic_ids.append(episodic_id)
                    entry.cue_anchors = []
                    store.abstraction_embeddings = []
                    return entry

        entry = MemoryEntry(
            id=f"{conversation_id}:mem:{len(store.entries) + 1}",
            conversation_id=conversation_id,
            primary_abstraction=str(response.get("updated_index") or primary_abstraction),
            memory_value=str(response.get("updated_value") or memory_value),
            episodic_ids=[episodic_id],
        )
        store.entries.append(entry)
        store.abstraction_embeddings = []
        return entry

    def find_consolidation_candidates(
        self, store: MemoryStore, primary_abstraction: str
    ) -> list[tuple[MemoryEntry, float]]:
        if not store.entries:
            return []
        if len(store.abstraction_embeddings) != len(store.entries):
            self.refresh_abstraction_embeddings(store)
        query_embedding = self.client.embed(model=self.embedding_model, texts=[primary_abstraction])[0]
        matches = top_k_cosine(query_embedding, store.abstraction_embeddings, self.consolidation_top_k)
        return [
            (store.entries[i], score)
            for i, score in matches
            if score >= self.consolidation_similarity_threshold
        ]

    @staticmethod
    def format_candidates(candidates: list[tuple[MemoryEntry, float]]) -> str:
        lines = []
        for entry, score in candidates:
            lines.append(
                f"ID: {entry.id}\nSimilarity: {score:.3f}\nIndex: {entry.primary_abstraction}\nValue: {entry.memory_value}"
            )
        return "\n\n".join(lines)

    def ensure_cues(self, store: MemoryStore, entry: MemoryEntry) -> None:
        response = self.client.chat_json(
            model=self.memory_model,
            messages=cue_anchor_prompt([(entry.primary_abstraction, entry.memory_value)]),
            temperature=0.0,
        )
        items = response.get("items", [])
        if items:
            cues = items[0].get("cue_anchors", [])
            entry.cue_anchors = [str(cue).strip() for cue in cues if str(cue).strip()]

    def refresh_abstraction_embeddings(self, store: MemoryStore) -> None:
        texts = [entry.primary_abstraction for entry in store.entries]
        store.abstraction_embeddings = self.client.embed(model=self.embedding_model, texts=texts) if texts else []

    def refresh_embeddings(self, store: MemoryStore) -> None:
        self.refresh_abstraction_embeddings(store)
        cues = sorted({cue for entry in store.entries for cue in entry.cue_anchors})
        if cues:
            embeddings = self.client.embed(model=self.embedding_model, texts=cues)
            store.cue_embeddings = dict(zip(cues, embeddings, strict=True))


def save_store(store: MemoryStore, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(store), ensure_ascii=False, indent=2), encoding="utf-8")


def load_store(path: str | Path) -> MemoryStore:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MemoryStore(
        conversation_id=data["conversation_id"],
        segments=[Segment(**row) for row in data.get("segments", [])],
        episodic_memories=[EpisodicMemory(**row) for row in data.get("episodic_memories", [])],
        entries=[MemoryEntry(**row) for row in data.get("entries", [])],
        abstraction_embeddings=data.get("abstraction_embeddings", []),
        cue_embeddings=data.get("cue_embeddings", {}),
    )
