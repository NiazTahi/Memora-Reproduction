from __future__ import annotations

from dataclasses import dataclass

from memora_repro.llm.openai_client import OpenAIClient
from memora_repro.memory.schemas import EpisodicMemory, MemoryEntry, MemoryStore
from memora_repro.retrieval.vector import top_k_cosine


@dataclass
class RetrievedMemory:
    entry: MemoryEntry
    score: float


def format_memory_context(store: MemoryStore, retrieved: list[RetrievedMemory]) -> str:
    episodic_by_id: dict[str, EpisodicMemory] = {
        episodic.id: episodic for episodic in store.episodic_memories
    }
    blocks = []
    for item in retrieved:
        entry = item.entry
        episodes = [episodic_by_id[eid] for eid in entry.episodic_ids if eid in episodic_by_id]
        episode_text = "\n".join(
            f"Episodic Context: {episode.index}\n{episode.value}" for episode in episodes[:2]
        )
        cues = ", ".join(entry.cue_anchors)
        blocks.append(
            f"Primary Abstraction: {entry.primary_abstraction}\n"
            f"Memory Value: {entry.memory_value}\n"
            f"Cue Anchors: {cues}\n"
            f"{episode_text}".strip()
        )
    return "\n\n---\n\n".join(blocks)


class MemoraSemanticRetriever:
    def __init__(
        self,
        *,
        client: OpenAIClient,
        embedding_model: str,
        top_k_abstractions: int,
        top_k_cues: int,
        max_memories: int,
    ):
        self.client = client
        self.embedding_model = embedding_model
        self.top_k_abstractions = top_k_abstractions
        self.top_k_cues = top_k_cues
        self.max_memories = max_memories

    def retrieve(self, query: str, store: MemoryStore) -> list[RetrievedMemory]:
        query_embedding = self.client.embed(model=self.embedding_model, texts=[query])[0]
        scores: dict[str, float] = {}

        for index, score in top_k_cosine(
            query_embedding, store.abstraction_embeddings, self.top_k_abstractions
        ):
            scores[store.entries[index].id] = max(scores.get(store.entries[index].id, 0.0), score)

        cues = list(store.cue_embeddings.keys())
        cue_embeddings = [store.cue_embeddings[cue] for cue in cues]
        for cue_index, score in top_k_cosine(query_embedding, cue_embeddings, self.top_k_cues):
            cue = cues[cue_index]
            for entry in store.entries:
                if cue in entry.cue_anchors:
                    scores[entry.id] = max(scores.get(entry.id, 0.0), score)

        by_id = {entry.id: entry for entry in store.entries}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            RetrievedMemory(entry=by_id[entry_id], score=score)
            for entry_id, score in ranked[: self.max_memories]
            if entry_id in by_id
        ]


def policy_messages(
    query: str,
    working: list[RetrievedMemory],
    frontier: list[RetrievedMemory],
    step: int,
    frontier_limit: int = 20,
) -> list[dict[str, str]]:
    working_text = "\n".join(
        f"{i + 1}. ID={item.entry.id}; {item.entry.primary_abstraction}: {item.entry.memory_value}"
        for i, item in enumerate(working)
    ) or "None"
    shown_frontier = frontier[:frontier_limit]
    frontier_text = "\n".join(
        f"{i + 1}. ID={item.entry.id}; score={item.score:.3f}; {item.entry.primary_abstraction}: {item.entry.memory_value}"
        for i, item in enumerate(shown_frontier)
    ) or "None"
    return [
        {
            "role": "system",
            "content": "You are a memory retrieval policy. Output strict JSON.",
        },
        {
            "role": "user",
            "content": f"""
Question:
{query}

Step: {step}

Working set:
{working_text}

Frontier candidates:
{frontier_text}

Choose one action:
- EXPAND: select useful frontier memory IDs.
- REFINE: provide a better search query.
- STOP: enough evidence has been gathered.

Output JSON:
{{
  "action": "EXPAND" or "REFINE" or "STOP",
  "selected_memory_ids": ["id1", "id2"],
  "refined_query": "new query or null"
}}
""".strip(),
        },
    ]


class MemoraPolicyRetriever:
    def __init__(
        self,
        *,
        client: OpenAIClient,
        policy_model: str,
        semantic_retriever: MemoraSemanticRetriever,
        max_steps: int,
        expand_k: int,
    ):
        self.client = client
        self.policy_model = policy_model
        self.semantic_retriever = semantic_retriever
        self.max_steps = max_steps
        self.expand_k = expand_k

    def retrieve(self, query: str, store: MemoryStore) -> list[RetrievedMemory]:
        current_query = query
        working: list[RetrievedMemory] = []
        working_ids: set[str] = set()
        frontier = self.semantic_retriever.retrieve(current_query, store)

        for step in range(self.max_steps):
            try:
                response = self.client.chat_json(
                    model=self.policy_model,
                    messages=policy_messages(current_query, working, frontier, step),
                    temperature=0.0,
                )
            except Exception:
                response = {
                    "action": "EXPAND",
                    "selected_memory_ids": [item.entry.id for item in frontier[: self.expand_k]],
                    "refined_query": None,
                }
            action = str(response.get("action", "STOP")).upper()
            if action == "STOP":
                break
            if action == "REFINE":
                refined = response.get("refined_query")
                if refined:
                    current_query = str(refined)
                    frontier = self.semantic_retriever.retrieve(current_query, store)
                    frontier = [item for item in frontier if item.entry.id not in working_ids]
                continue
            if action == "EXPAND":
                selected_ids = set(response.get("selected_memory_ids") or [])
                if selected_ids:
                    selected = [item for item in frontier if item.entry.id in selected_ids]
                else:
                    selected = frontier[: self.expand_k]
                for item in selected[: self.expand_k]:
                    if item.entry.id not in working_ids:
                        working.append(item)
                        working_ids.add(item.entry.id)
                frontier = self.expand_frontier(store, working_ids, selected)

        if not working:
            working = frontier[: self.semantic_retriever.max_memories]
        return working[: self.semantic_retriever.max_memories]

    def expand_frontier(
        self, store: MemoryStore, working_ids: set[str], selected: list[RetrievedMemory]
    ) -> list[RetrievedMemory]:
        selected_cues = {cue for item in selected for cue in item.entry.cue_anchors}
        selected_episodes = {eid for item in selected for eid in item.entry.episodic_ids}
        candidates = []
        for entry in store.entries:
            if entry.id in working_ids:
                continue
            cue_overlap = len(selected_cues.intersection(entry.cue_anchors))
            episode_overlap = len(selected_episodes.intersection(entry.episodic_ids))
            if cue_overlap or episode_overlap:
                candidates.append(RetrievedMemory(entry=entry, score=float(cue_overlap + episode_overlap)))
        return sorted(candidates, key=lambda item: item.score, reverse=True)
