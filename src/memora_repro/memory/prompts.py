from __future__ import annotations


def segment_messages_prompt(messages: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are an expert conversation segmentation specialist. Output strict JSON.",
        },
        {
            "role": "user",
            "content": f"""
Read the conversation and segment it into coherent topical episodes.

Rules:
- Group consecutive messages around the same subject, event, or theme.
- Prefer episodes of 2-8 messages.
- Include all messages exactly once.
- Use 1-based message indices from the provided list.

Output JSON:
{{
  "episodes": [
    {{"topic": "brief topic description", "indices": [1, 2, 3]}}
  ]
}}

Conversation:
{messages}
""".strip(),
        },
    ]


def episodic_memory_prompt(content: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You generate concise episodic memory summaries. Output strict JSON.",
        },
        {
            "role": "user",
            "content": f"""
Generate an episodic memory from this conversation segment.

Output JSON:
{{
  "episodic_index": "6-8 word summary capturing main topic/entity/event",
  "episodic_value": "1-3 sentence self-contained summary"
}}

Use only information in the segment.

Segment:
{content}
""".strip(),
        },
    ]


def factual_memory_prompt(content: str, timestamp: str | None) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You extract factual memories from conversation segments. Output strict JSON.",
        },
        {
            "role": "user",
            "content": f"""
Extract all factual memories useful for future reference.

Rules:
- Use only explicitly supported information.
- Exclude greetings and filler.
- Split distinct facts into separate entries.
- Replace pronouns with specific names/entities.
- Convert relative dates using timestamp if possible.

Timestamp: {timestamp or "unknown"}

Output JSON:
{{
  "memories": [
    {{"mem_index": "short unambiguous memory index", "mem_value": "one or two factual sentences"}}
  ]
}}

Segment:
{content}
""".strip(),
        },
    ]


def consolidation_prompt(new_index: str, new_value: str, candidates: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You decide whether to update an existing memory or create a new one. Output strict JSON.",
        },
        {
            "role": "user",
            "content": f"""
Given a new memory and similar existing memories, decide whether to update one existing entry or create a new entry.

New memory:
Index: {new_index}
Value: {new_value}

Existing candidates:
{candidates or "None"}

Output JSON:
{{
  "action": "update" or "create",
  "target_id": "existing memory id or null",
  "updated_index": "best primary abstraction",
  "updated_value": "merged memory value"
}}
""".strip(),
        },
    ]


def cue_anchor_prompt(memories: list[tuple[str, str]]) -> list[dict[str, str]]:
    formatted = "\n\n".join(
        f"Memory {i + 1}\nPrimary Abstraction: {idx}\nMemory Value: {val}"
        for i, (idx, val) in enumerate(memories)
    )
    return [
        {
            "role": "system",
            "content": "You create short cue anchors for memory retrieval. Output strict JSON.",
        },
        {
            "role": "user",
            "content": f"""
For each memory, generate 1-3 short cue anchors.

Cue anchor rules:
- 2-4 words.
- Format: main entity/topic + key aspect.
- Do not repeat the primary abstraction exactly.
- Avoid generic words and near-duplicates.

Output JSON:
{{
  "items": [
    {{"memory_number": 1, "cue_anchors": ["cue one", "cue two"]}}
  ]
}}

Memories:
{formatted}
""".strip(),
        },
    ]

