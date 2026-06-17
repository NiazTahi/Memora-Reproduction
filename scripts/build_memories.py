from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from memora_repro.data.locomo import load_locomo
from memora_repro.llm.openai_client import OpenAIClient
from memora_repro.memory.builder import MemoraBuilder, load_store, save_store
from memora_repro.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/locomo.yaml", type=Path)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--conversation-id", action="append", default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    conversations = load_locomo("data/raw/locomo_repo/data/locomo10.json")
    if args.conversation_id:
        wanted = set(args.conversation_id)
        conversations = [
            conversation
            for conversation in conversations
            if conversation.conversation_id in wanted
        ]
    if args.limit_conversations is not None:
        conversations = conversations[: args.limit_conversations]

    client = OpenAIClient(
        cache_dir=config["paths"]["cache_dir"],
        seed=config["openai"].get("seed"),
    )
    builder = MemoraBuilder(
        client=client,
        memory_model=config["openai"]["memory_model"],
        embedding_model=config["openai"]["embedding_model"],
        consolidation_top_k=int(config["memora"]["consolidation_top_k"]),
        consolidation_similarity_threshold=float(
            config["memora"]["consolidation_similarity_threshold"]
        ),
    )

    out_dir = Path(config["paths"]["processed_data_dir"]) / "memora_stores"
    out_dir.mkdir(parents=True, exist_ok=True)

    for conversation in tqdm(conversations, desc="Conversations"):
        out_file = out_dir / f"{conversation.conversation_id}.json"
        if out_file.exists() and not args.overwrite:
            existing = load_store(out_file)
            all_done = len(existing.episodic_memories) >= len(existing.segments) and existing.segments
            if all_done and args.max_segments is None:
                continue
        else:
            existing = None
        store = builder.build(
            conversation,
            max_segments=args.max_segments,
            existing_store=existing,
            checkpoint_path=out_file,
        )
        save_store(store, out_file)
        print(
            f"saved {out_file}: "
            f"{len(store.segments)} segments, "
            f"{len(store.episodic_memories)} episodes, "
            f"{len(store.entries)} memories"
        )


if __name__ == "__main__":
    main()
