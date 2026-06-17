from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from memora_repro.data.locomo import flatten_qa, load_locomo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        default="data/raw/locomo_repo/data/locomo10.json",
        type=Path,
    )
    args = parser.parse_args()

    conversations = load_locomo(args.data_file)
    qas = flatten_qa(conversations)
    category_counts = Counter(qa.category_name for qa in qas)

    print(f"conversations: {len(conversations)}")
    print(f"qa items excluding adversarial: {len(qas)}")
    print(f"category counts: {dict(category_counts)}")
    for conversation in conversations[:1]:
        print(f"first conversation: {conversation.conversation_id}")
        print(f"messages: {len(conversation.messages)}")
        print(f"qa: {len(conversation.qa)}")
        print(conversation.messages[0].render())
        print(conversation.qa[0])


if __name__ == "__main__":
    main()

