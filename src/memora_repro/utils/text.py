from __future__ import annotations


def rough_token_count(text: str) -> int:
    return max(1, len(text.split()))


def chunk_by_words(text: str, chunk_size: int, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks

