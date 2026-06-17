from __future__ import annotations

import numpy as np


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def top_k_cosine(query: list[float], matrix: list[list[float]] | np.ndarray, k: int) -> list[tuple[int, float]]:
    if len(matrix) == 0:
        return []
    mat = normalize_matrix(np.asarray(matrix, dtype=np.float32))
    q = np.asarray(query, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    scores = mat @ q
    top_indices = np.argsort(-scores)[:k]
    return [(int(index), float(scores[index])) for index in top_indices]

