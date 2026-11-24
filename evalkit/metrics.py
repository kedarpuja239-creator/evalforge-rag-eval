"""Retrieval metrics computed by hand. Ground truth is at document level (source ids),
so changing chunking does not invalidate the golden set."""
from __future__ import annotations


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant docs that appear in the top-k retrieved source ids."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1 if at least one relevant doc is in the top k, else 0."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / position of the first relevant doc (positions start at 1); 0 if none was retrieved."""
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
