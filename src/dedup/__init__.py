"""Semantic deduplication module."""

from .deduplicator import (
    FrameDeduplicator,
    GreedyCosineDedup,
    KMeansDedup,
    DeduplicationResult,
    create_deduplicator,
)

__all__ = [
    "FrameDeduplicator",
    "GreedyCosineDedup",
    "KMeansDedup",
    "DeduplicationResult",
    "create_deduplicator",
]

