"""Retrieval module for semantic and keyword search."""

from core.retrieval.vector_retriever import (
    RetrievalResult,
    RetrieverError,
    RetrieverTimeoutError,
    RetrieverUnavailableError,
    VectorRetriever,
    retriever,
)
from core.retrieval.hybrid import (
    BM25Index,
    ChangelogEntry,
    HybridResult,
    HybridRetriever,
    HybridRetrieverError,
    hybrid_retriever,
)

__all__ = [
    "RetrievalResult",
    "RetrieverError",
    "RetrieverTimeoutError",
    "RetrieverUnavailableError",
    "VectorRetriever",
    "retriever",
    "BM25Index",
    "ChangelogEntry",
    "HybridResult",
    "HybridRetriever",
    "HybridRetrieverError",
    "hybrid_retriever",
]
