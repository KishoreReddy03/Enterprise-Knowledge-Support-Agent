"""
Hybrid retrieval module combining vector search with BM25 keyword search.

Uses Reciprocal Rank Fusion (RRF) to merge results from both methods,
providing better coverage for both semantic and exact keyword queries.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from rank_bm25 import BM25Okapi

from config import settings
from core.retrieval.vector_retriever import (
    RetrievalResult,
    RetrieverError,
    VectorRetriever,
)

logger = logging.getLogger(__name__)


class HybridRetrieverError(RetrieverError):
    """Raised when both vector and BM25 retrieval fail."""
    
    def __init__(self, vector_error: str, bm25_error: str) -> None:
        self.vector_error = vector_error
        self.bm25_error = bm25_error
        super().__init__(
            f"Both retrieval methods failed. Vector: {vector_error}, BM25: {bm25_error}"
        )


@dataclass
class ChangelogEntry:
    """
    Represents a changelog entry from search results.
    
    Attributes:
        text: The changelog entry text.
        date: Date of the changelog entry.
        source_url: URL to the changelog.
        affected_features: Features mentioned in the entry.
        chunk_id: Unique identifier for the chunk.
    """
    text: str
    date: str | None
    source_url: str
    affected_features: list[str] = field(default_factory=list)
    chunk_id: str = ""


@dataclass
class HybridResult(RetrievalResult):
    """
    Extended retrieval result with hybrid search metadata.
    
    Attributes:
        retrieval_method: How this result was found ('vector_only', 'bm25_only', 'hybrid').
        rrf_score: Reciprocal Rank Fusion score.
        vector_rank: Rank in vector search results (None if not found).
        bm25_rank: Rank in BM25 search results (None if not found).
    """
    retrieval_method: str = "hybrid"
    rrf_score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None


class BM25Index:
    """
    BM25 index for keyword search across chunks.
    
    Maintains an in-memory index of all chunks that can be
    rebuilt when ingestion updates occur.
    """

    def __init__(self) -> None:
        """Initialize empty BM25 index."""
        self._documents: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._is_built = False

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 indexing.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List of lowercase tokens.
        """
        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def build_from_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """
        Build BM25 index from chunk data.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and metadata.
        """
        logger.info(f"Building BM25 index from {len(chunks)} chunks...")
        
        self._documents = chunks
        self._tokenized_corpus = [
            self._tokenize(chunk.get("text", ""))
            for chunk in chunks
        ]
        
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            self._is_built = True
            logger.info(f"BM25 index built with {len(chunks)} documents")
        else:
            self._is_built = False
            logger.warning("BM25 index is empty - no documents to index")

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Search the BM25 index.
        
        Args:
            query: Search query.
            limit: Maximum results to return.
            
        Returns:
            List of (document, score) tuples sorted by score descending.
        """
        if not self._is_built or self._bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top results
        scored_docs: list[tuple[dict[str, Any], float]] = [
            (self._documents[i], float(scores[i]))
            for i in range(len(scores))
            if scores[i] > 0
        ]
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:limit]

    @property
    def is_built(self) -> bool:
        """Check if index is built and ready for search."""
        return self._is_built

    @property
    def document_count(self) -> int:
        """Return number of indexed documents."""
        return len(self._documents)


class HybridRetriever:
    """
    Combines vector search with BM25 keyword search using Reciprocal Rank Fusion.
    
    Provides better retrieval coverage by merging semantic similarity results
    with exact keyword matches.
    """

    # RRF constant (standard value from literature)
    RRF_K: int = 60

    # Default number of days for changelog lookback
    CHANGELOG_LOOKBACK_DAYS: int = 90

    def __init__(self) -> None:
        """
        Initialize hybrid retriever with vector search and BM25 index.
        """
        # Initialize vector retriever
        self._vector_retriever = VectorRetriever()
        
        # Initialize BM25 index
        self._bm25_index = BM25Index()
        
        # Initialize Qdrant client for index building
        self._qdrant = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        
        # Build initial index
        self._rebuild_bm25_index()
        
        logger.info("HybridRetriever initialized")

    def _rebuild_bm25_index(self) -> None:
        """
        Rebuild BM25 index from all Qdrant collections.
        
        Fetches all chunks from all collections and builds the index.
        Should be called after ingestion updates.
        """
        all_chunks: list[dict[str, Any]] = []
        
        collections = [
            settings.QDRANT_DOCS_COLLECTION,
            settings.QDRANT_ISSUES_COLLECTION,
            settings.QDRANT_STACKOVERFLOW_COLLECTION,
        ]
        
        for collection_name in collections:
            try:
                # Scroll through all points in collection
                offset = None
                while True:
                    results, offset = self._qdrant.scroll(
                        collection_name=collection_name,
                        limit=1000,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    
                    if not results:
                        break
                    
                    for point in results:
                        payload = point.payload or {}
                        all_chunks.append({
                            "text": payload.get("text", ""),
                            "chunk_id": payload.get("chunk_id", str(point.id)),
                            "source_url": payload.get("source_url", ""),
                            "source_type": payload.get("source_type", ""),
                            "title": payload.get("title", ""),
                            "date": payload.get("date"),
                            "metadata": payload.get("metadata", {}),
                            "potentially_stale": payload.get("potentially_stale", False),
                            "collection": collection_name,
                        })
                    
                    if offset is None:
                        break
                        
            except Exception as e:
                logger.warning(f"Failed to load chunks from {collection_name}: {e}")
        
        self._bm25_index.build_from_chunks(all_chunks)

    def refresh_index(self) -> None:
        """
        Refresh the BM25 index.
        
        Call this after ingestion updates to include new chunks.
        """
        logger.info("Refreshing BM25 index...")
        self._rebuild_bm25_index()

    def _calculate_rrf_scores(
        self,
        vector_results: list[RetrievalResult],
        bm25_results: list[tuple[dict[str, Any], float]],
    ) -> dict[str, dict[str, Any]]:
        """
        Calculate RRF scores for all results.
        
        Args:
            vector_results: Results from vector search.
            bm25_results: Results from BM25 search.
            
        Returns:
            Dictionary mapping chunk_id to merged result data with RRF score.
        """
        merged: dict[str, dict[str, Any]] = {}

        # Process vector results
        for rank, result in enumerate(vector_results):
            chunk_id = result.chunk_id
            rrf_score = 1 / (rank + 1 + self.RRF_K)
            
            merged[chunk_id] = {
                "chunk_id": chunk_id,
                "text": result.text,
                "source_url": result.source_url,
                "source_type": result.source_type,
                "title": result.title,
                "date": result.date,
                "metadata": result.metadata,
                "is_potentially_stale": result.is_potentially_stale,
                "original_score": result.score,
                "rrf_score": rrf_score,
                "vector_rank": rank + 1,
                "bm25_rank": None,
                "retrieval_method": "vector_only",
            }

        # Process BM25 results
        for rank, (doc, bm25_score) in enumerate(bm25_results):
            chunk_id = doc.get("chunk_id", "")
            rrf_contribution = 1 / (rank + 1 + self.RRF_K)
            
            if chunk_id in merged:
                # Found in both - update to hybrid
                merged[chunk_id]["rrf_score"] += rrf_contribution
                merged[chunk_id]["bm25_rank"] = rank + 1
                merged[chunk_id]["retrieval_method"] = "hybrid"
            else:
                # BM25 only
                merged[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": doc.get("text", ""),
                    "source_url": doc.get("source_url", ""),
                    "source_type": doc.get("source_type", ""),
                    "title": doc.get("title", ""),
                    "date": doc.get("date"),
                    "metadata": doc.get("metadata", {}),
                    "is_potentially_stale": doc.get("potentially_stale", False),
                    "original_score": bm25_score,
                    "rrf_score": rrf_contribution,
                    "vector_rank": None,
                    "bm25_rank": rank + 1,
                    "retrieval_method": "bm25_only",
                }

        return merged

    def _to_hybrid_results(
        self,
        merged: dict[str, dict[str, Any]],
        limit: int,
    ) -> list[HybridResult]:
        """
        Convert merged results to HybridResult objects.
        
        Args:
            merged: Dictionary of merged result data.
            limit: Maximum results to return.
            
        Returns:
            List of HybridResult objects sorted by RRF score.
        """
        # Sort by RRF score descending
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:limit]

        return [
            HybridResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["original_score"],
                source_url=r["source_url"],
                source_type=r["source_type"],
                title=r["title"],
                date=r["date"],
                metadata=r["metadata"],
                is_potentially_stale=r["is_potentially_stale"],
                retrieval_method=r["retrieval_method"],
                rrf_score=r["rrf_score"],
                vector_rank=r["vector_rank"],
                bm25_rank=r["bm25_rank"],
            )
            for r in sorted_results
        ]

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[HybridResult]:
        """
        Perform hybrid retrieval combining vector and BM25 search.
        
        Uses Reciprocal Rank Fusion to merge results from both methods.
        Falls back to single method if one fails.
        
        Args:
            query: The search query.
            limit: Maximum number of results to return.
            
        Returns:
            List of HybridResult objects with retrieval_method populated.
            
        Raises:
            HybridRetrieverError: If both retrieval methods fail.
        """
        expanded_limit = limit * 2
        vector_results: list[RetrievalResult] = []
        bm25_results: list[tuple[dict[str, Any], float]] = []
        vector_error: str | None = None
        bm25_error: str | None = None

        # Run vector search
        try:
            all_vector_results = await self._vector_retriever.search_all_collections(
                query=query,
                limit_per_collection=expanded_limit,
            )
            # Flatten results from all collections
            for results in all_vector_results.values():
                vector_results.extend(results)
            # Sort by score and limit
            vector_results.sort(key=lambda x: x.score, reverse=True)
            vector_results = vector_results[:expanded_limit]
            logger.debug(f"Vector search returned {len(vector_results)} results")
        except Exception as e:
            vector_error = str(e)
            logger.warning(f"Vector search failed: {e}")

        # Run BM25 search
        try:
            if self._bm25_index.is_built:
                bm25_results = self._bm25_index.search(query, limit=expanded_limit)
                logger.debug(f"BM25 search returned {len(bm25_results)} results")
            else:
                bm25_error = "BM25 index not built"
                logger.warning("BM25 index not available")
        except Exception as e:
            bm25_error = str(e)
            logger.warning(f"BM25 search failed: {e}")

        # Handle fallback cases
        if not vector_results and not bm25_results:
            raise HybridRetrieverError(
                vector_error or "No results",
                bm25_error or "No results",
            )

        if not vector_results:
            # BM25 only fallback
            logger.info("Using BM25-only results (vector search failed)")
            return self._bm25_only_results(bm25_results, limit)

        if not bm25_results:
            # Vector only fallback
            logger.info("Using vector-only results (BM25 search failed)")
            return self._vector_only_results(vector_results, limit)

        # Merge with RRF
        merged = self._calculate_rrf_scores(vector_results, bm25_results)
        results = self._to_hybrid_results(merged, limit)

        # Log retrieval method distribution
        method_counts = {}
        for r in results:
            method_counts[r.retrieval_method] = method_counts.get(r.retrieval_method, 0) + 1
        logger.info(f"Hybrid retrieval: {method_counts}")

        return results

    def _vector_only_results(
        self,
        vector_results: list[RetrievalResult],
        limit: int,
    ) -> list[HybridResult]:
        """Convert vector results to HybridResult with vector_only method."""
        return [
            HybridResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.score,
                source_url=r.source_url,
                source_type=r.source_type,
                title=r.title,
                date=r.date,
                metadata=r.metadata,
                is_potentially_stale=r.is_potentially_stale,
                retrieval_method="vector_only",
                rrf_score=1 / (i + 1 + self.RRF_K),
                vector_rank=i + 1,
                bm25_rank=None,
            )
            for i, r in enumerate(vector_results[:limit])
        ]

    def _bm25_only_results(
        self,
        bm25_results: list[tuple[dict[str, Any], float]],
        limit: int,
    ) -> list[HybridResult]:
        """Convert BM25 results to HybridResult with bm25_only method."""
        return [
            HybridResult(
                chunk_id=doc.get("chunk_id", ""),
                text=doc.get("text", ""),
                score=score,
                source_url=doc.get("source_url", ""),
                source_type=doc.get("source_type", ""),
                title=doc.get("title", ""),
                date=doc.get("date"),
                metadata=doc.get("metadata", {}),
                is_potentially_stale=doc.get("potentially_stale", False),
                retrieval_method="bm25_only",
                rrf_score=1 / (i + 1 + self.RRF_K),
                vector_rank=None,
                bm25_rank=i + 1,
            )
            for i, (doc, score) in enumerate(bm25_results[:limit])
        ]

    async def retrieve_with_changelog(
        self,
        query: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        """
        Retrieve knowledge base results with changelog context.
        
        Searches the main knowledge base using hybrid retrieval, then
        separately searches for recent changelog entries that may affect
        the query topic.
        
        Args:
            query: The search query.
            limit: Maximum knowledge base results to return.
            
        Returns:
            Dictionary with:
            - knowledge_base_results: List of HybridResult from main KB
            - changelog_results: List of ChangelogEntry from recent changes
            - has_recent_changes: Whether any relevant recent changes exist
        """
        # Get main knowledge base results
        kb_results = await self.retrieve(query, limit)

        # Search changelog with date filter
        changelog_results = await self._search_changelog(query)

        return {
            "knowledge_base_results": kb_results,
            "changelog_results": changelog_results,
            "has_recent_changes": len(changelog_results) > 0,
        }

    async def _search_changelog(
        self,
        query: str,
        limit: int = 5,
    ) -> list[ChangelogEntry]:
        """
        Search for recent changelog entries related to the query.
        
        Args:
            query: The search query.
            limit: Maximum changelog entries to return.
            
        Returns:
            List of ChangelogEntry objects.
        """
        cutoff_date = datetime.utcnow() - timedelta(days=self.CHANGELOG_LOOKBACK_DAYS)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")

        try:
            # Search with changelog filter
            results = await self._vector_retriever.search(
                query=query,
                collection=settings.QDRANT_DOCS_COLLECTION,
                limit=limit,
                filters={
                    "source_type": "changelog",
                    "date_after": cutoff_str,
                },
                score_threshold=0.6,  # Lower threshold for changelog
            )

            # Convert to ChangelogEntry
            changelog_entries: list[ChangelogEntry] = []
            for result in results:
                # Extract affected features from text (simple keyword extraction)
                affected_features = self._extract_features_from_text(result.text)
                
                changelog_entries.append(
                    ChangelogEntry(
                        text=result.text,
                        date=result.date,
                        source_url=result.source_url,
                        affected_features=affected_features,
                        chunk_id=result.chunk_id,
                    )
                )

            return changelog_entries

        except Exception as e:
            logger.warning(f"Changelog search failed: {e}")
            return []

    def _extract_features_from_text(self, text: str) -> list[str]:
        """
        Extract Stripe feature names from changelog text.
        
        Args:
            text: Changelog entry text.
            
        Returns:
            List of detected feature names.
        """
        # Common Stripe features to look for
        stripe_features = {
            "webhooks", "webhook",
            "payments", "payment intents", "payment methods",
            "subscriptions", "billing", "invoices",
            "customers", "customer portal",
            "checkout", "checkout sessions",
            "connect", "connected accounts", "transfers",
            "radar", "fraud",
            "disputes", "refunds",
            "payouts", "balance",
            "tax", "tax rates",
            "coupons", "promotions",
            "elements", "payment element",
            "api", "api version",
        }
        
        text_lower = text.lower()
        found_features = [
            feature for feature in stripe_features
            if feature in text_lower
        ]
        
        return list(set(found_features))


# Module-level instance
hybrid_retriever = HybridRetriever()
