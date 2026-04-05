"""
Vector retrieval module for semantic search.

Provides semantic search against Qdrant collections with filtering,
score thresholds, and graceful error handling.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from config import settings
from core.ingestion.embedder import DocumentEmbedder

logger = logging.getLogger(__name__)


class RetrieverError(Exception):
    """Base exception for retriever errors."""
    pass


class RetrieverTimeoutError(RetrieverError):
    """Raised when Qdrant request times out."""
    
    def __init__(self, collection: str, timeout_seconds: float) -> None:
        self.collection = collection
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Qdrant request to collection '{collection}' timed out after {timeout_seconds}s"
        )


class RetrieverUnavailableError(RetrieverError):
    """Raised when Qdrant is unavailable or unreachable."""
    
    def __init__(self, collection: str, reason: str) -> None:
        self.collection = collection
        self.reason = reason
        super().__init__(
            f"Qdrant collection '{collection}' unavailable: {reason}"
        )


@dataclass
class RetrievalResult:
    """
    Represents a single retrieval result from vector search.
    
    Attributes:
        chunk_id: Unique identifier for the chunk.
        text: The text content of the chunk.
        score: Similarity score (0-1 for cosine similarity).
        source_url: URL where the original content can be found.
        source_type: Type of source (stripe_doc, github_issue, etc.).
        title: Title of the parent document.
        date: Optional date associated with the content.
        metadata: Additional metadata from the chunk.
        is_potentially_stale: Whether this chunk may contain outdated info.
    """
    chunk_id: str
    text: str
    score: float
    source_url: str
    source_type: str
    title: str
    date: str | None
    metadata: dict = field(default_factory=dict)
    is_potentially_stale: bool = False


class VectorRetriever:
    """
    Handles semantic search against Qdrant vector collections.
    
    Provides filtered search with score thresholds and graceful
    error handling for timeouts and unavailability.
    """

    DEFAULT_TIMEOUT_SECONDS: float = 10.0

    def __init__(self) -> None:
        """Initialize retriever with Qdrant client and embedder."""
        try:
            self._qdrant = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=self.DEFAULT_TIMEOUT_SECONDS,
            )
            logger.info(f"Connected to Qdrant at {settings.QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise RetrieverUnavailableError("all", str(e))

        self._embedder = DocumentEmbedder()
        
        # Collection name mapping
        self._collections = {
            "stripe_docs": settings.QDRANT_DOCS_COLLECTION,
            "stripe_github_issues": settings.QDRANT_ISSUES_COLLECTION,
            "stripe_stackoverflow": settings.QDRANT_STACKOVERFLOW_COLLECTION,
        }

    def _build_qdrant_filter(
        self,
        filters: dict[str, Any] | None,
    ) -> qdrant_models.Filter | None:
        """
        Build Qdrant filter from user-provided filter dict.
        
        Supported filter keys:
        - source_type: Exact match on source type
        - date_after: Filter to chunks with date after this value
        - date_before: Filter to chunks with date before this value
        - is_resolution: Boolean filter for GitHub resolution comments
        - is_accepted: Boolean filter for SO accepted answers
        
        Args:
            filters: Dictionary of filter conditions.
            
        Returns:
            Qdrant Filter object or None if no filters.
        """
        if not filters:
            return None

        conditions: list[qdrant_models.FieldCondition] = []

        # Source type filter
        if "source_type" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="source_type",
                    match=qdrant_models.MatchValue(value=filters["source_type"]),
                )
            )

        # Date range filters
        if "date_after" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="date",
                    range=qdrant_models.Range(gte=filters["date_after"]),
                )
            )

        if "date_before" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="date",
                    range=qdrant_models.Range(lte=filters["date_before"]),
                )
            )

        # Boolean metadata filters
        if "is_resolution" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="metadata.is_resolution",
                    match=qdrant_models.MatchValue(value=filters["is_resolution"]),
                )
            )

        if "is_accepted" in filters:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="metadata.is_accepted",
                    match=qdrant_models.MatchValue(value=filters["is_accepted"]),
                )
            )

        if not conditions:
            return None

        return qdrant_models.Filter(must=conditions)

    def _parse_result(
        self,
        scored_point: qdrant_models.ScoredPoint,
    ) -> RetrievalResult:
        """
        Parse a Qdrant ScoredPoint into a RetrievalResult.
        
        Args:
            scored_point: Raw result from Qdrant search.
            
        Returns:
            Parsed RetrievalResult object.
        """
        payload = scored_point.payload or {}

        return RetrievalResult(
            chunk_id=payload.get("chunk_id", str(scored_point.id)),
            text=payload.get("text", ""),
            score=scored_point.score,
            source_url=payload.get("source_url", ""),
            source_type=payload.get("source_type", ""),
            title=payload.get("title", ""),
            date=payload.get("date"),
            metadata=payload.get("metadata", {}),
            is_potentially_stale=payload.get("potentially_stale", False),
        )

    async def search(
        self,
        query: str,
        collection: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Perform semantic search against a Qdrant collection.
        
        Args:
            query: The search query text.
            collection: Name of the Qdrant collection to search.
            limit: Maximum number of results to return.
            filters: Optional filter conditions (source_type, date_after, etc.).
            score_threshold: Minimum similarity score. Defaults to
                           config.RETRIEVAL_SIMILARITY_THRESHOLD.
                           
        Returns:
            List of RetrievalResult objects sorted by score descending.
            
        Raises:
            RetrieverTimeoutError: If Qdrant request times out.
            RetrieverUnavailableError: If Qdrant is unavailable.
        """
        if score_threshold is None:
            score_threshold = settings.RETRIEVAL_SIMILARITY_THRESHOLD

        # Resolve collection name
        resolved_collection = self._collections.get(collection, collection)

        logger.debug(
            f"Searching '{resolved_collection}' for: {query[:50]}... "
            f"(limit={limit}, threshold={score_threshold})"
        )

        try:
            # Generate query embedding
            query_embedding = self._embedder.embed_text(query)

            # Build filter
            qdrant_filter = self._build_qdrant_filter(filters)

            # Execute search with timeout
            search_result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._qdrant.search,
                    collection_name=resolved_collection,
                    query_vector=query_embedding,
                    limit=limit,
                    query_filter=qdrant_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                ),
                timeout=self.DEFAULT_TIMEOUT_SECONDS,
            )

            # Parse results
            results = [self._parse_result(point) for point in search_result]

            logger.info(
                f"Found {len(results)} results in '{resolved_collection}' "
                f"(query: {query[:30]}...)"
            )

            return results

        except asyncio.TimeoutError:
            logger.error(
                f"Timeout searching '{resolved_collection}' after "
                f"{self.DEFAULT_TIMEOUT_SECONDS}s"
            )
            raise RetrieverTimeoutError(resolved_collection, self.DEFAULT_TIMEOUT_SECONDS)

        except (ResponseHandlingException, UnexpectedResponse) as e:
            logger.error(f"Qdrant error searching '{resolved_collection}': {e}")
            raise RetrieverUnavailableError(resolved_collection, str(e))

        except Exception as e:
            logger.error(f"Unexpected error searching '{resolved_collection}': {e}")
            raise RetrieverUnavailableError(resolved_collection, str(e))

    async def search_all_collections(
        self,
        query: str,
        limit_per_collection: int = 3,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, list[RetrievalResult]]:
        """
        Search all collections simultaneously.
        
        Executes searches in parallel using asyncio.gather. If one collection
        fails, returns empty list for that collection and continues with others.
        
        Args:
            query: The search query text.
            limit_per_collection: Maximum results per collection.
            filters: Optional filter conditions applied to all collections.
            score_threshold: Minimum similarity score.
            
        Returns:
            Dictionary mapping collection names to lists of results.
            Collections that failed will have empty lists.
        """
        logger.info(f"Searching all collections for: {query[:50]}...")

        # Prepare search tasks for all collections
        collection_names = list(self._collections.keys())
        
        async def safe_search(collection: str) -> tuple[str, list[RetrievalResult]]:
            """
            Wrapper that catches exceptions and returns empty list on failure.
            """
            try:
                results = await self.search(
                    query=query,
                    collection=collection,
                    limit=limit_per_collection,
                    filters=filters,
                    score_threshold=score_threshold,
                )
                return (collection, results)
            except RetrieverError as e:
                logger.warning(f"Search failed for {collection}: {e}")
                return (collection, [])
            except Exception as e:
                logger.error(f"Unexpected error searching {collection}: {e}")
                return (collection, [])

        # Execute all searches in parallel
        tasks = [safe_search(collection) for collection in collection_names]
        results = await asyncio.gather(*tasks)

        # Build result dictionary
        result_dict = {collection: results_list for collection, results_list in results}

        # Log summary
        total_results = sum(len(r) for r in result_dict.values())
        collections_with_results = sum(1 for r in result_dict.values() if r)
        logger.info(
            f"Found {total_results} total results across "
            f"{collections_with_results}/{len(collection_names)} collections"
        )

        return result_dict

    async def get_chunk_by_id(
        self,
        chunk_id: str,
        collection: str,
    ) -> RetrievalResult | None:
        """
        Retrieve a specific chunk by its ID.
        
        Args:
            chunk_id: The chunk_id to look up.
            collection: Collection to search in.
            
        Returns:
            RetrievalResult if found, None otherwise.
        """
        resolved_collection = self._collections.get(collection, collection)

        try:
            # Search by chunk_id in payload
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    self._qdrant.scroll,
                    collection_name=resolved_collection,
                    scroll_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="chunk_id",
                                match=qdrant_models.MatchValue(value=chunk_id),
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                ),
                timeout=self.DEFAULT_TIMEOUT_SECONDS,
            )

            points, _ = results
            if points:
                # Create a mock ScoredPoint for parsing
                point = points[0]
                return RetrievalResult(
                    chunk_id=point.payload.get("chunk_id", str(point.id)),
                    text=point.payload.get("text", ""),
                    score=1.0,  # Direct lookup, no similarity score
                    source_url=point.payload.get("source_url", ""),
                    source_type=point.payload.get("source_type", ""),
                    title=point.payload.get("title", ""),
                    date=point.payload.get("date"),
                    metadata=point.payload.get("metadata", {}),
                    is_potentially_stale=point.payload.get("potentially_stale", False),
                )

            return None

        except Exception as e:
            logger.warning(f"Failed to get chunk {chunk_id}: {e}")
            return None


# Module-level instance
retriever = VectorRetriever()
