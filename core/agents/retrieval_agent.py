"""
Retrieval agent for parallel source querying.

This is Level 1 in the orchestration hierarchy. Queries all sources
simultaneously to meet the 20-second response time target.
"""

import asyncio
import logging
import time
from typing import Any

from langfuse import observe

from config import settings
from core.agents.state import SourceResult, TicketState
from core.retrieval.hybrid import HybridRetriever
from core.retrieval.vector_retriever import (
    RetrievalResult,
    RetrieverError,
    VectorRetriever,
)

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Queries all knowledge sources in parallel.
    
    Retrieves relevant information from Stripe docs, GitHub issues,
    StackOverflow, and changelog simultaneously using asyncio.gather.
    """

    # Results per source
    RESULTS_PER_SOURCE: int = 5

    def __init__(self) -> None:
        """Initialize retrieval agent with retrievers."""
        self._vector_retriever = VectorRetriever()
        self._hybrid_retriever = HybridRetriever()
        logger.info("RetrievalAgent initialized")

    def _build_query(self, state: TicketState) -> str:
        """
        Build optimized search query from ticket state.
        
        Combines ticket content with error codes and topic keywords
        for better retrieval relevance.
        
        Args:
            state: Current ticket state.
            
        Returns:
            Optimized search query string.
        """
        parts: list[str] = []
        
        # Prepend error codes if present
        error_codes = state.get("error_codes", [])
        if error_codes:
            parts.append(" ".join(error_codes))
        
        # Add main ticket content
        ticket_content = state.get("ticket_content", "")
        parts.append(ticket_content)
        
        # Append topic keyword if known and not 'other'
        primary_topic = state.get("primary_topic", "other")
        if primary_topic and primary_topic != "other":
            # Only append if not already in content
            if primary_topic.lower() not in ticket_content.lower():
                parts.append(primary_topic)
        
        query = " ".join(parts)
        logger.debug(f"Built query: {query[:100]}...")
        return query

    def _convert_to_source_result(
        self,
        result: RetrievalResult,
    ) -> SourceResult:
        """
        Convert RetrievalResult to SourceResult for state storage.
        
        Args:
            result: RetrievalResult from retriever.
            
        Returns:
            SourceResult dict for state.
        """
        return SourceResult(
            chunk_id=result.chunk_id,
            text=result.text,
            score=result.score,
            source_url=result.source_url,
            source_type=result.source_type,
            title=result.title,
            date=result.date,
            is_stale=result.is_potentially_stale,
            retrieval_method=getattr(result, "retrieval_method", "vector"),
        )

    async def _search_stripe_docs(
        self,
        query: str,
    ) -> list[SourceResult]:
        """
        Search Stripe documentation.
        
        Args:
            query: Search query.
            
        Returns:
            List of SourceResult from docs.
        """
        results = await self._vector_retriever.search(
            query=query,
            collection=settings.QDRANT_DOCS_COLLECTION,
            limit=self.RESULTS_PER_SOURCE,
            filters={"source_type": "stripe_doc"},
        )
        return [self._convert_to_source_result(r) for r in results]

    async def _search_github_issues(
        self,
        query: str,
    ) -> list[SourceResult]:
        """
        Search GitHub issues.
        
        Args:
            query: Search query.
            
        Returns:
            List of SourceResult from GitHub.
        """
        results = await self._vector_retriever.search(
            query=query,
            collection=settings.QDRANT_ISSUES_COLLECTION,
            limit=self.RESULTS_PER_SOURCE,
        )
        return [self._convert_to_source_result(r) for r in results]

    async def _search_stackoverflow(
        self,
        query: str,
    ) -> list[SourceResult]:
        """
        Search StackOverflow questions and answers.
        
        Args:
            query: Search query.
            
        Returns:
            List of SourceResult from StackOverflow.
        """
        results = await self._vector_retriever.search(
            query=query,
            collection=settings.QDRANT_STACKOVERFLOW_COLLECTION,
            limit=self.RESULTS_PER_SOURCE,
        )
        return [self._convert_to_source_result(r) for r in results]

    async def _search_changelog(
        self,
        query: str,
        topic: str,
    ) -> list[SourceResult]:
        """
        Search changelog for recent changes.
        
        Args:
            query: Search query.
            topic: Primary topic for filtering.
            
        Returns:
            List of SourceResult from changelog.
        """
        # Add topic to query for changelog search
        changelog_query = f"{query} {topic}" if topic != "other" else query
        
        results = await self._vector_retriever.search(
            query=changelog_query,
            collection=settings.QDRANT_DOCS_COLLECTION,
            limit=self.RESULTS_PER_SOURCE,
            filters={"source_type": "changelog"},
            score_threshold=0.6,  # Lower threshold for changelog
        )
        return [self._convert_to_source_result(r) for r in results]

    @observe(name="retrieval_agent")
    async def process(self, state: TicketState) -> TicketState:
        """
        Query all sources in parallel.
        
        Executes searches against all 4 sources simultaneously using
        asyncio.gather. Failures are captured in retrieval_errors;
        this method never raises exceptions.
        
        Args:
            state: Current ticket state.
            
        Returns:
            Updated state with retrieval results.
        """
        start_time = time.time()
        query = self._build_query(state)
        primary_topic = state.get("primary_topic", "other")
        
        # Initialize result containers
        docs_results: list[SourceResult] = []
        github_results: list[SourceResult] = []
        stackoverflow_results: list[SourceResult] = []
        changelog_results: list[SourceResult] = []
        retrieval_errors: list[str] = []

        # Run all searches in parallel
        try:
            results = await asyncio.gather(
                self._search_stripe_docs(query),
                self._search_github_issues(query),
                self._search_stackoverflow(query),
                self._search_changelog(query, primary_topic),
                return_exceptions=True,
            )

            # Process docs results
            if isinstance(results[0], Exception):
                error_msg = f"docs: {type(results[0]).__name__}: {results[0]}"
                retrieval_errors.append(error_msg)
                logger.warning(f"Docs retrieval failed: {results[0]}")
            else:
                docs_results = results[0]

            # Process GitHub results
            if isinstance(results[1], Exception):
                error_msg = f"github: {type(results[1]).__name__}: {results[1]}"
                retrieval_errors.append(error_msg)
                logger.warning(f"GitHub retrieval failed: {results[1]}")
            else:
                github_results = results[1]

            # Process StackOverflow results
            if isinstance(results[2], Exception):
                error_msg = f"stackoverflow: {type(results[2]).__name__}: {results[2]}"
                retrieval_errors.append(error_msg)
                logger.warning(f"StackOverflow retrieval failed: {results[2]}")
            else:
                stackoverflow_results = results[2]

            # Process changelog results
            if isinstance(results[3], Exception):
                error_msg = f"changelog: {type(results[3]).__name__}: {results[3]}"
                retrieval_errors.append(error_msg)
                logger.warning(f"Changelog retrieval failed: {results[3]}")
            else:
                changelog_results = results[3]

        except Exception as e:
            # This should never happen with return_exceptions=True
            # but we catch it just in case
            error_msg = f"gather_failed: {type(e).__name__}: {e}"
            retrieval_errors.append(error_msg)
            logger.error(f"asyncio.gather failed unexpectedly: {e}")

        # Update state with results
        state["docs_results"] = docs_results
        state["github_results"] = github_results
        state["stackoverflow_results"] = stackoverflow_results
        state["changelog_results"] = changelog_results
        state["retrieval_errors"] = retrieval_errors

        # Set derived flags
        state["has_recent_changes"] = len(changelog_results) > 0
        state["has_stale_content"] = any(
            r.get("is_stale", False)
            for r in docs_results + github_results + stackoverflow_results
        )

        # Track agent path
        state["agent_path"] = state.get("agent_path", []) + ["retrieval"]

        # Log metrics
        elapsed = time.time() - start_time
        total_results = (
            len(docs_results)
            + len(github_results)
            + len(stackoverflow_results)
            + len(changelog_results)
        )
        
        logger.info(
            f"Retrieval complete in {elapsed:.2f}s: "
            f"docs={len(docs_results)}, github={len(github_results)}, "
            f"so={len(stackoverflow_results)}, changelog={len(changelog_results)}, "
            f"errors={len(retrieval_errors)}"
        )

        # Log to error_log if any failures
        if retrieval_errors:
            state["error_log"] = state.get("error_log", []) + [
                f"retrieval: {len(retrieval_errors)} source(s) failed"
            ]

        return state

    def has_sufficient_results(self, state: TicketState) -> bool:
        """
        Check if we have enough results to proceed.
        
        Args:
            state: Current ticket state.
            
        Returns:
            True if we have at least some results from any source.
        """
        total = (
            len(state.get("docs_results", []))
            + len(state.get("github_results", []))
            + len(state.get("stackoverflow_results", []))
        )
        return total > 0

    def get_result_summary(self, state: TicketState) -> dict[str, Any]:
        """
        Get a summary of retrieval results for logging/tracing.
        
        Args:
            state: Current ticket state.
            
        Returns:
            Summary dict with counts and flags.
        """
        return {
            "docs_count": len(state.get("docs_results", [])),
            "github_count": len(state.get("github_results", [])),
            "stackoverflow_count": len(state.get("stackoverflow_results", [])),
            "changelog_count": len(state.get("changelog_results", [])),
            "error_count": len(state.get("retrieval_errors", [])),
            "has_recent_changes": state.get("has_recent_changes", False),
            "has_stale_content": state.get("has_stale_content", False),
        }


# Module-level instance
retrieval_agent = RetrievalAgent()
