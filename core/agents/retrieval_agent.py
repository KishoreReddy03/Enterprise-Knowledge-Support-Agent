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
from core.ingestion.embedder import DocumentEmbedder
from core.retrieval.hybrid import HybridRetriever
from core.retrieval.reranker import CrossEncoderReranker
from core.retrieval.vector_retriever import (
    RetrievalResult,
    RetrieverError,
)

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Queries all knowledge sources in parallel.

    Retrieves relevant information from Stripe docs, GitHub issues,
    StackOverflow, and changelog simultaneously using asyncio.gather.

    Retrieval depth adapts dynamically based on ticket complexity,
    urgency, and intake confidence from the IntakeAgent classification.
    """

    # Dynamic retrieval total budget by complexity
    _BUDGET_BY_COMPLEXITY: dict[str, int] = {
        "simple": 6,      # Targeted retrieval - fewer sources, low latency
        "moderate": 14,   # Balanced retrieval
        "complex": 24,    # Broad retrieval - wide search net across all sources
    }

    # Urgency multipliers applied on top of total budget
    _URGENCY_MULTIPLIER: dict[str, float] = {
        "low": 0.8,       # Low urgency - slightly reduced budget
        "medium": 1.0,    # Baseline
        "high": 1.2,      # Broad search - customer blocked
        "critical": 1.5,  # Maximum budget - production issue
    }

    def __init__(self) -> None:
        """Initialize retrieval agent with retrievers and cross-encoder reranker."""
        self._hybrid_retriever = HybridRetriever()
        self._embedder = DocumentEmbedder()
        self._reranker = CrossEncoderReranker()
        logger.info("RetrievalAgent initialized")

    def _allocate_retrieval_budget(self, state: TicketState) -> dict[str, int]:
        """
        Dynamically allocate the total retrieval result budget across all sources.

        Logic:
          1. Determine the total chunk budget based on complexity, urgency, and confidence.
          2. Distribute the total budget proportionally among sources based on complexity
             and the ticket's primary topic.
          3. Enforce a minimum limit (e.g. 3) for active sources, or 0 to skip entirely.
        """
        complexity = state.get("complexity", "moderate")
        urgency = state.get("urgency", "medium")
        confidence = state.get("intake_confidence", 0.75)
        topic = state.get("primary_topic", "other")

        base_budget = self._BUDGET_BY_COMPLEXITY.get(complexity, 14)
        multiplier = self._URGENCY_MULTIPLIER.get(urgency, 1.0)
        total_budget = int(base_budget * multiplier)

        # Low classification confidence -> boost budget to search wider
        if confidence < 0.6:
            total_budget += 4

        # Clamp total budget: 4 to 35 chunks total
        total_budget = max(4, min(35, total_budget))

        # Base percentages based on complexity
        if complexity == "simple":
            # Targeted: focus 80% on Stripe Docs, 20% on Changelog. Skip community.
            shares = {
                "stripe_docs": 0.80,
                "stripe_changelogs": 0.20,
                "stripe_github_issues": 0.0,
                "stripe_stackoverflow": 0.0,
            }
        elif complexity == "moderate":
            # Balanced: Docs, GitHub, SO, Changelog
            shares = {
                "stripe_docs": 0.50,
                "stripe_github_issues": 0.25,
                "stripe_stackoverflow": 0.15,
                "stripe_changelogs": 0.10,
            }
        else:
            # Complex: Broad search, high limits on everything
            shares = {
                "stripe_docs": 0.40,
                "stripe_github_issues": 0.30,
                "stripe_stackoverflow": 0.20,
                "stripe_changelogs": 0.10,
            }

        # Fine-tune shares based on primary topic
        if topic in ("billing", "auth"):
            # Billing and auth are official/regulatory domains: shift from community to docs/changelog
            shares["stripe_docs"] += 0.10
            shares["stripe_changelogs"] += 0.05
            shares["stripe_github_issues"] = max(0.0, shares["stripe_github_issues"] - 0.08)
            shares["stripe_stackoverflow"] = max(0.0, shares["stripe_stackoverflow"] - 0.07)
        elif topic in ("webhook", "connect", "api"):
            # Webhook, connect, and API are integration/code-heavy: shift from docs to community
            shares["stripe_github_issues"] += 0.10
            shares["stripe_stackoverflow"] += 0.05
            shares["stripe_docs"] = max(0.20, shares["stripe_docs"] - 0.15)

        # Re-normalize shares to sum to 1.0 (excluding skipped sources)
        total_shares = sum(shares.values())
        if total_shares > 0:
            for k in shares:
                shares[k] /= total_shares

        # Allocate limits
        allocations: dict[str, int] = {}
        for source_name, share in shares.items():
            if share <= 0.0:
                allocations[source_name] = 0
            else:
                allocated = int(total_budget * share)
                # Enforce minimum of 3 for active sources, or 0 if skipped
                allocations[source_name] = max(3, allocated) if allocated > 0 else 0

        logger.info(
            f"[BUDGET ALLOCATION] complexity={complexity}, urgency={urgency}, topic={topic} | "
            f"total_budget={total_budget} -> {allocations}"
        )
        return allocations

    def _build_query(self, state: TicketState) -> str:
        """
        Build optimized search query from ticket state.
        
        Combines extracted search_keywords with error codes and topic keywords
        for highly relevant retrieval.
        """
        rewritten = state.get("rewritten_query", "")
        # If there's a rewritten query and we're in a multi-turn conversation and it's not a topic shift:
        if rewritten and state.get("chat_history") and not state.get("topic_shift", False):
            logger.info(f"[RETRIEVAL QUERY] Using rephrased query: '{rewritten}'")
            return rewritten

        parts: list[str] = []
        
        # Prepend error codes if present
        error_codes = state.get("error_codes", [])
        if error_codes:
            parts.append(" ".join(error_codes))
            
        # Use extracted search keywords for cleaner vector/FTS search
        search_keywords = state.get("search_keywords", [])
        if search_keywords:
            parts.append(" ".join(search_keywords))
        else:
            # Fallback to full ticket content if no keywords extracted
            ticket_content = state.get("ticket_content", "")
            parts.append(ticket_content)
        
        # Append topic keyword if known and not 'other'
        primary_topic = state.get("primary_topic", "other")
        if primary_topic and primary_topic != "other":
            parts.append(primary_topic)
        
        query = " ".join(parts)
        logger.debug(f"Built query: {query[:100]}...")
        return query

    def _apply_dynamic_source_weights(
        self,
        scored_pairs: list[tuple[RetrievalResult, float]],
        state: TicketState,
    ) -> list[tuple[RetrievalResult, float]]:
        """
        Dynamically adjust cross-encoder relevance scores based on source trust.
        
        Applies bonuses/penalties depending on:
          1. Base source trust (stripe_docs/changelogs vs community)
          2. Ticket topic & complexity (e.g. billing demands official, dev integration trusts GitHub issues)
          3. Staleness / Outdated content penalties
        """
        complexity = state.get("complexity", "moderate")
        topic = state.get("primary_topic", "other")
        has_recent_changes = state.get("has_recent_changes", False)

        adjusted_pairs = []
        for result, raw_score in scored_pairs:
            source = result.source_type
            is_stale = getattr(result, "is_potentially_stale", False)
            if not is_stale and isinstance(getattr(result, "metadata", None), dict):
                is_stale = result.metadata.get("is_stale", False)
            
            # 1. Base trust adjustment
            trust_adjustment = 0.0
            if source == "stripe_docs":
                trust_adjustment += 1.5  # Highly trusted official docs
            elif source == "stripe_changelogs":
                trust_adjustment += 1.2  # Trusted official changelogs
            elif source == "stripe_github_issues":
                trust_adjustment += 0.5  # Moderate trust for engineering/github issues
            elif source == "stripe_stackoverflow":
                trust_adjustment += 0.0  # Baseline trust for StackOverflow

            # 2. Dynamic topic & complexity weighting
            if topic in ("billing", "auth"):
                # Financial/regulatory topics: heavily penalize community sources
                if source in ("stripe_github_issues", "stripe_stackoverflow"):
                    trust_adjustment -= 1.2
            elif topic in ("webhook", "connect", "api"):
                # Integration/code-heavy topics: boost engineering/github issues trust
                if source == "stripe_github_issues":
                    trust_adjustment += 0.8
                elif source == "stripe_stackoverflow":
                    trust_adjustment += 0.3

            if complexity == "simple":
                # Simple queries should stay extremely focused on official documentation
                if source in ("stripe_github_issues", "stripe_stackoverflow"):
                    trust_adjustment -= 1.5

            # 3. Staleness / Outdated content penalties
            if is_stale:
                trust_adjustment -= 2.0
                if source in ("stripe_github_issues", "stripe_stackoverflow"):
                    trust_adjustment -= 1.0  # Stale community Q&A is doubly untrusted
            
            if has_recent_changes and source in ("stripe_github_issues", "stripe_stackoverflow"):
                trust_adjustment -= 0.8

            adjusted_score = raw_score + trust_adjustment
            result.score = adjusted_score
            adjusted_pairs.append((result, adjusted_score))

        return adjusted_pairs

    def _rerank_cross_source(
        self,
        query_text: str,
        all_results: list[SourceResult],
        top_k: int,
        state: TicketState,
    ) -> list[SourceResult]:
        """
        Rerank merged results from all sources using the Cross-Encoder and dynamic source trust weights.

        Converts SourceResult dicts → lightweight RetrievalResult objects,
        runs the cross-encoder scoring, applies dynamic source trust adjustments,
        and returns the top-k adjusted results sorted by adjusted score descending.

        Args:
            query_text: Original query string used as the reranking anchor.
            all_results: Flat merged list from all 4 source searches.
            top_k: Maximum number of results to return after reranking.
            state: Current ticket state containing classification details.

        Returns:
            Reranked list of SourceResult, best first based on adjusted scores.
        """
        if not all_results:
            return []

        # Wrap SourceResult dicts as RetrievalResult for the cross-encoder API
        retrieval_results = [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
                source_url=r.get("source_url", ""),
                source_type=r.get("source_type", ""),
                title=r.get("title", ""),
                date=r.get("date"),
                is_potentially_stale=r.get("is_stale", False),
                retrieval_method=r.get("retrieval_method", "hybrid"),
            )
            for r in all_results
        ]

        # Get raw cross-encoder scores (no truncation yet to ensure we score all candidates)
        scored_pairs = self._reranker.score_pairs(
            query=query_text,
            results=retrieval_results,
        )

        # Apply learned/dynamic source weighting adjustments
        adjusted_pairs = self._apply_dynamic_source_weights(scored_pairs, state)

        # Sort descending by adjusted score
        adjusted_pairs.sort(key=lambda x: x[1], reverse=True)

        # Map the top-k adjusted results back to SourceResult
        id_to_source: dict[str, SourceResult] = {
            r["chunk_id"]: r for r in all_results
        }
        reranked_source_results: list[SourceResult] = []
        for r, adj_score in adjusted_pairs[:top_k]:
            source = id_to_source.get(r.chunk_id)
            if source:
                # Update score with the more precise adjusted score
                updated = dict(source)
                updated["score"] = adj_score
                reranked_source_results.append(SourceResult(**updated))

        logger.info(
            f"[RERANK] Cross-source reranking with dynamic trust: "
            f"{len(all_results)} candidates → top {len(reranked_source_results)}"
        )
        return reranked_source_results

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
            retrieval_method=getattr(result, "retrieval_method", "hybrid"),
        )

    async def _search_stripe_docs(
        self,
        query_text: str,
        query_vector: list[float],
        limit: int,
    ) -> list[SourceResult]:
        """
        Search Stripe documentation.

        Args:
            query_text: Text query for hybrid search.
            query_vector: Embedding vector for semantic search.
            limit: Number of results to return (adaptive).

        Returns:
            List of SourceResult from docs.
        """
        print("EXECUTION HIT: RetrievalAgent._search_stripe_docs()")
        if limit <= 0:
            logger.info("Skipping Stripe docs search (budget allocated: 0)")
            return []
        try:
            results = await self._hybrid_retriever.search(
                query_text=query_text,
                query_vector=query_vector,
                collection_name="stripe_docs",
                limit=limit,
            )
            return [self._convert_to_source_result(r) for r in results]
        except Exception as e:
            logger.error(f"Stripe docs search failed: {e}")
            return []

    async def _search_github_issues(
        self,
        query_text: str,
        query_vector: list[float],
        limit: int,
    ) -> list[SourceResult]:
        """
        Search GitHub issues.

        Args:
            query_text: Text query for hybrid search.
            query_vector: Embedding vector for semantic search.
            limit: Number of results to return (adaptive).

        Returns:
            List of SourceResult from GitHub.
        """
        print("EXECUTION HIT: RetrievalAgent._search_github_issues()")
        if limit <= 0:
            logger.info("Skipping GitHub issues search (budget allocated: 0)")
            return []
        try:
            results = await self._hybrid_retriever.search(
                query_text=query_text,
                query_vector=query_vector,
                collection_name="stripe_github_issues",
                limit=limit,
            )
            return [self._convert_to_source_result(r) for r in results]
        except Exception as e:
            logger.error(f"GitHub issues search failed: {e}")
            return []

    async def _search_stackoverflow(
        self,
        query_text: str,
        query_vector: list[float],
        limit: int,
    ) -> list[SourceResult]:
        """
        Search StackOverflow questions and answers.

        Args:
            query_text: Text query for hybrid search.
            query_vector: Embedding vector for semantic search.
            limit: Number of results to return (adaptive).

        Returns:
            List of SourceResult from StackOverflow.
        """
        print("EXECUTION HIT: RetrievalAgent._search_stackoverflow()")
        if limit <= 0:
            logger.info("Skipping StackOverflow search (budget allocated: 0)")
            return []
        try:
            results = await self._hybrid_retriever.search(
                query_text=query_text,
                query_vector=query_vector,
                collection_name="stripe_stackoverflow",
                limit=limit,
            )
            return [self._convert_to_source_result(r) for r in results]
        except Exception as e:
            logger.error(f"StackOverflow search failed: {e}")
            return []

    async def _search_changelog(
        self,
        query_text: str,
        query_vector: list[float],
        topic: str,
        limit: int,
    ) -> list[SourceResult]:
        """
        Search changelog for recent changes.

        Args:
            query_text: Text query.
            query_vector: Embedding vector.
            topic: Primary topic for filtering.
            limit: Number of results to return (adaptive).

        Returns:
            List of SourceResult from changelog.
        """
        if limit <= 0:
            logger.info("Skipping Stripe changelogs search (budget allocated: 0)")
            return []
        # Add topic to query for changelog search
        changelog_query = f"{query_text} {topic}" if topic != "other" else query_text

        try:
            results = await self._hybrid_retriever.search(
                query_text=changelog_query,
                query_vector=query_vector,
                collection_name="stripe_changelogs",
                limit=limit,
            )
            return [self._convert_to_source_result(r) for r in results]
        except Exception as e:
            logger.error(f"Changelog search failed: {e}")
            return []

    @observe(name="retrieval_agent")
    async def process(self, state: TicketState) -> TicketState:
        """
        Query all sources in parallel.
        
        Executes searches against all 3 sources simultaneously using
        asyncio.gather. Failures are captured in retrieval_errors;
        this method never raises exceptions.
        
        Args:
            state: Current ticket state.
            
        Returns:
            Updated state with retrieval results.
        """
        print("EXECUTION HIT: RetrievalAgent.process()")
        start_time = time.time()
        query_text = self._build_query(state)
        primary_topic = state.get("primary_topic", "other")

        # Compute dynamic resource allocation budget
        allocations = self._allocate_retrieval_budget(state)

        # Embed query once
        try:
            query_vector = self._embedder.embed_text(query_text)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            query_vector = [0.0] * 384  # Fallback: zero vector

        # Initialize result containers
        docs_results: list[SourceResult] = []
        github_results: list[SourceResult] = []
        stackoverflow_results: list[SourceResult] = []
        changelog_results: list[SourceResult] = []
        retrieval_errors: list[str] = []

        # Run all searches in parallel with dynamically allocated budget
        try:
            results = await asyncio.gather(
                self._search_stripe_docs(query_text, query_vector, allocations.get("stripe_docs", 0)),
                self._search_github_issues(query_text, query_vector, allocations.get("stripe_github_issues", 0)),
                self._search_stackoverflow(query_text, query_vector, allocations.get("stripe_stackoverflow", 0)),
                self._search_changelog(query_text, query_vector, primary_topic, allocations.get("stripe_changelogs", 0)),
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

        # Update state with per-source results (preserved for audit/debug)
        state["docs_results"] = docs_results
        state["github_results"] = github_results
        state["stackoverflow_results"] = stackoverflow_results
        state["changelog_results"] = changelog_results
        state["retrieval_errors"] = retrieval_errors

        # Cross-source reranking: merge all 4 source lists and rerank together
        all_results = (
            docs_results + github_results + stackoverflow_results + changelog_results
        )
        
        # Enforce dynamic top_k limit for reranked results based on complexity
        complexity = state.get("complexity", "moderate")
        if complexity == "simple":
            top_k = min(4, len(all_results))
        elif complexity == "moderate":
            top_k = min(8, len(all_results))
        else:
            top_k = min(12, len(all_results))

        # Set derived flags first so the cross-source reranker can use them for dynamic trust
        state["has_recent_changes"] = len(changelog_results) > 0
        state["has_stale_content"] = any(
            r.get("is_stale", False)
            for r in docs_results + github_results + stackoverflow_results
        )

        state["reranked_results"] = self._rerank_cross_source(
            query_text=query_text,
            all_results=all_results,
            top_k=top_k,
            state=state,
        )

        # Track agent path
        state["agent_path"] = ["retrieval"]

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
            state["error_log"] = [
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


# Lazy instance initialization to avoid idle connection timeout
_retrieval_agent_instance: RetrievalAgent | None = None

def get_retrieval_agent() -> RetrievalAgent:
    """Get or create retrieval agent instance (lazy initialization)."""
    global _retrieval_agent_instance
    if _retrieval_agent_instance is None:
        _retrieval_agent_instance = RetrievalAgent()
    return _retrieval_agent_instance


# Backward compatibility
def retrieval_agent() -> RetrievalAgent:
    """Backward compatibility wrapper."""
    return get_retrieval_agent()
