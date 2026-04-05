"""
Ingestion orchestration and scheduling module.

Handles full and incremental ingestion pipelines, coordinating
scraping, chunking, embedding, and staleness detection.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import anthropic
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from config import settings
from core.ingestion.chunker import Chunk, SemanticChunker
from core.ingestion.embedder import DocumentEmbedder
from core.ingestion.scrapers import RawDocument, StripeDataScraper

logger = logging.getLogger(__name__)


@dataclass
class SourceReport:
    """
    Report for a single source ingestion.
    
    Attributes:
        source: Source name (docs, github, stackoverflow, changelog).
        success: Whether ingestion completed successfully.
        documents_scraped: Number of documents fetched.
        chunks_created: Number of chunks generated.
        chunks_inserted: Number of new chunks added to Qdrant.
        chunks_updated: Number of existing chunks updated.
        chunks_failed: Number of chunks that failed to upsert.
        error: Error message if ingestion failed.
        duration_seconds: Time taken for this source.
    """
    source: str
    success: bool
    documents_scraped: int = 0
    chunks_created: int = 0
    chunks_inserted: int = 0
    chunks_updated: int = 0
    chunks_failed: int = 0
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class IngestionReport:
    """
    Report for a complete ingestion run.
    
    Attributes:
        started_at: Timestamp when ingestion started.
        completed_at: Timestamp when ingestion completed.
        sources: List of per-source reports.
        total_documents: Total documents across all sources.
        total_chunks: Total chunks created across all sources.
        success: True if at least one source succeeded.
    """
    started_at: datetime
    completed_at: datetime | None = None
    sources: list[SourceReport] = field(default_factory=list)
    total_documents: int = 0
    total_chunks: int = 0
    success: bool = False


@dataclass
class UpdateReport:
    """
    Report for an incremental update run.
    
    Attributes:
        started_at: Timestamp when update started.
        completed_at: Timestamp when update completed.
        new_documents: Number of new documents processed.
        stale_chunks_detected: Number of chunks marked as stale.
        features_affected: List of features affected by changes.
        knowledge_updates_logged: Number of entries added to knowledge_updates table.
    """
    started_at: datetime
    completed_at: datetime | None = None
    new_documents: int = 0
    stale_chunks_detected: int = 0
    features_affected: list[str] = field(default_factory=list)
    knowledge_updates_logged: int = 0


class IngestionOrchestrator:
    """
    Orchestrates the complete ingestion pipeline.
    
    Coordinates scraping, chunking, embedding, and storage across
    multiple sources with parallel execution and failure isolation.
    """

    VALID_SOURCES: set[str] = {"docs", "github", "stackoverflow", "changelog"}

    def __init__(self) -> None:
        """Initialize the orchestrator with all required components."""
        self._scraper = StripeDataScraper()
        self._chunker = SemanticChunker()
        self._embedder = DocumentEmbedder()
        
        # Initialize Qdrant client for stats
        self._qdrant = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        
        # Initialize Anthropic client for staleness detection
        self._anthropic = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        
        logger.info("IngestionOrchestrator initialized")

    async def run_full_ingestion(
        self,
        sources: list[str] | None = None,
    ) -> IngestionReport:
        """
        Run complete ingestion pipeline for specified sources.
        
        Executes all sources in parallel. Each source failure is isolated
        and does not affect other sources.
        
        Args:
            sources: List of sources to ingest. Valid values: 'docs', 'github',
                     'stackoverflow', 'changelog'. Defaults to all sources.
                     
        Returns:
            IngestionReport with results from all sources.
        """
        if sources is None:
            sources = list(self.VALID_SOURCES)
        
        # Validate sources
        invalid_sources = set(sources) - self.VALID_SOURCES
        if invalid_sources:
            raise ValueError(
                f"Invalid sources: {invalid_sources}. "
                f"Valid sources: {self.VALID_SOURCES}"
            )

        report = IngestionReport(started_at=datetime.utcnow())
        logger.info(f"Starting full ingestion for sources: {sources}")

        # Run all sources in parallel
        tasks = [
            self._ingest_source(source)
            for source in sources
        ]
        
        source_reports = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for source, result in zip(sources, source_reports):
            if isinstance(result, Exception):
                report.sources.append(
                    SourceReport(
                        source=source,
                        success=False,
                        error=str(result),
                    )
                )
                logger.error(f"Source {source} failed with exception: {result}")
            else:
                report.sources.append(result)
                report.total_documents += result.documents_scraped
                report.total_chunks += result.chunks_created

        report.completed_at = datetime.utcnow()
        report.success = any(sr.success for sr in report.sources)

        duration = (report.completed_at - report.started_at).total_seconds()
        logger.info(
            f"Full ingestion completed in {duration:.1f}s. "
            f"Documents: {report.total_documents}, Chunks: {report.total_chunks}"
        )

        return report

    async def _ingest_source(self, source: str) -> SourceReport:
        """
        Run ingestion pipeline for a single source.
        
        Args:
            source: Source name to ingest.
            
        Returns:
            SourceReport with results.
        """
        start_time = datetime.utcnow()
        report = SourceReport(source=source, success=False)

        try:
            # Step 1: Scrape
            logger.info(f"[{source}] Starting scrape...")
            raw_documents = await self._scrape_source(source)
            report.documents_scraped = len(raw_documents)
            logger.info(f"[{source}] Scraped {len(raw_documents)} documents")

            if not raw_documents:
                report.success = True
                report.duration_seconds = (
                    datetime.utcnow() - start_time
                ).total_seconds()
                return report

            # Step 2: Chunk
            logger.info(f"[{source}] Starting chunking...")
            all_chunks: list[Chunk] = []
            for doc in raw_documents:
                chunks = self._chunk_document(doc)
                all_chunks.extend(chunks)
            
            report.chunks_created = len(all_chunks)
            logger.info(f"[{source}] Created {len(all_chunks)} chunks")

            if not all_chunks:
                report.success = True
                report.duration_seconds = (
                    datetime.utcnow() - start_time
                ).total_seconds()
                return report

            # Step 3: Embed + Upsert
            logger.info(f"[{source}] Starting embed and upsert...")
            collection_name = self._get_collection_name(source)
            upsert_result = self._embedder.upsert_chunks(all_chunks, collection_name)
            
            report.chunks_inserted = upsert_result["inserted"]
            report.chunks_updated = upsert_result["updated"]
            report.chunks_failed = upsert_result["failed"]
            
            logger.info(
                f"[{source}] Upserted: {upsert_result['inserted']} new, "
                f"{upsert_result['updated']} updated, {upsert_result['failed']} failed"
            )

            report.success = True

        except Exception as e:
            report.error = str(e)
            logger.error(f"[{source}] Ingestion failed: {e}")

        report.duration_seconds = (datetime.utcnow() - start_time).total_seconds()
        return report

    async def _scrape_source(self, source: str) -> list[RawDocument]:
        """
        Scrape documents from a specific source.
        
        Args:
            source: Source name.
            
        Returns:
            List of RawDocument objects.
        """
        if source == "docs":
            return await self._scraper.scrape_stripe_docs()
        elif source == "changelog":
            return await self._scraper.scrape_stripe_changelog()
        elif source == "github":
            return await self._scraper.fetch_github_issues()
        elif source == "stackoverflow":
            return await self._scraper.fetch_stackoverflow_questions()
        else:
            raise ValueError(f"Unknown source: {source}")

    def _chunk_document(self, doc: RawDocument) -> list[Chunk]:
        """
        Chunk a raw document based on its source type.
        
        Args:
            doc: RawDocument to chunk.
            
        Returns:
            List of Chunk objects.
        """
        if doc.source_type == "github_issue":
            # Build structured data for GitHub issue chunker
            issue_data = {
                "title": doc.title,
                "body": doc.metadata.get("body", doc.content),
                "url": doc.url,
                "labels": doc.metadata.get("labels", []),
                "comments": doc.metadata.get("comments", []),
            }
            return self._chunker.chunk_github_issue(issue_data)
        
        elif doc.source_type == "stackoverflow":
            # Build structured data for StackOverflow chunker
            question_data = {
                "title": doc.title,
                "body": doc.metadata.get("body", doc.content),
                "url": doc.url,
                "score": doc.metadata.get("score", 0),
                "accepted_answer": doc.metadata.get("accepted_answer"),
                "other_answers": doc.metadata.get("other_answers", []),
            }
            return self._chunker.chunk_stackoverflow(question_data)
        
        else:
            # stripe_doc or changelog
            return self._chunker.chunk(
                text=doc.content,
                source_type=doc.source_type,
                source_url=doc.url,
                title=doc.title,
                metadata=doc.metadata,
            )

    def _get_collection_name(self, source: str) -> str:
        """
        Get Qdrant collection name for a source.
        
        Args:
            source: Source name.
            
        Returns:
            Collection name from settings.
        """
        if source == "docs" or source == "changelog":
            return settings.QDRANT_DOCS_COLLECTION
        elif source == "github":
            return settings.QDRANT_ISSUES_COLLECTION
        elif source == "stackoverflow":
            return settings.QDRANT_STACKOVERFLOW_COLLECTION
        else:
            return settings.QDRANT_DOCS_COLLECTION

    async def run_incremental_update(self) -> UpdateReport:
        """
        Run daily incremental update.
        
        Fetches only new/changed content from the last 7 days and
        performs staleness detection on existing chunks.
        
        Returns:
            UpdateReport with results.
        """
        report = UpdateReport(started_at=datetime.utcnow())
        logger.info("Starting incremental update...")

        try:
            # Fetch recent changelog entries
            changelog_docs = await self._scraper.scrape_stripe_changelog(max_entries=50)
            
            # Filter to last 7 days
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            recent_docs = [
                doc for doc in changelog_docs
                if self._is_recent(doc.date, cutoff_date)
            ]
            
            report.new_documents = len(recent_docs)
            logger.info(f"Found {len(recent_docs)} recent changelog entries")

            # Process each new changelog entry for staleness detection
            for doc in recent_docs:
                affected_features = await self._extract_affected_features(doc.content)
                report.features_affected.extend(affected_features)

                # Find and mark stale chunks
                stale_count = await self._mark_stale_chunks(affected_features)
                report.stale_chunks_detected += stale_count

                # Log to knowledge_updates (would use Supabase client)
                if affected_features:
                    await self._log_knowledge_update(doc, affected_features)
                    report.knowledge_updates_logged += 1

            # Also process GitHub issues updated recently
            github_docs = await self._scraper.fetch_github_issues(
                max_per_repo=100,
                state="all",  # Include open issues too
            )
            
            # Filter to issues updated in last 7 days
            recent_github = [
                doc for doc in github_docs
                if self._is_recent(doc.date, cutoff_date)
            ]
            
            if recent_github:
                logger.info(f"Processing {len(recent_github)} recent GitHub issues")
                all_chunks: list[Chunk] = []
                for doc in recent_github:
                    chunks = self._chunk_document(doc)
                    all_chunks.extend(chunks)
                
                if all_chunks:
                    self._embedder.upsert_chunks(
                        all_chunks, settings.QDRANT_ISSUES_COLLECTION
                    )
                    report.new_documents += len(recent_github)

        except Exception as e:
            logger.error(f"Incremental update failed: {e}")

        report.completed_at = datetime.utcnow()
        duration = (report.completed_at - report.started_at).total_seconds()
        logger.info(
            f"Incremental update completed in {duration:.1f}s. "
            f"New docs: {report.new_documents}, Stale chunks: {report.stale_chunks_detected}"
        )

        return report

    def _is_recent(self, date_str: str | None, cutoff: datetime) -> bool:
        """
        Check if a date string is more recent than cutoff.
        
        Args:
            date_str: Date string in various formats.
            cutoff: Cutoff datetime.
            
        Returns:
            True if date is after cutoff.
        """
        if not date_str:
            return True  # If no date, assume it's recent
        
        try:
            # Try ISO format first
            if "T" in date_str:
                doc_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                # Try common date formats
                for fmt in ["%Y-%m-%d", "%B %d, %Y", "%B %d %Y"]:
                    try:
                        doc_date = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return True  # Can't parse, assume recent
            
            return doc_date.replace(tzinfo=None) > cutoff
        except Exception:
            return True

    async def _extract_affected_features(self, changelog_content: str) -> list[str]:
        """
        Use Claude Haiku to extract affected features from changelog entry.
        
        Args:
            changelog_content: Changelog entry text.
            
        Returns:
            List of affected feature names.
        """
        try:
            response = self._anthropic.messages.create(
                model=settings.HAIKU_MODEL,
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Extract the Stripe features affected by this changelog entry.
Return only a comma-separated list of feature names (e.g., "webhooks, payment intents, subscriptions").
If no specific features are mentioned, return "general".

Changelog entry:
{changelog_content[:2000]}

Affected features:"""
                    }
                ],
            )
            
            # Parse response
            features_text = response.content[0].text.strip()
            features = [f.strip().lower() for f in features_text.split(",")]
            return [f for f in features if f and f != "general"]
            
        except anthropic.APIError as e:
            logger.error(f"Claude API error extracting features: {e}")
            return []
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return []

    async def _mark_stale_chunks(self, features: list[str]) -> int:
        """
        Find and mark chunks as potentially stale based on affected features.
        
        Args:
            features: List of affected feature names.
            
        Returns:
            Number of chunks marked as stale.
        """
        if not features:
            return 0

        stale_count = 0
        collection_name = settings.QDRANT_DOCS_COLLECTION

        try:
            # Search for chunks mentioning affected features
            for feature in features:
                # Embed feature name to find semantically related chunks
                feature_embedding = self._embedder.embed_text(feature)
                
                # Search for related chunks
                results = self._qdrant.search(
                    collection_name=collection_name,
                    query_vector=feature_embedding,
                    limit=50,
                    score_threshold=0.7,
                )

                # Mark as potentially stale
                point_ids = [result.id for result in results]
                if point_ids:
                    self._qdrant.set_payload(
                        collection_name=collection_name,
                        payload={"potentially_stale": True},
                        points=point_ids,
                    )
                    stale_count += len(point_ids)
                    logger.info(
                        f"Marked {len(point_ids)} chunks as stale for feature: {feature}"
                    )

        except Exception as e:
            logger.error(f"Error marking stale chunks: {e}")

        return stale_count

    async def _log_knowledge_update(
        self,
        doc: RawDocument,
        affected_features: list[str],
    ) -> None:
        """
        Log a knowledge update to Supabase.
        
        Note: This is a placeholder that logs locally. 
        Full Supabase integration will be in core/memory/supabase_client.py.
        
        Args:
            doc: The changelog document.
            affected_features: List of affected features.
        """
        content_hash = hashlib.sha256(doc.content.encode()).hexdigest()[:32]
        
        logger.info(
            f"Knowledge update: {doc.title} affects {affected_features} "
            f"(hash: {content_hash})"
        )

    def get_ingestion_stats(self) -> dict[str, Any]:
        """
        Get current state of the knowledge base.
        
        Returns:
            Dictionary with:
            - total_chunks_per_collection: Chunk counts per collection
            - last_ingestion_per_source: Not implemented (requires Supabase)
            - stale_chunks_count: Number of chunks marked as stale
            - collection_sizes_mb: Approximate storage per collection
        """
        stats: dict[str, Any] = {
            "total_chunks_per_collection": {},
            "last_ingestion_per_source": {},
            "stale_chunks_count": {},
            "collection_sizes_mb": {},
        }

        collections = [
            settings.QDRANT_DOCS_COLLECTION,
            settings.QDRANT_ISSUES_COLLECTION,
            settings.QDRANT_STACKOVERFLOW_COLLECTION,
        ]

        for collection_name in collections:
            try:
                # Get collection info
                collection_info = self._qdrant.get_collection(collection_name)
                
                stats["total_chunks_per_collection"][collection_name] = (
                    collection_info.points_count
                )
                
                # Estimate size (384 dims * 4 bytes * points + payload overhead)
                estimated_mb = (
                    collection_info.points_count * 384 * 4 / 1024 / 1024
                ) * 1.5  # 1.5x for payload overhead
                stats["collection_sizes_mb"][collection_name] = round(estimated_mb, 2)

                # Count stale chunks
                try:
                    stale_results = self._qdrant.scroll(
                        collection_name=collection_name,
                        scroll_filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="potentially_stale",
                                    match=qdrant_models.MatchValue(value=True),
                                )
                            ]
                        ),
                        limit=1,
                        with_payload=False,
                        with_vectors=False,
                    )
                    # This is a rough count - would need proper counting for accuracy
                    stats["stale_chunks_count"][collection_name] = (
                        len(stale_results[0]) if stale_results[0] else 0
                    )
                except Exception:
                    stats["stale_chunks_count"][collection_name] = 0

            except Exception as e:
                logger.warning(f"Could not get stats for {collection_name}: {e}")
                stats["total_chunks_per_collection"][collection_name] = 0
                stats["collection_sizes_mb"][collection_name] = 0
                stats["stale_chunks_count"][collection_name] = 0

        return stats


# Module-level instance
orchestrator = IngestionOrchestrator()
