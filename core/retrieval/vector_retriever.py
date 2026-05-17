import logging
import psycopg2
from psycopg2.extras import execute_values
from dataclasses import dataclass, field
from typing import Any, Optional
from config import settings

logger = logging.getLogger(__name__)

class RetrieverError(Exception):
    """Base exception for retrieval errors."""
    pass

class RetrieverTimeoutError(RetrieverError):
    """Timeout during retrieval operations."""
    pass

class RetrieverUnavailableError(RetrieverError):
    """Retrieval service is unavailable."""
    pass

@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    source_url: str
    source_type: str
    title: Optional[str] = None
    date: Optional[str] = None
    is_potentially_stale: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_method: str = "vector"

    def __getitem__(self, key):
        """Allows dictionary-style subscription for backwards compatibility."""
        if key == "id":
            return self.chunk_id
        elif key == "payload":
            return {
                "content": self.text,
                "url": self.source_url,
                "title": self.title,
                "source_type": self.source_type,
                "is_stale": self.is_potentially_stale
            }
        elif key == "score":
            return self.score
        raise KeyError(key)

    def get(self, key, default=None):
        """Allows dictionary-style get for backwards compatibility."""
        try:
            return self[key]
        except KeyError:
            return default

class VectorRetriever:
    """
    Handles vector storage and similarity search in Neon Postgres.
    Supports advanced metadata-aware filtering (Freshness, Source Trust, Dates).
    """
    
    COLLECTION_TO_TABLE = {
        "stripe_docs": "stripe_docs",
        "stripe_github_issues": "stripe_github_issues",
        "stripe_stackoverflow": "stripe_stackoverflow",
        "stripe_changelogs": "stripe_changelogs"
    }

    # Trust tiers to boost official sources and penalize unverified ones
    SOURCE_TRUST_BOOSTS = {
        "stripe_docs": 0.15,          # Highly official
        "stripe_changelogs": 0.10,    # Official updates
        "stripe_github_issues": 0.00, # Community/Technical
        "stripe_stackoverflow": -0.05 # Unverified community Q&A
    }

    def __init__(self):
        self.db_url = settings.NEON_DB_URL

    def _get_connection(self):
        return psycopg2.connect(self.db_url)

    def _get_table_name(self, collection_name: str) -> str:
        return self.COLLECTION_TO_TABLE.get(collection_name, collection_name)

    def close(self):
        """No-op for compatibility with hybrid connection management."""
        pass

    def _put_connection(self, conn):
        """Put connection back. No-op since we open/close ad-hoc here."""
        try:
            conn.close()
        except Exception:
            pass

    def upsert(self, collection_name: str, points: list[dict]) -> bool:
        table_name = self.COLLECTION_TO_TABLE.get(collection_name)
        if not table_name:
            logger.error(f"Unknown collection: {collection_name}")
            return False

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Prepare data for batch insert using explicit columns
                data = []
                for p in points:
                    payload = p["payload"]
                    data.append((
                        p["id"],
                        payload.get("url", ""),
                        payload.get("title", ""),
                        payload.get("content", ""),
                        p["vector"],
                        payload.get("is_stale", False)
                    ))
                
                execute_values(cur, f"""
                    INSERT INTO {table_name} (id, url, title, content, embedding, is_stale)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        url = EXCLUDED.url,
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        is_stale = EXCLUDED.is_stale,
                        updated_at = NOW()
                """, data)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to upsert to {table_name}: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.3,
        threshold: Optional[float] = None,
        is_stale: Optional[bool] = False,      # Freshness filter
        min_date: Optional[str] = None,         # Date range filter
        boost_trust: bool = True,               # Source trust boost
        enable_decay: bool = True               # Soft Temporal Decay
    ) -> list[RetrievalResult]:
        table_name = self.COLLECTION_TO_TABLE.get(collection_name)
        if not table_name:
            return []

        active_threshold = threshold if threshold is not None else score_threshold
        conn = self._get_connection()
        
        # Build dynamic metadata-aware SQL query
        query = f"""
            SELECT id, url, title, content, 1 - (embedding <=> %s::vector) as score, is_stale, updated_at
            FROM {table_name}
            WHERE 1 - (embedding <=> %s::vector) >= %s
        """
        params = [query_vector, query_vector, active_threshold]

        if is_stale is not None:
            query += " AND is_stale = %s"
            params.append(is_stale)

        if min_date is not None:
            query += " AND updated_at >= %s"
            params.append(min_date)

        query += " ORDER BY score DESC LIMIT %s"
        params.append(limit)

        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                
                results = []
                for row in cur.fetchall():
                    raw_score = float(row[4])
                    updated_at = row[6]
                    
                    # Apply Source Trust boosting if requested
                    boost = self.SOURCE_TRUST_BOOSTS.get(collection_name, 0.0) if boost_trust else 0.0
                    boosted_score = raw_score + boost

                    # Apply Soft Temporal Decay
                    decay_penalty = 0.0
                    age_days = 0
                    if enable_decay and updated_at:
                        from datetime import datetime
                        now = datetime.now(updated_at.tzinfo) if updated_at.tzinfo else datetime.now()
                        age_days = (now - updated_at).days
                        # Decay Rate: 0.0001 per day
                        decay_penalty = max(0.0, age_days * 0.0001)
                        boosted_score -= decay_penalty

                    results.append(
                        RetrievalResult(
                            chunk_id=row[0],
                            text=row[3],
                            score=boosted_score,
                            source_url=row[1] or "",
                            source_type=collection_name,
                            title=row[2] or "",
                            date=str(row[6]) if row[6] else None,
                            is_potentially_stale=bool(row[5]),
                            metadata={
                                "title": row[2] or "",
                                "raw_score": raw_score,
                                "trust_boost": boost,
                                "age_days": age_days,
                                "decay_penalty": decay_penalty
                            },
                            retrieval_method="vector"
                        )
                    )
                # Re-sort results based on the final boosted/decayed score in Python
                results.sort(key=lambda r: r.score, reverse=True)
                return results
        finally:
            conn.close()

