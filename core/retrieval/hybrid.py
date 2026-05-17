"""
Hybrid retrieval module combining vector search with full-text search.

Uses pgvector for semantic search and PostgreSQL full-text search (tsvector)
for keyword matching. Results are fused using Reciprocal Rank Fusion (RRF).

Both vector and FTS queries go through Neon via the VectorRetriever's
ThreadedConnectionPool. Vector and FTS searches run in PARALLEL for performance.
"""

import asyncio
import logging
import time
from typing import Any, Optional

import psycopg2.extras

from config import settings
from core.retrieval.vector_retriever import VectorRetriever, RetrievalResult
from core.intelligence.query_classifier import QueryClassifier, QueryType
from core.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines pgvector semantic search with PostgreSQL full-text search.
    
    Merges results using Reciprocal Rank Fusion (RRF) with k=60
    for balanced contribution from both methods.
    
    FTS queries are filtered to remove English stopwords for improved precision.
    """

    RRF_K: int = 60
    
    # English stopwords to filter from FTS queries
    # Reduces noise while preserving signal in topic-focused chunks
    STOPWORDS: set[str] = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
        'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
        'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cant', 'cannot',
        'could', 'couldnt', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing',
        'dont', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
        'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he',
        'hed', 'hell', 'hes', 'her', 'here', 'heres', 'hers', 'herself', 'him',
        'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im', 'ive', 'if',
        'in', 'into', 'is', 'isnt', 'it', 'its', 'itself', 'just', 'ks', 'me',
        'might', 'mightnt', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off',
        'on', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out',
        'over', 'own', 'same', 'shant', 'she', 'shed', 'shell', 'shes', 'should',
        'shouldnt', 'so', 'some', 'such', 'than', 'that', 'thats', 'the',
        'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres',
        'these', 'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those',
        'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasnt',
        'we', 'wed', 'well', 'were', 'werent', 'weve', 'what', 'whats', 'when',
        'whens', 'where', 'wheres', 'which', 'who', 'whos', 'whom', 'why', 'whys',
        'with', 'wont', 'would', 'wouldnt', 'you', 'youd', 'youll', 'youre',
        'youve', 'your', 'yours', 'yourself', 'yourselves',
    }

    def __init__(self) -> None:
        self._vector_retriever = VectorRetriever()
        self._classifier = QueryClassifier()
        self._reranker = CrossEncoderReranker()
        # Semaphore to limit concurrent database operations
        # Prevents connection pool exhaustion by serializing some concurrent access
        self._db_semaphore = asyncio.Semaphore(15)
        logger.info("HybridRetriever initialized with pgvector, FTS, query classification, and Cross-Encoder reranker")
        logger.info("  - DB semaphore: max 15 concurrent operations")
        logger.info("  - Connection pool: maxconn=30 (increased from 20)")
        logger.info("  - Query classifier: enabled for adaptive weighting")

    def _fts_search(
        self,
        table_name: str,
        query_text: str,
        limit: int = 10,
        is_stale: Optional[bool] = False,
        min_date: Optional[str] = None,
        boost_trust: bool = True,
        enable_decay: bool = True
    ) -> list[RetrievalResult]:
        """
        Full-text search against tsvector column with stopword filtering.
        Supports advanced metadata-aware constraints (freshness, dates, trust boosts, and soft decay).
        """
        import re
        
        words = [w.lower() for w in re.split(r'\W+', query_text) if w]
        filtered_words = [w for w in words if w not in self.STOPWORDS]
        
        if not filtered_words:
            filtered_words = words
        
        if not filtered_words:
            return []
        
        or_query = ' OR '.join(filtered_words)
        
        logger.info(
            f"[FTS STOPWORD FILTER] Input: {len(words)} words, "
            f"Output: {len(filtered_words)} words | "
            f"Query: {' OR '.join(words[:5])}{'...' if len(words) > 5 else ''} → "
            f"{' OR '.join(filtered_words[:5])}{'...' if len(filtered_words) > 5 else ''}"
        )

        conn = None
        try:
            conn = self._vector_retriever._get_connection()
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # Dynamic metadata-aware SQL query
                query = f"""
                    SELECT 
                        id::text as id,
                        title,
                        content,
                        url,
                        ts_rank(fts_vector, websearch_to_tsquery('english', %s)) as score,
                        is_stale,
                        updated_at
                    FROM {table_name}
                    WHERE fts_vector @@ websearch_to_tsquery('english', %s)
                """
                params = [or_query, or_query]

                if is_stale is not None:
                    query += " AND is_stale = %s"
                    params.append(is_stale)

                if min_date is not None:
                    query += " AND updated_at >= %s"
                    params.append(min_date)

                query += " ORDER BY ts_rank(fts_vector, websearch_to_tsquery('english', %s)) DESC LIMIT %s"
                params.extend([or_query, limit])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
            
            logger.debug(f"FTS search returned {len(rows)} rows from {table_name}")
            
            # Apply Source Trust boosting
            boost = self._vector_retriever.SOURCE_TRUST_BOOSTS.get(table_name, 0.0) if boost_trust else 0.0

            results = []
            for row in rows:
                raw_score = float(row["score"])
                boosted_score = raw_score + boost
                updated_at = row["updated_at"]

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

                result = RetrievalResult(
                    chunk_id=row["id"],
                    text=row["content"],
                    score=boosted_score,
                    source_url=row["url"] or "",
                    source_type=table_name,
                    title=row["title"],
                    date=str(row["updated_at"]) if row["updated_at"] else None,
                    is_potentially_stale=bool(row["is_stale"]),
                    metadata={
                        "title": row["title"],
                        "source_type": "docs",
                        "raw_score": raw_score,
                        "trust_boost": boost,
                        "age_days": age_days,
                        "decay_penalty": decay_penalty
                    },
                    retrieval_method="fts",
                )
                results.append(result)
            
            # Re-sort results based on final score in Python
            results.sort(key=lambda r: r.score, reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"FTS search failed for {table_name}: {e}")
            return []
        finally:
            if conn is not None:
                self._vector_retriever._put_connection(conn)

    async def search(
        self,
        query_text: str,
        query_vector: list[float],
        collection_name: str,
        limit: int = 10,
        is_stale: Optional[bool] = False,      # Freshness filter
        min_date: Optional[str] = None,         # Date range filter
        boost_trust: bool = True,               # Source trust boost
        enable_decay: bool = True               # Soft Temporal Decay
    ) -> list[RetrievalResult]:
        """
        Perform hybrid search with adaptive RRF fusion based on query classification.
        Supports advanced metadata constraints (freshness, dates, trust boosting, and temporal decay).
        """
        print(f"\n=== HYBRID_RETRIEVER.SEARCH HIT ===")
        print(f"Query text: {query_text[:50]}")
        print(f"Collection: {collection_name}")
        
        # Classify query for adaptive weighting
        classification = self._classifier.classify(query_text)
        fts_weight, vector_weight = self._classifier.get_weights(classification.query_type)
        
        print(f"Query type: {classification.query_type.value} (confidence={classification.confidence:.2f})")
        print(f"Adaptive weights: FTS={fts_weight:.0%}, Vector={vector_weight:.0%}")
        
        expanded_limit = limit * 2
        table_name = self._vector_retriever._get_table_name(collection_name)
        
        # RUN VECTOR + FTS IN PARALLEL
        print(f"Starting parallel vector + FTS search...")
        search_start = time.time()
        
        try:
            vector_results, fts_results = await asyncio.gather(
                asyncio.to_thread(
                    self._vector_retriever.search,
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=expanded_limit,
                    score_threshold=settings.RETRIEVAL_SIMILARITY_THRESHOLD,
                    is_stale=is_stale,
                    min_date=min_date,
                    boost_trust=boost_trust,
                    enable_decay=enable_decay
                ),
                asyncio.to_thread(
                    self._fts_search,
                    table_name=table_name,
                    query_text=query_text,
                    limit=expanded_limit,
                    is_stale=is_stale,
                    min_date=min_date,
                    boost_trust=boost_trust,
                    enable_decay=enable_decay
                ),
                return_exceptions=True,
            )
        except Exception as e:
            logger.error(f"Parallel hybrid search failed for {collection_name}: {e}")
            return []
        
        # Handle exceptions from gather (return_exceptions=True)
        if isinstance(vector_results, Exception):
            logger.warning(f"Vector search exception for {collection_name}: {vector_results}")
            vector_results = []
        if isinstance(fts_results, Exception):
            logger.warning(f"FTS search exception for {collection_name}: {fts_results}")
            fts_results = []
        
        search_elapsed = time.time() - search_start
        print(f"[PERF] Vector search returned: {len(vector_results)} results")
        print(f"[PERF] FTS search returned: {len(fts_results)} results")
        print(f"[PERF] Parallel search took {search_elapsed:.2f}s")
        
        if not vector_results and not fts_results:
            logger.warning(f"No results for collection {collection_name}")
            return []
        
        if not vector_results:
            logger.warning(f"FTS-only results for {collection_name} (vector search failed)")
            return fts_results[:limit]
        
        if not fts_results:
            logger.warning(f"Vector-only results for {collection_name} (FTS failed)")
            return vector_results[:limit]
        
        merged: dict[str, dict[str, Any]] = {}
        
        # Apply adaptive vector weight
        for rank, result in enumerate(vector_results):
            result_id = result.chunk_id
            rrf_score = vector_weight / (rank + 1 + self.RRF_K)
            
            merged[result_id] = {
                "result": result,
                "rrf_score": rrf_score,
                "vector_rank": rank + 1,
                "fts_rank": None,
            }
        
        # Apply adaptive FTS weight
        for rank, result in enumerate(fts_results):
            result_id = result.chunk_id
            rrf_contribution = fts_weight / (rank + 1 + self.RRF_K)
            
            if result_id in merged:
                merged[result_id]["rrf_score"] += rrf_contribution
                merged[result_id]["fts_rank"] = rank + 1
            else:
                merged[result_id] = {
                    "result": result,
                    "rrf_score": rrf_contribution,
                    "vector_rank": None,
                    "fts_rank": rank + 1,
                }
        
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )
        
        # Extract candidates for cross-encoder reranking
        candidates = [item["result"] for item in sorted_results]
        
        # Log Phase 2 RRF complete
        logger.info(f"[PHASE2] RRF fusion complete: query_type={classification.query_type.value} fts_weight={fts_weight:.0%} candidates={len(candidates)}")

        # Run Phase 3: Cross-Encoder Reranking
        final_results = self._reranker.rerank(
            query=query_text,
            results=candidates,
            top_k=limit
        )

        # FIX 2: Debug print — exact keys returned by retrieval for synthesis matching
        if final_results:
            sample = final_results[0]
            print(f"[HYBRID] Result keys returned (RetrievalResult attrs): "
                  f"{[a for a in vars(sample) if not a.startswith('_')]}")
            print(f"[HYBRID] Returning {len(final_results)} reranked results")

        return final_results

    def close(self) -> None:
        """Close the underlying VectorRetriever's pool."""
        self._vector_retriever.close()

    def __del__(self) -> None:
        """Close pool on cleanup."""
        try:
            if hasattr(self, "_vector_retriever"):
                self._vector_retriever.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")