import logging
import psycopg2
from psycopg2.extras import execute_values
from config import settings

logger = logging.getLogger(__name__)

class VectorRetriever:
    """
    Handles vector storage and similarity search in Neon Postgres.
    """
    
    COLLECTION_TO_TABLE = {
        "stripe_docs": "stripe_docs",
        "stripe_github_issues": "stripe_github_issues",
        "stripe_stackoverflow": "stripe_stackoverflow",
        "stripe_changelogs": "stripe_changelogs"
    }

    def __init__(self):
        self.db_url = settings.NEON_DB_URL

    def _get_connection(self):
        return psycopg2.connect(self.db_url)

    def collection_exists(self, collection_name: str) -> bool:
        table_name = self.COLLECTION_TO_TABLE.get(collection_name)
        if not table_name:
            return False
            
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    );
                """, (table_name,))
                return cur.fetchone()[0]
        finally:
            conn.close()

    def upsert(self, collection_name: str, points: list[dict]) -> bool:
        table_name = self.COLLECTION_TO_TABLE.get(collection_name)
        if not table_name:
            logger.error(f"Unknown collection: {collection_name}")
            return False

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Prepare data for batch insert
                data = []
                for p in points:
                    data.append((
                        p["id"],
                        p["vector"],
                        psycopg2.extras.Json(p["payload"]),
                        False # is_stale
                    ))
                
                execute_values(cur, f"""
                    INSERT INTO {table_name} (id, embedding, metadata, is_stale)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        is_stale = EXCLUDED.is_stale
                """, data)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to upsert to {table_name}: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def search(self, collection_name: str, query_vector: list[float], limit: int = 5, threshold: float = 0.3):
        table_name = self.COLLECTION_TO_TABLE.get(collection_name)
        if not table_name:
            return []

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # pgvector cosine similarity search (<=> is distance, so 1 - distance is similarity)
                cur.execute(f"""
                    SELECT id, metadata, 1 - (embedding <=> %s::vector) as score
                    FROM {table_name}
                    WHERE 1 - (embedding <=> %s::vector) >= %s
                    AND is_stale = FALSE
                    ORDER BY score DESC
                    LIMIT %s
                """, (query_vector, query_vector, threshold, limit))
                
                results = []
                for row in cur.fetchall():
                    results.append({
                        "id": row[0],
                        "payload": row[1],
                        "score": row[2]
                    })
                return results
        finally:
            conn.close()
