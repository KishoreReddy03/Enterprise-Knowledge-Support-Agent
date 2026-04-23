"""
Document embedding module for vector storage.

Uses sentence-transformers all-MiniLM-L6-v2 for local embeddings (zero API cost)
and Qdrant for vector storage and retrieval.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import settings
from core.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Simple LRU cache implementation with max size limit.
    
    Used to cache embeddings and avoid redundant computation
    for repeated text inputs.
    """

    def __init__(self, max_size: int = 10000) -> None:
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries before eviction.
        """
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> list[float] | None:
        """
        Get value from cache, moving it to end (most recently used).
        
        Args:
            key: Cache key.
            
        Returns:
            Cached value or None if not found.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: list[float]) -> None:
        """
        Set value in cache, evicting oldest if at capacity.
        
        Args:
            key: Cache key.
            value: Value to cache.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)


class DocumentEmbedder:
    """
    Handles document embedding and Qdrant vector storage.
    
    Uses sentence-transformers for local embedding generation (no API costs)
    and Qdrant for scalable vector storage with payload indexing.
    """

    VECTOR_SIZE: int = 384  # all-MiniLM-L6-v2 output dimension
    CACHE_MAX_SIZE: int = 10000

    def __init__(self) -> None:
        """
        Initialize embedder with model and Qdrant client.
        
        Loads the sentence-transformer model and establishes
        connection to Qdrant. Logs model load time.
        """
        # Load embedding model
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        start_time = time.time()
        
        try:
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        load_time = time.time() - start_time
        logger.info(f"Embedding model loaded in {load_time:.2f} seconds")

        # Initialize Qdrant client
        try:
            self._qdrant = QdrantClient(
                                url=settings.QDRANT_URL,
                                api_key=settings.QDRANT_API_KEY,
                                timeout=60,
                                check_compatibility=False,
                                        )
            logger.info(f"Connected to Qdrant at {settings.QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

        # Initialize embedding cache
        self._cache = LRUCache(max_size=self.CACHE_MAX_SIZE)

    def _hash_text(self, text: str) -> str:
        """
        Generate SHA256 hash of text for cache key.
        
        Args:
            text: Text to hash.
            
        Returns:
            Hex digest of SHA256 hash.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Uses LRU cache to avoid redundant computation.
        
        Args:
            text: Text to embed.
            
        Returns:
            384-dimensional embedding vector.
        """
        # Check cache first
        cache_key = self._hash_text(text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Generate embedding
        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            embedding_list = embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise

        # Cache result
        self._cache.set(cache_key, embedding_list)

        return embedding_list

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per batch.
            
        Returns:
            List of 384-dimensional embedding vectors.
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._hash_text(text)
            cached = self._cache.get(cache_key)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append([])  # Placeholder
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch process uncached texts
        if uncached_texts:
            num_batches = (len(uncached_texts) + batch_size - 1) // batch_size

            with tqdm(
                total=len(uncached_texts),
                desc="Embedding texts",
                unit="text",
            ) as pbar:
                for batch_idx in range(num_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, len(uncached_texts))
                    batch_texts = uncached_texts[start:end]

                    try:
                        batch_embeddings = self._model.encode(
                            batch_texts,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                        )

                        for j, embedding in enumerate(batch_embeddings):
                            embedding_list = embedding.tolist()
                            original_idx = uncached_indices[start + j]
                            embeddings[original_idx] = embedding_list

                            # Cache the result
                            cache_key = self._hash_text(batch_texts[j])
                            self._cache.set(cache_key, embedding_list)

                    except Exception as e:
                        logger.error(f"Failed to embed batch {batch_idx}: {e}")
                        raise

                    pbar.update(len(batch_texts))

        return embeddings

    def ensure_collection(self, collection_name: str) -> None:
                """
                Temporarily skip collection existence check because collection
                is already created manually in Qdrant dashboard.
                """
                logger.info(f"Skipping ensure_collection for '{collection_name}' (already created manually)")
                return

    def _chunk_to_payload(self, chunk: Chunk) -> dict[str, Any]:
        """
        Convert Chunk to Qdrant payload dictionary.
        
        Args:
            chunk: Chunk object to convert.
            
        Returns:
            Dictionary suitable for Qdrant payload.
        """
        return {
            "text": chunk.text,
            "source_type": chunk.source_type,
            "source_url": chunk.source_url,
            "title": chunk.title,
            "section_path": chunk.section_path,
            "date": chunk.date,
            "chunk_id": chunk.chunk_id,
            "metadata": chunk.metadata,
        }

    def _chunk_id_to_point_id(self, chunk_id: str) -> str:
        """
        Convert chunk_id (hex string) to valid Qdrant point ID.
        
        Qdrant accepts either unsigned integers or UUIDs.
        We use the chunk_id directly as it's already a valid hex string.
        
        Args:
            chunk_id: 16-character hex string from chunk.
            
        Returns:
            UUID-formatted string for Qdrant.
        """
        # Pad to 32 chars for UUID format, then format as UUID
        padded = chunk_id.ljust(32, "0")
        return f"{padded[:8]}-{padded[8:12]}-{padded[12:16]}-{padded[16:20]}-{padded[20:32]}"

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        collection_name: str,
    ) -> dict[str, int]:
        """
        Embed and upsert chunks to Qdrant collection.
        
        Uses chunk_id as deterministic point ID for idempotent re-ingestion.
        Continues on individual chunk failures to avoid losing entire batches.
        
        Args:
            chunks: List of Chunk objects to upsert.
            collection_name: Target Qdrant collection.
            
        Returns:
            Dictionary with counts: {inserted: int, updated: int, failed: int}
        """
        if not chunks:
            return {"inserted": 0, "updated": 0, "failed": 0}

        # Ensure collection exists
        self.ensure_collection(collection_name)

        # Check which chunks already exist
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        point_ids = [self._chunk_id_to_point_id(cid) for cid in chunk_ids]

        existing_ids: set[str] = set()
        try:
            # Check existence in batches
            batch_size = 100
            for i in range(0, len(point_ids), batch_size):
                batch_ids = point_ids[i : i + batch_size]
                try:
                    results = self._qdrant.retrieve(
                        collection_name=collection_name,
                        ids=batch_ids,
                        with_payload=False,
                        with_vectors=False,
                    )
                    existing_ids.update(str(r.id) for r in results)
                except UnexpectedResponse:
                    # Collection might be empty or IDs don't exist
                    pass
        except Exception as e:
            logger.warning(f"Could not check existing points: {e}")

        # Embed all chunk texts
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_batch(texts)

        # Upsert to Qdrant
        inserted = 0
        updated = 0
        failed = 0

        # Process in batches for upsert
        upsert_batch_size = 5

        with tqdm(
            total=len(chunks),
            desc="Upserting to Qdrant",
            unit="chunk",
        ) as pbar:
            for i in range(0, len(chunks), upsert_batch_size):
                batch_chunks = chunks[i : i + upsert_batch_size]
                batch_embeddings = embeddings[i : i + upsert_batch_size]
                batch_point_ids = point_ids[i : i + upsert_batch_size]

                points: list[qdrant_models.PointStruct] = []

                for j, (chunk, embedding, point_id) in enumerate(
                    zip(batch_chunks, batch_embeddings, batch_point_ids)
                ):
                    try:
                        point = qdrant_models.PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=self._chunk_to_payload(chunk),
                        )
                        points.append(point)

                        if point_id in existing_ids:
                            updated += 1
                        else:
                            inserted += 1

                    except Exception as e:
                        logger.error(
                            f"Failed to prepare chunk {chunk.chunk_id}: {e}"
                        )
                        failed += 1
                        continue

                # Upsert batch
                if points:
                    try:
                        self._qdrant.upsert(
                            collection_name=collection_name,
                            points=points,
                        )
                    except Exception as e:
                        logger.error(f"Failed to upsert batch: {e}")
                        # Mark all points in this batch as failed
                        failed += len(points)
                        inserted -= sum(
                            1 for p in points if str(p.id) not in existing_ids
                        )
                        updated -= sum(
                            1 for p in points if str(p.id) in existing_ids
                        )

                pbar.update(len(batch_chunks))

        result = {"inserted": inserted, "updated": updated, "failed": failed}
        logger.info(
            f"Upsert complete: {inserted} inserted, {updated} updated, {failed} failed"
        )
        return result


# Module-level instance for convenience
embedder = DocumentEmbedder()
