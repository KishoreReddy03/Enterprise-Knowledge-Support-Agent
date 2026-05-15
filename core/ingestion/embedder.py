import logging
import time
from sentence_transformers import SentenceTransformer
from config import settings
from core.ingestion.chunker import Chunk
from core.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """
    Handles document embedding and Neon pgvector storage.
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self._retriever = VectorRetriever()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._model.encode(texts).tolist()

    def upsert_chunks(self, chunks: list[Chunk], collection_name: str) -> dict:
        if not chunks:
            return {"inserted": 0, "failed": 0}

        # Embed all texts
        texts = [c.text for c in chunks]
        embeddings = self.embed_batch(texts)

        points = []
        for chunk, vector in zip(chunks, embeddings):
            points.append({
                "id": chunk.chunk_id,
                "vector": vector,
                "payload": {
                    "content": chunk.text,
                    "source": chunk.source_type,
                    "url": chunk.source_url,
                    "title": chunk.title,
                    "section_path": chunk.section_path,
                    "date": chunk.date,
                    "metadata": chunk.metadata
                }
            })

        success = self._retriever.upsert(collection_name, points)
        
        if success:
            return {"inserted": len(chunks), "failed": 0}
        else:
            return {"inserted": 0, "failed": len(chunks)}
