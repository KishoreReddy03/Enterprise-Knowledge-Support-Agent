import hashlib
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_type: str
    source_url: str
    title: str
    section_path: str = ""
    date: str = ""
    metadata: dict[str, Any] = None

class Chunker:
    """
    Intelligent document chunking for Stripe-related content.
    """
    
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _generate_id(self, text: str, source_url: str) -> str:
        # Deterministic ID based on content and URL
        return hashlib.sha256(f"{source_url}:{text}".encode()).hexdigest()

    def chunk_document(self, doc: dict) -> list[Chunk]:
        """
        Splits a document into a list of Chunks.
        """
        text = doc.get("content", "")
        if not text:
            return []
            
        # Very basic sliding window chunking for demonstration
        # In production, use LangChain's RecursiveCharacterTextSplitter
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunk_id = self._generate_id(chunk_text, doc["url"])
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source_type=doc.get("source_type", "unknown"),
                source_url=doc["url"],
                title=doc.get("title", "Untitled"),
                section_path=doc.get("section_path", ""),
                date=doc.get("date", ""),
                metadata=doc.get("metadata", {})
            ))
            
            start += self.chunk_size - self.chunk_overlap
            
        return chunks
