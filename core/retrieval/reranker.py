"""Cross-encoder based reranking for improved retrieval precision.

Uses sentence-transformers cross-encoder models to score query-document pairs
and rerank retrieval results. Cross-encoders are more accurate than bi-encoders
for relevance scoring but slower, so they're applied as a final reranking step.
"""

import logging
from typing import Optional

from sentence_transformers import CrossEncoder

from core.retrieval.vector_retriever import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieval results using a cross-encoder model.
    
    Cross-encoders take a query-document pair and directly predict relevance,
    which is more accurate than separate vector representations but slower.
    Used as a final reranking step after hybrid retrieval.
    """
    
    # Model choices (smaller = faster):
    # - mxbai-embed-large: 335M params, 512 dim, ~100ms per batch
    # - cross-encoder-mxbai-v1: ~50M params, faster, recommended for production
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name. Defaults to DEFAULT_MODEL.
            device: Device to run model on (cpu, cuda, mps).
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        
        try:
            self.model = CrossEncoder(self.model_name, device=device, max_length=512)
            logger.info(f"CrossEncoderReranker initialized with model: {self.model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model {self.model_name}: {e}")
            self.model = None
    
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> list[RetrievalResult]:
        """
        Rerank retrieval results using cross-encoder scoring.
        
        Args:
            query: User query string.
            results: List of RetrievalResult objects from hybrid search.
            top_k: Number of top results to return.
            batch_size: Batch size for scoring (trade-off between speed and memory).
            
        Returns:
            Top-k reranked results sorted by cross-encoder score.
        """
        if not results:
            logger.debug("No results to rerank")
            return []
        
        if self.model is None:
            logger.warning("Cross-encoder model not loaded, returning original results")
            return results[:top_k]
        
        if len(results) <= top_k:
            logger.debug(f"Results count ({len(results)}) <= top_k ({top_k}), skipping reranking")
            return results
        
        try:
            # Prepare query-document pairs for scoring
            pairs = [[query, result.text] for result in results]
            
            # Score all pairs using cross-encoder
            logger.debug(f"Cross-encoder scoring {len(pairs)} query-document pairs")
            scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            
            # Attach scores and sort by descending score
            scored_results = []
            for result, score in zip(results, scores):
                scored_results.append((result, float(score)))
            
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k
            top_results = [result for result, score in scored_results[:top_k]]
            
            logger.info(
                f"[RERANK] Cross-encoder reranking complete: "
                f"input={len(results)} results, output={len(top_results)} top-k results, "
                f"top_score={scored_results[0][1]:.4f}, bottom_score={scored_results[top_k-1][1]:.4f}"
            )
            
            return top_results
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            # Fallback to original ordering
            return results[:top_k]
    
    def score_pairs(
        self,
        query: str,
        results: list[RetrievalResult],
        batch_size: int = 32,
    ) -> list[tuple[RetrievalResult, float]]:
        """
        Score query-document pairs without reranking.
        
        Useful for analyzing scores or custom ranking logic.
        
        Args:
            query: User query string.
            results: List of RetrievalResult objects.
            batch_size: Batch size for scoring.
            
        Returns:
            List of (result, score) tuples in original order.
        """
        if not results or self.model is None:
            return [(result, 0.0) for result in results]
        
        try:
            pairs = [[query, result.text] for result in results]
            scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            return list(zip(results, scores))
        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}")
            return [(result, 0.0) for result in results]
