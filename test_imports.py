#!/usr/bin/env python3
"""Test if imports work correctly."""

try:
    from core.retrieval.vector_retriever import RetrievalResult, VectorRetriever
    print("✓ VectorRetriever imports OK")
    
    from core.retrieval.hybrid import HybridRetriever
    print("✓ HybridRetriever imports OK")
    
    from core.retrieval import RetrievalResult as RR
    print("✓ Package imports OK")
    
    # Test dataclass creation
    r = RetrievalResult(chunk_id="test", text="hello", score=0.9)
    print(f"✓ RetrievalResult creation OK: {r.chunk_id}")
    
    print("\n✅ All imports successful!")
    
