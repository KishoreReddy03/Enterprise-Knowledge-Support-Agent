import asyncio
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.retrieval.vector_retriever import VectorRetriever
from core.retrieval.reranker import CrossEncoderReranker

async def test_pure_retrieval_with_reranker():
    print("\n⚡ PURE RETRIEVAL & RERANKING TEST")
    print("=" * 60)

    try:
        # 1. Initialize Vector Retriever (Neon DB Interface)
        retriever = VectorRetriever()
        dummy_vector = [0.0] * 384
        
        print("1. Fetching candidate chunks from Neon Postgres...")
        candidates = retriever.search(
            collection_name="stripe_docs",
            query_vector=dummy_vector,
            limit=4,
            threshold=0.0  # Retrieve top 4 candidates to rerank
        )
        
        if not candidates:
            print("⚠️ No chunks retrieved from Neon. Please check database seeding.")
            return

        print(f"   Successfully fetched {len(candidates)} candidates.")
        for i, c in enumerate(candidates):
            print(f"   Candidate [{i+1}] ID: {c.chunk_id} | Title: {c.title}")

        # 2. Initialize the Cross-Encoder Reranker
        print("\n2. Initializing Local Cross-Encoder Reranker...")
        start_init = time.time()
        reranker = CrossEncoderReranker()
        print(f"   Loaded model in {time.time() - start_init:.2f} seconds.")

        # 3. Rerank the candidates based on a test query
        query = "How do I secure my webhook endpoints by verifying signatures?"
        print(f"\n3. Reranking candidates for query: '{query}'...")
        
        start_rerank = time.time()
        reranked_results = reranker.rerank(
            query=query,
            results=candidates,
            top_k=3
        )
        print(f"   Reranking complete in {time.time() - start_rerank:.4f} seconds.")

        # 4. Compare before-and-after rankings
        print("\n📊 RERANKER COMPARISON SCORECARD:")
        print("-" * 50)
        for i, r in enumerate(reranked_results):
            # Find its original rank in the database candidates
            original_rank = next((idx + 1 for idx, c in enumerate(candidates) if c.chunk_id == r.chunk_id), "N/A")
            print(f"Rank [{i+1}] (Was Rank {original_rank}):")
            print(f"  - Chunk ID   : {r.chunk_id}")
            print(f"  - Title      : {r.title}")
            print(f"  - Excerpt    : {r.text[:100]}...\n")
        print("-" * 50)
        
        print("✅ Success: Pure retrieval and reranking engine are fully operational!")
            
    except Exception as e:
        print(f"❌ Retrieval/Reranking failed: {e}")
        
    print("=" * 60 + "\n")

if __name__ == "__main__":
    asyncio.run(test_pure_retrieval_with_reranker())
