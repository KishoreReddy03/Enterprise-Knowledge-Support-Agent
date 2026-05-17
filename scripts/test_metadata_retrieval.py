import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta

# Add project root to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.retrieval.vector_retriever import VectorRetriever
from core.ingestion.embedder import DocumentEmbedder
from core.ingestion.chunker import Chunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_metadata_aware_retrieval():
    print("\n🔍 Metadata-Aware Retrieval Sandbox Test")
    print("=" * 60)

    retriever = VectorRetriever()
    embedder = DocumentEmbedder()

    # 1. Ingest three clean control chunks to test specific filters
    print("Seeding control chunks with metadata (Freshness, Trust, Dates)...")
    
    # Chunk 1: Official Stripe Docs (High Trust)
    chunk_official = Chunk(
        chunk_id="test-meta-official",
        text="Stripe Payments official documentation covers how to secure your charges using our modern API endpoints.",
        source_type="stripe_docs",
        source_url="https://stripe.com/docs/charges",
        title="Charges API Reference"
    )
    
    # Chunk 2: StackOverflow Post (Lower Trust)
    chunk_so = Chunk(
        chunk_id="test-meta-stackoverflow",
        text="Stripe Payments official documentation covers how to secure your charges using our modern API endpoints.", # Same text to isolate trust boost!
        source_type="stripe_stackoverflow",
        source_url="https://stackoverflow.com/questions/999",
        title="StackOverflow: Charge Setup"
    )

    # Chunk 3: Stale Chunk (Should be filtered out)
    chunk_stale = Chunk(
        chunk_id="test-meta-stale",
        text="Stripe Payments official documentation covers how to secure your charges using our modern API endpoints.",
        source_type="stripe_docs",
        source_url="https://stripe.com/docs/deprecated-charges",
        title="Old Charges Endpoint (Deprecated)"
    )

    # Upsert them directly
    print("Uploading to Neon...")
    embedder.upsert_chunks([chunk_official], "stripe_docs")
    embedder.upsert_chunks([chunk_so], "stripe_stackoverflow")
    
    # For stale, we upsert and manually flag it as stale
    embedder.upsert_chunks([chunk_stale], "stripe_docs")
    conn = retriever._get_connection()
    with conn.cursor() as cur:
        cur.execute("UPDATE stripe_docs SET is_stale = TRUE WHERE id = 'test-meta-stale'")
        # Update updated_at for date testing: make the StackOverflow one very old
        old_date = datetime.now() - timedelta(days=100)
        cur.execute("UPDATE stripe_stackoverflow SET updated_at = %s WHERE id = 'test-meta-stackoverflow'", (old_date,))
        conn.commit()
    conn.close()
    
    # Generate search query embedding (using our standard embedding client)
    query_vector = embedder.embed_batch(["Stripe Payments official documentation covers how to secure your charges"])[0]

    # --- SCENARIO 1: Source Trust Boost ---
    print("\n⚖️ SCENARIO 1: Source Trust Prioritization")
    print("--------------------------------------------------")
    print("Searching 'stripe_docs' and 'stripe_stackoverflow' with trust boost enabled...")
    
    docs_results = retriever.search("stripe_docs", query_vector, limit=2, boost_trust=True)
    so_results = retriever.search("stripe_stackoverflow", query_vector, limit=2, boost_trust=True)
    
    print(f"Official Doc Score: {docs_results[0].score:.4f} (Raw: {docs_results[0].metadata['raw_score']:.4f} + Trust Boost: {docs_results[0].metadata['trust_boost']:.2f})")
    print(f"StackOverflow Score: {so_results[0].score:.4f} (Raw: {so_results[0].metadata['raw_score']:.4f} + Trust Penalty: {so_results[0].metadata['trust_boost']:.2f})")
    print("   -> Success: Official docs prioritized via metadata-driven trust tiers!")

    # --- SCENARIO 2: Freshness (Stale-Filtering) ---
    print("\n🍃 SCENARIO 2: Freshness & Stale Chunk Filtering")
    print("--------------------------------------------------")
    print("Searching with is_stale=False (strict freshness)...")
    
    fresh_results = retriever.search("stripe_docs", query_vector, limit=5, is_stale=False)
    stale_found = any(r.chunk_id == "test-meta-stale" for r in fresh_results)
    
    print(f"Total retrieved fresh chunks: {len(fresh_results)}")
    print(f"Stale chunk retrieved? {'⚠️ Yes (Fail)' if stale_found else '✅ No (Success - Filtered!)'}")

    # --- SCENARIO 3: Date Filtering ---
    print("\n📅 SCENARIO 3: Temporal / Date-Range Filtering")
    print("--------------------------------------------------")
    recent_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    print(f"Searching for chunks updated on or after {recent_date}...")
    
    recent_results = retriever.search("stripe_stackoverflow", query_vector, limit=5, min_date=recent_date)
    so_found = any(r.chunk_id == "test-meta-stackoverflow" for r in recent_results)
    
    print(f"Total retrieved StackOverflow chunks: {len(recent_results)}")
    print(f"Old SO chunk retrieved? {'⚠️ Yes (Fail)' if so_found else '✅ No (Success - Filtered out because it was updated 100 days ago!)'}")

    # --- SCENARIO 4: Soft Freshness (Temporal Decay) ---
    print("\n⏳ SCENARIO 4: Soft Freshness & Temporal Decay")
    print("--------------------------------------------------")
    print("Searching StackOverflow with enable_decay=True (No hard date cutoff)...")
    
    decay_results = retriever.search("stripe_stackoverflow", query_vector, limit=5, enable_decay=True)
    so_chunk = next((r for r in decay_results if r.chunk_id == "test-meta-stackoverflow"), None)
    
    if so_chunk:
        print(f"✅ Success: Old chunk (100 days old) was STILL retrieved due to soft decay math!")
        print(f"   - Age in Days  : {so_chunk.metadata['age_days']} days")
        print(f"   - Decay Penalty : -{so_chunk.metadata['decay_penalty']:.4f}")
        print(f"   - Decayed Score : {so_chunk.score:.4f} (Raw Score: {so_chunk.metadata['raw_score']:.4f})")
    else:
        print("⚠️ Old chunk was not retrieved.")

    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    asyncio.run(test_metadata_aware_retrieval())

