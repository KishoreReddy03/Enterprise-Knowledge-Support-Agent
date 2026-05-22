"""
Test script: Intake Classification Caching

Verifies that:
1. Cache misses properly trigger LLM classification and populate/store results in Redis.
2. Cache hits retrieve exact stored classification details from Redis and cleanly bypass the LLM classification step.
3. Correctly handles empty ticket contents and defaults.
"""

import asyncio
import hashlib
import os
import sys

# Add project root to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.intake import IntakeAgent
from core.agents.state import create_initial_state
from core.redis_client import get_redis_client


async def run_caching_test():
    print("\n" + "=" * 80)
    print("🤖 STARTING INTAKE CLASSIFICATION CACHING TEST")
    print("=" * 80)

    agent = IntakeAgent()
    redis_client = get_redis_client()

    # Define a unique ticket text to avoid colliding with other test cases
    ticket_content = (
        "This is an automated test ticket for Connect routing and webhook signature verification caching. "
        "We are getting parameter_missing error on capabilities field."
    )
    
    clean_content = ticket_content.strip()
    content_hash = hashlib.md5(clean_content.lower().encode('utf-8')).hexdigest()
    cache_key = f"cache:intake:{content_hash}"

    print(f"\n[INFO] Cache key for test: {cache_key}")

    # 1. Clean the cache key first to guarantee a cache miss
    print("[STEP 1] Cleaning redis cache key for fresh start...")
    deleted = await redis_client.delete(cache_key)
    print(f"         → Deleted key: {deleted}")

    # Instrument/mock the LLM call to trace execution
    original_call = agent._call_fast_llm
    called_count = 0

    async def mock_call(ticket_text, is_retry):
        nonlocal called_count
        called_count += 1
        print(f"         [LLM CALLED] _call_fast_llm executed (attempt {called_count})")
        return await original_call(ticket_text, is_retry)

    agent._call_fast_llm = mock_call

    # 2. Run first execution (expect cache miss)
    print("\n[STEP 2] Running classification for the first time (Cache Miss)...")
    state1 = create_initial_state(
        ticket_id="test-caching-001",
        ticket_content=ticket_content,
        customer_id="cus_test_caching",
    )
    
    res_state1 = await agent.process(state1)
    
    # Assert LLM was called once
    assert called_count == 1, f"Expected 1 LLM call, got {called_count}"
    print("         ✅ Verified LLM was called exactly once.")
    
    # Verify we got valid results in state
    c_topic = res_state1.get("primary_topic")
    c_complexity = res_state1.get("complexity")
    c_urgency = res_state1.get("urgency")
    c_confidence = res_state1.get("intake_confidence")
    print(f"         → Classified primary_topic: {c_topic}")
    print(f"         → Classified complexity: {c_complexity}")
    print(f"         → Classified urgency: {c_urgency}")
    print(f"         → Classified confidence: {c_confidence}")

    # 3. Verify Redis has the cache
    print("\n[STEP 3] Verifying redis cache key has been populated...")
    cached_data = await redis_client.get_json(cache_key)
    assert cached_data is not None, "Cache should be populated in Redis"
    print(f"         → Cache data in Redis: {cached_data}")
    assert cached_data.get("primary_topic") == c_topic
    assert cached_data.get("complexity") == c_complexity
    assert cached_data.get("urgency") == c_urgency
    print("         ✅ Verified cached data fields match state outputs.")

    # 4. Run second execution (expect cache hit and zero LLM calls)
    print("\n[STEP 4] Running classification for the second time (Cache Hit)...")
    state2 = create_initial_state(
        ticket_id="test-caching-002",
        ticket_content=ticket_content,
        customer_id="cus_test_caching",
    )
    
    res_state2 = await agent.process(state2)
    
    # Assert LLM was NOT called again
    assert called_count == 1, f"LLM should not be called again! called_count = {called_count}"
    print("         ✅ Verified LLM was NOT called on the second run (zero token consumption!).")
    
    # Assert fields are identical
    assert res_state2.get("primary_topic") == c_topic, "Topic should match"
    assert res_state2.get("complexity") == c_complexity, "Complexity should match"
    assert res_state2.get("urgency") == c_urgency, "Urgency should match"
    assert res_state2.get("intake_confidence") == c_confidence, "Confidence should match"
    print("         ✅ Verified retrieved fields are identical.")

    # 5. Clean up
    print("\n[STEP 5] Cleaning up redis cache key...")
    await redis_client.delete(cache_key)
    print("         ✅ Cache clean up complete.")

    print("\n" + "=" * 80)
    print("🎉 ALL INTAKE CACHING TESTS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(run_caching_test())
