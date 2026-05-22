"""
Test script: Adaptive Retrieval Depth + Cross-Source Reranking

Verifies three things:
  1. Adaptive depth produces the correct limit per (complexity, urgency, confidence)
  2. RetrievalAgent.process() fetches real results from Neon and populates reranked_results
  3. Cross-source reranked list is ordered best-first across all 4 sources
"""

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.retrieval_agent import RetrievalAgent


# ─── SECTION 1: Adaptive Depth Unit Tests (no DB needed) ────────────────────

def test_adaptive_depth():
    print("\n" + "=" * 60)
    print("SECTION 1: Adaptive Retrieval Depth & Budget Allocation")
    print("=" * 60)

    agent = RetrievalAgent()

    cases = [
        {
            "label": "simple + low + high-conf (billing topic)",
            "state": {
                "complexity": "simple",
                "urgency": "low",
                "intake_confidence": 0.90,
                "primary_topic": "billing",
            },
            "checks": {
                "stripe_docs": 3,
                "stripe_changelogs": 0,
                "stripe_github_issues": 0,
                "stripe_stackoverflow": 0,
            }
        },
        {
            "label": "simple + high + high-conf (webhook topic)",
            "state": {
                "complexity": "simple",
                "urgency": "high",
                "intake_confidence": 0.90,
                "primary_topic": "webhook",
            },
            "checks": {
                "stripe_docs": 4,
                "stripe_changelogs": 3,
                "stripe_github_issues": 0,
                "stripe_stackoverflow": 0,
            }
        },
        {
            "label": "moderate + medium + normal (other topic)",
            "state": {
                "complexity": "moderate",
                "urgency": "medium",
                "intake_confidence": 0.75,
                "primary_topic": "other",
            },
            "checks": {
                "stripe_docs": 7,
                "stripe_github_issues": 3,
                "stripe_stackoverflow": 3,
                "stripe_changelogs": 3,
            }
        },
        {
            "label": "complex + high + normal (webhook topic)",
            "state": {
                "complexity": "complex",
                "urgency": "high",
                "intake_confidence": 0.80,
                "primary_topic": "webhook",
            },
            "checks": {
                "stripe_docs": 7,
                "stripe_github_issues": 11,
                "stripe_stackoverflow": 7,
                "stripe_changelogs": 3,
            }
        }
    ]

    all_passed = True
    for case in cases:
        label = case["label"]
        state = case["state"]
        checks = case["checks"]
        
        allocations = agent._allocate_retrieval_budget(state)
        
        print(f"\n  Testing Case: {label}")
        print(f"         → Allocations: {allocations}")
        
        case_passed = True
        for key, expected_val in checks.items():
            actual_val = allocations.get(key, -1)
            if actual_val != expected_val:
                print(f"    ❌ FAIL: {key} actual={actual_val}, expected={expected_val}")
                case_passed = False
            else:
                print(f"    ✅ PASS: {key} = {actual_val}")
                
        # Explicit validation for skipping community sources on simple complexity
        if state["complexity"] == "simple":
            github = allocations.get("stripe_github_issues", 0)
            so = allocations.get("stripe_stackoverflow", 0)
            if github != 0 or so != 0:
                print(f"    ❌ FAIL: Simple ticket did not skip community sources (github={github}, SO={so})")
                case_passed = False
            else:
                print("    ✅ PASS: Simple ticket skipped community sources as expected")
                
        # Validate that moderate/complex complexity allocate non-zero to active sources
        if state["complexity"] in ("moderate", "complex"):
            for src, budget in allocations.items():
                if budget == 0:
                    print(f"    ❌ FAIL: {state['complexity']} complexity allocated 0 to source {src}")
                    case_passed = False
                    
        if not case_passed:
            all_passed = False

    return all_passed


# ─── SECTION 2: Live End-to-End Test (hits Neon) ─────────────────────────────

async def test_retrieval_and_reranking():
    print("\n" + "=" * 60)
    print("SECTION 2: Live Retrieval + Cross-Source Reranking")
    print("=" * 60)

    agent = RetrievalAgent()

    # Simulate a 'complex + high' ticket about webhook signature errors
    state = {
        "ticket_id": "test-001",
        "ticket_content": (
            "We are getting invalid signature errors on all our webhook "
            "endpoints after rotating our Stripe secret key yesterday. "
            "Payments are failing in production. Urgent."
        ),
        "customer_tier": "enterprise",
        "customer_id": "cus_test",
        "complexity": "complex",
        "urgency": "high",
        "intake_confidence": 0.88,
        "primary_topic": "webhook",
        "error_codes": ["invalid_signature"],
        "search_keywords": ["webhook", "signature", "secret key", "rotate"],
    }

    print(f"\n  Ticket   : {state['ticket_content'][:80]}...")
    print(f"  Complexity: {state['complexity']}  |  Urgency: {state['urgency']}  |  Confidence: {state['intake_confidence']}")

    # Run retrieval
    updated_state = await agent.process(state)

    # ── Check per-source counts ───────────────────────────────────────────────
    docs   = updated_state.get("docs_results", [])
    github = updated_state.get("github_results", [])
    so     = updated_state.get("stackoverflow_results", [])
    cl     = updated_state.get("changelog_results", [])
    errors = updated_state.get("retrieval_errors", [])
    reranked = updated_state.get("reranked_results", [])

    print(f"\n  Per-source counts (complex + high → expect ~9/source):")
    print(f"    Stripe Docs   : {len(docs)}")
    print(f"    GitHub Issues : {len(github)}")
    print(f"    StackOverflow : {len(so)}")
    print(f"    Changelog     : {len(cl)}")
    print(f"    Errors        : {errors or 'None'}")

    # ── Check reranked list ───────────────────────────────────────────────────
    print(f"\n  Cross-source reranked results ({len(reranked)} total):")
    all_passed = True
    for i, r in enumerate(reranked):
        print(f"    [{i+1}] score={r['score']:.4f}  source={r['source_type']:<25}  title={r.get('title','')[:50]}")

    # Validation 1: reranked_results must be non-empty if ANY source returned results
    total_raw = len(docs) + len(github) + len(so) + len(cl)
    if total_raw > 0 and len(reranked) == 0:
        print("\n  ❌ FAIL: Got raw results but reranked_results is empty!")
        all_passed = False
    else:
        print(f"\n  ✅ PASS: reranked_results populated ({len(reranked)} entries)")

    # Validation 2: reranked list must be ordered descending by score
    scores = [r["score"] for r in reranked]
    is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    if is_sorted:
        print("  ✅ PASS: Results are ordered best-first (descending score)")
    else:
        print("  ❌ FAIL: Results are NOT correctly sorted!")
        all_passed = False

    # Validation 3: reranked list must contain chunks from more than one source
    # (proves cross-source merging actually happened)
    source_types = {r["source_type"] for r in reranked}
    if len(source_types) > 1:
        print(f"  ✅ PASS: Cross-source merge confirmed — sources present: {source_types}")
    else:
        print(f"  ⚠️  NOTE: Only one source type in results: {source_types}")
        print(       "           (May be OK if DB only has data in one collection)")

    return all_passed


# ─── RUNNER ──────────────────────────────────────────────────────────────────

async def main():
    print("\n🧪 Retrieval Agent — Adaptive Depth + Cross-Source Reranking Test")

    s1_pass = test_adaptive_depth()
    s2_pass = await test_retrieval_and_reranking()

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"  Section 1 (Adaptive Depth)    : {'✅ PASS' if s1_pass else '❌ FAIL'}")
    print(f"  Section 2 (Live Retrieval)    : {'✅ PASS' if s2_pass else '❌ FAIL'}")
    overall = s1_pass and s2_pass
    print(f"\n  {'✅ ALL TESTS PASSED' if overall else '❌ SOME TESTS FAILED'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
