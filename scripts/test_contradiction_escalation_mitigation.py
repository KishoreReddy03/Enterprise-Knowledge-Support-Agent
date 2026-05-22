"""
Test script: Contradiction Escalation Mitigation & Conflict-Tolerant Drafting

Verifies that:
1. When retrieved sources contain high-severity contradictions, the SynthesisAgent resolves them based on authority (changelog > docs) and freshness.
2. The DraftingAgent proactively calls out and explains the resolved contradiction in the customer reply.
3. The EscalationAgent and QualityGate bypass immediate human escalation if the contradiction was successfully resolved, keeping human handoffs low.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

# Add project root to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.state import create_initial_state, TicketState
from core.agents.synthesis import synthesis_agent
from core.agents.drafting import drafting_agent
from core.agents.quality_gate import quality_gate_agent, escalation_agent
from core.agents.orchestrator import build_graph


async def run_mitigation_tests():
    print("\n" + "=" * 80)
    print("🛡️ STARTING CONTRADICTION ESCALATION MITIGATION & CONFLICT-TOLERANT TESTS")
    print("=" * 80)

    # 1. Define conflicting sources:
    # Source A: Stale Official Doc (says Connect uses custom_metadata_fields)
    # Source B: Fresh Stripe Changelog (Highest authority, confirms custom_metadata_fields is deprecated)
    docs_results = [
        {
            "chunk_id": "doc_001",
            "text": "Stripe Connect Custom accounts should use the 'custom_metadata_fields' property for tagging custom application attributes.",
            "title": "Stripe Connect Custom Accounts Reference Guide",
            "source_url": "https://stripe.com/docs/connect/custom-accounts-legacy",
            "source_type": "stripe_docs",
            "date": "2023-01-10",
            "is_stale": True,
        }
    ]

    changelog_results = [
        {
            "chunk_id": "changelog_001",
            "text": "Connect API Update: The legacy 'custom_metadata_fields' parameter has been fully deprecated and removed. All custom parameters must now be passed under the unified 'metadata' parameter across all accounts.",
            "title": "Stripe API Changelog - November 2025 Connect API Changes",
            "source_url": "https://stripe.com/docs/changelog/2025-11-connect",
            "source_type": "stripe_changelogs",
            "date": "2025-11-15",
            "is_stale": False,
        }
    ]

    # Create initial ticket state
    state = create_initial_state(
        ticket_id="test-contradiction-mitigation",
        ticket_content="Can I use custom_metadata_fields when creating Stripe Connect Custom accounts?",
        customer_id="cus_test_999",
    )
    state["docs_results"] = docs_results
    state["changelog_results"] = changelog_results

    # --- STEP 1: TEST SYNTHESIS AGENT CONTRADICTION DETECTION ---
    print("\n[STEP 1] Running contradiction check via SynthesisAgent...")
    
    mock_sufficiency_response = """
    {
        "coverage": "complete",
        "gaps": [],
        "has_direct_answer": true,
        "best_source": "documentation"
    }
    """
    
    mock_contradiction_response = """
    [
        {
            "topic": "Use of custom_metadata_fields vs metadata inside Connect creation request",
            "source_a": "Stripe Connect Custom Accounts Reference Guide recommends custom_metadata_fields",
            "source_b": "Stripe API Changelog states legacy custom_metadata_fields is deprecated and standard metadata should be used",
            "likely_correct": "The Stripe API Changelog (stripe_changelogs) has the absolute highest authority and is active/fresh (November 2025). The official guide is stale (2023) and legacy custom_metadata_fields is deprecated, so metadata is standard.",
            "severity": "high"
        }
    ]
    """

    with patch("core.agents.synthesis.call_fast", AsyncMock(return_value=mock_sufficiency_response)), \
         patch("core.agents.synthesis.call_strong", AsyncMock(return_value=mock_contradiction_response)):
        state = await synthesis_agent.process(state)

    print(f"         → Synthesis Decision    : '{state.get('synthesis_decision')}'")
    print(f"         → Contradictions Found  : {len(state.get('contradictions', []))}")
    
    # Assertions for Synthesis
    assert len(state.get("contradictions", [])) == 1, "Must detect exactly 1 contradiction"
    assert state.get("contradictions")[0].get("severity") == "high", "Contradiction must have high severity"
    assert "stripe_changelogs" in state.get("contradictions")[0].get("resolution").lower(), "Must identify changelog resolution"
    print("         ✅ Verified: Synthesis agent correctly parsed and resolved contradiction in favor of changelog.")

    # --- STEP 2: TEST DRAFTING AGENT CONFLICT-TOLERANT DRAFTING ---
    print("\n[STEP 2] Running drafting check via DraftingAgent...")
    
    mock_drafting_response = """
    {
        "draft_reply": "Thanks for reaching out! While older community discussions and legacy guides referenced the custom_metadata_fields parameter [1], Stripe's official API changelog updated on November 2025 specifies that you should use the standard unified metadata parameter instead [2] as custom_metadata_fields has been fully deprecated. Let me know if you run into anything else!",
        "confidence_score": 0.95,
        "rep_guidance": "HIGH_CONFIDENCE",
        "sources_cited": [
            {"chunk_id": "doc_001", "url": "https://stripe.com/docs/connect/custom-accounts-legacy", "title": "Stripe Connect Custom Accounts Reference Guide", "relevance": "legacy parameter details"},
            {"chunk_id": "changelog_001", "url": "https://stripe.com/docs/changelog/2025-11-connect", "title": "Stripe API Changelog - November 2025 Connect API Changes", "relevance": "unified metadata update"}
        ],
        "missing_information": null
    }
    """

    # Spy/Check if the system prompt has the dynamic resolved contradiction warning instructions
    system_prompt_built = drafting_agent._build_system_prompt(state)
    assert "PROACTIVELY CALL OUT AND EXPLAIN" in system_prompt_built.upper(), "System prompt must instruct on resolved discrepancies"
    assert "WHILE OLDER COMMUNITY DISCUSSIONS" in system_prompt_built.upper(), "System prompt must include the correct conversational example"
    
    with patch("core.agents.drafting.call_strong", AsyncMock(return_value=mock_drafting_response)):
        state = await drafting_agent.process(state)

    print(f"         → Draft Reply Generated : '{state.get('draft_reply')[:140]}...'")
    print(f"         → Rep Guidance          : '{state.get('rep_guidance')}'")
    
    # Assertions for Drafting
    assert "While older community discussions" in state.get("draft_reply"), "Draft must proactively call out and explain discrepancy"
    assert state.get("rep_guidance") == "HIGH_CONFIDENCE", "Rep guidance must remain HIGH_CONFIDENCE for resolved contradiction"
    print("         ✅ Verified: Drafting agent successfully generated conflict-tolerant customer response.")

    # --- STEP 3: TEST QUALITY GATE & ESCALATION BYPASS ---
    print("\n[STEP 3] Running QualityGate and EscalationAgent checks...")
    
    # Simulate Quality Gate checklist approval
    mock_quality_response = """
    {
        "answers_the_question": true,
        "all_claims_have_sources": true,
        "no_hallucinated_api_behavior": true,
        "appropriate_for_customer_tier": true,
        "specific_issues": [],
        "improvement_instruction": null
    }
    """

    class DummyGroundingReport:
        grounding_score = 1.0
        is_safe = True
        ungrounded_segments = []

    with patch("core.agents.quality_gate.call_fast", AsyncMock(return_value=mock_quality_response)), \
         patch("core.guardrails.grounding_verifier.GroundingVerifier.verify_grounding", AsyncMock(return_value=DummyGroundingReport())):
        state = await quality_gate_agent.process(state)

    # Resolve next routing decision
    next_route = quality_gate_agent.route(state)
    print(f"         → Quality Score          : {state.get('quality_score')}")
    print(f"         → Quality Routing Route  : '{next_route}'")
    
    # Assertions for Quality Gate Routing
    assert next_route == "approved", "Route must be 'approved' rather than escalated"
    
    # Run EscalationAgent process just to be absolutely sure it doesn't trigger escalation reason 'contradictions'
    reason = escalation_agent._determine_escalation_reason(state)
    print(f"         → Escalation Reason      : '{reason}' (if escalated)")
    
    # Assertions for Escalation
    assert reason != "contradictions", "Resolved contradiction must not trigger contradiction escalation"
    
    print("         ✅ Verified: Quality gate correctly bypassed human escalation for resolved contradictions!")
    
    print("\n" + "=" * 80)
    print("🎉 ALL CONTRADICTION ESCALATION MITIGATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(run_mitigation_tests())
