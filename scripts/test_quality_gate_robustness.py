"""
Test script: Quality Gate Robustness & Deterministic Safety Contracts

Verifies that:
1. QualityGateAgent behaves deterministically when checklist criteria are failed.
2. Even if the LLM self-report yields an 'approved' decision or high score,
   technical checklist failures (hallucination, missing sources, or unanswered questions)
   trigger a Python contract override that forces the route to 'revise'.
3. JSON parse failures safely default to 'revise' to avoid boundary issues.
"""

import asyncio
import os
import sys

# Add project root to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.quality_gate import QualityGateAgent
from core.agents.state import create_initial_state, TicketState


async def run_quality_gate_robustness_tests():
    print("\n" + "=" * 80)
    print("🛡️ STARTING QUALITY GATE ROBUSTNESS & SAFETY CONTRACT TESTS")
    print("=" * 80)

    agent = QualityGateAgent()

    # Define base ticket state with draft reply
    base_state = create_initial_state(
        ticket_id="test-quality-robustness",
        ticket_content="How do I list customer invoices?",
        customer_id="cus_test_001",
    )
    base_state["draft_reply"] = "To list invoices, call Stripe.invoices.list()."
    base_state["sources_cited"] = [
        {"title": "Invoices API Documentation", "url": "https://stripe.com/docs/api/invoices", "relevance": "listing invoices"}
    ]

    # CASE 1: Deterministic Override for Hallucination Check Failure
    # LLM returns approved and 0.85 score, but flags hallucination = False
    print("\n[CASE 1] Simulating Hallucination checklist failure with LLM approval...")
    
    # We mock _parse_response to return the simulated LLM response
    original_parse = agent._parse_response
    agent._parse_response = lambda text: {
        "answers_the_question": True,
        "all_claims_have_sources": True,
        "no_hallucinated_api_behavior": False,  # Hallucinated API behavior detected!
        "appropriate_for_customer_tier": True,
        "overall_score": 0.85,
        "routing_decision": "approved",
        "specific_issues": ["Simulated hallucination issue"],
        "improvement_instruction": "Fix simulated hallucination"
    }

    state1 = base_state.copy()
    
    # We call process. (We temporarily mock call_fast to return a dummy string)
    from unittest.mock import AsyncMock, patch
    
    with patch("core.agents.quality_gate.call_fast", AsyncMock(return_value="{}")):
        res_state1 = await agent.process(state1)

    print(f"         → Quality Score          : {res_state1.get('quality_score')}")
    print(f"         → LLM Routing Decision   : '{res_state1.get('llm_routing_decision')}'")
    print(f"         → Final Router Decision  : '{agent.route(res_state1)}'")
    
    # Assertions
    assert res_state1.get("llm_routing_decision") == "revise", "Must override routing decision to 'revise'"
    assert res_state1.get("quality_score") < 0.60, "Must override quality score to below 0.60"
    assert agent.route(res_state1) == "revise", "Final route must be 'revise'"
    print("         ✅ Verified deterministic safety override for hallucinated behavior successfully.")

    # CASE 2: Deterministic Override for Missing Sources Failure
    # LLM returns approved and 0.90 score, but flags all_claims_have_sources = False
    print("\n[CASE 2] Simulating Missing Sources checklist failure with LLM approval...")
    agent._parse_response = lambda text: {
        "answers_the_question": True,
        "all_claims_have_sources": False,  # Claims are missing sources!
        "no_hallucinated_api_behavior": True,
        "appropriate_for_customer_tier": True,
        "overall_score": 0.90,
        "routing_decision": "approved",
        "specific_issues": ["Simulated missing source issue"],
        "improvement_instruction": "Fix simulated missing source"
    }

    state2 = base_state.copy()
    with patch("core.agents.quality_gate.call_fast", AsyncMock(return_value="{}")):
        res_state2 = await agent.process(state2)

    print(f"         → Quality Score          : {res_state2.get('quality_score')}")
    print(f"         → LLM Routing Decision   : '{res_state2.get('llm_routing_decision')}'")
    print(f"         → Final Router Decision  : '{agent.route(res_state2)}'")
    
    # Assertions
    assert res_state2.get("llm_routing_decision") == "revise", "Must override routing decision to 'revise'"
    assert res_state2.get("quality_score") < 0.60, "Must override quality score to below 0.60"
    assert agent.route(res_state2) == "revise", "Final route must be 'revise'"
    print("         ✅ Verified deterministic safety override for missing sources successfully.")

    # CASE 3: Deterministic Override for Unanswered Question Failure
    # LLM returns approved and 0.80 score, but flags answers_the_question = False
    print("\n[CASE 3] Simulating Unanswered Question checklist failure with LLM approval...")
    agent._parse_response = lambda text: {
        "answers_the_question": False,  # Question was not answered!
        "all_claims_have_sources": True,
        "no_hallucinated_api_behavior": True,
        "appropriate_for_customer_tier": True,
        "overall_score": 0.80,
        "routing_decision": "approved",
        "specific_issues": ["Simulated unanswered issue"],
        "improvement_instruction": "Fix simulated unanswered question"
    }

    state3 = base_state.copy()
    with patch("core.agents.quality_gate.call_fast", AsyncMock(return_value="{}")):
        res_state3 = await agent.process(state3)

    print(f"         → Quality Score          : {res_state3.get('quality_score')}")
    print(f"         → LLM Routing Decision   : '{res_state3.get('llm_routing_decision')}'")
    print(f"         → Final Router Decision  : '{agent.route(res_state3)}'")
    
    # Assertions
    assert res_state3.get("llm_routing_decision") == "revise", "Must override routing decision to 'revise'"
    assert res_state3.get("quality_score") < 0.60, "Must override quality score to below 0.60"
    assert agent.route(res_state3) == "revise", "Final route must be 'revise'"
    print("         ✅ Verified deterministic safety override for unanswered question successfully.")

    # CASE 4: Parse Failure Default
    # Parse returns None, should default to 'revise' route with 0.40 score
    print("\n[CASE 4] Simulating JSON parse failure...")
    agent._parse_response = lambda text: None

    state4 = base_state.copy()
    with patch("core.agents.quality_gate.call_fast", AsyncMock(return_value="{}")):
        res_state4 = await agent.process(state4)

    print(f"         → Quality Score          : {res_state4.get('quality_score')}")
    print(f"         → LLM Routing Decision   : '{res_state4.get('llm_routing_decision')}'")
    print(f"         → Final Router Decision  : '{agent.route(res_state4)}'")
    
    # Assertions
    assert res_state4.get("llm_routing_decision") == "revise", "JSON parse failure must default to 'revise'"
    assert res_state4.get("quality_score") == 0.40, "JSON parse failure must set quality score to 0.40"
    assert agent.route(res_state4) == "revise", "Final route must be 'revise'"
    print("         ✅ Verified safe default routing under JSON parse failures successfully.")

    # Restore original _parse_response
    agent._parse_response = original_parse

    print("\n" + "=" * 80)
    print("🎉 ALL QUALITY GATE ROBUSTNESS CONTRACT TESTS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(run_quality_gate_robustness_tests())
