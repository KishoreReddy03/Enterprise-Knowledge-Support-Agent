"""
Test script: Pluggable Intake Routers & Adaptive Routing

Verifies that:
1. The pluggable router architecture correctly supports custom routing logic.
2. ConfidenceAdaptiveRouter dynamically adapts routing decisions based on confidence and customer tier.
3. Decoupled router contracts keep full backward compatibility.
"""

import asyncio
import os
import sys

# Add project root to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.intake import IntakeAgent, ConfidenceAdaptiveRouter, StaticRuleRouter
from core.agents.state import create_initial_state


async def run_router_tests():
    print("\n" + "=" * 80)
    print("🤖 STARTING PLUGGABLE ROUTER INTEGRATION TESTS")
    print("=" * 80)

    adaptive_router = ConfidenceAdaptiveRouter()
    agent = IntakeAgent(router=adaptive_router)

    # Test Case 1: Low-Confidence Safety Fallback for Urgent Enterprise Ticket
    # Even if classified as 'simple' complexity, low confidence (0.4) on an enterprise high-urgency ticket should escalate!
    print("\n[CASE 1] Testing Low-Confidence Safety Fallback...")
    state1 = create_initial_state(
        ticket_id="test-router-001",
        ticket_content="Critical webhook failure in prod!",
        customer_id="cus_ent_001",
        customer_tier="enterprise",
    )
    # Manually simulate classification outputs with low confidence
    state1["complexity"] = "simple"
    state1["urgency"] = "high"
    state1["intake_confidence"] = 0.40
    state1["primary_topic"] = "webhook"

    route1 = agent.route(state1)
    assert route1 == "escalate", f"Expected escalate, got {route1}"
    print("         ✅ Low confidence safety fallback routed to 'escalate' as expected.")

    # Test Case 2: High-Touch Escalation for Sensitive Enterprise Billing
    # A moderate complexity billing ticket on an enterprise tier should escalate directly
    print("\n[CASE 2] Testing Enterprise Billing High-Touch Escalation...")
    state2 = create_initial_state(
        ticket_id="test-router-002",
        ticket_content="Double charge on my business subscription.",
        customer_id="cus_ent_002",
        customer_tier="enterprise",
    )
    state2["complexity"] = "moderate"
    state2["urgency"] = "medium"
    state2["intake_confidence"] = 0.90
    state2["primary_topic"] = "billing"

    route2 = agent.route(state2)
    assert route2 == "escalate", f"Expected escalate, got {route2}"
    print("         ✅ Sensitive Enterprise billing query routed directly to 'escalate' as expected.")

    # Test Case 3: Enterprise Simple Billing gets broad parallel retrieval instead of simple retrieval
    print("\n[CASE 3] Testing Enterprise Simple Billing Broad Search Shift...")
    state3 = create_initial_state(
        ticket_id="test-router-003",
        ticket_content="How do I change my billing card details?",
        customer_id="cus_ent_003",
        customer_tier="enterprise",
    )
    state3["complexity"] = "simple"
    state3["urgency"] = "low"
    state3["intake_confidence"] = 0.95
    state3["primary_topic"] = "billing"

    route3 = agent.route(state3)
    assert route3 == "parallel_retrieval", f"Expected parallel_retrieval, got {route3}"
    print("         ✅ Simple Enterprise billing query elevated to 'parallel_retrieval' for broader search context.")

    # Test Case 4: Backwards Compatibility Default Router
    print("\n[CASE 4] Verifying Default Router Backwards Compatibility...")
    default_agent = IntakeAgent()
    assert isinstance(default_agent.router, StaticRuleRouter), "Default router must be StaticRuleRouter"
    print("         ✅ Verified default router initialization maps to StaticRuleRouter.")

    print("\n" + "=" * 80)
    print("🎉 ALL PLUGGABLE ROUTER TESTS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(run_router_tests())
