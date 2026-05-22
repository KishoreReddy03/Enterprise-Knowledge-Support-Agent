"""
Test script: Grounding-to-Drafting Loop End-to-End Test

Verifies the integration of the GroundingVerifier directly into the QualityGateAgent
evaluation node, ensuring that grounding failures successfully set a categorical routing
decision to "revise", capture ungrounded claims into "grounding_feedback", and feed
them back dynamically to the DraftingAgent's prompt during the revision attempt.
"""

import asyncio
import os
import sys

# Add project root to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.state import create_initial_state, SourceResult
from core.agents.quality_gate import quality_gate_agent
from core.agents.drafting import drafting_agent

async def run_grounding_loop_test():
    print("\n" + "=" * 80)
    print("🔄 TESTING GROUNDING VERIFICATION IN-GRAPH REVISION LOOP")
    print("=" * 80)

    # 1. Create a mock initial ticket state
    state = create_initial_state(
        ticket_id="ticket_test_grounding_loop",
        ticket_content="How do I create a Stripe Connect Custom account in Python?",
        customer_id="cus_test_loop",
        customer_tier="enterprise"
    )

    # 2. Add standard retrieved source chunks that ONLY support basic connect endpoints
    state["docs_results"] = [
        {
            "chunk_id": "chunk_doc_connect_1",
            "source_type": "stripe_doc",
            "title": "Creating Connect Accounts",
            "text": (
                "To create a custom account, make a POST request to /v1/accounts with "
                "type='custom' and capabilities set. Use the official stripe python library "
                "via stripe.Account.create(type='custom')."
            ),
            "source_url": "https://stripe.com/docs/api/accounts/create"
        }
    ]

    # 3. Fabricate a draft reply that contains ungrounded technical parameters & endpoints
    # "/v1/custom_registrations" and "billing_grace_period" do not exist in the source chunk.
    unsupported_claim_1 = "hit the /v1/custom_registrations endpoint"
    unsupported_claim_2 = "pass the billing_grace_period parameter to set the trial duration"
    
    state["draft_reply"] = (
        "To set up a custom account in Python, first make sure you hit the /v1/custom_registrations endpoint "
        "and pass the billing_grace_period parameter to set the trial duration. "
        "Then you can call stripe.Account.create(type='custom') [1] to complete the creation process."
    )
    state["sources_cited"] = [
        {
            "chunk_id": "chunk_doc_connect_1",
            "title": "Creating Connect Accounts",
            "url": "https://stripe.com/docs/api/accounts/create",
            "relevance": "Supports creating custom connect account"
        }
    ]
    state["synthesized_context"] = "Use stripe.Account.create(type='custom') to create a custom connect account."
    
    print("\n[PHASE 1] Raw Draft Reply to evaluate:")
    print(f"\"\"\"\n{state['draft_reply']}\n\"\"\"")

    # 4. Process the state using QualityGateAgent (with mocked LLM calls)
    print("\nProcessing draft reply through QualityGateAgent (evaluating quality + grounding)...")
    from unittest.mock import AsyncMock, patch

    mock_qg_response_1 = """{
        "answers_the_question": true,
        "all_claims_have_sources": true,
        "no_hallucinated_api_behavior": false,
        "appropriate_for_customer_tier": true,
        "specific_issues": ["Draft reply contains ungrounded endpoint /v1/custom_registrations and parameter billing_grace_period."],
        "improvement_instruction": "Remove ungrounded custom_registrations endpoint and billing_grace_period parameter."
    }"""

    mock_qg_response_2 = """{
        "answers_the_question": true,
        "all_claims_have_sources": true,
        "no_hallucinated_api_behavior": true,
        "appropriate_for_customer_tier": true,
        "specific_issues": [],
        "improvement_instruction": null
    }"""

    mock_gv_response_1 = """[
      {
        "segment": "To set up a custom account in Python, first make sure you hit the /v1/custom_registrations endpoint and pass the billing_grace_period parameter to set the trial duration.",
        "is_factual_claim": true,
        "is_grounded": false,
        "source_chunk_ids": ["chunk_doc_connect_1"],
        "reason": "Absent in source chunks."
      },
      {
        "segment": "Then you can call stripe.Account.create(type='custom') [1] to complete the creation process.",
        "is_factual_claim": true,
        "is_grounded": true,
        "source_chunk_ids": ["chunk_doc_connect_1"],
        "reason": "Supported by chunk_doc_connect_1."
      }
    ]"""

    mock_gv_response_2 = """[
      {
        "segment": "To create a custom account in Python, call stripe.Account.create(type='custom') [1].",
        "is_factual_claim": true,
        "is_grounded": true,
        "source_chunk_ids": ["chunk_doc_connect_1"],
        "reason": "Directly supported by docs."
      },
      {
        "segment": "This will set up the custom account according to your needs.",
        "is_factual_claim": false,
        "is_grounded": true,
        "source_chunk_ids": [],
        "reason": "Introductory statement."
      }
    ]"""

    mock_draft_response = """{
        "draft_reply": "To create a custom account in Python, call stripe.Account.create(type='custom') [1]. This will set up the custom account according to your needs.",
        "confidence_score": 0.85,
        "rep_guidance": "HIGH_CONFIDENCE",
        "sources_cited": [
            {"chunk_id": "chunk_doc_connect_1", "url": "https://stripe.com/docs/api/accounts/create", "title": "Creating Connect Accounts", "relevance": "Supports creating custom connect account"}
        ],
        "missing_information": null
    }"""

    qg_mock = AsyncMock(side_effect=[mock_qg_response_1, mock_qg_response_2])
    gv_mock = AsyncMock(side_effect=[mock_gv_response_1, mock_gv_response_2])
    draft_mock = AsyncMock(return_value=mock_draft_response)

    with patch("core.agents.quality_gate.call_fast", qg_mock), \
         patch("core.guardrails.grounding_verifier.call_strong", gv_mock), \
         patch("core.agents.drafting.call_strong", draft_mock):
        
        evaluated_state = await quality_gate_agent.process(state)

        print("\n[PHASE 2] Evaluation Results:")
        print(f"   - Quality Score      : {evaluated_state.get('quality_score'):.2f}")
        print(f"   - Quality Issues     : {evaluated_state.get('quality_issues')}")
        print(f"   - Routing Decision   : {evaluated_state.get('llm_routing_decision')}")
        print(f"   - Grounding Feedback : {evaluated_state.get('grounding_feedback')}")
        print(f"   - Revision Count     : {evaluated_state.get('revision_count')}")

        # Assertions for QualityGate integration
        assert evaluated_state.get("llm_routing_decision") == "revise", "Routing decision must be 'revise' due to grounding failure."
        assert evaluated_state.get("revision_count") == 1, "Revision count must be incremented to 1."
        assert len(evaluated_state.get("grounding_feedback", [])) > 0, "Grounding feedback list must contain the ungrounded claims."
        
        print("\n✅ Quality Gate Grounding Evaluation assertions passed successfully!")

        # 5. Build prompt for DraftingAgent using the updated state to verify feedback injection
        print("\n[PHASE 3] Generating system prompt for DraftingAgent revision...")
        system_prompt = drafting_agent._build_system_prompt(evaluated_state)
        
        print("\nDraftingAgent System Prompt containing revision warnings:")
        print("=" * 80)
        print(system_prompt)
        print("=" * 80)

        # Assert that the grounding feedback warnings are successfully formatted inside the prompt template
        assert "WARNING: PREVIOUS DRAFT WAS REJECTED DUE TO UNGROUNDED/HALLUCINATED CLAIMS!" in system_prompt, "System prompt must contain the ungrounded claims warning header."
        assert "billing_grace_period" in system_prompt or "/v1/custom_registrations" in system_prompt, "System prompt must explicitly list the rejected ungrounded claims."
        print("\n✅ Drafting Agent Grounding Feedback Warning assertions passed successfully!")

        # 6. Run DraftingAgent again to generate the revised reply
        print("\n[PHASE 4] Running DraftingAgent to generate a revised, grounded reply...")
        revised_state = await drafting_agent.process(evaluated_state)

        print("\n[PHASE 5] Revised Response:")
        print(f"   - Revised Reply:\n\"\"\"\n{revised_state.get('draft_reply')}\n\"\"\"")
        print(f"   - Confidence Score: {revised_state.get('confidence_score'):.2f}")
        print(f"   - Rep Guidance    : {revised_state.get('rep_guidance')}")

        # 7. Evaluate the revised response again to prove the loop works and terminates
        print("\n[PHASE 6] Evaluating the revised reply again through QualityGateAgent...")
        final_state = await quality_gate_agent.process(revised_state)

        print("\n[PHASE 7] Final Evaluation Results after revision:")
        print(f"   - Quality Score      : {final_state.get('quality_score'):.2f}")
        print(f"   - Quality Issues     : {final_state.get('quality_issues')}")
        print(f"   - Routing Decision   : {final_state.get('llm_routing_decision')}")
        print(f"   - Grounding Feedback : {final_state.get('grounding_feedback')}")
        print(f"   - Revision Count     : {final_state.get('revision_count')}")

    print("\n" + "=" * 80)
    print("🎉 ALL IN-GRAPH GROUNDING ENFORCEMENT LOOP TESTS PASSED!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(run_grounding_loop_test())
