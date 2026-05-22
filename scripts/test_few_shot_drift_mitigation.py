"""
Test script: Dynamic Few-Shot Drift Risk Mitigation Test Suite

Verifies:
1. Outdated or deprecated examples receive appropriate score decay.
2. Examples below the similarity threshold are rejected.
3. Structurally inconsistent or inactive examples are skipped.
4. Fallbacks work seamlessly when metadata fields are missing.
"""

import os
import sys
from datetime import date, timedelta
from unittest.mock import MagicMock

# Add project root to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intelligence.few_shot_selector import FewShotSelector

def test_drift_mitigation():
    print("\n" + "=" * 80)
    print("🛡️ TESTING DYNAMIC FEW-SHOT DRIFT RISK MITIGATION & GUARDRAILS")
    print("=" * 80)

    # 1. Create a selector instance
    selector = FewShotSelector()
    
    # 2. Setup mock embedder
    mock_embedder = MagicMock()
    # Assume 3-dimensional embeddings for simplicity
    mock_embedder.embed_text.return_value = [1.0, 0.0, 0.0]
    selector._embedder = mock_embedder
    
    # 3. Define standard and boundary examples for testing
    today_str = date.today().strftime("%Y-%m-%d")
    two_years_ago_str = (date.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    
    test_examples = [
        # Perfect baseline example
        {
            "id": "ex_perfect",
            "ticket_content": "Query ticket content",
            "ideal_response": "This is a perfect standard response [1]. It cites documentation.",
            "active": True,
            "deprecated": False,
            "date": today_str,
            "classification": {"primary_topic": "billing"}
        },
        # Inactive example (should be skipped)
        {
            "id": "ex_inactive",
            "ticket_content": "Query ticket content",
            "ideal_response": "This response is inactive [1].",
            "active": False,
            "deprecated": False,
            "date": today_str,
            "classification": {"primary_topic": "billing"}
        },
        # Deprecated example (should be skipped)
        {
            "id": "ex_deprecated",
            "ticket_content": "Query ticket content",
            "ideal_response": "This response is deprecated [1].",
            "active": True,
            "deprecated": True,
            "date": today_str,
            "classification": {"primary_topic": "billing"}
        },
        # Weak example (should be rejected by threshold check)
        {
            "id": "ex_weak",
            "ticket_content": "Completely unrelated content",
            "ideal_response": "Unrelated answer [1].",
            "active": True,
            "deprecated": False,
            "date": today_str,
            "classification": {"primary_topic": "billing"}
        },
        # Outdated example (should receive age-based decay)
        {
            "id": "ex_outdated",
            "ticket_content": "Query ticket content",
            "ideal_response": "This response is two years old [1].",
            "active": True,
            "deprecated": False,
            "date": two_years_ago_str,
            "classification": {"primary_topic": "billing"}
        },
        # Structurally inconsistent example: placeholder
        {
            "id": "ex_placeholder",
            "ticket_content": "Query ticket content",
            "ideal_response": "This contains TODO inside ideal response [1].",
            "active": True,
            "deprecated": False,
            "date": today_str,
            "classification": {"primary_topic": "billing"}
        },
        # Structurally inconsistent example: mismatched brackets
        {
            "id": "ex_mismatched_brackets",
            "ticket_content": "Query ticket content",
            "ideal_response": "This has unbalanced [brackets in response.",
            "active": True,
            "deprecated": False,
            "date": today_str,
            "classification": {"primary_topic": "billing"}
        },
        # Structurally inconsistent example: empty ticket
        {
            "id": "ex_empty_ticket",
            "ticket_content": "",
            "ideal_response": "Valid response [1].",
            "active": True,
            "deprecated": False,
            "date": today_str,
            "classification": {"primary_topic": "billing"}
        },
        # Minimal fallback example: missing active, deprecated, date
        {
            "id": "ex_missing_metadata",
            "ticket_content": "Query ticket content",
            "ideal_response": "Valid response with missing metadata [1].",
            "classification": {"primary_topic": "billing"}
        }
    ]
    
    # Pre-configure embeddings to simulate similarities
    # We'll set embeddings such that:
    # - "ex_perfect", "ex_inactive", "ex_deprecated", "ex_outdated", "ex_placeholder", "ex_mismatched_brackets", "ex_empty_ticket", "ex_missing_metadata": exact match (similarity = 1.0)
    # - "ex_weak": perpendicular (similarity = 0.0)
    test_embeddings = []
    for ex in test_examples:
        if ex["id"] == "ex_weak":
            test_embeddings.append([0.0, 1.0, 0.0])  # Perpendicular -> similarity 0.0
        else:
            test_embeddings.append([1.0, 0.0, 0.0])  # Identical -> similarity 1.0

    selector._examples = test_examples
    selector._example_embeddings = test_embeddings
    selector._loaded = True
    
    # -------------------------------------------------------------------------
    # CASE 1: Inactive and Deprecated examples are skipped
    # -------------------------------------------------------------------------
    print("\n[CASE 1] Verifying inactive and deprecated filter...")
    selected = selector.select("Query ticket content", primary_topic="billing", n=10)
    selected_ids = [ex["id"] for ex in selected]
    print(f"         → Selected IDs: {selected_ids}")
    assert "ex_inactive" not in selected_ids, "Inactive example should not be selected."
    assert "ex_deprecated" not in selected_ids, "Deprecated example should not be selected."
    print("         ✅ Passed! Inactive and deprecated examples were correctly skipped.")

    # -------------------------------------------------------------------------
    # CASE 2: Weak examples are rejected (below SIMILARITY_THRESHOLD = 0.45)
    # -------------------------------------------------------------------------
    print("\n[CASE 2] Verifying similarity threshold (reject below 0.45)...")
    assert "ex_weak" not in selected_ids, "Weak example with similarity below threshold should be rejected."
    print("         ✅ Passed! Weak examples below similarity threshold were rejected.")

    # -------------------------------------------------------------------------
    # CASE 3: Structurally and style inconsistent examples are rejected
    # -------------------------------------------------------------------------
    print("\n[CASE 3] Verifying structural and style consistency checks...")
    assert "ex_placeholder" not in selected_ids, "Example with TODO placeholder should be rejected."
    assert "ex_mismatched_brackets" not in selected_ids, "Example with mismatched brackets should be rejected."
    assert "ex_empty_ticket" not in selected_ids, "Example with empty ticket content should be rejected."
    print("         ✅ Passed! Inconsistent examples were rejected.")

    # -------------------------------------------------------------------------
    # CASE 4: Outdated examples receive score decay
    # -------------------------------------------------------------------------
    print("\n[CASE 4] Verifying age-based score decay...")
    selected_n1 = selector.select("Query ticket content", primary_topic="billing", n=1)
    print(f"         → Selected ID with n=1: {selected_n1[0]['id']}")
    assert selected_n1[0]["id"] == "ex_perfect", "ex_perfect should be preferred over ex_outdated due to age decay."
    print("         ✅ Passed! Age-based score decay was correctly applied.")

    # -------------------------------------------------------------------------
    # CASE 5: Fallbacks work seamlessly when metadata is missing
    # -------------------------------------------------------------------------
    print("\n[CASE 5] Verifying fallback with missing metadata...")
    assert "ex_missing_metadata" in selected_ids, "Example with missing metadata should be selected successfully."
    print("         ✅ Passed! Selection fell back gracefully on missing metadata without crashing.")

    print("\n" + "=" * 80)
    print("🎉 ALL DYNAMIC FEW-SHOT DRIFT RISK MITIGATION TESTS PASSED!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    test_drift_mitigation()
