"""
Evaluation module for Stripe Support Agent.

Contains test cases and evaluation harness for measuring
retrieval quality, faithfulness, and overall system performance.
"""

from evaluation.run_full_eval import (
    CaseResult,
    EvalReport,
    evaluate_faithfulness,
    evaluate_mentions,
    evaluate_retrieval,
    run_complete_evaluation,
)

__all__ = [
    "CaseResult",
    "EvalReport",
    "run_complete_evaluation",
    "evaluate_retrieval",
    "evaluate_faithfulness",
    "evaluate_mentions",
]
