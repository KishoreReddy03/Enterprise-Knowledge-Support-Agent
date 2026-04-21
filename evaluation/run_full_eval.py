"""
Complete evaluation suite runner.

Runs all test cases from ground_truth.json through the pipeline
and generates metrics for the README.

Usage:
    python -m evaluation.run_full_eval
    python -m evaluation.run_full_eval --limit 10  # Quick test
    python -m evaluation.run_full_eval --category webhook  # Single category
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from core.agents.orchestrator import process_ticket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)



# DATA MODELS


@dataclass
class CaseResult:
    """Result from evaluating a single test case."""
    case_id: str
    category: str
    passed: bool
    retrieval_precision: float
    faithfulness: float
    mention_compliance: bool
    escalation_correct: bool
    latency_seconds: float
    confidence: float
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Complete evaluation report with all metrics."""
    
    # Metadata
    run_date: str
    total_cases: int
    cases_evaluated: int
    cases_errored: int
    
    # Overall metrics
    pass_rate: float
    retrieval_precision_at_3: float
    faithfulness_score: float
    mention_compliance_rate: float
    escalation_accuracy: float
    
    # By category
    by_category: dict[str, dict[str, float]]
    
    # Latency
    p50_latency: float
    p95_latency: float
    avg_latency: float
    
    # Special metrics
    out_of_scope_rejection_rate: float
    deprecated_api_detection_rate: float
    contradiction_detection_rate: float
    
    # Confidence calibration
    avg_confidence: float
    confidence_when_correct: float
    confidence_when_incorrect: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_date": self.run_date,
            "total_cases": self.total_cases,
            "cases_evaluated": self.cases_evaluated,
            "cases_errored": self.cases_errored,
            "pass_rate": self.pass_rate,
            "retrieval_precision_at_3": self.retrieval_precision_at_3,
            "faithfulness_score": self.faithfulness_score,
            "mention_compliance_rate": self.mention_compliance_rate,
            "escalation_accuracy": self.escalation_accuracy,
            "by_category": self.by_category,
            "p50_latency": self.p50_latency,
            "p95_latency": self.p95_latency,
            "avg_latency": self.avg_latency,
            "out_of_scope_rejection_rate": self.out_of_scope_rejection_rate,
            "deprecated_api_detection_rate": self.deprecated_api_detection_rate,
            "contradiction_detection_rate": self.contradiction_detection_rate,
            "avg_confidence": self.avg_confidence,
            "confidence_when_correct": self.confidence_when_correct,
            "confidence_when_incorrect": self.confidence_when_incorrect,
        }

    def to_readme_table(self) -> str:
        """Generate markdown table for README."""
        return f"""
## Evaluation Results

**Run Date:** {self.run_date}
**Test Cases:** {self.cases_evaluated}/{self.total_cases} evaluated ({self.cases_errored} errors)

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Pass Rate** | {self.pass_rate:.1%} |
| **Retrieval Precision@3** | {self.retrieval_precision_at_3:.1%} |
| **Faithfulness Score** | {self.faithfulness_score:.1%} |
| **Mention Compliance** | {self.mention_compliance_rate:.1%} |
| **Escalation Accuracy** | {self.escalation_accuracy:.1%} |

### Latency

| Percentile | Time |
|------------|------|
| P50 | {self.p50_latency:.2f}s |
| P95 | {self.p95_latency:.2f}s |
| Average | {self.avg_latency:.2f}s |

### Performance by Category

| Category | Pass Rate | Cases |
|----------|-----------|-------|
{self._format_category_rows()}

### Special Capabilities

| Capability | Rate |
|------------|------|
| Out-of-Scope Rejection | {self.out_of_scope_rejection_rate:.1%} |
| Deprecated API Detection | {self.deprecated_api_detection_rate:.1%} |
| Contradiction Detection | {self.contradiction_detection_rate:.1%} |

### Confidence Calibration

| Condition | Avg Confidence |
|-----------|----------------|
| When Correct | {self.confidence_when_correct:.1%} |
| When Incorrect | {self.confidence_when_incorrect:.1%} |
| Overall | {self.avg_confidence:.1%} |
"""

    def _format_category_rows(self) -> str:
        """Format category rows for table."""
        rows = []
        for cat, metrics in sorted(self.by_category.items()):
            rows.append(
                f"| {cat} | {metrics['pass_rate']:.1%} | {metrics['count']} |"
            )
        return "\n".join(rows)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)



# EVALUATION FUNCTIONS


def percentile(values: list[float], p: int) -> float:
    """Calculate percentile of a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * p / 100
    lower = int(index)
    upper = lower + 1
    if upper >= len(sorted_values):
        return sorted_values[-1]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def evaluate_retrieval(
    sources_cited: list[dict[str, Any]],
    expected_sources: list[str],
) -> float:
    """
    Evaluate retrieval precision.
    
    Checks if expected source types appear in cited sources.
    
    Args:
        sources_cited: Sources cited in the response.
        expected_sources: Expected source types (stripe_doc, stackoverflow, etc.)
        
    Returns:
        Precision score 0-1.
    """
    if not expected_sources:
        return 1.0  # No expectations = pass
    
    if not sources_cited:
        return 0.0
    
    # Extract source types from URLs or metadata
    found_types: set[str] = set()
    for source in sources_cited:
        url = source.get("url", "").lower()
        if "stripe.com/docs" in url:
            found_types.add("stripe_doc")
        elif "github.com" in url:
            found_types.add("github_issue")
        elif "stackoverflow.com" in url:
            found_types.add("stackoverflow")
        elif "stripe.com/changelog" in url:
            found_types.add("changelog")
    
    # Calculate precision
    expected_set = set(expected_sources)
    matches = len(found_types & expected_set)
    
    return matches / len(expected_set) if expected_set else 1.0


def evaluate_mentions(
    response: str,
    must_mention: list[str],
) -> bool:
    """
    Check if response mentions all required terms.
    
    Args:
        response: The generated response text.
        must_mention: List of terms that must appear.
        
    Returns:
        True if all terms mentioned.
    """
    if not must_mention:
        return True
    
    response_lower = response.lower()
    
    for term in must_mention:
        # Allow partial matches and common variations
        term_lower = term.lower()
        if term_lower not in response_lower:
            # Try without spaces/hyphens
            term_normalized = term_lower.replace(" ", "").replace("-", "")
            response_normalized = response_lower.replace(" ", "").replace("-", "")
            if term_normalized not in response_normalized:
                return False
    
    return True


async def evaluate_faithfulness(
    ticket: str,
    response: str,
    must_not_hallucinate: list[str],
) -> float:
    """
    Evaluate faithfulness - check for hallucinations.
    
    Uses Claude Haiku to check if the response contains
    any of the forbidden hallucinated claims.
    
    Args:
        ticket: Original ticket text.
        response: Generated response.
        must_not_hallucinate: Claims that should NOT appear.
        
    Returns:
        Faithfulness score 0-1 (1 = no hallucinations).
    """
    if not must_not_hallucinate:
        return 1.0
    
    if not response:
        return 1.0  # Empty response = no hallucinations
    
    # Quick regex check first
    response_lower = response.lower()
    violations = []
    
    for forbidden in must_not_hallucinate:
        forbidden_lower = forbidden.lower()
        if forbidden_lower in response_lower:
            violations.append(forbidden)
    
    if not violations:
        return 1.0
    
    # Use LLM for nuanced check
    try:
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        
        prompt = f"""Check if this response contains any of these FALSE claims:

RESPONSE:
{response[:1500]}

FALSE CLAIMS TO CHECK:
{json.dumps(violations)}

For each claim, determine if the response actually states this (not just mentions related topic).

Return JSON only:
{{
    "violations_found": ["list of false claims actually stated"],
    "score": 0.0-1.0  // 1.0 = no violations
}}"""

        result = client.messages.create(
            model=settings.HAIKU_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        
        text = result.content[0].text.strip()
        # Parse JSON
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])
        text = text.replace("```json", "").replace("```", "").strip()
        
        parsed = json.loads(text)
        return parsed.get("score", 0.5)
        
    except Exception as e:
        logger.warning(f"Faithfulness check error: {e}")
        # Fall back to simple ratio
        return 1.0 - (len(violations) / len(must_not_hallucinate))


async def evaluate_single_case(
    case: dict[str, Any],
) -> CaseResult:
    """
    Evaluate a single test case.
    
    Args:
        case: Test case from ground_truth.json.
        
    Returns:
        CaseResult with all metrics.
    """
    case_id = case["id"]
    category = case["category"]
    
    logger.info(f"Evaluating case {case_id}...")
    
    start = time.time()
    
    try:
        # Run through pipeline
        response = await process_ticket(
            ticket_content=case["ticket"],
            customer_id="eval_customer",
            customer_tier="standard",
        )
        
        latency = time.time() - start
        
        # Get response details
        final_response = response.get("final_response", {})
        draft_reply = final_response.get("reply_text", "") if final_response else ""
        confidence = response.get("confidence_score", 0.0)
        sources_cited = final_response.get("sources", []) if final_response else []
        escalated = response.get("escalated", False)
        
        # Evaluate retrieval
        retrieval_precision = evaluate_retrieval(
            sources_cited,
            case.get("source_should_include", []),
        )
        
        # Evaluate faithfulness
        faithfulness = await evaluate_faithfulness(
            ticket=case["ticket"],
            response=draft_reply,
            must_not_hallucinate=case.get("must_not_hallucinate", []),
        )
        
        # Evaluate mention compliance
        mention_compliance = evaluate_mentions(
            response=draft_reply,
            must_mention=case.get("must_mention", []),
        )
        
        # Evaluate escalation behavior
        expected_escalation = case.get("escalation_expected", False)
        escalation_correct = escalated == expected_escalation
        
        # Check confidence range
        conf_range = case.get("acceptable_confidence_range", [0.0, 1.0])
        confidence_in_range = conf_range[0] <= confidence <= conf_range[1]
        
        # Determine pass/fail
        passed = all([
            faithfulness >= 0.85,
            mention_compliance or category == "out_of_scope",
            escalation_correct,
        ])
        
        return CaseResult(
            case_id=case_id,
            category=category,
            passed=passed,
            retrieval_precision=retrieval_precision,
            faithfulness=faithfulness,
            mention_compliance=mention_compliance,
            escalation_correct=escalation_correct,
            latency_seconds=latency,
            confidence=confidence,
            details={
                "confidence_in_range": confidence_in_range,
                "agent_path": response.get("agent_path", []),
                "sources_count": len(sources_cited),
            },
        )
        
    except Exception as e:
        logger.error(f"Error evaluating {case_id}: {e}")
        return CaseResult(
            case_id=case_id,
            category=category,
            passed=False,
            retrieval_precision=0.0,
            faithfulness=0.0,
            mention_compliance=False,
            escalation_correct=False,
            latency_seconds=time.time() - start,
            confidence=0.0,
            error=str(e),
        )


def build_report(results: list[CaseResult], total_cases: int) -> EvalReport:
    """
    Build evaluation report from results.
    
    Args:
        results: List of case results.
        total_cases: Total number of test cases.
        
    Returns:
        Complete EvalReport.
    """
    if not results:
        return EvalReport(
            run_date=datetime.utcnow().isoformat(),
            total_cases=total_cases,
            cases_evaluated=0,
            cases_errored=0,
            pass_rate=0.0,
            retrieval_precision_at_3=0.0,
            faithfulness_score=0.0,
            mention_compliance_rate=0.0,
            escalation_accuracy=0.0,
            by_category={},
            p50_latency=0.0,
            p95_latency=0.0,
            avg_latency=0.0,
            out_of_scope_rejection_rate=0.0,
            deprecated_api_detection_rate=0.0,
            contradiction_detection_rate=0.0,
            avg_confidence=0.0,
            confidence_when_correct=0.0,
            confidence_when_incorrect=0.0,
        )
    
    # Filter errors
    valid_results = [r for r in results if r.error is None]
    errored = len(results) - len(valid_results)
    
    # Basic metrics
    pass_rate = mean([r.passed for r in valid_results]) if valid_results else 0.0
    retrieval_precision = mean([r.retrieval_precision for r in valid_results]) if valid_results else 0.0
    faithfulness = mean([r.faithfulness for r in valid_results]) if valid_results else 0.0
    mention_rate = mean([r.mention_compliance for r in valid_results]) if valid_results else 0.0
    escalation_acc = mean([r.escalation_correct for r in valid_results]) if valid_results else 0.0
    
    # Latency
    latencies = [r.latency_seconds for r in valid_results]
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    avg_latency = mean(latencies) if latencies else 0.0
    
    # By category
    categories = set(r.category for r in valid_results)
    by_category = {}
    for cat in categories:
        cat_results = [r for r in valid_results if r.category == cat]
        by_category[cat] = {
            "pass_rate": mean([r.passed for r in cat_results]),
            "faithfulness": mean([r.faithfulness for r in cat_results]),
            "count": len(cat_results),
        }
    
    # Special metrics
    oos_results = [r for r in valid_results if r.category == "out_of_scope"]
    oos_rate = mean([r.escalation_correct for r in oos_results]) if oos_results else 0.0
    
    deprecated_results = [r for r in valid_results if r.category == "deprecated_api"]
    deprecated_rate = mean([r.passed for r in deprecated_results]) if deprecated_results else 0.0
    
    contradiction_results = [r for r in valid_results if r.category == "contradiction_detection"]
    contradiction_rate = mean([r.passed for r in contradiction_results]) if contradiction_results else 0.0
    
    # Confidence calibration
    confidences = [r.confidence for r in valid_results]
    avg_conf = mean(confidences) if confidences else 0.0
    
    correct_results = [r for r in valid_results if r.passed]
    incorrect_results = [r for r in valid_results if not r.passed]
    
    conf_correct = mean([r.confidence for r in correct_results]) if correct_results else 0.0
    conf_incorrect = mean([r.confidence for r in incorrect_results]) if incorrect_results else 0.0
    
    return EvalReport(
        run_date=datetime.utcnow().isoformat(),
        total_cases=total_cases,
        cases_evaluated=len(valid_results),
        cases_errored=errored,
        pass_rate=pass_rate,
        retrieval_precision_at_3=retrieval_precision,
        faithfulness_score=faithfulness,
        mention_compliance_rate=mention_rate,
        escalation_accuracy=escalation_acc,
        by_category=by_category,
        p50_latency=p50,
        p95_latency=p95,
        avg_latency=avg_latency,
        out_of_scope_rejection_rate=oos_rate,
        deprecated_api_detection_rate=deprecated_rate,
        contradiction_detection_rate=contradiction_rate,
        avg_confidence=avg_conf,
        confidence_when_correct=conf_correct,
        confidence_when_incorrect=conf_incorrect,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def run_complete_evaluation(
    limit: int | None = None,
    category: str | None = None,
    output_file: str | None = None,
) -> EvalReport:
    """
    Run complete evaluation suite.
    
    Args:
        limit: Maximum number of cases to run (for quick testing).
        category: Filter to specific category.
        output_file: Optional file to write results.
        
    Returns:
        Complete EvalReport.
    """
    # Load test cases
    ground_truth_path = Path(__file__).parent / "ground_truth.json"
    
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    test_cases = data["test_cases"]
    total_cases = len(test_cases)
    
    logger.info(f"Loaded {total_cases} test cases from ground_truth.json")
    
    # Filter by category if specified
    if category:
        test_cases = [tc for tc in test_cases if tc["category"] == category]
        logger.info(f"Filtered to {len(test_cases)} cases in category '{category}'")
    
    # Limit if specified
    if limit:
        test_cases = test_cases[:limit]
        logger.info(f"Limited to {len(test_cases)} cases")
    
    # Run evaluations
    results: list[CaseResult] = []
    
    for i, case in enumerate(test_cases):
        logger.info(f"[{i+1}/{len(test_cases)}] Evaluating {case['id']}...")
        
        result = await evaluate_single_case(case)
        results.append(result)
        
        status = "PASS" if result.passed else "FAIL"
        if result.error:
            status = "ERROR"
        
        logger.info(
            f"  -> {status} | faithfulness={result.faithfulness:.2f} | "
            f"latency={result.latency_seconds:.2f}s"
        )
    
    # Build report
    report = build_report(results, total_cases)
    
    # Output results
    if output_file:
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report.to_json())
        logger.info(f"Results written to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print(report.to_readme_table())
    print("=" * 60)
    
    # Print failing cases
    failing = [r for r in results if not r.passed and r.error is None]
    if failing:
        print("\nFailing Cases:")
        for r in failing[:10]:
            print(f"  - {r.case_id}: faithfulness={r.faithfulness:.2f}, "
                  f"mentions={r.mention_compliance}, escalation={r.escalation_correct}")
    
    return report


async def run_quick_test() -> None:
    """Run a quick test with 5 cases from each category."""
    logger.info("Running quick test (5 cases per category)...")
    
    ground_truth_path = Path(__file__).parent / "ground_truth.json"
    
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    test_cases = data["test_cases"]
    
    # Sample 5 from each category
    from collections import defaultdict
    by_category: dict[str, list] = defaultdict(list)
    for tc in test_cases:
        by_category[tc["category"]].append(tc)
    
    sampled = []
    for cat, cases in by_category.items():
        sampled.extend(cases[:5])
    
    logger.info(f"Sampled {len(sampled)} cases across {len(by_category)} categories")
    
    # Run evaluation
    results: list[CaseResult] = []
    for i, case in enumerate(sampled):
        logger.info(f"[{i+1}/{len(sampled)}] {case['id']}...")
        result = await evaluate_single_case(case)
        results.append(result)
    
    # Quick summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and r.error is None)
    errors = sum(1 for r in results if r.error is not None)
    
    print(f"\nQuick Test Results: {passed} passed, {failed} failed, {errors} errors")
    
    # Show failures
    for r in results:
        if not r.passed:
            print(f"  FAIL: {r.case_id} - {r.error or 'evaluation failed'}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run evaluation suite for Stripe Support Agent"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of cases to evaluate",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=[
            "webhook",
            "error_codes", 
            "deprecated_api",
            "out_of_scope",
            "contradiction_detection",
            "missing_information",
            "multi_source_reasoning",
        ],
        help="Filter to specific category",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for JSON results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test (5 cases per category)",
    )
    
    args = parser.parse_args()
    
    if args.quick:
        asyncio.run(run_quick_test())
    else:
        asyncio.run(
            run_complete_evaluation(
                limit=args.limit,
                category=args.category,
                output_file=args.output,
            )
        )


if __name__ == "__main__":
    main()
