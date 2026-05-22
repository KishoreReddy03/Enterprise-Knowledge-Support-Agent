"""
Evaluation API endpoints.

Provides endpoints to trigger RAG evaluation runs, view results,
and get metric summaries for dashboard display.

Usage:
    GET /api/v1/eval/run?limit=3        → Run quick eval
    GET /api/v1/eval/metrics             → Get latest metrics summary
    POST /api/v1/eval/rag               → Run RAG eval on custom inputs
"""

import json
import logging
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from core.agents.orchestrator import process_ticket
from core.evaluation.rag_evaluator import RAGEvaluator, RAGMetrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["evaluation"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class RAGEvalRequest(BaseModel):
    """Request to evaluate a single RAG pipeline run."""

    question: str = Field(description="The customer question")
    answer: str = Field(description="The AI-generated answer")
    contexts: list[str] = Field(
        description="Retrieved context chunks used to generate the answer"
    )


class RAGEvalResponse(BaseModel):
    """Response from RAG evaluation."""

    context_relevance: float = Field(description="How relevant retrieved docs are (0-1)")
    context_recall: float = Field(description="Coverage of answer by sources (0-1)")
    answer_faithfulness: float = Field(description="Grounding in sources — hallucination check (0-1)")
    answer_relevance: float = Field(description="How well answer addresses the question (0-1)")
    overall_score: float = Field(description="Weighted average of all metrics (0-1)")
    evaluation_time_ms: int = Field(description="Time taken for evaluation")


class PipelineEvalCase(BaseModel):
    """Result from evaluating a single ticket through the full pipeline."""

    ticket_preview: str
    draft_reply_preview: str
    confidence_score: float
    escalated: bool
    rag_metrics: dict[str, Any]
    processing_time_ms: int
    agent_path: list[str]
    guardrail_warnings: list[str]


class PipelineEvalResponse(BaseModel):
    """Response from running pipeline evaluation."""

    total_cases: int
    cases_evaluated: int
    avg_confidence: float
    avg_rag_overall: float
    avg_latency_ms: float
    escalation_rate: float
    results: list[PipelineEvalCase]
    run_date: str
    run_duration_ms: int


class MetricsSummary(BaseModel):
    """Summary metrics for the dashboard."""

    last_eval_date: str | None
    total_cases_evaluated: int
    avg_confidence: float
    avg_faithfulness: float
    avg_context_relevance: float
    avg_answer_relevance: float
    avg_overall_rag: float
    escalation_rate: float
    avg_latency_ms: float
    guardrail_trigger_rate: float


# ═══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY STORE (replace with DB in production)
# ═══════════════════════════════════════════════════════════════════════════════

_latest_eval: dict[str, Any] | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE TEST CASES
# ═══════════════════════════════════════════════════════════════════════════════

QUICK_EVAL_TICKETS = [
    {
        "ticket_content": "We're receiving duplicate payment_intent.succeeded webhooks. How do we handle this?",
        "customer_id": "eval_cust_1",
        "customer_tier": "standard",
    },
    {
        "ticket_content": "Getting card_declined error on our checkout page. What should we display to the user?",
        "customer_id": "eval_cust_2",
        "customer_tier": "standard",
    },
    {
        "ticket_content": "How do I use the Charges API to create a payment?",
        "customer_id": "eval_cust_3",
        "customer_tier": "free",
    },
    {
        "ticket_content": "Can you help me set up a merchant account with my bank?",
        "customer_id": "eval_cust_4",
        "customer_tier": "free",
    },
    {
        "ticket_content": "Our webhook endpoint is returning 200 but Stripe keeps retrying. What's happening?",
        "customer_id": "eval_cust_5",
        "customer_tier": "enterprise",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/rag",
    response_model=RAGEvalResponse,
    summary="Evaluate RAG quality for a single response",
    description="Run RAGAS-style metrics on a question/answer/context triple",
)
async def evaluate_rag(body: RAGEvalRequest) -> RAGEvalResponse:
    """
    Evaluate a single RAG pipeline output.

    Submit a question, the AI's answer, and the retrieved contexts.
    Returns four quality scores + an overall score.
    """
    start = time.time()

    evaluator = RAGEvaluator()
    metrics = await evaluator.evaluate(
        question=body.question,
        answer=body.answer,
        contexts=body.contexts,
    )

    eval_time = int((time.time() - start) * 1000)

    return RAGEvalResponse(
        context_relevance=metrics.context_relevance,
        context_recall=metrics.context_recall,
        answer_faithfulness=metrics.answer_faithfulness,
        answer_relevance=metrics.answer_relevance,
        overall_score=metrics.overall_score,
        evaluation_time_ms=eval_time,
    )


@router.get(
    "/run",
    response_model=PipelineEvalResponse,
    summary="Run end-to-end pipeline evaluation",
    description="Process test tickets and evaluate RAG quality for each",
)
async def run_pipeline_eval(
    limit: int = Query(default=3, ge=1, le=10, description="Number of test cases to run"),
) -> PipelineEvalResponse:
    """
    Run end-to-end evaluation on sample tickets.

    Processes tickets through the full pipeline, then runs RAG evaluation
    on each result. Returns detailed metrics per case and aggregate scores.
    """
    global _latest_eval

    run_start = time.time()
    cases = QUICK_EVAL_TICKETS[:limit]
    results: list[PipelineEvalCase] = []
    evaluator = RAGEvaluator()

    for i, ticket in enumerate(cases):
        logger.info(f"Eval [{i+1}/{len(cases)}]: {ticket['ticket_content'][:60]}...")

        try:
            # Run through full pipeline
            result = await process_ticket(
                ticket_content=ticket["ticket_content"],
                customer_id=ticket["customer_id"],
                customer_tier=ticket["customer_tier"],
            )

            # Extract data for RAG evaluation
            final_response = result.get("final_response", {})
            draft_reply = ""
            sources = []

            if isinstance(final_response, dict):
                draft_reply = final_response.get("reply_text", "")
                sources = final_response.get("sources", [])

            # Run RAG evaluation on the result
            contexts = [
                f"{s.get('title', '')}: {s.get('relevance', '')}"
                for s in sources
            ] if sources else []

            # Also use synthesized_context if available
            synth_context = result.get("synthesized_context", "")
            if synth_context:
                contexts.insert(0, synth_context)

            rag_metrics = RAGMetrics()
            if draft_reply and contexts:
                rag_metrics = await evaluator.evaluate(
                    question=ticket["ticket_content"],
                    answer=draft_reply,
                    contexts=contexts,
                )

            results.append(PipelineEvalCase(
                ticket_preview=ticket["ticket_content"][:100],
                draft_reply_preview=draft_reply[:200] if draft_reply else "[escalated]",
                confidence_score=result.get("confidence_score", 0.0),
                escalated=result.get("escalated", False),
                rag_metrics=rag_metrics.to_dict(),
                processing_time_ms=result.get("processing_time_ms", 0),
                agent_path=result.get("agent_path", []),
                guardrail_warnings=result.get("guardrail_warnings", []),
            ))

        except Exception as e:
            logger.error(f"Eval case failed: {e}")
            results.append(PipelineEvalCase(
                ticket_preview=ticket["ticket_content"][:100],
                draft_reply_preview=f"[ERROR: {str(e)[:100]}]",
                confidence_score=0.0,
                escalated=True,
                rag_metrics=RAGMetrics().to_dict(),
                processing_time_ms=0,
                agent_path=[],
                guardrail_warnings=[f"eval_error: {str(e)}"],
            ))

    # Compute aggregates
    valid_results = [r for r in results if not r.escalated]
    run_duration = int((time.time() - run_start) * 1000)

    response = PipelineEvalResponse(
        total_cases=len(cases),
        cases_evaluated=len(results),
        avg_confidence=(
            sum(r.confidence_score for r in results) / len(results)
            if results else 0.0
        ),
        avg_rag_overall=(
            sum(r.rag_metrics.get("overall_score", 0) for r in valid_results)
            / len(valid_results) if valid_results else 0.0
        ),
        avg_latency_ms=(
            sum(r.processing_time_ms for r in results) / len(results)
            if results else 0.0
        ),
        escalation_rate=(
            sum(1 for r in results if r.escalated) / len(results)
            if results else 0.0
        ),
        results=results,
        run_date=datetime.utcnow().isoformat(),
        run_duration_ms=run_duration,
    )

    # Store for dashboard
    _latest_eval = response.model_dump()

    logger.info(
        f"Eval complete: {len(results)} cases, "
        f"avg_rag={response.avg_rag_overall:.2f}, "
        f"duration={run_duration}ms"
    )

    return response


@router.get(
    "/metrics",
    response_model=MetricsSummary,
    summary="Get latest evaluation metrics",
    description="Returns a summary of the most recent evaluation run for dashboard display",
)
async def get_metrics() -> MetricsSummary:
    """
    Get the latest evaluation metrics summary.

    Returns aggregate metrics from the most recent /eval/run call.
    If no evaluation has been run yet, returns zeroed metrics.
    """
    if _latest_eval is None:
        return MetricsSummary(
            last_eval_date=None,
            total_cases_evaluated=0,
            avg_confidence=0.0,
            avg_faithfulness=0.0,
            avg_context_relevance=0.0,
            avg_answer_relevance=0.0,
            avg_overall_rag=0.0,
            escalation_rate=0.0,
            avg_latency_ms=0.0,
            guardrail_trigger_rate=0.0,
        )

    results = _latest_eval.get("results", [])
    valid = [r for r in results if not r.get("escalated", False)]

    guardrail_triggered = sum(
        1 for r in results if r.get("guardrail_warnings")
    )

    return MetricsSummary(
        last_eval_date=_latest_eval.get("run_date"),
        total_cases_evaluated=len(results),
        avg_confidence=_latest_eval.get("avg_confidence", 0.0),
        avg_faithfulness=(
            sum(r.get("rag_metrics", {}).get("answer_faithfulness", 0) for r in valid)
            / len(valid) if valid else 0.0
        ),
        avg_context_relevance=(
            sum(r.get("rag_metrics", {}).get("context_relevance", 0) for r in valid)
            / len(valid) if valid else 0.0
        ),
        avg_answer_relevance=(
            sum(r.get("rag_metrics", {}).get("answer_relevance", 0) for r in valid)
            / len(valid) if valid else 0.0
        ),
        avg_overall_rag=_latest_eval.get("avg_rag_overall", 0.0),
        escalation_rate=_latest_eval.get("escalation_rate", 0.0),
        avg_latency_ms=_latest_eval.get("avg_latency_ms", 0.0),
        guardrail_trigger_rate=(
            guardrail_triggered / len(results) if results else 0.0
        ),
    )
