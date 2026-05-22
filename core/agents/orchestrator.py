"""
LangGraph master orchestrator.

Wires all agents together into a complete ticket processing pipeline.
This is the entry point for processing support tickets through the
multi-agent system.

PRODUCTION FEATURES:
- Input guardrails (prompt injection, PII masking, rate limiting)
- Output guardrails (hallucination check, topic guardrail, PII leak prevention)
- Circuit breaker (fallback when Groq is down)
- Semantic cache (skip pipeline for duplicate queries via Redis)

Pipeline flow:
  intake → retrieval → synthesis → drafting → quality_gate → output
                ↑          ↓              ↑           ↓
                └─ need_more              └── revise ─┘
                                                   ↓
                                              escalation
"""

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from langgraph.graph import END, StateGraph
from langfuse import observe

from config import settings
from core.agents.state import (
    CitedSource,
    FinalResponse,
    QualityRoute,
    SynthesisRoute,
    TicketState,
    create_initial_state,
)
from core.agents.intake import intake_agent
from core.agents.retrieval_agent import retrieval_agent
from core.agents.synthesis import synthesis_agent
from core.agents.drafting import drafting_agent
from core.agents.quality_gate import quality_gate_agent, escalation_agent
from core.guardrails.input_guard import get_input_guard
from core.guardrails.output_guard import get_output_guard
from core.guardrails.circuit_breaker import get_circuit_breaker
from core.redis_client import get_redis_client

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

async def format_final_output(state: TicketState) -> TicketState:
    """
    Format the final output based on pipeline results.
    
    Creates a FinalResponse if approved, or escalation response
    if escalated.
    
    Args:
        state: Final pipeline state.
        
    Returns:
        State with final_response populated.
    """
    state["agent_path"] = ["output_formatter"]
    
    if state.get("escalation_needed", False):
        # Escalation response
        state["final_response"] = FinalResponse(
            reply_text="",
            confidence=0.0,
            sources=[],
            needs_review=True,
            review_reason=state.get("escalation_reason", "Escalated to human"),
        )
        logger.info(
            f"Ticket {state.get('ticket_id')} escalated: "
            f"{state.get('escalation_reason')}"
        )
    else:
        # Approved response
        state["final_response"] = FinalResponse(
            reply_text=state.get("draft_reply", ""),
            confidence=state.get("confidence_score", 0.0),
            sources=state.get("sources_cited", []),
            needs_review=state.get("quality_score", 0.0) < 0.9,
            review_reason=_get_review_reason(state),
        )
        logger.info(
            f"Ticket {state.get('ticket_id')} completed: "
            f"confidence={state.get('confidence_score', 0):.2f}"
        )
    
    # Record downstream performance feedback in the retrieval feedback loop
    ticket_content = state.get("ticket_content", "").strip()
    if ticket_content:
        import hashlib
        content_hash = hashlib.md5(ticket_content.lower().encode('utf-8')).hexdigest()
        feedback_key = f"feedback:history:{content_hash}"
        try:
            redis = get_redis_client()
            feedback_data = {
                "synthesis_confidence": state.get("synthesis_confidence", 1.0),
                "quality_score": state.get("quality_score", 1.0),
                "knowledge_gaps": state.get("knowledge_gaps", []),
                "quality_issues": state.get("quality_issues", []),
                "escalated": state.get("escalation_needed", False),
                "route_taken": state.get("complexity", "moderate"),
            }
            await redis.set_json(feedback_key, feedback_data, ttl_seconds=604800)  # Store for 7 days
            logger.info(f"[FEEDBACK LOOP] Successfully recorded downstream performance feedback for hash {content_hash}")
        except Exception as e:
            logger.warning(f"Failed to record performance feedback in loop: {e}")

    return state


def _get_review_reason(state: TicketState) -> str:
    """
    Determine reason for review based on state.
    
    Args:
        state: Current ticket state.
        
    Returns:
        Review reason or empty string.
    """
    reasons: list[str] = []
    
    rep_guidance = state.get("rep_guidance", "")
    if rep_guidance == "VERIFY_CHANGELOG":
        reasons.append("Recent Stripe changes may affect this answer")
    elif rep_guidance == "VERIFY_WITH_ENG":
        reasons.append("Conflicting sources - verify with engineering")
    elif rep_guidance == "DO_NOT_SEND":
        reasons.append("Low confidence - thorough review required")
    
    if state.get("has_stale_content", False):
        reasons.append("Some sources may be outdated")
    
    quality_score = state.get("quality_score", 1.0)
    if quality_score < 0.85:
        reasons.append(f"Quality score: {quality_score:.0%}")
    
    return "; ".join(reasons) if reasons else ""


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS WITH LOOP PREVENTION
# ═══════════════════════════════════════════════════════════════════════════════

def intake_router(state: TicketState) -> str:
    """
    Route after intake based on classification.
    
    Routes:
        - simple_retrieval/parallel_retrieval → retrieval
        - escalate → escalation (critical + data loss)
    """
    route = intake_agent.route(state)
    logger.debug(f"Intake routing: {route}")
    
    if route == "escalate":
        return "escalation"
    
    # Both simple and parallel go to retrieval
    return "retrieval"


def synthesis_router(state: TicketState) -> str:
    """
    Route after synthesis.
    """
    decision = synthesis_agent.route(state)
    logger.debug(f"Synthesis routing: {decision}")
    
    if decision == "escalate":
        return "escalation"
    
    if decision == "need_more" or decision == "additional_retrieval":
        return "retrieval"
    
    # 'ready' goes to drafting
    return "drafting"


def quality_router(state: TicketState) -> str:
    """
    Route after quality gate with loop prevention.
    
    Uses quality_gate_agent.route() which respects MAX_AGENT_RETRIES.
    """
    route = quality_gate_agent.route(state)
    logger.debug(f"Quality gate routing: {route}")
    
    if route == "revise":
        return "drafting"
    elif route == "escalate":
        return "escalation"
    
    # 'approved'
    return "output_formatter"


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

async def _retrieval_node(state: TicketState) -> TicketState:
    """Wrapper for retrieval agent with lazy initialization."""
    return await retrieval_agent().process(state)


def build_graph() -> StateGraph:
    """
    Build the LangGraph state machine for ticket processing.
    
    Returns:
        Configured StateGraph ready for compilation.
    """
    graph = StateGraph(TicketState)
    
    # Add nodes
    graph.add_node("intake", intake_agent.process)
    graph.add_node("retrieval", _retrieval_node)
    graph.add_node("synthesis", synthesis_agent.process)
    graph.add_node("drafting", drafting_agent.process)
    graph.add_node("quality_gate", quality_gate_agent.process)
    graph.add_node("escalation", escalation_agent.process)
    graph.add_node("output_formatter", format_final_output)
    
    # Set entry point
    graph.set_entry_point("intake")
    
    # Add edges with conditional routing
    graph.add_conditional_edges(
        "intake",
        intake_router,
        {
            "retrieval": "retrieval",
            "escalation": "escalation",
        }
    )
    
    graph.add_edge("retrieval", "synthesis")
    
    graph.add_conditional_edges(
        "synthesis",
        synthesis_router,
        {
            "drafting": "drafting",
            "retrieval": "retrieval",
            "escalation": "escalation",
        }
    )
    
    graph.add_edge("drafting", "quality_gate")
    
    graph.add_conditional_edges(
        "quality_gate",
        quality_router,
        {
            "output_formatter": "output_formatter",
            "drafting": "drafting",
            "escalation": "escalation",
        }
    )
    
    graph.add_edge("escalation", "output_formatter")
    graph.add_edge("output_formatter", END)
    
    logger.info("LangGraph pipeline constructed")
    
    return graph


# Pre-built graph instance
_graph: StateGraph | None = None


def get_graph() -> StateGraph:
    """
    Get or create the graph instance (lazy initialization).
    
    Returns:
        The StateGraph instance.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

@observe(name="full_ticket_pipeline")
async def process_ticket(
    ticket_content: str,
    customer_id: str = "unknown",
    customer_tier: str = "standard",
    ticket_id: str | None = None,
) -> dict[str, Any]:
    """
    Process a support ticket through the complete agent pipeline.
    
    This is the main entry point for the system. It:
    1. Runs input guardrails (injection, PII, rate limiting)
    2. Checks semantic cache for duplicate queries
    3. Checks circuit breaker for Groq availability
    4. Creates initial state and runs LangGraph pipeline
    5. Runs output guardrails (hallucination, topic, PII leak)
    6. Caches the response for future duplicate queries
    7. Returns the final response
    
    Args:
        ticket_content: The customer's support message.
        customer_id: Customer identifier.
        customer_tier: Customer tier (free, standard, enterprise).
        ticket_id: Optional ticket ID (generated if not provided).
        
    Returns:
        Dictionary with final_response, agent_path, guardrail_warnings, etc.
        
    Raises:
        Never raises - all errors are captured in the response.
    """
    start_time = datetime.utcnow()
    guardrail_warnings: list[str] = []
    generated_ticket_id = ticket_id or str(uuid4())
    
    # ─── STEP 1: Input Guardrails ────────────────────────────────────────────
    input_guard = get_input_guard()
    guard_result = await input_guard.check(ticket_content, customer_id)
    
    # Use sanitized content (PII masked) for the pipeline
    sanitized_content = guard_result.sanitized_content
    guardrail_warnings.extend(guard_result.warnings)
    
    if guard_result.warnings:
        logger.info(
            f"Input guardrails triggered for {customer_id}: "
            f"{guard_result.warnings}"
        )
    
    # ─── STEP 2: Semantic Cache Check ────────────────────────────────────────
    redis = get_redis_client()
    cached = await redis.get_cached_response(guard_result.query_hash)
    if cached:
        logger.info(
            f"Cache HIT for query hash {guard_result.query_hash} — "
            f"skipping pipeline"
        )
        cached["cache_hit"] = True
        cached["guardrail_warnings"] = guardrail_warnings
        return cached
    
    # ─── STEP 3: Circuit Breaker Check ───────────────────────────────────────
    circuit_breaker = get_circuit_breaker()
    if not await circuit_breaker.is_available():
        logger.warning("Circuit breaker OPEN — returning fallback response")
        fallback = circuit_breaker.get_fallback_response()
        fallback["ticket_id"] = generated_ticket_id
        fallback["guardrail_warnings"] = guardrail_warnings
        return fallback
    
    # ─── STEP 4: Create State & Run Pipeline ─────────────────────────────────
    initial_state = create_initial_state(
        ticket_id=generated_ticket_id,
        ticket_content=sanitized_content,
        customer_id=customer_id,
        customer_tier=customer_tier,  # type: ignore
    )
    
    logger.info(
        f"Processing ticket {initial_state['ticket_id']}: "
        f"tier={customer_tier}, content_length={len(sanitized_content)}"
    )
    
    try:
        # Compile and run graph
        graph = get_graph()
        compiled = graph.compile()
        final_state = await compiled.ainvoke(initial_state)
        
        # Record success for circuit breaker
        await circuit_breaker.record_success()
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # ─── STEP 5: Output Guardrails ────────────────────────────────────────────
        _grounding_score: float | None = None  # captured below if guardrails run
        draft_reply = ""
        final_response = final_state.get("final_response")
        if final_response and not final_state.get("escalation_needed", False):
            draft_reply = final_response.get("reply_text", "")
            
            from core.agents.state import get_all_results
            retrieved_chunks = get_all_results(final_state)
            
            output_guard = get_output_guard()
            output_result = await output_guard.check(
                draft_reply=draft_reply,
                synthesized_context=final_state.get("synthesized_context", ""),
                retrieved_chunks=retrieved_chunks,
                sources_cited=final_state.get("sources_cited", []),
            )
            
            guardrail_warnings.extend(output_result.warnings)
            _grounding_score = output_result.grounding_score  # float | None
            
            # Replace reply with redacted version if needed
            if output_result.modified_reply != draft_reply:
                final_response["reply_text"] = output_result.modified_reply
            
            # Auto-escalate / flag review if guardrails flagged issues
            reasons = []
            if output_result.hallucination_flags:
                final_response["needs_review"] = True
                reasons.append(
                    f"Hallucination risk: {len(output_result.hallucination_flags)} "
                    f"ungrounded claims detected"
                )
            
            if output_result.citation_mismatches:
                final_response["needs_review"] = True
                reasons.append(
                    f"Citation mismatches: {len(output_result.citation_mismatches)} "
                    f"incorrect citations"
                )
                
            if reasons:
                existing_reason = final_response.get("review_reason", "")
                new_reason = " | ".join(reasons)
                if existing_reason and existing_reason not in ("No review needed", "None", ""):
                    final_response["review_reason"] = f"{existing_reason} | {new_reason}"
                else:
                    final_response["review_reason"] = new_reason
        
        # Force escalation if input guardrails flagged it
        if guard_result.should_escalate:
            final_state["escalation_needed"] = True
            guardrail_warnings.append(
                "auto_escalated: input guardrail flagged this ticket"
            )
        
        # ─── STEP 6: Build Response ──────────────────────────────────────────
        response = {
            "ticket_id": final_state.get("ticket_id"),
            "final_response": final_state.get("final_response"),
            "agent_path": final_state.get("agent_path", []),
            "total_tokens": final_state.get("total_tokens", 0),
            "escalated": final_state.get("escalation_needed", False),
            "escalation_brief": final_state.get("escalation_brief", ""),
            "processing_time_ms": processing_time_ms,
            "quality_score": final_state.get("quality_score", 0.0),
            "confidence_score": final_state.get("confidence_score", 0.0),
            "grounding_score": _grounding_score,
            "guardrail_warnings": guardrail_warnings,
            "pii_detected": guard_result.pii_detected,
            "cache_hit": False,
        }
        
        # ─── STEP 7: Cache Response ──────────────────────────────────────────
        if not response["escalated"] and response["confidence_score"] >= 0.6:
            await redis.cache_response(
                query_hash=guard_result.query_hash,
                response_data=response,
                ttl_seconds=3600,  # 1 hour cache
            )
            logger.info(f"Response cached for hash {guard_result.query_hash}")
        
        logger.info(
            f"Ticket {response['ticket_id']} processed: "
            f"path={response['agent_path']}, "
            f"tokens={response['total_tokens']}, "
            f"time={processing_time_ms}ms, "
            f"escalated={response['escalated']}, "
            f"guardrail_warnings={len(guardrail_warnings)}"
        )
        
        return response

    except Exception as e:
        # Record failure for circuit breaker
        await circuit_breaker.record_failure()
        
        logger.error(f"Pipeline error for ticket: {e}", exc_info=True)
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Return error response
        return {
            "ticket_id": generated_ticket_id,
            "final_response": FinalResponse(
                reply_text="",
                confidence=0.0,
                sources=[],
                needs_review=True,
                review_reason=f"Pipeline error: {str(e)}",
            ),
            "agent_path": initial_state.get("agent_path", []),
            "total_tokens": initial_state.get("total_tokens", 0),
            "escalated": True,
            "escalation_brief": f"Pipeline failed with error: {str(e)}",
            "processing_time_ms": processing_time_ms,
            "quality_score": 0.0,
            "confidence_score": 0.0,
            "guardrail_warnings": guardrail_warnings,
            "error": str(e),
        }


async def process_ticket_debug(
    ticket_content: str,
    customer_id: str = "unknown",
    customer_tier: str = "standard",
) -> TicketState:
    """
    Process a ticket and return the full final state (for debugging).
    
    Unlike process_ticket(), this returns the complete TicketState
    instead of just the response summary.
    
    Args:
        ticket_content: The customer's support message.
        customer_id: Customer identifier.
        customer_tier: Customer tier.
        
    Returns:
        Complete final TicketState.
    """
    initial_state = create_initial_state(
        ticket_id=str(uuid4()),
        ticket_content=ticket_content,
        customer_id=customer_id,
        customer_tier=customer_tier,  # type: ignore
    )
    
    graph = get_graph()
    compiled = graph.compile()
    final_state = await compiled.ainvoke(initial_state)
    
    return final_state


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_graph() -> str:
    """
    Generate a Mermaid diagram of the pipeline graph.
    
    Returns:
        Mermaid markdown string.
    """
    return """```mermaid
graph TD
    A[intake] -->|classify| B{intake_router}
    B -->|retrieval| C[retrieval]
    B -->|escalate| F[escalation]
    
    C --> D[synthesis]
    D --> E{synthesis_router}
    E -->|ready| G[drafting]
    E -->|need_more| C
    E -->|escalate| F
    
    G --> H[quality_gate]
    H --> I{quality_router}
    I -->|approved| J[output_formatter]
    I -->|revise| G
    I -->|escalate| F
    
    F --> J
    J --> K[END]
```"""
