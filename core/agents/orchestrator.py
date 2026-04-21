"""
LangGraph master orchestrator.

Wires all agents together into a complete ticket processing pipeline.
This is the entry point for processing support tickets through the
multi-agent system.

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
from core.agents.quality_gate import quality_gate_agent
from core.agents.escalation import escalation_agent

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
    state["agent_path"] = state.get("agent_path", []) + ["output_formatter"]
    
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
    Route after synthesis with loop prevention.
    
    CRITICAL: 'need_more' can only trigger retrieval retry ONCE.
    After that, force to drafting to prevent infinite loops.
    """
    decision = synthesis_agent.route(state)
    logger.debug(f"Synthesis routing: {decision}")
    
    if decision == "escalate":
        return "escalation"
    
    if decision == "need_more":
        # LOOP PREVENTION: Only allow one retrieval retry
        retry_count = state.get("retrieval_retry_count", 0)
        if retry_count >= 1:
            logger.warning(
                f"Forcing to drafting after {retry_count} retrieval retries"
            )
            return "drafting"
        
        # Increment counter for next time
        state["retrieval_retry_count"] = retry_count + 1
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

def build_graph() -> StateGraph:
    """
    Build the LangGraph state machine for ticket processing.
    
    Returns:
        Configured StateGraph ready for compilation.
    """
    graph = StateGraph(TicketState)
    
    # Add nodes
    graph.add_node("intake", intake_agent.process)
    graph.add_node("retrieval", retrieval_agent.process)
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
    1. Creates initial state
    2. Runs the LangGraph pipeline
    3. Returns the final response
    
    Args:
        ticket_content: The customer's support message.
        customer_id: Customer identifier.
        customer_tier: Customer tier (free, standard, enterprise).
        ticket_id: Optional ticket ID (generated if not provided).
        
    Returns:
        Dictionary containing:
        - final_response: The FinalResponse object
        - agent_path: List of agents that ran
        - total_tokens: Total tokens used
        - escalated: Whether ticket was escalated
        - escalation_brief: Brief for human (if escalated)
        - processing_time_ms: Total processing time
        
    Raises:
        Never raises - all errors are captured in the response.
    """
    start_time = datetime.utcnow()
    
    # Create initial state
    initial_state = create_initial_state(
        ticket_id=ticket_id or str(uuid4()),
        ticket_content=ticket_content,
        customer_id=customer_id,
        customer_tier=customer_tier,  # type: ignore
    )
    
    logger.info(
        f"Processing ticket {initial_state['ticket_id']}: "
        f"tier={customer_tier}, content_length={len(ticket_content)}"
    )
    
    try:
        # Compile and run graph
        graph = get_graph()
        compiled = graph.compile()
        final_state = await compiled.ainvoke(initial_state)
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Build response
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
        }
        
        logger.info(
            f"Ticket {response['ticket_id']} processed: "
            f"path={response['agent_path']}, "
            f"tokens={response['total_tokens']}, "
            f"time={processing_time_ms}ms, "
            f"escalated={response['escalated']}"
        )
        
        return response

    except Exception as e:
        logger.error(f"Pipeline error for ticket: {e}", exc_info=True)
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Return error response
        return {
            "ticket_id": initial_state["ticket_id"],
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
