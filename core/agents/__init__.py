"""Agent modules for the LangGraph pipeline."""

from core.agents.state import (
    CitedSource,
    ComplexityRoute,
    ContradictionInfo,
    CustomerTier,
    FinalResponse,
    PrimaryTopic,
    QualityRoute,
    SourceResult,
    SynthesisRoute,
    TicketState,
    UrgencyLevel,
    count_total_results,
    create_initial_state,
    get_all_results,
)
from core.agents.intake import IntakeAgent, intake_agent
from core.agents.retrieval_agent import RetrievalAgent, retrieval_agent
from core.agents.synthesis import SynthesisAgent, synthesis_agent
from core.agents.drafting import DraftingAgent, drafting_agent
from core.agents.quality_gate import QualityGateAgent, quality_gate_agent
from core.agents.escalation import EscalationAgent, escalation_agent
from core.agents.orchestrator import (
    process_ticket,
    process_ticket_debug,
    build_graph,
    get_graph,
    visualize_graph,
)

__all__ = [
    # State
    "TicketState",
    # Routing types
    "ComplexityRoute",
    "SynthesisRoute",
    "QualityRoute",
    "UrgencyLevel",
    "PrimaryTopic",
    "CustomerTier",
    # Sub-types
    "SourceResult",
    "ContradictionInfo",
    "CitedSource",
    "FinalResponse",
    # Factory functions
    "create_initial_state",
    "get_all_results",
    "count_total_results",
    # Agents
    "IntakeAgent",
    "intake_agent",
    "RetrievalAgent",
    "retrieval_agent",
    "SynthesisAgent",
    "synthesis_agent",
    "DraftingAgent",
    "drafting_agent",
    "QualityGateAgent",
    "quality_gate_agent",
    "EscalationAgent",
    "escalation_agent",
    # Orchestrator
    "process_ticket",
    "process_ticket_debug",
    "build_graph",
    "get_graph",
    "visualize_graph",
]
