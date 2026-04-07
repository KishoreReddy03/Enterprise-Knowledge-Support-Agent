"""
LangGraph pipeline state definitions.

This module defines the complete state that flows through the agent pipeline.
Every agent reads from and writes to this state. Changes to this file will
affect all agents in the system.

IMPORTANT: Modifying field names or types is a breaking change.
Add new fields at the end, never remove existing fields.
"""

from typing import Annotated, Any, Literal, TypedDict

import operator


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING TYPES
# ═══════════════════════════════════════════════════════════════════════════════

ComplexityRoute = Literal["simple", "moderate", "complex"]
"""
Ticket complexity classification used for routing decisions.

- simple: Direct question, single topic, likely answerable from docs
- moderate: Multi-part question, may need multiple sources
- complex: Edge case, potential bug, needs synthesis from many sources
"""

SynthesisRoute = Literal["ready", "need_more", "escalate"]
"""
Synthesis agent decision for next step.

- ready: Have enough context to draft a response
- need_more: Need additional retrieval or clarification
- escalate: Cannot resolve confidently, needs human
"""

QualityRoute = Literal["approved", "revise", "escalate"]
"""
Quality gate decision for draft response.

- approved: Response meets quality bar, ready to send
- revise: Response needs improvement, send back to drafting
- escalate: Response cannot meet quality bar, needs human
"""

UrgencyLevel = Literal["low", "medium", "high", "critical"]
"""
Ticket urgency classification.

- low: Can wait, no business impact
- medium: Normal priority
- high: Time-sensitive, customer waiting
- critical: Revenue at risk, SLA breach imminent
"""

PrimaryTopic = Literal["webhook", "billing", "connect", "auth", "api", "other"]
"""
Primary topic classification for routing and retrieval focus.
"""

CustomerTier = Literal["free", "standard", "enterprise"]
"""
Customer tier for prioritization and escalation rules.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE RESULT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class SourceResult(TypedDict, total=False):
    """
    Standardized format for retrieval results from any source.
    
    Attributes:
        chunk_id: Unique identifier for the chunk.
        text: The retrieved text content.
        score: Relevance/similarity score.
        source_url: URL to the original source.
        source_type: Type of source (stripe_doc, github_issue, etc.).
        title: Title of the parent document.
        date: Date associated with the content.
        is_stale: Whether this content may be outdated.
        retrieval_method: How this was retrieved (vector, bm25, hybrid).
    """
    chunk_id: str
    text: str
    score: float
    source_url: str
    source_type: str
    title: str
    date: str | None
    is_stale: bool
    retrieval_method: str


class ContradictionInfo(TypedDict):
    """
    Information about a detected contradiction between sources.
    
    Attributes:
        source_a: First source chunk_id.
        source_b: Second source chunk_id.
        description: Description of the contradiction.
        resolution: How the agent resolved it (if at all).
    """
    source_a: str
    source_b: str
    description: str
    resolution: str


class CitedSource(TypedDict):
    """
    A source citation in the final response.
    
    Attributes:
        chunk_id: ID of the cited chunk.
        title: Title for display.
        url: Link to the source.
        relevance: Why this source was relevant.
    """
    chunk_id: str
    title: str
    url: str
    relevance: str


class FinalResponse(TypedDict):
    """
    The final packaged response ready for delivery.
    
    Attributes:
        reply_text: The customer-facing response text.
        confidence: Overall confidence score.
        sources: List of cited sources.
        needs_review: Whether a human should review before sending.
        review_reason: Why review is needed (if applicable).
    """
    reply_text: str
    confidence: float
    sources: list[CitedSource]
    needs_review: bool
    review_reason: str


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE STATE
# ═══════════════════════════════════════════════════════════════════════════════

class TicketState(TypedDict, total=False):
    """
    Complete state that flows through the LangGraph pipeline.
    
    Every agent reads from and writes to this state. The state is divided
    into logical sections based on which agent produces the data.
    
    Fields marked with Annotated[..., operator.add] are accumulator fields
    that append values rather than overwriting.
    """
    
    # ─── INPUT ───────────────────────────────────────────────────────────────────
    # Set by the API layer when creating the ticket
    
    ticket_id: str
    """Unique identifier for this support ticket."""
    
    ticket_content: str
    """The customer's message/question."""
    
    customer_id: str
    """Customer identifier for context lookup."""
    
    customer_tier: CustomerTier
    """Customer tier: 'free', 'standard', or 'enterprise'."""
    
    # ─── INTAKE AGENT OUTPUT ─────────────────────────────────────────────────────
    # Produced by the intake/classification agent
    
    complexity: ComplexityRoute
    """Ticket complexity: 'simple', 'moderate', or 'complex'."""
    
    urgency: UrgencyLevel
    """Ticket urgency: 'low', 'medium', 'high', or 'critical'."""
    
    primary_topic: PrimaryTopic
    """Primary topic: 'webhook', 'billing', 'connect', 'auth', 'api', or 'other'."""
    
    error_codes: list[str]
    """Extracted error codes from the ticket (e.g., ['card_declined', 'rate_limit'])."""
    
    intake_confidence: float
    """Confidence score (0-1) in the classification."""
    
    # ─── RETRIEVAL AGENT OUTPUT ──────────────────────────────────────────────────
    # Produced by the retrieval agent
    
    docs_results: list[SourceResult]
    """Results from Stripe documentation search."""
    
    github_results: list[SourceResult]
    """Results from GitHub issues search."""
    
    stackoverflow_results: list[SourceResult]
    """Results from StackOverflow search."""
    
    changelog_results: list[SourceResult]
    """Results from changelog search (recent changes)."""
    
    retrieval_errors: list[str]
    """Which sources failed during retrieval (for logging)."""
    
    has_recent_changes: bool
    """Whether changelog contains recent relevant changes."""
    
    has_stale_content: bool
    """Whether any retrieved content is marked as potentially stale."""
    
    # ─── SYNTHESIS AGENT OUTPUT ──────────────────────────────────────────────────
    # Produced by the synthesis/reasoning agent
    
    synthesis_decision: SynthesisRoute
    """Next step: 'ready', 'need_more', or 'escalate'."""
    
    synthesized_context: str
    """Consolidated context prepared for the drafting agent."""
    
    contradictions: list[ContradictionInfo]
    """Detected contradictions between sources."""
    
    knowledge_gaps: list[str]
    """Identified gaps in available knowledge."""
    
    synthesis_confidence: float
    """Confidence score (0-1) in the synthesized context."""
    
    # ─── DRAFTING AGENT OUTPUT ───────────────────────────────────────────────────
    # Produced by the response drafting agent
    
    draft_reply: str
    """The drafted customer response."""
    
    confidence_score: float
    """Overall confidence (0-1) in the draft response."""
    
    rep_guidance: str
    """Guidance for the human rep reviewing this response."""
    
    sources_cited: list[CitedSource]
    """Sources cited in the response."""
    
    # ─── QUALITY GATE OUTPUT ─────────────────────────────────────────────────────
    # Produced by the quality check agent
    
    quality_score: float
    """Quality score (0-1) of the draft response."""
    
    quality_issues: list[str]
    """Specific quality issues identified."""
    
    revision_count: int
    """Number of revision loops (to prevent infinite loops)."""
    
    retrieval_retry_count: int
    """Number of retrieval retry loops (synthesis -> need_more -> retrieval)."""
    
    # ─── ESCALATION OUTPUT ───────────────────────────────────────────────────────
    # Produced when escalation is needed
    
    escalation_needed: bool
    """Whether this ticket needs human escalation."""
    
    escalation_reason: str
    """Why escalation is needed."""
    
    escalation_brief: str
    """Context summary for the human rep taking over."""
    
    # ─── FINAL OUTPUT ────────────────────────────────────────────────────────────
    # The packaged final response
    
    final_response: FinalResponse | None
    """The final response package, or None if escalated."""
    
    # ─── METADATA ────────────────────────────────────────────────────────────────
    # Pipeline execution metadata
    
    session_id: str
    """Unique session ID for tracing."""
    
    started_at: str
    """ISO timestamp when processing started."""
    
    agent_path: Annotated[list[str], operator.add]
    """
    Tracks which agents ran, in order.
    Uses operator.add for accumulation across nodes.
    """
    
    total_tokens: int
    """Total tokens used across all LLM calls."""
    
    error_log: Annotated[list[str], operator.add]
    """
    Accumulating log of non-fatal errors.
    Uses operator.add for accumulation across nodes.
    """


# ═══════════════════════════════════════════════════════════════════════════════
# STATE FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_initial_state(
    ticket_id: str,
    ticket_content: str,
    customer_id: str,
    customer_tier: CustomerTier = "standard",
    session_id: str | None = None,
) -> TicketState:
    """
    Create a new TicketState with required input fields and defaults.
    
    Args:
        ticket_id: Unique ticket identifier.
        ticket_content: The customer's message.
        customer_id: Customer identifier.
        customer_tier: Customer tier level.
        session_id: Optional session ID (generated if not provided).
        
    Returns:
        Initialized TicketState ready for pipeline processing.
    """
    import uuid
    from datetime import datetime
    
    return TicketState(
        # Input
        ticket_id=ticket_id,
        ticket_content=ticket_content,
        customer_id=customer_id,
        customer_tier=customer_tier,
        
        # Intake defaults
        complexity="moderate",
        urgency="medium",
        primary_topic="other",
        error_codes=[],
        intake_confidence=0.0,
        
        # Retrieval defaults
        docs_results=[],
        github_results=[],
        stackoverflow_results=[],
        changelog_results=[],
        retrieval_errors=[],
        has_recent_changes=False,
        has_stale_content=False,
        
        # Synthesis defaults
        synthesis_decision="need_more",
        synthesized_context="",
        contradictions=[],
        knowledge_gaps=[],
        synthesis_confidence=0.0,
        
        # Drafting defaults
        draft_reply="",
        confidence_score=0.0,
        rep_guidance="",
        sources_cited=[],
        
        # Quality defaults
        quality_score=0.0,
        quality_issues=[],
        revision_count=0,
        retrieval_retry_count=0,
        
        # Escalation defaults
        escalation_needed=False,
        escalation_reason="",
        escalation_brief="",
        
        # Final output
        final_response=None,
        
        # Metadata
        session_id=session_id or str(uuid.uuid4()),
        started_at=datetime.utcnow().isoformat(),
        agent_path=[],
        total_tokens=0,
        error_log=[],
    )


def get_all_results(state: TicketState) -> list[SourceResult]:
    """
    Get all retrieval results from all sources.
    
    Args:
        state: Current ticket state.
        
    Returns:
        Combined list of all source results.
    """
    results: list[SourceResult] = []
    results.extend(state.get("docs_results", []))
    results.extend(state.get("github_results", []))
    results.extend(state.get("stackoverflow_results", []))
    results.extend(state.get("changelog_results", []))
    return results


def count_total_results(state: TicketState) -> int:
    """
    Count total retrieval results across all sources.
    
    Args:
        state: Current ticket state.
        
    Returns:
        Total number of retrieved chunks.
    """
    return len(get_all_results(state))
