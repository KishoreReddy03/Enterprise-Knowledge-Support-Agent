"""
Intake agent for ticket classification.

This is Level 0 in the orchestration hierarchy. Runs on EVERY ticket.
Uses fast LLM for cost efficiency.
"""

import json
import logging
from typing import Any

from langfuse import observe

from config import settings
from core.agents.state import (
    ComplexityRoute,
    PrimaryTopic,
    TicketState,
    UrgencyLevel,
)
from core.llm_client import call_fast

logger = logging.getLogger(__name__)


class IntakeAgent:
    """
    Classifies incoming support tickets for routing.
    
    Determines ticket complexity, urgency, primary topic, and extracts
    any error codes. Uses fast LLM for cost-efficient classification.
    """

    INTAKE_PROMPT = """Analyze this support ticket. Return JSON only.
No explanation. No markdown. Just the JSON object.

Ticket: {ticket_content}

{{
    "complexity": "simple|moderate|complex",
    "urgency": "low|medium|high|critical",
    "primary_topic": "webhook|billing|connect|auth|api|radar|other",
    "error_codes": ["list of any error codes mentioned, empty list if none"],
    "is_about_deprecated_feature": false,
    "confidence": 0.85
}}

Complexity guide:
- simple = single concept, likely answered in one doc
- moderate = requires 2-3 sources to answer
- complex = multi-system issue, edge case, no obvious answer

Urgency guide:
- low = general question, no time pressure
- medium = normal support request
- high = customer blocked, needs quick resolution
- critical = ONLY if: data loss, payment failure in production, or account suspension

Return valid JSON only."""

    RETRY_PROMPT = """Your previous response was not valid JSON. 
Return ONLY a valid JSON object with no additional text.
No markdown code blocks. No explanation.

Ticket: {ticket_content}

Required JSON structure:
{{
    "complexity": "simple",
    "urgency": "medium", 
    "primary_topic": "other",
    "error_codes": [],
    "is_about_deprecated_feature": false,
    "confidence": 0.5
}}"""

    def __init__(self) -> None:
        """Initialize intake agent."""
        logger.info("IntakeAgent initialized")

    @observe(name="intake_agent")
    async def process(self, state: TicketState) -> TicketState:
        """
        Classify ticket complexity, urgency, and topic.
        
        Uses Claude Haiku for cost-efficient classification.
        Extracts error codes if present in the ticket content.
        
        Args:
            state: Current ticket state with ticket_content.
            
        Returns:
            Updated state with intake classification outputs.
        """
        ticket_content = state.get("ticket_content", "")
        
        if not ticket_content:
            logger.warning("Empty ticket content, using defaults")
            return self._apply_defaults(state)

        # First attempt
        result = await self._call_haiku(ticket_content, is_retry=False)
        
        if result is None:
            # Retry with explicit JSON instruction
            logger.warning("First parse failed, retrying with explicit prompt")
            result = await self._call_haiku(ticket_content, is_retry=True)
        
        if result is None:
            # Both attempts failed - use safe defaults
            logger.error("Both parse attempts failed, using defaults")
            return self._apply_defaults(state)

        # Update state with parsed results
        state["complexity"] = self._validate_complexity(result.get("complexity"))
        state["urgency"] = self._validate_urgency(result.get("urgency"))
        state["primary_topic"] = self._validate_topic(result.get("primary_topic"))
        state["error_codes"] = self._validate_error_codes(result.get("error_codes"))
        state["intake_confidence"] = self._validate_confidence(result.get("confidence"))
        
        # Track agent path
        state["agent_path"] = state.get("agent_path", []) + ["intake"]
        
        logger.info(
            f"Intake complete: complexity={state['complexity']}, "
            f"urgency={state['urgency']}, topic={state['primary_topic']}, "
            f"confidence={state['intake_confidence']:.2f}"
        )
        
        return state

    async def _call_haiku(
        self,
        ticket_content: str,
        is_retry: bool,
    ) -> dict[str, Any] | None:
        """
        Call fast LLM and parse JSON response.
        
        Args:
            ticket_content: The ticket text to classify.
            is_retry: Whether this is a retry attempt.
            
        Returns:
            Parsed JSON dict or None if parsing failed.
        """
        prompt = self.RETRY_PROMPT if is_retry else self.INTAKE_PROMPT
        formatted_prompt = prompt.format(ticket_content=ticket_content)
        
        try:
            response_text = await call_fast(formatted_prompt, max_tokens=256)
            
            logger.debug(f"Fast model response received")
            
            # Parse JSON
            return self._parse_json(response_text)
            
        except Exception as e:
            logger.error(f"Error calling fast LLM: {e}")
            return None

    def _parse_json(self, text: str) -> dict[str, Any] | None:
        """
        Parse JSON from model response.
        
        Handles common issues like markdown code blocks.
        
        Args:
            text: Raw response text.
            
        Returns:
            Parsed dict or None if parsing failed.
        """
        # Strip markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        # Also handle ```json prefix without closing
        text = text.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            logger.debug(f"Failed to parse: {text[:200]}")
            return None

    def _validate_complexity(self, value: Any) -> ComplexityRoute:
        """Validate and normalize complexity value."""
        if value in ("simple", "moderate", "complex"):
            return value
        return "moderate"

    def _validate_urgency(self, value: Any) -> UrgencyLevel:
        """Validate and normalize urgency value."""
        if value in ("low", "medium", "high", "critical"):
            return value
        return "medium"

    def _validate_topic(self, value: Any) -> PrimaryTopic:
        """Validate and normalize primary topic value."""
        valid_topics = ("webhook", "billing", "connect", "auth", "api", "radar", "other")
        if value in valid_topics:
            return value
        return "other"

    def _validate_error_codes(self, value: Any) -> list[str]:
        """Validate and normalize error codes list."""
        if isinstance(value, list):
            return [str(code) for code in value if code]
        return []

    def _validate_confidence(self, value: Any) -> float:
        """Validate and normalize confidence score."""
        try:
            conf = float(value)
            return max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            return 0.5

    def _apply_defaults(self, state: TicketState) -> TicketState:
        """
        Apply safe defaults when classification fails.
        
        Args:
            state: Current ticket state.
            
        Returns:
            State with default classification values.
        """
        state["complexity"] = "moderate"
        state["urgency"] = "medium"
        state["primary_topic"] = "other"
        state["error_codes"] = []
        state["intake_confidence"] = 0.5
        state["agent_path"] = state.get("agent_path", []) + ["intake"]
        
        # Log error
        state["error_log"] = state.get("error_log", []) + [
            "intake: classification failed, using defaults"
        ]
        
        return state

    def route(self, state: TicketState) -> str:
        """
        Determine routing based on intake classification.
        
        Args:
            state: Ticket state with intake classification.
            
        Returns:
            Route name: 'simple_retrieval', 'parallel_retrieval', or 'escalate'.
        """
        complexity = state.get("complexity", "moderate")
        urgency = state.get("urgency", "medium")
        ticket_content = state.get("ticket_content", "").lower()
        
        # Immediate escalation for critical + data loss
        if urgency == "critical" and "data_loss" in ticket_content:
            logger.info("Routing to escalate: critical urgency with data_loss")
            return "escalate"
        
        # Also escalate for critical + production failures
        if urgency == "critical" and any(
            term in ticket_content
            for term in ["production", "prod", "all payments", "completely down"]
        ):
            logger.info("Routing to escalate: critical production issue")
            return "escalate"
        
        # Simple tickets get focused retrieval
        if complexity == "simple":
            logger.info("Routing to simple_retrieval")
            return "simple_retrieval"
        
        # Moderate and complex both get parallel retrieval
        logger.info("Routing to parallel_retrieval")
        return "parallel_retrieval"


# Module-level instance
intake_agent = IntakeAgent()
