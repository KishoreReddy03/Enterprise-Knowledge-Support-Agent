"""
Quality gate agent for pre-send reply evaluation.

This is Level 4 in the orchestration hierarchy. Evaluates AI-generated
replies before showing them to support reps. Uses Claude Haiku for
cost-effective quality assessment.

THIS IS THE KEY DIFFERENTIATOR - no generic RAG project has this.
"""

import json
import logging
from typing import Any

import anthropic
from langfuse.decorators import observe

from config import settings
from core.agents.state import CitedSource, QualityRoute, TicketState

logger = logging.getLogger(__name__)


class QualityGateAgent:
    """
    Evaluates draft replies for quality before showing to reps.
    
    Checks for:
    - Whether the reply actually answers the question
    - All claims are properly sourced
    - No hallucinated API behavior
    - Appropriate tone for customer tier
    
    Routes to: approved, revise, or escalate
    """

    EVALUATION_PROMPT = """You are a quality assurance system for AI-generated support replies.
Evaluate this reply STRICTLY before it goes to a human support rep.

ORIGINAL CUSTOMER TICKET:
{ticket_content}

AI-GENERATED REPLY:
{draft_reply}

SOURCES THE AI USED:
{sources_formatted}

CUSTOMER TIER: {customer_tier}

Evaluate each criterion carefully:

1. ANSWERS_THE_QUESTION: Does the reply directly address what the customer asked?
   - Not just related information, but actually answers their specific question

2. ALL_CLAIMS_HAVE_SOURCES: Every factual claim about Stripe API behavior must cite a source.
   - Code examples must come from documentation
   - "Stripe does X" statements need citations
   - General advice doesn't need citations

3. NO_HALLUCINATED_API_BEHAVIOR: Check for invented API behavior:
   - Made-up parameter names
   - Incorrect endpoint paths
   - Wrong error codes
   - Features that don't exist
   
4. APPROPRIATE_FOR_CUSTOMER_TIER: 
   - Enterprise: detailed technical depth appropriate
   - Standard: balanced technical detail
   - Free: may need to mention documentation links more

Return JSON ONLY (no markdown, no explanation):
{
    "answers_the_question": true/false,
    "all_claims_have_sources": true/false,
    "no_hallucinated_api_behavior": true/false,
    "appropriate_for_customer_tier": true/false,
    "overall_score": 0.0-1.0,
    "specific_issues": ["list each specific problem found"],
    "improvement_instruction": "specific rewrite instruction if score < 0.8, null if score >= 0.8"
}

SCORING GUIDE:
- 0.9-1.0: Excellent - all criteria pass, ready to send
- 0.8-0.9: Good - minor issues, still approvable
- 0.6-0.8: Issues found - needs revision
- Below 0.6: Serious problems - should escalate to senior rep"""

    def __init__(self) -> None:
        """Initialize quality gate agent with Anthropic client."""
        self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        logger.info("QualityGateAgent initialized with Haiku model")

    def _format_sources(self, sources: list[CitedSource]) -> str:
        """
        Format cited sources for evaluation prompt.
        
        Args:
            sources: List of cited sources from drafting agent.
            
        Returns:
            Formatted string of sources.
        """
        if not sources:
            return "(No sources cited)"
        
        lines: list[str] = []
        for i, src in enumerate(sources, 1):
            title = src.get("title", "Unknown")
            url = src.get("url", "No URL")
            relevance = src.get("relevance", "")
            lines.append(f"{i}. {title}\n   URL: {url}\n   Used for: {relevance}")
        
        return "\n".join(lines)

    def _parse_response(self, text: str) -> dict[str, Any] | None:
        """
        Parse JSON response from model.
        
        Args:
            text: Raw response text.
            
        Returns:
            Parsed dict or None if parsing failed.
        """
        # Strip markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
        text = text.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error in quality gate: {e}")
            logger.debug(f"Failed to parse: {text[:500]}")
            return None

    def _calculate_score_from_criteria(
        self,
        answers_question: bool,
        has_sources: bool,
        no_hallucination: bool,
        appropriate_tier: bool,
    ) -> float:
        """
        Calculate quality score from boolean criteria.
        
        Used as fallback when model doesn't return a score.
        
        Args:
            answers_question: Whether reply answers the question.
            has_sources: Whether all claims have sources.
            no_hallucination: Whether reply avoids hallucinations.
            appropriate_tier: Whether tone matches customer tier.
            
        Returns:
            Calculated quality score between 0.0 and 1.0.
        """
        weights = {
            "answers_question": 0.35,
            "has_sources": 0.25,
            "no_hallucination": 0.30,
            "appropriate_tier": 0.10,
        }
        
        score = 0.0
        if answers_question:
            score += weights["answers_question"]
        if has_sources:
            score += weights["has_sources"]
        if no_hallucination:
            score += weights["no_hallucination"]
        if appropriate_tier:
            score += weights["appropriate_tier"]
        
        return score

    @observe(name="quality_gate")
    async def process(self, state: TicketState) -> TicketState:
        """
        Evaluate draft reply quality before showing to rep.
        
        Uses Claude Haiku for cost-effective evaluation.
        
        Args:
            state: Current ticket state with draft reply.
            
        Returns:
            Updated state with quality assessment.
        """
        # Handle empty draft
        if not state.get("draft_reply"):
            logger.warning("Quality gate received empty draft")
            state["quality_score"] = 0.0
            state["quality_issues"] = ["No draft reply to evaluate"]
            state["agent_path"] = state.get("agent_path", []) + ["quality_gate"]
            return state
        
        prompt = self.EVALUATION_PROMPT.format(
            ticket_content=state.get("ticket_content", ""),
            draft_reply=state.get("draft_reply", ""),
            sources_formatted=self._format_sources(
                state.get("sources_cited", [])
            ),
            customer_tier=state.get("customer_tier", "standard"),
        )
        
        try:
            response = self._client.messages.create(
                model=settings.HAIKU_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            
            response_text = response.content[0].text.strip()
            
            # Track token usage
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            state["total_tokens"] = state.get("total_tokens", 0) + tokens_used
            
            # Parse response
            parsed = self._parse_response(response_text)
            
            if parsed:
                # Extract quality assessment
                state["quality_score"] = float(parsed.get("overall_score", 0.5))
                state["quality_issues"] = parsed.get("specific_issues", [])
                
                # Store improvement instruction for potential revision
                improvement = parsed.get("improvement_instruction")
                if improvement and improvement != "null":
                    state["error_log"] = state.get("error_log", []) + [
                        f"quality_gate: improvement_needed: {improvement}"
                    ]
                
                # Log detailed evaluation results
                logger.info(
                    f"Quality evaluation: score={state['quality_score']:.2f}, "
                    f"answers={parsed.get('answers_the_question')}, "
                    f"sourced={parsed.get('all_claims_have_sources')}, "
                    f"no_hallucination={parsed.get('no_hallucinated_api_behavior')}, "
                    f"issues={len(state['quality_issues'])}"
                )
            else:
                # Parse failed - use conservative fallback
                logger.error("Failed to parse quality gate response")
                state["quality_score"] = 0.6  # Require revision
                state["quality_issues"] = ["Quality evaluation parse failed"]
                state["error_log"] = state.get("error_log", []) + [
                    "quality_gate: JSON parse failed, defaulting to revision"
                ]

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error in quality gate: {e}")
            # On API error, be conservative - require revision
            state["quality_score"] = 0.6
            state["quality_issues"] = [f"Quality evaluation failed: {str(e)}"]
            state["error_log"] = state.get("error_log", []) + [
                f"quality_gate: API error: {str(e)}"
            ]

        except Exception as e:
            logger.error(f"Unexpected error in quality gate: {e}")
            state["quality_score"] = 0.5
            state["quality_issues"] = [f"Unexpected evaluation error: {str(e)}"]
            state["error_log"] = state.get("error_log", []) + [
                f"quality_gate: unexpected error: {str(e)}"
            ]

        # Track agent path
        state["agent_path"] = state.get("agent_path", []) + ["quality_gate"]
        
        return state

    def route(self, state: TicketState) -> QualityRoute:
        """
        Determine next step based on quality evaluation.
        
        Routing logic:
        - 'approved' if quality_score >= 0.80
        - 'revise' if score >= 0.60 AND under retry limit
        - 'escalate' if score < 0.60 OR exceeded retries
        
        Args:
            state: Current ticket state with quality assessment.
            
        Returns:
            QualityRoute literal: 'approved', 'revise', or 'escalate'
        """
        quality_score = state.get("quality_score", 0.0)
        revision_count = state.get("revision_count", 0)
        
        # High quality - approve
        if quality_score >= 0.80:
            logger.info(
                f"Quality gate APPROVED: score={quality_score:.2f}"
            )
            return "approved"
        
        # Medium quality with retries remaining - revise
        if quality_score >= 0.60 and revision_count < settings.MAX_AGENT_RETRIES:
            state["revision_count"] = revision_count + 1
            logger.info(
                f"Quality gate REVISE: score={quality_score:.2f}, "
                f"revision {state['revision_count']}/{settings.MAX_AGENT_RETRIES}"
            )
            return "revise"
        
        # Low quality or out of retries - escalate
        if revision_count >= settings.MAX_AGENT_RETRIES:
            logger.warning(
                f"Quality gate ESCALATE: max retries ({settings.MAX_AGENT_RETRIES}) exceeded"
            )
        else:
            logger.warning(
                f"Quality gate ESCALATE: score too low ({quality_score:.2f})"
            )
        
        return "escalate"

    def get_revision_instruction(self, state: TicketState) -> str:
        """
        Get specific instruction for drafting agent revision.
        
        Extracts the improvement instruction from error log or
        generates one from quality issues.
        
        Args:
            state: Current ticket state.
            
        Returns:
            Instruction string for revision.
        """
        error_log = state.get("error_log", [])
        
        # Look for improvement instruction in error log
        for entry in reversed(error_log):
            if "improvement_needed:" in entry:
                return entry.split("improvement_needed:")[1].strip()
        
        # Generate from quality issues
        issues = state.get("quality_issues", [])
        if issues:
            return f"Fix these issues: {'; '.join(issues)}"
        
        return "Improve the response quality and ensure all claims are sourced."


# Module-level instance
quality_gate_agent = QualityGateAgent()
