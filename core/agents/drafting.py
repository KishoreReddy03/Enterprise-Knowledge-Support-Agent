"""
Drafting agent for generating support replies.

This is Level 3A in the orchestration hierarchy. Generates the actual
customer reply using Claude Sonnet. This is the only place in the pipeline
where Sonnet is used for generation (cost control).
"""

import json
import logging
from typing import Any

import anthropic
from langfuse.decorators import observe

from config import settings
from core.agents.state import (
    CitedSource,
    ContradictionInfo,
    SourceResult,
    TicketState,
)

logger = logging.getLogger(__name__)


class DraftingAgent:
    """
    Generates support reply drafts from synthesized context.
    
    Uses Claude Sonnet for high-quality response generation with
    proper citation and conflict handling.
    """

    SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant helping a support rep draft a reply to a Stripe API support ticket.

RULES:
- Only state facts that appear in the provided sources
- Cite every factual claim with [Source: url]
- If sources conflict: state both versions explicitly, do not pick sides
- Never invent API behavior not documented in the sources
- If information is missing to fully answer: say so clearly
- Write for the support rep, not directly to the customer (rep will edit before sending)
- Be concise but thorough
- Use code examples from sources when relevant
- End with clear guidance for the rep

{contradiction_warning}
{stale_warning}

OUTPUT FORMAT: Return valid JSON only, no markdown code blocks."""

    USER_PROMPT_TEMPLATE = """SUPPORT TICKET:
{ticket_content}

CUSTOMER INFO:
- Customer ID: {customer_id}
- Tier: {customer_tier}
- Ticket Complexity: {complexity}
- Urgency: {urgency}

RETRIEVED INFORMATION:
{synthesized_context}

RECENT STRIPE CHANGES (if relevant):
{changelog_summary}

Generate a reply draft as JSON:
{{
    "draft_reply": "The actual reply text for the support rep to review and send",
    "confidence_score": 0.85,
    "rep_guidance": "HIGH_CONFIDENCE|VERIFY_CHANGELOG|VERIFY_WITH_ENG|DO_NOT_SEND",
    "sources_cited": [
        {{"url": "https://...", "title": "Source title", "relevance": "Why this source was used"}}
    ],
    "missing_information": "What additional info would make the answer better, or null if complete"
}}

REP_GUIDANCE VALUES:
- HIGH_CONFIDENCE: Safe to send after quick review
- VERIFY_CHANGELOG: Recent Stripe changes detected, verify current behavior
- VERIFY_WITH_ENG: Contradictory sources found, consult engineering
- DO_NOT_SEND: Not enough information, escalate to senior rep"""

    def __init__(self) -> None:
        """Initialize drafting agent with Anthropic client."""
        self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        logger.info("DraftingAgent initialized with Sonnet model")

    def _format_contradictions(
        self,
        contradictions: list[ContradictionInfo],
    ) -> str:
        """
        Format contradictions for inclusion in prompt.
        
        Args:
            contradictions: List of detected contradictions.
            
        Returns:
            Formatted string describing conflicts.
        """
        if not contradictions:
            return ""
        
        lines: list[str] = []
        for i, c in enumerate(contradictions, 1):
            lines.append(
                f"Conflict {i}: {c.get('description', 'Unknown')}\n"
                f"  - One source says: {c.get('source_a', 'N/A')}\n"
                f"  - Another says: {c.get('source_b', 'N/A')}\n"
                f"  - Analysis: {c.get('resolution', 'Unclear which is correct')}"
            )
        return "\n".join(lines)

    def _format_changelog(
        self,
        changelog_results: list[SourceResult],
    ) -> str:
        """
        Format changelog results for prompt.
        
        Args:
            changelog_results: List of changelog source results.
            
        Returns:
            Formatted changelog summary.
        """
        if not changelog_results:
            return "(No recent changes found related to this topic)"
        
        lines: list[str] = []
        for result in changelog_results[:3]:  # Max 3 entries
            date = result.get("date", "Unknown date")
            text = result.get("text", "")[:300]
            lines.append(f"[{date}] {text}")
        
        return "\n\n".join(lines)

    def _build_system_prompt(self, state: TicketState) -> str:
        """
        Build system prompt with appropriate warnings.
        
        Args:
            state: Current ticket state.
            
        Returns:
            Formatted system prompt.
        """
        contradiction_warning = ""
        contradictions = state.get("contradictions", [])
        
        if contradictions:
            # Check for high severity contradictions
            high_severity = [
                c for c in contradictions
                if c.get("severity") == "high"
            ]
            
            if high_severity:
                contradiction_warning = f"""
⚠️ WARNING: CONFLICTING INFORMATION DETECTED

{self._format_contradictions(high_severity)}

IMPORTANT: Surface this conflict explicitly in your reply.
Do NOT pick one source over another without evidence.
Recommend the rep verify with engineering before sending."""
            else:
                contradiction_warning = f"""
Note: Minor discrepancies found between sources:
{self._format_contradictions(contradictions)}
Prefer official documentation over community sources."""

        stale_warning = ""
        if state.get("has_stale_content", False):
            stale_warning = """
⚠️ WARNING: Some retrieved content may be outdated due to recent Stripe API changes.
Note this uncertainty in the reply and recommend verifying current behavior."""

        return self.SYSTEM_PROMPT_TEMPLATE.format(
            contradiction_warning=contradiction_warning,
            stale_warning=stale_warning,
        )

    def _build_user_prompt(self, state: TicketState) -> str:
        """
        Build user prompt with ticket and context.
        
        Args:
            state: Current ticket state.
            
        Returns:
            Formatted user prompt.
        """
        return self.USER_PROMPT_TEMPLATE.format(
            ticket_content=state.get("ticket_content", ""),
            customer_id=state.get("customer_id", "Unknown"),
            customer_tier=state.get("customer_tier", "standard"),
            complexity=state.get("complexity", "moderate"),
            urgency=state.get("urgency", "medium"),
            synthesized_context=state.get("synthesized_context", "(No context available)"),
            changelog_summary=self._format_changelog(
                state.get("changelog_results", [])
            ),
        )

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
            logger.warning(f"JSON parse error: {e}")
            logger.debug(f"Failed to parse: {text[:500]}")
            return None

    def _validate_guidance(self, guidance: str) -> str:
        """
        Validate and normalize rep guidance value.
        
        Args:
            guidance: Raw guidance string from model.
            
        Returns:
            Normalized guidance value.
        """
        valid_values = {
            "HIGH_CONFIDENCE",
            "VERIFY_CHANGELOG",
            "VERIFY_WITH_ENG",
            "DO_NOT_SEND",
        }
        
        guidance_upper = guidance.upper().replace(" ", "_")
        
        if guidance_upper in valid_values:
            return guidance_upper
        
        # Try to infer from partial matches
        if "HIGH" in guidance_upper or "CONFIDENCE" in guidance_upper:
            return "HIGH_CONFIDENCE"
        if "CHANGELOG" in guidance_upper:
            return "VERIFY_CHANGELOG"
        if "ENG" in guidance_upper or "CONTRADICT" in guidance_upper:
            return "VERIFY_WITH_ENG"
        if "NOT" in guidance_upper or "ESCALATE" in guidance_upper:
            return "DO_NOT_SEND"
        
        return "HIGH_CONFIDENCE"  # Default

    def _convert_to_cited_sources(
        self,
        sources: list[dict[str, str]],
    ) -> list[CitedSource]:
        """
        Convert raw source dicts to CitedSource typed dicts.
        
        Args:
            sources: List of source dicts from model.
            
        Returns:
            List of CitedSource typed dicts.
        """
        cited: list[CitedSource] = []
        for src in sources:
            cited.append(
                CitedSource(
                    chunk_id=src.get("chunk_id", ""),
                    title=src.get("title", "Unknown"),
                    url=src.get("url", ""),
                    relevance=src.get("relevance", ""),
                )
            )
        return cited

    @observe(name="drafting_agent")
    async def process(self, state: TicketState) -> TicketState:
        """
        Generate a draft reply for the support ticket.
        
        Uses Claude Sonnet to generate a well-cited, accurate response
        based on the synthesized context.
        
        Args:
            state: Current ticket state with synthesized context.
            
        Returns:
            Updated state with draft reply and metadata.
        """
        system_prompt = self._build_system_prompt(state)
        user_prompt = self._build_user_prompt(state)
        
        try:
            response = self._client.messages.create(
                model=settings.SONNET_MODEL,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            
            response_text = response.content[0].text.strip()
            
            # Track token usage
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            state["total_tokens"] = state.get("total_tokens", 0) + tokens_used
            
            # Parse response
            parsed = self._parse_response(response_text)
            
            if parsed:
                state["draft_reply"] = parsed.get("draft_reply", "")
                state["confidence_score"] = float(parsed.get("confidence_score", 0.5))
                state["rep_guidance"] = self._validate_guidance(
                    parsed.get("rep_guidance", "HIGH_CONFIDENCE")
                )
                state["sources_cited"] = self._convert_to_cited_sources(
                    parsed.get("sources_cited", [])
                )
                
                # Log missing info to error_log if present
                missing = parsed.get("missing_information")
                if missing and missing != "null":
                    state["error_log"] = state.get("error_log", []) + [
                        f"drafting: missing_info: {missing}"
                    ]
                
                logger.info(
                    f"Draft generated: confidence={state['confidence_score']:.2f}, "
                    f"guidance={state['rep_guidance']}, "
                    f"sources={len(state['sources_cited'])}"
                )
            else:
                # Parse failed - generate fallback
                logger.error("Failed to parse Sonnet response, using fallback")
                state["draft_reply"] = (
                    "I was unable to generate a confident response for this ticket. "
                    "Please review the retrieved sources manually."
                )
                state["confidence_score"] = 0.3
                state["rep_guidance"] = "DO_NOT_SEND"
                state["sources_cited"] = []
                state["error_log"] = state.get("error_log", []) + [
                    "drafting: JSON parse failed"
                ]

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error in drafting: {e}")
            state["draft_reply"] = ""
            state["confidence_score"] = 0.0
            state["rep_guidance"] = "DO_NOT_SEND"
            state["sources_cited"] = []
            state["error_log"] = state.get("error_log", []) + [
                f"drafting: API error: {str(e)}"
            ]

        except Exception as e:
            logger.error(f"Unexpected error in drafting: {e}")
            state["draft_reply"] = ""
            state["confidence_score"] = 0.0
            state["rep_guidance"] = "DO_NOT_SEND"
            state["sources_cited"] = []
            state["error_log"] = state.get("error_log", []) + [
                f"drafting: unexpected error: {str(e)}"
            ]

        # Track agent path
        state["agent_path"] = state.get("agent_path", []) + ["drafting"]
        
        return state

    def needs_revision(self, state: TicketState) -> bool:
        """
        Check if draft needs revision based on guidance.
        
        Args:
            state: Current ticket state.
            
        Returns:
            True if draft should be revised.
        """
        guidance = state.get("rep_guidance", "")
        confidence = state.get("confidence_score", 0.0)
        
        # Low confidence always needs revision
        if confidence < 0.5:
            return True
        
        # DO_NOT_SEND needs revision or escalation
        if guidance == "DO_NOT_SEND":
            return True
        
        return False


# Module-level instance
drafting_agent = DraftingAgent()
