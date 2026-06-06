"""
Drafting agent for generating support replies.

This is Level 3A in the orchestration hierarchy. Generates the actual
customer reply using strong LLM. This is the only place in the pipeline
where strong model is used for generation (cost control).
"""

import json
import logging
from typing import Any

from langfuse import observe

from core.agents.state import (
    CitedSource,
    ContradictionInfo,
    SourceResult,
    TicketState,
)
from core.llm_client import call_strong
from core.intelligence.few_shot_selector import get_few_shot_selector

logger = logging.getLogger(__name__)


class DraftingAgent:
    """
    Generates support reply drafts from synthesized context.
    
    Uses strong LLM for high-quality response generation with
    proper citation and conflict handling.
    """

    SYSTEM_PROMPT_TEMPLATE = """You are a senior Stripe support engineer drafting an email reply for a support rep to review before sending to the customer.

TONE & STYLE:
- Write like a real human support engineer writing an email, NOT like a documentation page
- Open with a warm acknowledgment ("Thanks for reaching out!", "Great question!", "I can see what's happening here")
- If this is a follow-up question (check the CONVERSATION HISTORY), do NOT repeat generic introductory greetings like "Thanks for reaching out!" or "Hi there!". Instead, directly address their follow-up (e.g., "Sure, let's look at that...", "I see what you mean, let's debug that code snippet...")
- Be conversational and empathetic — the customer is frustrated and needs help
- Explain the root cause in plain English (1-2 sentences, no jargon headers)
- Provide the fix naturally — use code snippets inline when helpful
- Close warmly ("That should get you unblocked!", "Let me know if you run into anything else")
- Keep it concise: a real support email is 100-200 words, not an essay

WHAT NOT TO DO:
- Do NOT use section headers like "Root Cause:", "Solution:", "Prevention:"
- Do NOT write in a robotic, documentation-style format
- Do NOT write a wall of text — use short paragraphs (2-3 sentences each)
- Do NOT start with "Dear Customer" — be casual and professional

ACCURACY & CITATION RULES:
- Only state facts that appear in the provided sources
- Never invent API behavior not in the sources
- If sources conflict, mention both possibilities
- Every factual claim or sentence in the customer-facing reply MUST be followed by an inline citation marker, e.g. [1] if supported by the first source in 'sources_cited', [2] for the second, etc.
- Do NOT aggregate all citations at the end of the text; place them inline immediately after the claim.
- Every citation marker [i] you write must map to a valid source in the 'sources_cited' list.

INTERNAL REP NOTE:
- After the customer-facing reply, add a separate "---\\nRep Note:" section
- This is internal-only guidance the rep sees but does NOT send to the customer
- Include: things to verify, edge cases, whether to escalate

{contradiction_warning}
{stale_warning}
{grounding_feedback_warning}

OUTPUT FORMAT: Return valid JSON only. No markdown code blocks wrapping the JSON."""

    USER_PROMPT_TEMPLATE = """{chat_history_block}SUPPORT TICKET:
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
    "draft_reply": "A support email with inline citations mapped to sources (e.g. '...use capabilities parameter [1]...'). Short paragraphs, code snippets if relevant, friendly close. Followed by ---\\nRep Note: internal guidance.",
    "confidence_score": 0.85,
    "rep_guidance": "HIGH_CONFIDENCE|VERIFY_CHANGELOG|VERIFY_WITH_ENG|DO_NOT_SEND",
    "sources_cited": [
        {{"chunk_id": "chunk_doc_001", "url": "https://...", "title": "Source title", "relevance": "Why this source was used"}}
    ],
    "missing_information": "What additional info would make the answer better, or null if complete"
}}

REP_GUIDANCE VALUES:
- HIGH_CONFIDENCE: Safe to send after quick review
- VERIFY_CHANGELOG: Recent Stripe changes detected, verify current behavior
- VERIFY_WITH_ENG: Contradictory sources found, consult engineering
- DO_NOT_SEND: Not enough information, escalate to senior rep"""

    def __init__(self) -> None:
        """Initialize drafting agent."""
        self._few_shot_selector = get_few_shot_selector()
        logger.info("DraftingAgent initialized")

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

    def _is_contradiction_unresolved(self, c: ContradictionInfo) -> bool:
        """
        Determine if a contradiction was left unresolved by the synthesis agent.
        """
        resolution = c.get("resolution") or c.get("likely_correct") or ""
        if not isinstance(resolution, str) or not resolution.strip():
            return True
        
        resolution_lower = resolution.lower()
        unresolved_phrases = [
            "unresolved",
            "cannot resolve",
            "unable to resolve",
            "no resolution",
            "cannot determine",
            "no clear resolution",
            "could not resolve",
            "impossible to resolve",
            "remains contradictory",
            "equally authoritative",
            "both are stale",
            "no winner",
            "both are outdated",
            "unknown",
        ]
        return any(phrase in resolution_lower for phrase in unresolved_phrases)

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
            high_severity = [c for c in contradictions if c.get("severity") == "high"]
            resolved_high = [c for c in high_severity if not self._is_contradiction_unresolved(c)]
            unresolved_high = [c for c in high_severity if self._is_contradiction_unresolved(c)]
            minor_discrepancies = [c for c in contradictions if c.get("severity") != "high"]
            
            warnings_list = []
            
            if resolved_high:
                resolved_text = self._format_contradictions(resolved_high)
                warnings_list.append(f"""
👉 RESOLVED CONTRADICTIONS (High Severity):
{resolved_text}

IMPORTANT INSTRUCTION FOR RESOLVED DISCREPANCIES:
Proactively call out and explain these resolved discrepancies inside the customer-facing response to avoid customer confusion.
Use the resolution reasoning provided by the synthesis agent (under 'likely_correct' / 'Analysis') to explain why the correct source overrides the outdated or community source.
For example, write: 'While older community discussions referenced endpoint A, Stripe's official API changelog updated on X specifies that you should use endpoint B [1]'.
Be explicit, helpful, and reference the citation numbers appropriately.""")

            if unresolved_high:
                unresolved_text = self._format_contradictions(unresolved_high)
                warnings_list.append(f"""
⚠️ UNRESOLVED CONTRADICTIONS DETECTED (High Severity):
{unresolved_text}

IMPORTANT INSTRUCTION FOR UNRESOLVED DISCREPANCIES:
Surface this conflict explicitly in your customer-facing reply and explain the ambiguity.
Do NOT arbitrarily pick one source over another.
Clearly note the uncertainty, present both possibilities, and instruct the rep to verify with engineering before sending by setting 'rep_guidance' to 'VERIFY_WITH_ENG'.""")

            if minor_discrepancies:
                minor_text = self._format_contradictions(minor_discrepancies)
                warnings_list.append(f"""
Note: Minor discrepancies found between sources:
{minor_text}
Prefer official documentation over community sources.""")

            contradiction_warning = "\n".join(warnings_list)

        stale_warning = ""
        if state.get("has_stale_content", False):
            stale_warning = """
⚠️ WARNING: Some retrieved content may be outdated due to recent Stripe API changes.
Note this uncertainty in the reply and recommend verifying current behavior."""

        grounding_feedback_warning = ""
        grounding_feedback = state.get("grounding_feedback", [])
        if grounding_feedback:
            claims_formatted = "\n".join(f"- {claim}" for claim in grounding_feedback)
            grounding_feedback_warning = f"""
⚠️ WARNING: PREVIOUS DRAFT WAS REJECTED DUE TO UNGROUNDED/HALLUCINATED CLAIMS!
The following factual claims made in the previous draft were flagged as UNGROUNDED because they could not be traced back to or supported by the retrieved sources:
{claims_formatted}

IMPORTANT INSTRUCTIONS FOR REVISION:
1. You MUST either remove these ungrounded claims entirely, OR back them up strictly with facts directly from the retrieved sources.
2. Under no circumstances should you repeat these exact ungrounded claims unless you can cite a retrieved source chunk that explicitly contains them.
3. Be extremely precise: if a technical parameter, endpoint, or error code is not in the source chunks, do not use it. Keep the draft strictly faithful to the sources."""

        # Include other quality issues/revision instructions if available
        revision_count = state.get("revision_count", 0)
        quality_issues = state.get("quality_issues", [])
        if revision_count > 0 and quality_issues:
            issues_formatted = "\n".join(f"- {issue}" for issue in quality_issues if not issue.startswith("Grounding failure"))
            if issues_formatted:
                grounding_feedback_warning += f"""

⚠️ PREVIOUS QUALITY GATE CRITERIA FAILURES:
{issues_formatted}
Please make sure to address these quality issues as well in your revised draft."""

        return self.SYSTEM_PROMPT_TEMPLATE.format(
            contradiction_warning=contradiction_warning,
            stale_warning=stale_warning,
            grounding_feedback_warning=grounding_feedback_warning,
        )

    def _build_user_prompt(self, state: TicketState) -> str:
        """
        Build user prompt with ticket, context, and few-shot examples.
        
        Dynamically selects the 2 most relevant gold-standard examples
        and injects them before the actual ticket for in-context learning.
        
        Args:
            state: Current ticket state.
            
        Returns:
            Formatted user prompt with few-shot examples.
        """
        # Select and format the most relevant few-shot examples
        few_shot_block = ""
        try:
            examples = self._few_shot_selector.select(
                ticket_content=state.get("ticket_content", ""),
                primary_topic=state.get("primary_topic", "other"),
                n=2,
            )
            if examples:
                few_shot_block = self._few_shot_selector.format_for_prompt(examples)
                logger.info(
                    f"Injected {len(examples)} few-shot examples into drafting prompt"
                )
        except Exception as e:
            logger.warning(f"Few-shot selection failed, continuing without: {e}")

        # Format conversational history block if present
        chat_history = state.get("chat_history", [])
        chat_history_block = ""
        if chat_history:
            history_str = ""
            for turn in chat_history:
                role = "Customer" if turn.get("role") == "user" else "Agent"
                history_str += f"{role}: {turn.get('content', '')}\n"
            chat_history_block = f"CONVERSATION HISTORY:\n{history_str.strip()}\n\n"

        base_prompt = self.USER_PROMPT_TEMPLATE.format(
            chat_history_block=chat_history_block,
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

        # Prepend few-shot examples to the prompt
        if few_shot_block:
            return f"{few_shot_block}\n\n{base_prompt}"
        return base_prompt

    def _parse_response(self, text: str) -> dict[str, Any] | None:
        """
        Parse JSON response from model.
        
        Handles malformed JSON with control characters by:
        - Stripping markdown code blocks
        - Removing/escaping control characters
        - Attempting strict parsing, then lenient parsing
        
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
        
        # First attempt: strict JSON parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error (strict): {e}")
            
            # Second attempt: Remove/escape control characters more aggressively
            try:
                # Remove ALL non-ASCII control characters and replace with space
                # Keep only: printable ASCII, newlines, tabs, unicode letters/numbers
                import unicodedata
                cleaned = "".join(
                    c if (c.isprintable() or c in '\n\t\r') and ord(c) >= 32
                    else ' '
                    for c in text
                )
                
                # Also try to fix common JSON issues
                # Replace multiple spaces with single space
                cleaned = " ".join(cleaned.split())
                
                # Try parsing cleaned version
                result = json.loads(cleaned)
                logger.info("Successfully parsed after removing control characters")
                return result
            except json.JSONDecodeError as e2:
                logger.warning(f"JSON parse error (after cleaning): {e2}")
                
                # Third attempt: Try to extract JSON object from text
                try:
                    import re
                    # Find first { and last } and extract
                    match = re.search(r'\{.*\}', text, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        # Clean this extracted portion
                        json_str = "".join(
                            c if (c.isprintable() or c in '\n\t\r') and ord(c) >= 32
                            else ' '
                            for c in json_str
                        )
                        result = json.loads(json_str)
                        logger.info("Successfully parsed after extracting JSON object")
                        return result
                except (json.JSONDecodeError, AttributeError):
                    pass
                
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
        
        Uses strong LLM to generate a well-cited, accurate response
        based on the synthesized context.
        
        Args:
            state: Current ticket state with synthesized context.
            
        Returns:
            Updated state with draft reply and metadata.
        """
        system_prompt = self._build_system_prompt(state)
        user_prompt = self._build_user_prompt(state)
        
        try:
            response_text = await call_strong(
                user_prompt,
                max_tokens=2048,
                system=system_prompt,
            )
            
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
                    state["error_log"] = [
                        f"drafting: missing_info: {missing}"
                    ]
                
                logger.info(
                    f"Draft generated: confidence={state['confidence_score']:.2f}, "
                    f"guidance={state['rep_guidance']}, "
                    f"sources={len(state['sources_cited'])}"
                )
            else:
                # Parse failed - generate fallback
                logger.error("Failed to parse response, using fallback")
                state["draft_reply"] = (
                    "I was unable to generate a confident response for this ticket. "
                    "Please review the retrieved sources manually."
                )
                state["confidence_score"] = 0.3
                state["rep_guidance"] = "DO_NOT_SEND"
                state["sources_cited"] = []
                state["error_log"] = [
                    "drafting: JSON parse failed"
                ]

        except Exception as e:
            logger.error(f"Unexpected error in drafting: {e}")
            state["draft_reply"] = ""
            state["confidence_score"] = 0.0
            state["rep_guidance"] = "DO_NOT_SEND"
            state["sources_cited"] = []
            state["error_log"] = [
                f"drafting: unexpected error: {str(e)}"
            ]

        # Track agent path
        state["agent_path"] = ["drafting"]
        
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
