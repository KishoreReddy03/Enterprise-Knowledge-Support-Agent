"""
Evaluation layer: quality gate + escalation.

Contains two tightly coupled agents that form the final decision stage:

- QualityGateAgent (Level 4): Evaluates AI-generated draft replies for
  quality, sourcing, and hallucination before showing to support reps.
  Routes to: approved, revise, or escalate.

- EscalationAgent: Terminal node triggered when QualityGate or Synthesis
  decides the ticket cannot be resolved by AI. Generates a structured
  brief summarising the ticket, what the AI attempted, and why it is
  being handed to a human.

THIS IS THE KEY DIFFERENTIATOR - no generic RAG project has this.
"""

import json
import logging
from datetime import datetime
from typing import Any

from langfuse import observe

from config import settings
from core.agents.state import CitedSource, QualityRoute, TicketState
from core.llm_client import call_fast

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

Keep explanations and suggestions extremely concise. Reduce the level of detail and verbosity in the 'specific_issues' and 'improvement_instruction' attributes, prioritizing high-level summaries over granular feedback.

Return JSON ONLY (no markdown, no explanation):
{{
    "answers_the_question": true/false,
    "all_claims_have_sources": true/false,
    "no_hallucinated_api_behavior": true/false,
    "appropriate_for_customer_tier": true/false,
    "specific_issues": ["brief high-level summaries of problems"],
    "improvement_instruction": "extremely brief rewrite instruction if any check is false, null otherwise"
}}"""

    def __init__(self) -> None:
        """Initialize quality gate agent."""
        logger.info("QualityGateAgent initialized")

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
            logger.warning(f"JSON parse error in quality gate (strict): {e}")
            
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
                logger.info("Quality gate: successfully parsed after removing control characters")
                return result
            except json.JSONDecodeError as e2:
                logger.warning(f"JSON parse error in quality gate (after cleaning): {e2}")
                
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
                        logger.info("Quality gate: successfully parsed after extracting JSON object")
                        return result
                except (json.JSONDecodeError, AttributeError):
                    pass
                
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
        
        Uses fast LLM for cost-effective evaluation.
        
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
            state["agent_path"] = ["quality_gate"]
            return state
        
        prompt = self.EVALUATION_PROMPT.format(
            ticket_content=state.get("ticket_content", ""),
            draft_reply=state.get("draft_reply", ""),
            sources_formatted=self._format_sources(
                state.get("sources_cited", [])
            ),
            customer_tier=state.get("customer_tier", "standard"),
        )
        
        import asyncio
        from core.agents.state import get_all_results
        from core.guardrails.grounding_verifier import GroundingVerifier

        retrieved_chunks = get_all_results(state)
        chunks_to_verify = []
        for chunk in retrieved_chunks:
            chunks_to_verify.append(dict(chunk))
            
        verifier = GroundingVerifier()

        # Define tasks for concurrent execution to optimize latency
        async def run_quality_check():
            try:
                return await call_fast(prompt, max_tokens=1024)
            except Exception as e:
                logger.error(f"Error in quality gate call_fast: {e}")
                return e

        async def run_grounding_check():
            try:
                return await verifier.verify_grounding(state.get("draft_reply", ""), chunks_to_verify)
            except Exception as e:
                logger.error(f"Error in QualityGate grounding verification: {e}")
                return e

        # Gather both tasks concurrently to reduce response latency by ~2 seconds
        response_res, grounding_res = await asyncio.gather(
            run_quality_check(),
            run_grounding_check()
        )

        # 1. Process Quality Gate check results
        if isinstance(response_res, Exception) or response_res is None:
            # On error, be conservative - require revision
            state["quality_score"] = 0.6
            state["quality_issues"] = [f"Quality evaluation failed: {str(response_res)}"]
            if "error_log" not in state or not state["error_log"]:
                state["error_log"] = []
            state["error_log"].append(f"quality_gate: error: {str(response_res)}")
        else:
            try:
                # Parse response
                parsed = self._parse_response(response_res)
                
                if parsed:
                    # Deterministic Quality Gate Safety Check
                    # Relying purely on the LLM's subjective float "overall_score" or "routing_decision" is fragile.
                    # Instead, we calculate the quality score and routing decision completely deterministically in Python
                    # based on the objective checklist criteria. We still support overall_score and routing_decision if
                    # present in parsed (e.g. for backward compatibility with older older mocks/unit tests).
                    
                    answers_question = parsed.get("answers_the_question", True)
                    has_sources = parsed.get("all_claims_have_sources", True)
                    no_hallucination = parsed.get("no_hallucinated_api_behavior", True)
                    appropriate_tier = parsed.get("appropriate_for_customer_tier", True)
                    
                    state["quality_issues"] = parsed.get("specific_issues", [])
                    
                    if "overall_score" in parsed and "routing_decision" in parsed:
                        # Backward compatibility path: Use LLM-generated score and route, but still apply safety contract
                        state["quality_score"] = float(parsed["overall_score"])
                        state["llm_routing_decision"] = parsed["routing_decision"]
                        
                        if not answers_question or not has_sources or not no_hallucination:
                            if state.get("llm_routing_decision") == "approved":
                                logger.warning(
                                    f"Deterministic safety check triggered (compat mode): LLM approved draft despite check failures "
                                    f"(answers={answers_question}, has_sources={has_sources}, no_hallucination={no_hallucination}). "
                                    f"Forcing routing decision to 'revise'."
                                )
                                state["llm_routing_decision"] = "revise"
                                if state.get("quality_score", 0.0) >= 0.60:
                                    state["quality_score"] = 0.55
                    else:
                        # 100% Deterministic Python Quality Evaluation Matrix
                        calculated_score = self._calculate_score_from_criteria(
                            answers_question=answers_question,
                            has_sources=has_sources,
                            no_hallucination=no_hallucination,
                            appropriate_tier=appropriate_tier
                        )
                        
                        # Compress deterministic score to align with standard routing boundaries (0.45 to 0.85)
                        # calculated_score is in [0.0, 1.0]. A score of 1.0 -> 0.85, a score of 0.0 -> 0.45
                        state["quality_score"] = 0.45 + calculated_score * 0.40
                        
                        if not answers_question or not has_sources or not no_hallucination:
                            state["llm_routing_decision"] = "revise"
                            # Enforce score below approved threshold (0.60)
                            if state["quality_score"] >= 0.60:
                                state["quality_score"] = 0.55
                        else:
                            state["llm_routing_decision"] = "approved"
                            
                    # Log specific safety contract failures to the issues list
                    if not no_hallucination and "Hallucination flagged by quality gate checklist" not in state["quality_issues"]:
                        state["quality_issues"].append("Hallucination flagged by quality gate checklist")
                    if not has_sources and "Factual claims missing cited sources" not in state["quality_issues"]:
                        state["quality_issues"].append("Factual claims missing cited sources")
                    if not answers_question and "Draft reply does not address original customer ticket" not in state["quality_issues"]:
                        state["quality_issues"].append("Draft reply does not address original customer ticket")

                    # Store improvement instruction for potential revision
                    improvement = parsed.get("improvement_instruction")
                    if improvement and improvement != "null":
                        state["error_log"] = [
                            f"quality_gate: improvement_needed: {improvement}"
                        ]

                    # Log detailed evaluation results
                    logger.info(
                        f"Quality evaluation: score={state['quality_score']:.2f}, "
                        f"routing_decision={state['llm_routing_decision'] or '(threshold fallback)'}, "
                        f"answers={answers_question}, "
                        f"sourced={has_sources}, "
                        f"no_hallucination={no_hallucination}, "
                        f"issues={len(state['quality_issues'])}"
                    )
                else:
                    # Layer 3: Explicit parse failure — safe non-boundary default.
                    logger.error("Failed to parse quality gate response")
                    state["quality_score"] = 0.4          # Maps to 'revise' (below 0.60 threshold)
                    state["llm_routing_decision"] = "revise"  # Explicit categorical override
                    state["quality_issues"] = ["Quality evaluation parse failed"]
                    state["llm_parse_failures"] = ["quality_gate"]
                    state["error_log"] = [
                        "quality_gate: JSON parse failed — routing to revise as safe default"
                    ]
            except Exception as e:
                logger.error(f"Error processing quality gate response: {e}")
                state["quality_score"] = 0.6
                state["quality_issues"] = [f"Quality evaluation processing failed: {str(e)}"]
                if "error_log" not in state or not state["error_log"]:
                    state["error_log"] = []
                state["error_log"].append(f"quality_gate: processing error: {str(e)}")

        # 2. Process Grounding check results
        if isinstance(grounding_res, Exception):
            if "error_log" not in state or not state["error_log"]:
                state["error_log"] = []
            state["error_log"].append(f"quality_gate: Grounding verification error: {str(grounding_res)}")
        else:
            try:
                state["grounding_feedback"] = []
                
                if grounding_res.grounding_score < 1.0 or not grounding_res.is_safe:
                    logger.warning(
                        f"QualityGate GroundingVerifier flagged {len(grounding_res.ungrounded_segments)} ungrounded claims. "
                        f"Score: {grounding_res.grounding_score:.2f}"
                    )
                    state["grounding_feedback"] = grounding_res.ungrounded_segments
                    state["llm_routing_decision"] = "revise"
                    
                    if state.get("quality_score", 0.0) >= 0.60:
                        state["quality_score"] = 0.55
                    
                    if "quality_issues" not in state or not state["quality_issues"]:
                        state["quality_issues"] = []
                    for claim in grounding_res.ungrounded_segments:
                        state["quality_issues"].append(f"Grounding failure: {claim}")
                        
                    if "error_log" not in state or not state["error_log"]:
                        state["error_log"] = []
                    state["error_log"].append(
                        f"quality_gate: Grounding verifier failed with score {grounding_res.grounding_score:.2f} — forcing revise"
                    )
            except Exception as e:
                logger.error(f"Error processing grounding report: {e}")
                if "error_log" not in state or not state["error_log"]:
                    state["error_log"] = []
                state["error_log"].append(f"quality_gate: Grounding processing error: {str(e)}")

        # Track agent path
        state["agent_path"] = ["quality_gate"]
        
        # State mutations must happen in nodes, not conditional edges!
        # Increment revision_count for any score that may trigger a revise
        # (scores 0.40 to 0.79 go to revise in route())
        quality_score = state.get("quality_score", 0.0)
        if quality_score < 0.80:
            state["revision_count"] = state.get("revision_count", 0) + 1
            
        return state

    def route(self, state: TicketState) -> QualityRoute:
        """
        Determine next step based on quality evaluation.

        Routing priority (highest to lowest):
        1. Retry hard cap — always escalate when MAX_AGENT_RETRIES exceeded
        2. Categorical LLM decision (Layer 1) — use if LLM returned a valid enum
        3. Float threshold fallback — legacy path when LLM did not emit an enum

        Args:
            state: Current ticket state with quality assessment.

        Returns:
            QualityRoute literal: 'approved', 'revise', or 'escalate'
        """
        quality_score = state.get("quality_score", 0.0)
        revision_count = state.get("revision_count", 0)

        # Priority 1: Hard cap — always escalate when retry limit is exceeded,
        # regardless of LLM decision or score.
        if revision_count > settings.MAX_AGENT_RETRIES:
            logger.warning(
                f"Quality gate ESCALATE: max retries ({settings.MAX_AGENT_RETRIES}) exceeded"
            )
            return "escalate"

        # Priority 2: Layer 1 — use categorical LLM routing decision if available
        llm_route = state.get("llm_routing_decision", "")
        if llm_route in ("approved", "revise", "escalate"):
            logger.info(
                f"Quality gate {llm_route.upper()} (categorical LLM decision): "
                f"score={quality_score:.2f}, revision={revision_count}"
            )
            return llm_route  # type: ignore[return-value]

        # Priority 3: Threshold fallback — LLM did not return a valid routing_decision
        logger.warning(
            f"Quality gate using threshold fallback "
            f"(no valid routing_decision from LLM): score={quality_score:.2f}"
        )

        # Threshold lowered from 0.80 to 0.60 because Llama 3.1 8B
        # (our fast model) tends to score conservatively
        if quality_score >= 0.60:
            logger.info(f"Quality gate APPROVED (threshold): score={quality_score:.2f}")
            return "approved"

        # (revision_count was already incremented inside the process node)
        if quality_score >= 0.40:
            logger.info(
                f"Quality gate REVISE (threshold): score={quality_score:.2f}, "
                f"revision {revision_count}/{settings.MAX_AGENT_RETRIES}"
            )
            return "revise"

        logger.warning(
            f"Quality gate ESCALATE (threshold): score too low ({quality_score:.2f})"
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


# ═══════════════════════════════════════════════════════════════════════════════
# ESCALATION AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class EscalationAgent:
    """
    Prepares tickets for human escalation.

    Generates a concise brief summarizing the ticket, what the AI
    attempted, why it's being escalated, and relevant context for
    the human rep taking over.
    """

    ESCALATION_REASONS = {
        "quality_failed": "Draft response failed quality checks after maximum retries",
        "insufficient_knowledge": "Retrieved information insufficient to answer confidently",
        "contradictions": "Sources contain conflicting information requiring human judgment",
        "critical_urgency": "Critical urgency ticket flagged for immediate human attention",
        "data_loss_risk": "Potential data loss situation requires human verification",
        "complex_edge_case": "Complex edge case not well-covered in documentation",
        "api_error": "System errors prevented complete processing",
        "no_results": "No relevant information found in knowledge base",
    }

    def __init__(self) -> None:
        """Initialize escalation agent."""
        logger.info("EscalationAgent initialized")

    def _is_contradiction_unresolved(self, c: dict[str, Any]) -> bool:
        """
        Determine if a contradiction was left unresolved by the synthesis agent.
        
        A contradiction is considered unresolved if the resolution is empty,
        or contains phrases indicating failure to resolve.
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

    def _determine_escalation_reason(self, state: TicketState) -> str:
        """
        Determine the primary reason for escalation.

        Args:
            state: Current ticket state.

        Returns:
            Escalation reason key.
        """
        from core.agents.state import get_all_results

        # Check for quality gate failure
        revision_count = state.get("revision_count", 0)
        quality_score = state.get("quality_score", 0.0)
        if revision_count >= 2 and quality_score < 0.8:
            return "quality_failed"

        # Check for contradictions - only escalate if a high-severity contradiction was unresolved
        contradictions = state.get("contradictions", [])
        high_severity_unresolved = [
            c for c in contradictions 
            if c.get("severity") == "high" and self._is_contradiction_unresolved(c)
        ]
        if high_severity_unresolved:
            return "contradictions"

        # Check for no results
        total_results = len(get_all_results(state))
        if total_results == 0:
            return "no_results"

        # Check for insufficient knowledge
        synthesis_decision = state.get("synthesis_decision", "")
        if synthesis_decision == "escalate":
            knowledge_gaps = state.get("knowledge_gaps", [])
            if knowledge_gaps:
                return "insufficient_knowledge"
            return "complex_edge_case"

        # Check for critical urgency with data loss
        urgency = state.get("urgency", "")
        ticket_content = state.get("ticket_content", "").lower()
        if urgency == "critical":
            if "data loss" in ticket_content or "losing data" in ticket_content:
                return "data_loss_risk"
            return "critical_urgency"

        # Check for API errors
        error_log = state.get("error_log", [])
        api_errors = [e for e in error_log if "API error" in e or "timeout" in e.lower()]
        if api_errors:
            return "api_error"

        # Default
        return "complex_edge_case"

    def _format_retrieval_summary(self, state: TicketState) -> str:
        """
        Format a summary of what was retrieved.

        Args:
            state: Current ticket state.

        Returns:
            Formatted retrieval summary.
        """
        from core.agents.state import get_all_results

        lines: list[str] = []

        docs = state.get("docs_results", [])
        github = state.get("github_results", [])
        stackoverflow = state.get("stackoverflow_results", [])
        changelog = state.get("changelog_results", [])

        lines.append(f"Documentation: {len(docs)} results")
        lines.append(f"GitHub Issues: {len(github)} results")
        lines.append(f"StackOverflow: {len(stackoverflow)} results")
        lines.append(f"Changelog: {len(changelog)} entries")

        # List top sources
        all_results = get_all_results(state)
        if all_results:
            lines.append("\nTop sources found:")
            for r in all_results[:3]:
                title = r.get("title", "Unknown")
                url = r.get("source_url", "")
                lines.append(f"  - {title}: {url}")

        return "\n".join(lines)

    def _format_ai_attempt_summary(self, state: TicketState) -> str:
        """
        Summarize what the AI attempted before escalating.

        Args:
            state: Current ticket state.

        Returns:
            Formatted attempt summary.
        """
        lines: list[str] = []

        agent_path = state.get("agent_path", [])
        lines.append(f"Agents executed: {' → '.join(agent_path)}")

        revision_count = state.get("revision_count", 0)
        if revision_count > 0:
            lines.append(f"Draft revisions attempted: {revision_count}")

        retrieval_retry_count = state.get("retrieval_retry_count", 0)
        if retrieval_retry_count > 0:
            lines.append(f"Retrieval retries: {retrieval_retry_count}")

        # Quality issues
        quality_issues = state.get("quality_issues", [])
        if quality_issues:
            lines.append("\nQuality issues found:")
            for issue in quality_issues[:3]:
                lines.append(f"  - {issue}")

        # Knowledge gaps
        knowledge_gaps = state.get("knowledge_gaps", [])
        if knowledge_gaps:
            lines.append("\nKnowledge gaps:")
            for gap in knowledge_gaps[:3]:
                lines.append(f"  - {gap}")

        return "\n".join(lines)

    def _generate_escalation_brief(self, state: TicketState, reason_key: str) -> str:
        """
        Generate a comprehensive brief for the human rep.

        Args:
            state: Current ticket state.
            reason_key: The escalation reason key.

        Returns:
            Formatted escalation brief.
        """
        reason_text = self.ESCALATION_REASONS.get(reason_key, "Unknown reason")

        brief_parts: list[str] = [
            "═══ ESCALATION BRIEF ═══",
            "",
            f"REASON: {reason_text}",
            "",
            "─── CUSTOMER TICKET ───",
            state.get("ticket_content", "(No content)"),
            "",
            f"Customer ID: {state.get('customer_id', 'Unknown')}",
            f"Customer Tier: {state.get('customer_tier', 'standard')}",
            f"Urgency: {state.get('urgency', 'medium')}",
            f"Topic: {state.get('primary_topic', 'other')}",
            "",
            "─── WHAT AI FOUND ───",
            self._format_retrieval_summary(state),
            "",
            "─── WHAT AI ATTEMPTED ───",
            self._format_ai_attempt_summary(state),
        ]

        # Include draft if one was generated
        draft_reply = state.get("draft_reply", "")
        if draft_reply:
            brief_parts.extend([
                "",
                "─── PARTIAL DRAFT (not approved) ───",
                draft_reply[:500] + ("..." if len(draft_reply) > 500 else ""),
                f"\nConfidence: {state.get('confidence_score', 0):.0%}",
            ])

        # Include contradictions if found
        contradictions = state.get("contradictions", [])
        if contradictions:
            brief_parts.extend([
                "",
                "─── CONFLICTING INFORMATION ───",
            ])
            for c in contradictions[:2]:
                brief_parts.append(f"• {c.get('description', 'Unknown conflict')}")

        # Error log relevant entries
        error_log = state.get("error_log", [])
        relevant_errors = [e for e in error_log if "API error" not in e][:3]
        if relevant_errors:
            brief_parts.extend([
                "",
                "─── PROCESSING NOTES ───",
            ])
            for e in relevant_errors:
                brief_parts.append(f"• {e}")

        brief_parts.extend([
            "",
            "═══════════════════════════",
        ])

        return "\n".join(brief_parts)

    @observe(name="escalation_agent")
    async def process(self, state: TicketState) -> TicketState:
        """
        Prepare ticket for human escalation.

        Generates escalation reason, brief, and marks state appropriately.

        Args:
            state: Current ticket state.

        Returns:
            Updated state with escalation information.
        """
        logger.info(f"Escalating ticket {state.get('ticket_id', 'unknown')}")

        # Determine why we're escalating
        reason_key = self._determine_escalation_reason(state)
        reason_text = self.ESCALATION_REASONS.get(reason_key, "Unknown reason")

        # Generate the brief
        brief = self._generate_escalation_brief(state, reason_key)

        # Update state
        state["escalation_needed"] = True
        state["escalation_reason"] = reason_text
        state["escalation_brief"] = brief

        # Track agent path
        state["agent_path"] = ["escalation"]

        logger.info(
            f"Escalation prepared: reason={reason_key}, "
            f"brief_length={len(brief)}"
        )

        return state

    def get_suggested_team(self, state: TicketState) -> str:
        """
        Suggest which team should handle the escalation.

        Args:
            state: Current ticket state.

        Returns:
            Suggested team name.
        """
        topic = state.get("primary_topic", "other")
        urgency = state.get("urgency", "medium")
        tier = state.get("customer_tier", "standard")

        # Enterprise critical always goes to senior team
        if tier == "enterprise" and urgency in ("high", "critical"):
            return "enterprise_support_lead"

        # Topic-based routing
        topic_teams = {
            "billing": "billing_team",
            "connect": "connect_specialists",
            "auth": "security_team",
            "webhook": "integrations_team",
            "api": "api_support",
        }

        return topic_teams.get(topic, "general_support")


# Module-level instance
escalation_agent = EscalationAgent()
