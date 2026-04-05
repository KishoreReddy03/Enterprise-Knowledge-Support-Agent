"""
Synthesis agent for evaluating retrieval quality and detecting contradictions.

This is Level 2 in the orchestration hierarchy. Evaluates whether retrieved
information is sufficient and consistent, then decides next step.
"""

import json
import logging
from typing import Any

import anthropic
from langfuse.decorators import observe

from config import settings
from core.agents.state import ContradictionInfo, SourceResult, TicketState

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """
    Evaluates retrieved information and detects contradictions.
    
    Uses Claude Haiku for sufficiency checks (cheap) and
    Claude Sonnet for contradiction detection (high-value task).
    """

    SUFFICIENCY_PROMPT = """Evaluate whether these search results can answer the support question.

Support Question:
{ticket}

Search Results:

DOCUMENTATION:
{docs}

PAST RESOLUTIONS (GitHub):
{github}

COMMUNITY ANSWERS (StackOverflow):
{stackoverflow}

Return JSON only, no explanation:
{{
    "score": 0.0-1.0,
    "gaps": ["list of information gaps if any"],
    "has_direct_answer": true/false,
    "best_source": "documentation|past_resolutions|community|none"
}}

Score guide:
- 1.0: Perfect match, direct answer found
- 0.7-0.9: Good coverage, can answer confidently
- 0.4-0.6: Partial coverage, might need more info
- 0.0-0.3: Insufficient, escalation likely needed"""

    CONTRADICTION_PROMPT = """Compare these information sources about a Stripe support question.

Sources:
{formatted_results}

Find contradictions where sources give different information about the same thing.

Return JSON array only, no explanation:
[
    {{
        "topic": "what they disagree about",
        "source_a": "what source A says",
        "source_b": "what source B says", 
        "likely_correct": "which is more likely correct and why",
        "severity": "high|medium|low"
    }}
]

Return empty array [] if no contradictions found.
Only flag actual contradictions, not just different aspects of the same topic."""

    def __init__(self) -> None:
        """Initialize synthesis agent with Anthropic client."""
        self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        logger.info("SynthesisAgent initialized")

    def _format_results_for_prompt(
        self,
        results: list[SourceResult],
        max_per_source: int = 3,
    ) -> str:
        """
        Format results for inclusion in prompts.
        
        Args:
            results: List of source results.
            max_per_source: Max results to include.
            
        Returns:
            Formatted string for prompt.
        """
        if not results:
            return "(No results)"
        
        formatted: list[str] = []
        for i, result in enumerate(results[:max_per_source]):
            text = result.get("text", "")[:500]  # Truncate long texts
            title = result.get("title", "Untitled")
            source_url = result.get("source_url", "")
            formatted.append(f"[{i+1}] {title}\n{text}\nSource: {source_url}")
        
        return "\n\n".join(formatted)

    def _format_all_results_for_contradiction(
        self,
        all_results: dict[str, list[SourceResult]],
    ) -> str:
        """
        Format all results for contradiction detection prompt.
        
        Args:
            all_results: Dict of source type to results.
            
        Returns:
            Formatted string for prompt.
        """
        sections: list[str] = []
        
        for source_type, results in all_results.items():
            if results:
                formatted = self._format_results_for_prompt(results, max_per_source=5)
                sections.append(f"=== {source_type.upper()} ===\n{formatted}")
        
        return "\n\n".join(sections)

    async def _check_sufficiency(
        self,
        ticket: str,
        results: dict[str, list[SourceResult]],
    ) -> dict[str, Any]:
        """
        Check if retrieved results are sufficient to answer the ticket.
        
        Uses Claude Haiku for cost efficiency.
        
        Args:
            ticket: The support ticket content.
            results: Dict of source type to results.
            
        Returns:
            Dict with score (0-1), gaps list, and metadata.
        """
        prompt = self.SUFFICIENCY_PROMPT.format(
            ticket=ticket,
            docs=self._format_results_for_prompt(results.get("documentation", [])),
            github=self._format_results_for_prompt(results.get("past_resolutions", [])),
            stackoverflow=self._format_results_for_prompt(results.get("community_answers", [])),
        )

        try:
            response = self._client.messages.create(
                model=settings.HAIKU_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            
            response_text = response.content[0].text.strip()
            result = self._parse_json(response_text)
            
            if result:
                return {
                    "score": float(result.get("score", 0.5)),
                    "gaps": result.get("gaps", []),
                    "has_direct_answer": result.get("has_direct_answer", False),
                    "best_source": result.get("best_source", "none"),
                }
            
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error in sufficiency check: {e}")
        except Exception as e:
            logger.error(f"Error in sufficiency check: {e}")
        
        # Default on failure
        return {
            "score": 0.5,
            "gaps": ["Unable to evaluate sufficiency"],
            "has_direct_answer": False,
            "best_source": "none",
        }

    async def _detect_contradictions(
        self,
        results: dict[str, list[SourceResult]],
    ) -> list[ContradictionInfo]:
        """
        Detect contradictions between information sources.
        
        Uses Claude Sonnet for high-quality contradiction detection.
        
        Args:
            results: Dict of source type to results.
            
        Returns:
            List of ContradictionInfo dicts.
        """
        formatted_results = self._format_all_results_for_contradiction(results)
        
        if not formatted_results or formatted_results.count("===") < 2:
            # Not enough sources to have contradictions
            return []

        prompt = self.CONTRADICTION_PROMPT.format(
            formatted_results=formatted_results,
        )

        try:
            response = self._client.messages.create(
                model=settings.SONNET_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            
            response_text = response.content[0].text.strip()
            result = self._parse_json(response_text)
            
            if isinstance(result, list):
                contradictions: list[ContradictionInfo] = []
                for item in result:
                    contradictions.append(
                        ContradictionInfo(
                            source_a=item.get("source_a", ""),
                            source_b=item.get("source_b", ""),
                            description=item.get("topic", ""),
                            resolution=item.get("likely_correct", ""),
                        )
                    )
                return contradictions
            
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error in contradiction detection: {e}")
        except Exception as e:
            logger.error(f"Error in contradiction detection: {e}")
        
        return []

    def _parse_json(self, text: str) -> dict[str, Any] | list[Any] | None:
        """
        Parse JSON from model response.
        
        Args:
            text: Raw response text.
            
        Returns:
            Parsed dict/list or None if parsing failed.
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
            return None

    def _format_context(
        self,
        all_results: dict[str, list[SourceResult]],
        contradictions: list[ContradictionInfo],
    ) -> str:
        """
        Format synthesized context for the drafting agent.
        
        Args:
            all_results: All retrieved results by source.
            contradictions: Detected contradictions.
            
        Returns:
            Formatted context string for drafting.
        """
        sections: list[str] = []
        
        # Add results by source type
        source_names = {
            "documentation": "Official Stripe Documentation",
            "past_resolutions": "Past Issue Resolutions (GitHub)",
            "community_answers": "Community Solutions (StackOverflow)",
        }
        
        for source_key, display_name in source_names.items():
            results = all_results.get(source_key, [])
            if results:
                sections.append(f"## {display_name}\n")
                for i, result in enumerate(results[:5]):
                    title = result.get("title", "Untitled")
                    text = result.get("text", "")[:800]
                    url = result.get("source_url", "")
                    is_stale = result.get("is_stale", False)
                    stale_marker = " [⚠️ POTENTIALLY OUTDATED]" if is_stale else ""
                    
                    sections.append(
                        f"### Source {i+1}: {title}{stale_marker}\n{text}\n"
                        f"URL: {url}\n"
                    )
        
        # Add contradictions warning if any
        if contradictions:
            sections.append("\n## ⚠️ Detected Contradictions\n")
            for c in contradictions:
                sections.append(
                    f"**Topic:** {c['description']}\n"
                    f"- Source A says: {c['source_a']}\n"
                    f"- Source B says: {c['source_b']}\n"
                    f"- Resolution: {c['resolution']}\n"
                )
        
        return "\n".join(sections)

    @observe(name="synthesis_agent")
    async def process(self, state: TicketState) -> TicketState:
        """
        Evaluate retrieval quality and decide next step.
        
        Checks if retrieved information is sufficient and consistent,
        then routes to drafting, more retrieval, or escalation.
        
        Args:
            state: Current ticket state with retrieval results.
            
        Returns:
            Updated state with synthesis decision and context.
        """
        # Organize results by source type
        all_results: dict[str, list[SourceResult]] = {
            "documentation": state.get("docs_results", []),
            "past_resolutions": state.get("github_results", []),
            "community_answers": state.get("stackoverflow_results", []),
        }
        
        total_results = sum(len(v) for v in all_results.values())
        
        # Fast path: no results at all
        if total_results == 0:
            logger.warning("No retrieval results - escalating")
            state["synthesis_decision"] = "escalate"
            state["knowledge_gaps"] = ["No relevant content found in any source"]
            state["synthesized_context"] = ""
            state["synthesis_confidence"] = 0.0
            state["contradictions"] = []
            state["agent_path"] = state.get("agent_path", []) + ["synthesis"]
            return state

        # Sufficiency check with Haiku (cheap)
        logger.info("Running sufficiency check...")
        sufficiency = await self._check_sufficiency(
            ticket=state.get("ticket_content", ""),
            results=all_results,
        )
        
        state["knowledge_gaps"] = sufficiency.get("gaps", [])
        
        # Contradiction check with Sonnet (only if we have enough content)
        contradictions: list[ContradictionInfo] = []
        if sufficiency["score"] >= 0.6:
            logger.info("Running contradiction detection...")
            contradictions = await self._detect_contradictions(all_results)
            if contradictions:
                logger.info(f"Found {len(contradictions)} contradiction(s)")
        
        state["contradictions"] = contradictions
        
        # Routing decision
        if sufficiency["score"] >= 0.70:
            state["synthesis_decision"] = "ready"
            logger.info(f"Synthesis decision: ready (score={sufficiency['score']:.2f})")
        elif sufficiency["score"] >= 0.40:
            state["synthesis_decision"] = "need_more"
            logger.info(f"Synthesis decision: need_more (score={sufficiency['score']:.2f})")
        else:
            state["synthesis_decision"] = "escalate"
            logger.info(f"Synthesis decision: escalate (score={sufficiency['score']:.2f})")
        
        # Format context for drafting agent
        state["synthesized_context"] = self._format_context(all_results, contradictions)
        state["synthesis_confidence"] = sufficiency["score"]
        
        # Track agent path
        state["agent_path"] = state.get("agent_path", []) + ["synthesis"]
        
        return state

    def route(self, state: TicketState) -> str:
        """
        Get routing decision from synthesis output.
        
        Args:
            state: Ticket state with synthesis decision.
            
        Returns:
            Route name based on synthesis_decision.
        """
        decision = state.get("synthesis_decision", "need_more")
        
        if decision == "ready":
            return "drafting"
        elif decision == "need_more":
            return "additional_retrieval"
        else:
            return "escalate"


# Module-level instance
synthesis_agent = SynthesisAgent()
