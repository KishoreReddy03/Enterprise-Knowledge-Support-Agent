"""
Escalation agent for human routing logic.

Handles cases where AI cannot confidently resolve a ticket:
- Quality gate failures after max retries
- Synthesis determines knowledge is insufficient
- Critical urgency with potential data loss
- Detected contradictions requiring human judgment
"""

import logging
from datetime import datetime

from langfuse import observe

from core.agents.state import TicketState, get_all_results

logger = logging.getLogger(__name__)


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

    def _determine_escalation_reason(self, state: TicketState) -> str:
        """
        Determine the primary reason for escalation.
        
        Args:
            state: Current ticket state.
            
        Returns:
            Escalation reason key.
        """
        # Check for quality gate failure
        revision_count = state.get("revision_count", 0)
        quality_score = state.get("quality_score", 0.0)
        if revision_count >= 2 and quality_score < 0.8:
            return "quality_failed"
        
        # Check for contradictions
        contradictions = state.get("contradictions", [])
        high_severity = [c for c in contradictions if c.get("severity") == "high"]
        if high_severity:
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
            lines.append(f"\nQuality issues found:")
            for issue in quality_issues[:3]:
                lines.append(f"  - {issue}")
        
        # Knowledge gaps
        knowledge_gaps = state.get("knowledge_gaps", [])
        if knowledge_gaps:
            lines.append(f"\nKnowledge gaps:")
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
        state["agent_path"] = state.get("agent_path", []) + ["escalation"]
        
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
