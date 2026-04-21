"""
Pattern detector for weekly ticket analysis.

Runs weekly to find patterns humans miss. Identifies:
- Recurring issue patterns with frequency
- Documentation gaps where AI confidence is consistently low
- Potential regressions (spikes in recent issues)
- Knowledge base health metrics

THIS IS WHAT MAKES THE PROJECT UNFORGETTABLE.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from langfuse import observe

from core.llm_client import call_fast

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RecurringPattern:
    """
    A recurring issue pattern found in ticket analysis.
    
    Attributes:
        pattern: Description of the pattern.
        frequency: Number of occurrences in analysis period.
        avg_confidence: Average AI confidence for these tickets.
        example_tickets: Sample ticket IDs showing this pattern.
        recommendation: Suggested action to address this pattern.
    """
    pattern: str
    frequency: int
    avg_confidence: float
    example_tickets: list[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class DocumentationGap:
    """
    Identified gap in documentation coverage.
    
    When AI confidence is consistently low for a topic,
    it indicates missing or inadequate documentation.
    
    Attributes:
        topic: The topic with documentation gaps.
        frequency: How often this topic appears.
        avg_confidence: Average confidence (should be low).
        example_queries: Sample queries that exposed this gap.
        suggested_doc_title: Suggested documentation to create.
    """
    topic: str
    frequency: int
    avg_confidence: float
    example_queries: list[str] = field(default_factory=list)
    suggested_doc_title: str = ""


@dataclass
class PotentialRegression:
    """
    A potential regression or emerging issue.
    
    Detected when issue frequency spikes compared to baseline.
    
    Attributes:
        topic: The topic showing increased frequency.
        recent_count: Count in recent period (e.g., last 7 days).
        prior_count: Count in prior period (e.g., prior 3 weeks).
        spike_factor: Ratio of recent to expected rate.
        likely_cause: Hypothesized cause if identifiable.
    """
    topic: str
    recent_count: int
    prior_count: int
    spike_factor: float
    likely_cause: str = ""


@dataclass
class PatternReport:
    """
    Complete pattern analysis report.
    
    Generated weekly to provide insights into support trends,
    documentation gaps, and potential issues.
    
    Attributes:
        report_date: When the report was generated.
        analysis_period_days: How many days of data were analyzed.
        total_tickets_analyzed: Number of tickets in analysis.
        recurring_patterns: Top recurring issue patterns.
        documentation_gaps: Identified documentation gaps.
        potential_regressions: Potential regressions/spikes.
        overall_kb_health_score: Knowledge base effectiveness (0-1).
        top_recommendation: Single most important action item.
        insufficient_data: True if not enough data for analysis.
    """
    report_date: str = ""
    analysis_period_days: int = 30
    total_tickets_analyzed: int = 0
    recurring_patterns: list[RecurringPattern] = field(default_factory=list)
    documentation_gaps: list[DocumentationGap] = field(default_factory=list)
    potential_regressions: list[PotentialRegression] = field(default_factory=list)
    overall_kb_health_score: float = 0.0
    top_recommendation: str = ""
    insufficient_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for storage."""
        return {
            "report_date": self.report_date,
            "analysis_period_days": self.analysis_period_days,
            "total_tickets_analyzed": self.total_tickets_analyzed,
            "recurring_patterns": [
                {
                    "pattern": p.pattern,
                    "frequency": p.frequency,
                    "avg_confidence": p.avg_confidence,
                    "example_tickets": p.example_tickets,
                    "recommendation": p.recommendation,
                }
                for p in self.recurring_patterns
            ],
            "documentation_gaps": [
                {
                    "topic": g.topic,
                    "frequency": g.frequency,
                    "avg_confidence": g.avg_confidence,
                    "example_queries": g.example_queries,
                    "suggested_doc_title": g.suggested_doc_title,
                }
                for g in self.documentation_gaps
            ],
            "potential_regressions": [
                {
                    "topic": r.topic,
                    "recent_count": r.recent_count,
                    "prior_count": r.prior_count,
                    "spike_factor": r.spike_factor,
                    "likely_cause": r.likely_cause,
                }
                for r in self.potential_regressions
            ],
            "overall_kb_health_score": self.overall_kb_health_score,
            "top_recommendation": self.top_recommendation,
            "insufficient_data": self.insufficient_data,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK DATABASE CLIENT (until supabase_client.py is implemented)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TicketRecord:
    """Database ticket record."""
    id: str
    content: str
    primary_topic: str
    created_at: datetime


@dataclass
class ResponseRecord:
    """Database response record."""
    id: str
    ticket_id: str
    confidence_score: float
    was_escalated: bool
    created_at: datetime


@dataclass
class FeedbackRecord:
    """Database feedback record."""
    id: str
    response_id: str
    was_helpful: bool
    edit_distance: int
    created_at: datetime


class DatabaseClient:
    """
    Database operations for pattern detection.
    
    This is a placeholder that will be replaced with actual
    Supabase client implementation.
    """

    async def get_tickets_last_n_days(
        self,
        days: int = 30,
    ) -> list[TicketRecord]:
        """
        Fetch tickets from the last N days.
        
        Args:
            days: Number of days to look back.
            
        Returns:
            List of ticket records.
        """
        # Placeholder - will query Supabase tickets table
        logger.info(f"Fetching tickets from last {days} days")
        return []

    async def get_responses_for_tickets(
        self,
        ticket_ids: list[str],
    ) -> list[ResponseRecord]:
        """
        Fetch responses for given ticket IDs.
        
        Args:
            ticket_ids: List of ticket IDs.
            
        Returns:
            List of response records.
        """
        # Placeholder - will query Supabase agent_responses table
        logger.info(f"Fetching responses for {len(ticket_ids)} tickets")
        return []

    async def get_feedback_last_n_days(
        self,
        days: int = 30,
    ) -> list[FeedbackRecord]:
        """
        Fetch feedback from the last N days.
        
        Args:
            days: Number of days to look back.
            
        Returns:
            List of feedback records.
        """
        # Placeholder - will query Supabase rep_feedback table
        logger.info(f"Fetching feedback from last {days} days")
        return []

    async def store_pattern_report(
        self,
        report: dict[str, Any],
    ) -> str:
        """
        Store pattern report to database.
        
        Args:
            report: Report dictionary to store.
            
        Returns:
            Created report ID.
        """
        # Placeholder - will insert into Supabase pattern_reports table
        logger.info(f"Storing pattern report with {len(report)} fields")
        return f"report_{datetime.utcnow().strftime('%Y%m%d')}"


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class PatternDetector:
    """
    Weekly pattern analysis for support tickets.
    
    Identifies:
    - Recurring issue patterns that could be automated
    - Documentation gaps where AI struggles
    - Potential regressions from recent changes
    - Knowledge base health metrics
    """

    MINIMUM_TICKETS_FOR_ANALYSIS = 10
    CONFIDENCE_LOW_THRESHOLD = 0.65
    SPIKE_FACTOR_THRESHOLD = 2.0

    ANALYSIS_PROMPT = """Analyze {ticket_count} support tickets from the last {days} days.

TICKET DATA:
{ticket_summaries}

ANALYSIS TASKS:

1. RECURRING PATTERNS (top 5)
Find issues that appear multiple times with similar root causes.
Look for:
- Same error codes appearing repeatedly
- Similar API usage questions
- Common integration problems
- Frequently misunderstood features

2. DOCUMENTATION GAPS
Find topics where AI confidence was consistently low (<0.65).
These indicate areas where documentation is missing or inadequate.
Consider:
- Topics with low avg_confidence
- Topics with high escalation rates
- Topics where feedback was negative

3. POTENTIAL REGRESSIONS
Find issues that spiked in the last 7 days compared to the prior 3 weeks.
A spike_factor > 2.0 suggests a possible regression or breaking change.
Calculate: recent_7_day_count / (prior_21_day_count / 3)

4. KNOWLEDGE BASE HEALTH
Score 0-1 based on:
- Overall average confidence across tickets
- Percentage of tickets that needed escalation
- Diversity of topics handled successfully

Return ONLY valid JSON (no markdown, no explanation):
{{
    "recurring_patterns": [
        {{
            "pattern": "Brief description of the pattern",
            "frequency": 12,
            "avg_confidence": 0.72,
            "example_tickets": ["ticket_id_1", "ticket_id_2"],
            "recommendation": "Specific action to address this"
        }}
    ],
    "documentation_gaps": [
        {{
            "topic": "The topic needing documentation",
            "frequency": 8,
            "avg_confidence": 0.45,
            "example_queries": ["example question 1", "example question 2"],
            "suggested_doc_title": "Suggested documentation title"
        }}
    ],
    "potential_regressions": [
        {{
            "topic": "Topic showing spike",
            "recent_count": 15,
            "prior_count": 6,
            "spike_factor": 2.5,
            "likely_cause": "Possible cause if identifiable"
        }}
    ],
    "overall_kb_health_score": 0.75,
    "top_recommendation": "Single most important action to take"
}}"""

    def __init__(self, db: DatabaseClient | None = None) -> None:
        """
        Initialize pattern detector.
        
        Args:
            db: Database client for fetching data. Uses default if not provided.
        """
        self._db = db or DatabaseClient()
        logger.info("PatternDetector initialized")

    def _prepare_ticket_summaries(
        self,
        tickets: list[TicketRecord],
        responses: list[ResponseRecord],
        feedback: list[FeedbackRecord],
    ) -> list[dict[str, Any]]:
        """
        Prepare ticket summaries for analysis.
        
        Args:
            tickets: List of ticket records.
            responses: List of response records.
            feedback: List of feedback records.
            
        Returns:
            List of ticket summary dicts for the prompt.
        """
        # Create lookup maps
        response_by_ticket: dict[str, ResponseRecord] = {
            r.ticket_id: r for r in responses
        }
        feedback_by_response: dict[str, list[FeedbackRecord]] = {}
        for f in feedback:
            feedback_by_response.setdefault(f.response_id, []).append(f)
        
        summaries: list[dict[str, Any]] = []
        
        for ticket in tickets:
            response = response_by_ticket.get(ticket.id)
            
            summary = {
                "ticket_id": ticket.id,
                "content": ticket.content[:200],
                "topic": ticket.primary_topic,
                "created_at": ticket.created_at.isoformat(),
            }
            
            if response:
                summary["confidence"] = response.confidence_score
                summary["was_escalated"] = response.was_escalated
                
                ticket_feedback = feedback_by_response.get(response.id, [])
                summary["had_feedback"] = len(ticket_feedback) > 0
                if ticket_feedback:
                    summary["feedback_positive"] = any(
                        f.was_helpful for f in ticket_feedback
                    )
            else:
                summary["confidence"] = 0.0
                summary["was_escalated"] = True
                summary["had_feedback"] = False
            
            summaries.append(summary)
        
        return summaries

    def _calculate_local_metrics(
        self,
        tickets: list[TicketRecord],
        responses: list[ResponseRecord],
    ) -> dict[str, Any]:
        """
        Calculate local metrics before sending to Claude.
        
        Reduces token usage by pre-computing statistics.
        
        Args:
            tickets: List of ticket records.
            responses: List of response records.
            
        Returns:
            Dict of computed metrics.
        """
        response_by_ticket = {r.ticket_id: r for r in responses}
        
        confidences: list[float] = []
        escalations = 0
        topic_counts: dict[str, int] = {}
        topic_confidences: dict[str, list[float]] = {}
        
        # Last 7 days vs prior
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_topics: dict[str, int] = {}
        prior_topics: dict[str, int] = {}
        
        for ticket in tickets:
            topic = ticket.primary_topic
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            response = response_by_ticket.get(ticket.id)
            if response:
                confidences.append(response.confidence_score)
                topic_confidences.setdefault(topic, []).append(
                    response.confidence_score
                )
                if response.was_escalated:
                    escalations += 1
            
            # Time-based bucketing
            if ticket.created_at >= seven_days_ago:
                recent_topics[topic] = recent_topics.get(topic, 0) + 1
            else:
                prior_topics[topic] = prior_topics.get(topic, 0) + 1
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        escalation_rate = escalations / len(tickets) if tickets else 0.0
        
        # Topics with low confidence
        low_confidence_topics = [
            {
                "topic": topic,
                "avg_confidence": sum(confs) / len(confs),
                "count": len(confs),
            }
            for topic, confs in topic_confidences.items()
            if sum(confs) / len(confs) < self.CONFIDENCE_LOW_THRESHOLD
        ]
        
        # Potential spikes
        spikes: list[dict[str, Any]] = []
        for topic, recent in recent_topics.items():
            prior = prior_topics.get(topic, 0)
            expected_rate = prior / 3 if prior > 0 else 0.1
            spike_factor = recent / expected_rate if expected_rate > 0 else float(recent)
            
            if spike_factor >= self.SPIKE_FACTOR_THRESHOLD and recent >= 3:
                spikes.append({
                    "topic": topic,
                    "recent_count": recent,
                    "prior_count": prior,
                    "spike_factor": round(spike_factor, 2),
                })
        
        return {
            "total_tickets": len(tickets),
            "avg_confidence": round(avg_confidence, 3),
            "escalation_rate": round(escalation_rate, 3),
            "topic_distribution": topic_counts,
            "low_confidence_topics": sorted(
                low_confidence_topics,
                key=lambda x: x["avg_confidence"]
            )[:5],
            "potential_spikes": sorted(
                spikes,
                key=lambda x: x["spike_factor"],
                reverse=True
            )[:5],
        }

    def _parse_analysis_response(self, text: str) -> dict[str, Any] | None:
        """
        Parse JSON response from Claude.
        
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
            logger.error(f"Failed to parse analysis response: {e}")
            logger.debug(f"Response text: {text[:500]}")
            return None

    def _build_report_from_analysis(
        self,
        analysis: dict[str, Any],
        total_tickets: int,
    ) -> PatternReport:
        """
        Build PatternReport from Claude analysis output.
        
        Args:
            analysis: Parsed analysis dict from Claude.
            total_tickets: Total tickets analyzed.
            
        Returns:
            PatternReport instance.
        """
        # Parse recurring patterns
        patterns: list[RecurringPattern] = []
        for p in analysis.get("recurring_patterns", []):
            patterns.append(RecurringPattern(
                pattern=p.get("pattern", "Unknown"),
                frequency=p.get("frequency", 0),
                avg_confidence=p.get("avg_confidence", 0.0),
                example_tickets=p.get("example_tickets", []),
                recommendation=p.get("recommendation", ""),
            ))
        
        # Parse documentation gaps
        gaps: list[DocumentationGap] = []
        for g in analysis.get("documentation_gaps", []):
            gaps.append(DocumentationGap(
                topic=g.get("topic", "Unknown"),
                frequency=g.get("frequency", 0),
                avg_confidence=g.get("avg_confidence", 0.0),
                example_queries=g.get("example_queries", []),
                suggested_doc_title=g.get("suggested_doc_title", ""),
            ))
        
        # Parse regressions
        regressions: list[PotentialRegression] = []
        for r in analysis.get("potential_regressions", []):
            regressions.append(PotentialRegression(
                topic=r.get("topic", "Unknown"),
                recent_count=r.get("recent_count", 0),
                prior_count=r.get("prior_count", 0),
                spike_factor=r.get("spike_factor", 0.0),
                likely_cause=r.get("likely_cause", ""),
            ))
        
        return PatternReport(
            report_date=datetime.utcnow().isoformat(),
            analysis_period_days=30,
            total_tickets_analyzed=total_tickets,
            recurring_patterns=patterns,
            documentation_gaps=gaps,
            potential_regressions=regressions,
            overall_kb_health_score=analysis.get("overall_kb_health_score", 0.0),
            top_recommendation=analysis.get("top_recommendation", ""),
            insufficient_data=False,
        )

    @observe(name="pattern_detector_weekly")
    async def run_weekly_analysis(
        self,
        days: int = 30,
    ) -> PatternReport:
        """
        Run weekly pattern analysis on support tickets.
        
        Analyzes tickets from the specified period to identify:
        - Recurring patterns that could be automated
        - Documentation gaps needing attention
        - Potential regressions from recent changes
        - Overall knowledge base health
        
        Args:
            days: Number of days to analyze (default 30).
            
        Returns:
            PatternReport with analysis results.
        """
        logger.info(f"Starting weekly pattern analysis for last {days} days")
        
        # Fetch data
        tickets = await self._db.get_tickets_last_n_days(days)
        
        if len(tickets) < self.MINIMUM_TICKETS_FOR_ANALYSIS:
            logger.warning(
                f"Insufficient data: {len(tickets)} tickets "
                f"(minimum: {self.MINIMUM_TICKETS_FOR_ANALYSIS})"
            )
            return PatternReport(
                report_date=datetime.utcnow().isoformat(),
                analysis_period_days=days,
                total_tickets_analyzed=len(tickets),
                insufficient_data=True,
            )
        
        ticket_ids = [t.id for t in tickets]
        responses = await self._db.get_responses_for_tickets(ticket_ids)
        feedback = await self._db.get_feedback_last_n_days(days)
        
        # Prepare data for analysis
        ticket_summaries = self._prepare_ticket_summaries(
            tickets, responses, feedback
        )
        local_metrics = self._calculate_local_metrics(tickets, responses)
        
        # Build prompt with pre-computed metrics to reduce tokens
        enhanced_summaries = {
            "tickets": ticket_summaries[:50],  # Limit to top 50 for context
            "metrics": local_metrics,
        }
        
        prompt = self.ANALYSIS_PROMPT.format(
            ticket_count=len(tickets),
            days=days,
            ticket_summaries=json.dumps(enhanced_summaries, indent=2),
        )
        
        try:
            response = self._client.messages.create(
                model=settings.SONNET_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            
            response_text = response.content[0].text.strip()
            analysis = self._parse_analysis_response(response_text)
            
            if not analysis:
                logger.error("Failed to parse pattern analysis")
                return PatternReport(
                    report_date=datetime.utcnow().isoformat(),
                    analysis_period_days=days,
                    total_tickets_analyzed=len(tickets),
                    overall_kb_health_score=local_metrics["avg_confidence"],
                    top_recommendation="Analysis parsing failed - manual review needed",
                )
            
            # Build report
            report = self._build_report_from_analysis(analysis, len(tickets))
            
            # Store to database
            await self._db.store_pattern_report(report.to_dict())
            
            logger.info(
                f"Pattern analysis complete: "
                f"patterns={len(report.recurring_patterns)}, "
                f"gaps={len(report.documentation_gaps)}, "
                f"regressions={len(report.potential_regressions)}, "
                f"health_score={report.overall_kb_health_score:.2f}"
            )
            
            return report

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error in pattern analysis: {e}")
            return PatternReport(
                report_date=datetime.utcnow().isoformat(),
                analysis_period_days=days,
                total_tickets_analyzed=len(tickets),
                top_recommendation=f"Analysis failed: {str(e)}",
            )

        except Exception as e:
            logger.error(f"Unexpected error in pattern analysis: {e}")
            return PatternReport(
                report_date=datetime.utcnow().isoformat(),
                analysis_period_days=days,
                total_tickets_analyzed=len(tickets),
                top_recommendation=f"Analysis error: {str(e)}",
            )

    async def get_topic_health(self, topic: str) -> dict[str, Any]:
        """
        Get health metrics for a specific topic.
        
        Args:
            topic: The topic to analyze.
            
        Returns:
            Dict with topic health metrics.
        """
        tickets = await self._db.get_tickets_last_n_days(30)
        topic_tickets = [t for t in tickets if t.primary_topic == topic]
        
        if not topic_tickets:
            return {
                "topic": topic,
                "ticket_count": 0,
                "avg_confidence": None,
                "escalation_rate": None,
                "health_status": "no_data",
            }
        
        ticket_ids = [t.id for t in topic_tickets]
        responses = await self._db.get_responses_for_tickets(ticket_ids)
        
        confidences = [r.confidence_score for r in responses]
        escalations = sum(1 for r in responses if r.was_escalated)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        escalation_rate = escalations / len(responses) if responses else 0.0
        
        # Determine health status
        if avg_confidence >= 0.80 and escalation_rate <= 0.10:
            status = "healthy"
        elif avg_confidence >= 0.65 and escalation_rate <= 0.25:
            status = "moderate"
        else:
            status = "needs_attention"
        
        return {
            "topic": topic,
            "ticket_count": len(topic_tickets),
            "avg_confidence": round(avg_confidence, 3),
            "escalation_rate": round(escalation_rate, 3),
            "health_status": status,
        }


# Module-level instance
pattern_detector = PatternDetector()
