"""
Feedback loop processor for continuous learning.

Captures rep edits to AI-generated responses, classifies the edit type,
and uses this data to improve the system over time.

THE SINGLE FEATURE THAT MAKES THIS A LIVING SYSTEM.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from enum import Enum

import anthropic
from langfuse.decorators import observe

from config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class EditType(str, Enum):
    """Classification of edit types made by reps."""
    NONE = "none"
    FACTUAL_CORRECTION = "factual_correction"
    TONE_ADJUSTMENT = "tone_adjustment"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    STYLE = "style"


class EditSeverity(str, Enum):
    """Severity of the edit made."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class EditAnalysis:
    """
    Analysis of changes made by a rep to an AI response.
    
    Attributes:
        edit_type: Category of edit made.
        what_changed: Description of what was changed.
        severity: How significant the change was.
        affected_topics: Topics related to the correction.
    """
    edit_type: EditType
    what_changed: str
    severity: EditSeverity
    affected_topics: list[str] = field(default_factory=list)


@dataclass
class FeedbackRecord:
    """
    Record of rep feedback on an AI-generated response.
    
    Attributes:
        id: Unique feedback record ID.
        response_id: ID of the agent response being reviewed.
        original_reply: The AI-generated reply.
        edited_reply: The rep's edited version.
        edit_type: Classified type of edit.
        edit_analysis: Detailed analysis of the edit.
        was_sent: Whether the (edited) reply was sent to customer.
        rep_rating: Optional 1-5 rating from rep.
        rep_id: ID of the rep who made the edit.
        created_at: When the feedback was recorded.
    """
    response_id: str
    original_reply: str
    edited_reply: str
    edit_type: EditType
    was_sent: bool
    id: str = ""
    edit_analysis: EditAnalysis | None = None
    rep_rating: int | None = None
    rep_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "response_id": self.response_id,
            "original_reply": self.original_reply,
            "edited_reply": self.edited_reply,
            "edit_type": self.edit_type.value,
            "edit_analysis": {
                "edit_type": self.edit_analysis.edit_type.value,
                "what_changed": self.edit_analysis.what_changed,
                "severity": self.edit_analysis.severity.value,
                "affected_topics": self.edit_analysis.affected_topics,
            } if self.edit_analysis else None,
            "was_sent": self.was_sent,
            "rep_rating": self.rep_rating,
            "rep_id": self.rep_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ImprovementInsights:
    """
    Weekly summary of system improvement opportunities.
    
    Attributes:
        period_start: Start of analysis period.
        period_end: End of analysis period.
        total_feedback_count: Total feedback records analyzed.
        most_common_edit_type: Most frequent edit category.
        edit_type_distribution: Breakdown by edit type.
        factual_correction_rate: Rate of factual errors.
        avg_rep_rating: Average rep satisfaction.
        topics_with_most_corrections: Topics needing attention.
        high_severity_count: Number of high severity edits.
        recommendation: Primary improvement suggestion.
    """
    period_start: str
    period_end: str
    total_feedback_count: int
    most_common_edit_type: str
    edit_type_distribution: dict[str, int]
    factual_correction_rate: float
    avg_rep_rating: float | None
    topics_with_most_corrections: list[dict[str, Any]]
    high_severity_count: int
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_start": self.period_start,
            "period_end": self.period_end,
            "total_feedback_count": self.total_feedback_count,
            "most_common_edit_type": self.most_common_edit_type,
            "edit_type_distribution": self.edit_type_distribution,
            "factual_correction_rate": self.factual_correction_rate,
            "avg_rep_rating": self.avg_rep_rating,
            "topics_with_most_corrections": self.topics_with_most_corrections,
            "high_severity_count": self.high_severity_count,
            "recommendation": self.recommendation,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CLIENT (placeholder until supabase_client.py is implemented)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StoredFeedback:
    """Database feedback record."""
    id: str
    response_id: str
    edit_type: str
    severity: str
    was_sent: bool
    rep_rating: int | None
    topic: str
    created_at: datetime


class FeedbackDatabaseClient:
    """
    Database operations for feedback processing.
    
    Placeholder that will be replaced with actual Supabase implementation.
    """

    async def store_feedback(self, record: dict[str, Any]) -> str:
        """
        Store feedback record to database.
        
        Args:
            record: Feedback record dictionary.
            
        Returns:
            Created record ID.
        """
        logger.info(f"Storing feedback for response {record.get('response_id')}")
        return f"feedback_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    async def get_feedback_last_n_days(
        self,
        days: int = 7,
    ) -> list[StoredFeedback]:
        """
        Fetch feedback from the last N days.
        
        Args:
            days: Number of days to look back.
            
        Returns:
            List of feedback records.
        """
        logger.info(f"Fetching feedback from last {days} days")
        return []

    async def get_response_sources(
        self,
        response_id: str,
    ) -> list[dict[str, Any]]:
        """
        Get source chunks used in a response.
        
        Args:
            response_id: The response ID.
            
        Returns:
            List of source chunk references.
        """
        logger.info(f"Fetching sources for response {response_id}")
        return []

    async def flag_chunk_as_problematic(
        self,
        chunk_id: str,
        reason: str,
        response_id: str,
    ) -> None:
        """
        Flag a chunk as potentially problematic.
        
        Args:
            chunk_id: The chunk to flag.
            reason: Why it's being flagged.
            response_id: Response where issue was found.
        """
        logger.info(f"Flagging chunk {chunk_id}: {reason}")

    async def get_response_topic(self, response_id: str) -> str:
        """
        Get the primary topic for a response.
        
        Args:
            response_id: The response ID.
            
        Returns:
            Topic string.
        """
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# FEEDBACK PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class FeedbackProcessor:
    """
    Processes rep feedback to enable continuous system improvement.
    
    Captures edits made by reps, classifies them, and uses the data
    to identify knowledge base gaps and prompt improvements.
    """

    EDIT_CLASSIFICATION_PROMPT = """A support rep edited an AI-generated reply before sending to a customer.

ORIGINAL AI REPLY:
{original_reply}

REP'S EDITED VERSION:
{edited_reply}

Analyze the edit and classify it. Return JSON only (no markdown, no explanation):
{{
    "edit_type": "factual_correction|tone_adjustment|completeness|accuracy|style",
    "what_changed": "Brief description of what the rep changed",
    "severity": "high|medium|low",
    "affected_topics": ["list", "of", "topics", "affected"]
}}

EDIT TYPE DEFINITIONS:
- factual_correction: AI had wrong technical information (wrong API behavior, incorrect parameters, etc.)
- tone_adjustment: Response was too formal, informal, long, or short
- completeness: AI missed something important the customer asked about
- accuracy: Technically imprecise but not outright wrong
- style: Minor rewording, grammar, formatting only

SEVERITY:
- high: Could have caused customer harm or serious confusion
- medium: Noticeable quality issue
- low: Minor polish"""

    def __init__(self, db: FeedbackDatabaseClient | None = None) -> None:
        """
        Initialize feedback processor.
        
        Args:
            db: Database client. Uses default if not provided.
        """
        self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self._db = db or FeedbackDatabaseClient()
        logger.info("FeedbackProcessor initialized")

    def _calculate_edit_distance(self, original: str, edited: str) -> float:
        """
        Calculate normalized edit distance between texts.
        
        Simple word-level Jaccard distance as a proxy.
        
        Args:
            original: Original text.
            edited: Edited text.
            
        Returns:
            Distance score (0 = identical, 1 = completely different).
        """
        original_words = set(original.lower().split())
        edited_words = set(edited.lower().split())
        
        if not original_words and not edited_words:
            return 0.0
        
        intersection = len(original_words & edited_words)
        union = len(original_words | edited_words)
        
        return 1.0 - (intersection / union) if union > 0 else 0.0

    def _parse_classification_response(
        self,
        text: str,
    ) -> dict[str, Any] | None:
        """
        Parse JSON response from classification.
        
        Args:
            text: Raw response text.
            
        Returns:
            Parsed dict or None.
        """
        # Strip markdown if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
        text = text.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse classification: {e}")
            return None

    def _normalize_edit_type(self, edit_type_str: str) -> EditType:
        """
        Normalize edit type string to enum.
        
        Args:
            edit_type_str: Raw edit type string.
            
        Returns:
            EditType enum value.
        """
        mapping = {
            "factual_correction": EditType.FACTUAL_CORRECTION,
            "tone_adjustment": EditType.TONE_ADJUSTMENT,
            "completeness": EditType.COMPLETENESS,
            "accuracy": EditType.ACCURACY,
            "style": EditType.STYLE,
            "none": EditType.NONE,
        }
        return mapping.get(edit_type_str.lower(), EditType.STYLE)

    def _normalize_severity(self, severity_str: str) -> EditSeverity:
        """
        Normalize severity string to enum.
        
        Args:
            severity_str: Raw severity string.
            
        Returns:
            EditSeverity enum value.
        """
        mapping = {
            "high": EditSeverity.HIGH,
            "medium": EditSeverity.MEDIUM,
            "low": EditSeverity.LOW,
        }
        return mapping.get(severity_str.lower(), EditSeverity.LOW)

    @observe(name="feedback_capture_edit")
    async def capture_edit(
        self,
        response_id: str,
        original_reply: str,
        edited_reply: str,
        was_sent: bool,
        rep_rating: int | None = None,
        rep_id: str = "",
    ) -> FeedbackRecord:
        """
        Capture and analyze a rep's edit to an AI response.
        
        Classifies the edit type and stores it for learning.
        If it's a factual correction, flags the problematic sources.
        
        Args:
            response_id: ID of the agent response.
            original_reply: The AI-generated reply.
            edited_reply: The rep's edited version.
            was_sent: Whether the reply was sent to customer.
            rep_rating: Optional 1-5 quality rating from rep.
            rep_id: ID of the rep making the edit.
            
        Returns:
            FeedbackRecord with classification.
        """
        logger.info(f"Capturing edit for response {response_id}")
        
        # Check if no edit was made
        if original_reply.strip() == edited_reply.strip():
            logger.info(f"No edit detected for response {response_id}")
            record = FeedbackRecord(
                response_id=response_id,
                original_reply=original_reply,
                edited_reply=edited_reply,
                edit_type=EditType.NONE,
                was_sent=was_sent,
                rep_rating=rep_rating,
                rep_id=rep_id,
            )
            
            record_id = await self._db.store_feedback(record.to_dict())
            record.id = record_id
            
            return record
        
        # Calculate edit distance for quick assessment
        edit_distance = self._calculate_edit_distance(original_reply, edited_reply)
        logger.debug(f"Edit distance: {edit_distance:.2f}")
        
        # Classify the edit using Claude Haiku
        analysis: EditAnalysis | None = None
        edit_type = EditType.STYLE  # Default
        
        try:
            prompt = self.EDIT_CLASSIFICATION_PROMPT.format(
                original_reply=original_reply[:1000],
                edited_reply=edited_reply[:1000],
            )
            
            response = self._client.messages.create(
                model=settings.HAIKU_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            
            response_text = response.content[0].text.strip()
            parsed = self._parse_classification_response(response_text)
            
            if parsed:
                edit_type = self._normalize_edit_type(
                    parsed.get("edit_type", "style")
                )
                analysis = EditAnalysis(
                    edit_type=edit_type,
                    what_changed=parsed.get("what_changed", "Unknown"),
                    severity=self._normalize_severity(
                        parsed.get("severity", "low")
                    ),
                    affected_topics=parsed.get("affected_topics", []),
                )
                
                logger.info(
                    f"Edit classified: type={edit_type.value}, "
                    f"severity={analysis.severity.value}"
                )
            else:
                logger.warning("Failed to parse edit classification")
                
        except anthropic.APIError as e:
            logger.error(f"API error classifying edit: {e}")
            
        except Exception as e:
            logger.error(f"Unexpected error classifying edit: {e}")
        
        # Create feedback record
        record = FeedbackRecord(
            response_id=response_id,
            original_reply=original_reply,
            edited_reply=edited_reply,
            edit_type=edit_type,
            edit_analysis=analysis,
            was_sent=was_sent,
            rep_rating=rep_rating,
            rep_id=rep_id,
        )
        
        # Store to database
        record_id = await self._db.store_feedback(record.to_dict())
        record.id = record_id
        
        # If factual correction, flag the problematic sources
        if edit_type == EditType.FACTUAL_CORRECTION:
            await self._flag_problematic_sources(response_id, analysis)
        
        return record

    async def _flag_problematic_sources(
        self,
        response_id: str,
        analysis: EditAnalysis | None,
    ) -> None:
        """
        Flag source chunks that contributed to a factual error.
        
        Args:
            response_id: The response with the error.
            analysis: Edit analysis with details.
        """
        logger.info(f"Flagging problematic sources for response {response_id}")
        
        try:
            sources = await self._db.get_response_sources(response_id)
            
            reason = (
                analysis.what_changed if analysis 
                else "Factual correction required"
            )
            
            for source in sources:
                chunk_id = source.get("chunk_id")
                if chunk_id:
                    await self._db.flag_chunk_as_problematic(
                        chunk_id=chunk_id,
                        reason=reason,
                        response_id=response_id,
                    )
                    
            logger.info(f"Flagged {len(sources)} source chunks")
            
        except Exception as e:
            logger.error(f"Error flagging sources: {e}")

    @observe(name="feedback_get_insights")
    async def get_improvement_insights(
        self,
        days: int = 7,
    ) -> ImprovementInsights:
        """
        Generate weekly summary of improvement opportunities.
        
        Analyzes recent feedback to identify patterns in what
        the system gets wrong.
        
        Args:
            days: Number of days to analyze.
            
        Returns:
            ImprovementInsights with actionable recommendations.
        """
        logger.info(f"Generating improvement insights for last {days} days")
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        feedback = await self._db.get_feedback_last_n_days(days)
        
        if not feedback:
            return ImprovementInsights(
                period_start=start_date.isoformat(),
                period_end=end_date.isoformat(),
                total_feedback_count=0,
                most_common_edit_type="none",
                edit_type_distribution={},
                factual_correction_rate=0.0,
                avg_rep_rating=None,
                topics_with_most_corrections=[],
                high_severity_count=0,
                recommendation="No feedback data available for analysis.",
            )
        
        # Calculate metrics
        edit_type_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        topic_corrections: dict[str, int] = {}
        ratings: list[int] = []
        factual_corrections = 0
        
        for fb in feedback:
            # Count edit types
            edit_type_counts[fb.edit_type] = (
                edit_type_counts.get(fb.edit_type, 0) + 1
            )
            
            # Count severities
            if fb.severity in severity_counts:
                severity_counts[fb.severity] += 1
            
            # Track factual corrections by topic
            if fb.edit_type == "factual_correction":
                factual_corrections += 1
                topic_corrections[fb.topic] = (
                    topic_corrections.get(fb.topic, 0) + 1
                )
            
            # Collect ratings
            if fb.rep_rating is not None:
                ratings.append(fb.rep_rating)
        
        # Calculate derived metrics
        total = len(feedback)
        factual_rate = factual_corrections / total if total > 0 else 0.0
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        
        # Find most common edit type
        most_common = max(
            edit_type_counts.items(),
            key=lambda x: x[1],
            default=("none", 0)
        )[0]
        
        # Sort topics by correction count
        sorted_topics = sorted(
            topic_corrections.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        topics_with_corrections = [
            {"topic": topic, "correction_count": count}
            for topic, count in sorted_topics
        ]
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            most_common_edit_type=most_common,
            factual_rate=factual_rate,
            high_severity_count=severity_counts["high"],
            topics_with_corrections=topics_with_corrections,
        )
        
        return ImprovementInsights(
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat(),
            total_feedback_count=total,
            most_common_edit_type=most_common,
            edit_type_distribution=edit_type_counts,
            factual_correction_rate=round(factual_rate, 3),
            avg_rep_rating=round(avg_rating, 2) if avg_rating else None,
            topics_with_most_corrections=topics_with_corrections,
            high_severity_count=severity_counts["high"],
            recommendation=recommendation,
        )

    def _generate_recommendation(
        self,
        most_common_edit_type: str,
        factual_rate: float,
        high_severity_count: int,
        topics_with_corrections: list[dict[str, Any]],
    ) -> str:
        """
        Generate actionable recommendation based on feedback patterns.
        
        Args:
            most_common_edit_type: Most frequent edit type.
            factual_rate: Rate of factual corrections.
            high_severity_count: Number of high severity edits.
            topics_with_corrections: Topics needing attention.
            
        Returns:
            Recommendation string.
        """
        recommendations: list[str] = []
        
        # High factual correction rate
        if factual_rate > 0.2:
            recommendations.append(
                f"CRITICAL: {factual_rate:.0%} of responses need factual corrections. "
                "Review knowledge base accuracy and source freshness."
            )
        
        # Many high severity edits
        if high_severity_count >= 5:
            recommendations.append(
                f"{high_severity_count} high-severity edits detected. "
                "Investigate root causes immediately."
            )
        
        # Topic-specific issues
        if topics_with_corrections:
            top_topic = topics_with_corrections[0]
            recommendations.append(
                f"Topic '{top_topic['topic']}' has {top_topic['correction_count']} "
                "corrections. Review documentation coverage for this area."
            )
        
        # Edit type specific recommendations
        if most_common_edit_type == "tone_adjustment":
            recommendations.append(
                "Tone adjustments are most common. Consider tuning system prompts "
                "for more appropriate customer communication style."
            )
        elif most_common_edit_type == "completeness":
            recommendations.append(
                "Completeness issues are most common. Improve retrieval to ensure "
                "all aspects of customer questions are addressed."
            )
        elif most_common_edit_type == "accuracy":
            recommendations.append(
                "Accuracy issues are most common. Consider adding verification "
                "steps or expanding retrieval depth."
            )
        
        if not recommendations:
            recommendations.append(
                "Feedback patterns look healthy. Continue monitoring for changes."
            )
        
        return " | ".join(recommendations)

    async def get_rep_performance(
        self,
        rep_id: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get feedback statistics for a specific rep.
        
        Useful for understanding individual patterns.
        
        Args:
            rep_id: The rep to analyze.
            days: Period to analyze.
            
        Returns:
            Dict with rep's feedback patterns.
        """
        feedback = await self._db.get_feedback_last_n_days(days)
        rep_feedback = [f for f in feedback if f.id == rep_id]
        
        if not rep_feedback:
            return {
                "rep_id": rep_id,
                "feedback_count": 0,
                "avg_edit_rate": None,
                "common_edit_types": [],
            }
        
        edit_types: dict[str, int] = {}
        for fb in rep_feedback:
            edit_types[fb.edit_type] = edit_types.get(fb.edit_type, 0) + 1
        
        return {
            "rep_id": rep_id,
            "feedback_count": len(rep_feedback),
            "edit_type_distribution": edit_types,
            "most_common_edit": max(
                edit_types.items(),
                key=lambda x: x[1],
                default=("none", 0)
            )[0],
        }


# Module-level instance
feedback_processor = FeedbackProcessor()
