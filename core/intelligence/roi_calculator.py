"""
ROI calculator for business metrics.

Translates technical performance metrics into business value.
THIS IS WHAT MAKES EXECUTIVES APPROVE DEPLOYMENTS.
THIS IS WHAT MAKES YOUR README UNFORGETTABLE.

Key metrics:
- Time saved vs manual resolution
- Cost savings in USD
- Deflection and escalation rates
- Quality metrics (faithfulness, confidence)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean
from typing import Any

from langfuse.decorators import observe

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ROIReport:
    """
    Weekly ROI report translating technical metrics to business value.
    
    Attributes:
        report_date: When the report was generated.
        period_days: Days covered in this report.
        
        # Volume metrics
        tickets_handled: Total tickets processed by AI.
        tickets_escalated: Tickets requiring human intervention.
        tickets_deflected: High-confidence tickets resolved by AI.
        
        # Time metrics
        avg_resolution_seconds: Average AI response time.
        manual_baseline_seconds: Industry baseline for manual resolution.
        time_reduction_percent: Percentage faster than manual.
        hours_saved_this_week: Total hours saved.
        
        # Cost metrics
        cost_saved_usd_this_week: Dollar savings from time reduction.
        cost_per_query_usd: Average LLM cost per ticket.
        net_savings_usd: Savings minus LLM costs.
        
        # Quality metrics
        escalation_rate: Percentage of tickets escalated.
        deflection_rate: Percentage handled without human touch.
        avg_confidence_score: Average AI confidence.
        faithfulness_rate: Rate of factually correct responses.
        
        # Projections
        projected_monthly_savings_usd: Extrapolated monthly savings.
        projected_annual_savings_usd: Extrapolated annual savings.
        
        insufficient_data: True if not enough data for report.
    """
    # Metadata
    report_date: str = ""
    period_days: int = 7
    
    # Volume
    tickets_handled: int = 0
    tickets_escalated: int = 0
    tickets_deflected: int = 0
    
    # Time
    avg_resolution_seconds: float = 0.0
    manual_baseline_seconds: float = 510.0  # 8.5 minutes
    time_reduction_percent: float = 0.0
    hours_saved_this_week: float = 0.0
    
    # Cost
    cost_saved_usd_this_week: float = 0.0
    cost_per_query_usd: float = 0.0
    total_llm_cost_usd: float = 0.0
    net_savings_usd: float = 0.0
    
    # Quality
    escalation_rate: float = 0.0
    deflection_rate: float = 0.0
    avg_confidence_score: float = 0.0
    faithfulness_rate: float = 1.0
    
    # Projections
    projected_monthly_savings_usd: float = 0.0
    projected_annual_savings_usd: float = 0.0
    
    # Status
    insufficient_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/API response."""
        return {
            "report_date": self.report_date,
            "period_days": self.period_days,
            "volume": {
                "tickets_handled": self.tickets_handled,
                "tickets_escalated": self.tickets_escalated,
                "tickets_deflected": self.tickets_deflected,
            },
            "time": {
                "avg_resolution_seconds": round(self.avg_resolution_seconds, 2),
                "manual_baseline_seconds": self.manual_baseline_seconds,
                "time_reduction_percent": round(self.time_reduction_percent, 1),
                "hours_saved_this_week": round(self.hours_saved_this_week, 2),
            },
            "cost": {
                "cost_saved_usd_this_week": round(self.cost_saved_usd_this_week, 2),
                "cost_per_query_usd": round(self.cost_per_query_usd, 4),
                "total_llm_cost_usd": round(self.total_llm_cost_usd, 2),
                "net_savings_usd": round(self.net_savings_usd, 2),
            },
            "quality": {
                "escalation_rate": round(self.escalation_rate, 3),
                "deflection_rate": round(self.deflection_rate, 3),
                "avg_confidence_score": round(self.avg_confidence_score, 3),
                "faithfulness_rate": round(self.faithfulness_rate, 3),
            },
            "projections": {
                "monthly_savings_usd": round(self.projected_monthly_savings_usd, 2),
                "annual_savings_usd": round(self.projected_annual_savings_usd, 2),
            },
            "insufficient_data": self.insufficient_data,
        }

    def to_executive_summary(self) -> str:
        """Generate executive-friendly summary."""
        if self.insufficient_data:
            return "Insufficient data for ROI analysis."
        
        return f"""
═══ WEEKLY ROI SUMMARY ═══

📊 VOLUME
   • {self.tickets_handled:,} tickets processed
   • {self.tickets_deflected:,} fully automated ({self.deflection_rate:.0%} deflection rate)
   • {self.tickets_escalated:,} escalated to humans ({self.escalation_rate:.0%})

⏱️ TIME SAVINGS
   • AI resolution: {self.avg_resolution_seconds:.1f}s avg
   • Manual baseline: {self.manual_baseline_seconds/60:.1f} min
   • {self.time_reduction_percent:.0f}% faster than manual
   • {self.hours_saved_this_week:.1f} hours saved this week

💰 COST IMPACT
   • ${self.cost_saved_usd_this_week:,.2f} saved this week
   • ${self.cost_per_query_usd:.4f} per query (LLM cost)
   • ${self.net_savings_usd:,.2f} net savings
   • ${self.projected_monthly_savings_usd:,.2f} projected monthly
   • ${self.projected_annual_savings_usd:,.2f} projected annually

✅ QUALITY
   • {self.avg_confidence_score:.0%} average confidence
   • {self.faithfulness_rate:.0%} factual accuracy

═══════════════════════════
"""


@dataclass
class ResponseRecord:
    """Database response record for ROI calculation."""
    id: str
    ticket_id: str
    latency_ms: int
    confidence_score: float
    was_escalated: bool
    tokens_used: int
    created_at: datetime


@dataclass
class FeedbackRecord:
    """Database feedback record for quality metrics."""
    id: str
    response_id: str
    edit_type: str
    created_at: datetime


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CLIENT (placeholder)
# ═══════════════════════════════════════════════════════════════════════════════

class ROIDatabaseClient:
    """
    Database operations for ROI calculation.
    
    Placeholder that will be replaced with actual Supabase implementation.
    """

    async def get_responses_since(
        self,
        since: datetime,
    ) -> list[ResponseRecord]:
        """
        Fetch responses since a given date.
        
        Args:
            since: Start datetime.
            
        Returns:
            List of response records.
        """
        logger.info(f"Fetching responses since {since.isoformat()}")
        return []

    async def get_feedback_since(
        self,
        since: datetime,
    ) -> list[FeedbackRecord]:
        """
        Fetch feedback since a given date.
        
        Args:
            since: Start datetime.
            
        Returns:
            List of feedback records.
        """
        logger.info(f"Fetching feedback since {since.isoformat()}")
        return []

    async def store_roi_report(
        self,
        report: dict[str, Any],
    ) -> str:
        """
        Store ROI report to database.
        
        Args:
            report: Report dictionary.
            
        Returns:
            Created report ID.
        """
        logger.info("Storing ROI report")
        return f"roi_{datetime.utcnow().strftime('%Y%m%d')}"


# ═══════════════════════════════════════════════════════════════════════════════
# ROI CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ROICalculator:
    """
    Calculates ROI metrics for the AI support system.
    
    Translates technical performance into business value metrics
    that executives can understand and approve.
    """

    # Industry baselines
    MANUAL_RESOLUTION_MINUTES: float = 8.5  # Industry average
    SUPPORT_REP_HOURLY_COST_USD: float = 25.0  # Fully loaded cost
    
    # LLM pricing (Anthropic as of 2024)
    SONNET_INPUT_COST_PER_TOKEN: float = 0.000003  # $3 per 1M input
    SONNET_OUTPUT_COST_PER_TOKEN: float = 0.000015  # $15 per 1M output
    HAIKU_INPUT_COST_PER_TOKEN: float = 0.00000025  # $0.25 per 1M input
    HAIKU_OUTPUT_COST_PER_TOKEN: float = 0.00000125  # $1.25 per 1M output
    
    # Blended rate (assuming 70% Haiku, 30% Sonnet)
    BLENDED_COST_PER_TOKEN: float = 0.000002
    
    # Thresholds
    HIGH_CONFIDENCE_THRESHOLD: float = 0.90
    MINIMUM_RESPONSES_FOR_REPORT: int = 5

    def __init__(self, db: ROIDatabaseClient | None = None) -> None:
        """
        Initialize ROI calculator.
        
        Args:
            db: Database client. Uses default if not provided.
        """
        self._db = db or ROIDatabaseClient()
        logger.info("ROICalculator initialized")

    def _calculate_time_savings(
        self,
        avg_resolution_seconds: float,
        ticket_count: int,
    ) -> tuple[float, float]:
        """
        Calculate time and cost savings.
        
        Args:
            avg_resolution_seconds: Average AI resolution time.
            ticket_count: Number of tickets handled.
            
        Returns:
            Tuple of (hours_saved, cost_saved_usd).
        """
        manual_seconds = self.MANUAL_RESOLUTION_MINUTES * 60
        time_saved_per_ticket_seconds = manual_seconds - avg_resolution_seconds
        
        # Convert to hours
        time_saved_per_ticket_hours = time_saved_per_ticket_seconds / 3600
        total_hours_saved = time_saved_per_ticket_hours * ticket_count
        
        # Calculate cost savings
        cost_saved = total_hours_saved * self.SUPPORT_REP_HOURLY_COST_USD
        
        return max(total_hours_saved, 0), max(cost_saved, 0)

    def _calculate_llm_cost(
        self,
        responses: list[ResponseRecord],
    ) -> tuple[float, float]:
        """
        Calculate LLM costs.
        
        Args:
            responses: List of response records.
            
        Returns:
            Tuple of (total_cost, cost_per_query).
        """
        if not responses:
            return 0.0, 0.0
        
        total_tokens = sum(r.tokens_used for r in responses)
        total_cost = total_tokens * self.BLENDED_COST_PER_TOKEN
        cost_per_query = total_cost / len(responses)
        
        return total_cost, cost_per_query

    def _calculate_quality_metrics(
        self,
        responses: list[ResponseRecord],
        feedback: list[FeedbackRecord],
    ) -> dict[str, float]:
        """
        Calculate quality-related metrics.
        
        Args:
            responses: List of response records.
            feedback: List of feedback records.
            
        Returns:
            Dict with quality metrics.
        """
        if not responses:
            return {
                "escalation_rate": 0.0,
                "deflection_rate": 0.0,
                "avg_confidence": 0.0,
                "faithfulness_rate": 1.0,
            }
        
        total = len(responses)
        escalated = sum(1 for r in responses if r.was_escalated)
        deflected = sum(
            1 for r in responses
            if r.confidence_score >= self.HIGH_CONFIDENCE_THRESHOLD
            and not r.was_escalated
        )
        
        avg_confidence = mean(r.confidence_score for r in responses)
        
        # Faithfulness = 1 - (factual corrections / total feedback)
        factual_corrections = sum(
            1 for f in feedback
            if f.edit_type == "factual_correction"
        )
        faithfulness = 1 - (factual_corrections / max(len(feedback), 1))
        
        return {
            "escalation_rate": escalated / total,
            "deflection_rate": deflected / total,
            "avg_confidence": avg_confidence,
            "faithfulness_rate": faithfulness,
            "escalated_count": escalated,
            "deflected_count": deflected,
        }

    @observe(name="roi_weekly_metrics")
    async def calculate_weekly_metrics(
        self,
        days: int = 7,
    ) -> ROIReport:
        """
        Calculate weekly ROI metrics.
        
        Fetches data from the last N days and computes comprehensive
        ROI metrics including time savings, cost savings, and quality.
        
        Args:
            days: Number of days to analyze.
            
        Returns:
            ROIReport with all metrics.
        """
        logger.info(f"Calculating ROI metrics for last {days} days")
        
        since = datetime.utcnow() - timedelta(days=days)
        
        responses = await self._db.get_responses_since(since)
        feedback = await self._db.get_feedback_since(since)
        
        if len(responses) < self.MINIMUM_RESPONSES_FOR_REPORT:
            logger.warning(
                f"Insufficient data: {len(responses)} responses "
                f"(minimum: {self.MINIMUM_RESPONSES_FOR_REPORT})"
            )
            return ROIReport(
                report_date=datetime.utcnow().isoformat(),
                period_days=days,
                tickets_handled=len(responses),
                insufficient_data=True,
            )
        
        # Calculate latency metrics
        latencies_seconds = [r.latency_ms / 1000 for r in responses]
        avg_resolution_seconds = mean(latencies_seconds)
        manual_baseline_seconds = self.MANUAL_RESOLUTION_MINUTES * 60
        
        time_reduction_percent = (
            (1 - avg_resolution_seconds / manual_baseline_seconds) * 100
        )
        
        # Calculate time and cost savings
        hours_saved, cost_saved = self._calculate_time_savings(
            avg_resolution_seconds,
            len(responses),
        )
        
        # Calculate LLM costs
        total_llm_cost, cost_per_query = self._calculate_llm_cost(responses)
        
        # Net savings
        net_savings = cost_saved - total_llm_cost
        
        # Calculate quality metrics
        quality = self._calculate_quality_metrics(responses, feedback)
        
        # Projections (extrapolate weekly to monthly/annual)
        weeks_per_month = 4.33
        weeks_per_year = 52
        projected_monthly = net_savings * weeks_per_month
        projected_annual = net_savings * weeks_per_year
        
        report = ROIReport(
            report_date=datetime.utcnow().isoformat(),
            period_days=days,
            
            # Volume
            tickets_handled=len(responses),
            tickets_escalated=quality["escalated_count"],
            tickets_deflected=quality["deflected_count"],
            
            # Time
            avg_resolution_seconds=avg_resolution_seconds,
            manual_baseline_seconds=manual_baseline_seconds,
            time_reduction_percent=time_reduction_percent,
            hours_saved_this_week=hours_saved,
            
            # Cost
            cost_saved_usd_this_week=cost_saved,
            cost_per_query_usd=cost_per_query,
            total_llm_cost_usd=total_llm_cost,
            net_savings_usd=net_savings,
            
            # Quality
            escalation_rate=quality["escalation_rate"],
            deflection_rate=quality["deflection_rate"],
            avg_confidence_score=quality["avg_confidence"],
            faithfulness_rate=quality["faithfulness_rate"],
            
            # Projections
            projected_monthly_savings_usd=projected_monthly,
            projected_annual_savings_usd=projected_annual,
            
            insufficient_data=False,
        )
        
        # Store report
        await self._db.store_roi_report(report.to_dict())
        
        logger.info(
            f"ROI report generated: "
            f"tickets={report.tickets_handled}, "
            f"time_reduction={report.time_reduction_percent:.0f}%, "
            f"net_savings=${report.net_savings_usd:.2f}"
        )
        
        return report

    async def calculate_cost_per_resolution(
        self,
        days: int = 30,
    ) -> dict[str, float]:
        """
        Calculate detailed cost breakdown per resolution.
        
        Args:
            days: Period to analyze.
            
        Returns:
            Dict with cost breakdown.
        """
        since = datetime.utcnow() - timedelta(days=days)
        responses = await self._db.get_responses_since(since)
        
        if not responses:
            return {
                "ai_cost_per_resolution": 0.0,
                "manual_cost_per_resolution": 0.0,
                "savings_per_resolution": 0.0,
                "savings_percent": 0.0,
            }
        
        # AI cost per resolution
        total_llm_cost, _ = self._calculate_llm_cost(responses)
        ai_cost_per_resolution = total_llm_cost / len(responses)
        
        # Manual cost per resolution
        manual_minutes = self.MANUAL_RESOLUTION_MINUTES
        manual_cost_per_resolution = (
            (manual_minutes / 60) * self.SUPPORT_REP_HOURLY_COST_USD
        )
        
        # Savings
        savings_per_resolution = manual_cost_per_resolution - ai_cost_per_resolution
        savings_percent = (savings_per_resolution / manual_cost_per_resolution) * 100
        
        return {
            "ai_cost_per_resolution": round(ai_cost_per_resolution, 4),
            "manual_cost_per_resolution": round(manual_cost_per_resolution, 2),
            "savings_per_resolution": round(savings_per_resolution, 2),
            "savings_percent": round(savings_percent, 1),
        }

    async def get_trend_analysis(
        self,
        weeks: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Get ROI trends over multiple weeks.
        
        Args:
            weeks: Number of weeks to analyze.
            
        Returns:
            List of weekly metrics for trend analysis.
        """
        trends: list[dict[str, Any]] = []
        
        for week in range(weeks):
            end_date = datetime.utcnow() - timedelta(weeks=week)
            start_date = end_date - timedelta(days=7)
            
            responses = await self._db.get_responses_since(start_date)
            # Filter to only this week
            week_responses = [
                r for r in responses
                if r.created_at <= end_date
            ]
            
            if week_responses:
                avg_latency = mean(r.latency_ms / 1000 for r in week_responses)
                avg_confidence = mean(r.confidence_score for r in week_responses)
                escalation_rate = (
                    sum(1 for r in week_responses if r.was_escalated)
                    / len(week_responses)
                )
            else:
                avg_latency = 0.0
                avg_confidence = 0.0
                escalation_rate = 0.0
            
            trends.append({
                "week_ending": end_date.strftime("%Y-%m-%d"),
                "tickets_handled": len(week_responses),
                "avg_resolution_seconds": round(avg_latency, 2),
                "avg_confidence": round(avg_confidence, 3),
                "escalation_rate": round(escalation_rate, 3),
            })
        
        return list(reversed(trends))  # Oldest first

    def estimate_roi_at_scale(
        self,
        monthly_ticket_volume: int,
        current_escalation_rate: float = 0.15,
    ) -> dict[str, Any]:
        """
        Project ROI at different scale levels.
        
        Useful for capacity planning and executive presentations.
        
        Args:
            monthly_ticket_volume: Expected monthly tickets.
            current_escalation_rate: Current escalation rate.
            
        Returns:
            Dict with scale projections.
        """
        automated_tickets = monthly_ticket_volume * (1 - current_escalation_rate)
        
        # Time savings
        time_saved_minutes = automated_tickets * self.MANUAL_RESOLUTION_MINUTES
        time_saved_hours = time_saved_minutes / 60
        
        # Cost savings
        labor_savings = time_saved_hours * self.SUPPORT_REP_HOURLY_COST_USD
        
        # Estimated LLM cost (assuming 2000 tokens average)
        avg_tokens_per_ticket = 2000
        llm_cost = (
            automated_tickets * avg_tokens_per_ticket * self.BLENDED_COST_PER_TOKEN
        )
        
        # Net savings
        net_monthly = labor_savings - llm_cost
        net_annual = net_monthly * 12
        
        # FTE equivalent
        work_hours_per_month = 160  # 40 hrs/week * 4 weeks
        fte_equivalent = time_saved_hours / work_hours_per_month
        
        return {
            "monthly_ticket_volume": monthly_ticket_volume,
            "automated_tickets": int(automated_tickets),
            "escalated_tickets": int(monthly_ticket_volume * current_escalation_rate),
            "time_saved_hours_monthly": round(time_saved_hours, 1),
            "labor_savings_monthly_usd": round(labor_savings, 2),
            "llm_cost_monthly_usd": round(llm_cost, 2),
            "net_savings_monthly_usd": round(net_monthly, 2),
            "net_savings_annual_usd": round(net_annual, 2),
            "fte_equivalent": round(fte_equivalent, 2),
            "roi_percent": round((net_monthly / llm_cost) * 100, 0) if llm_cost > 0 else 0,
        }


# Module-level instance
roi_calculator = ROICalculator()
