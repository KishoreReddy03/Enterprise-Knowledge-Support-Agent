"""
Analytics endpoints.

Provides ROI metrics and pattern analysis for dashboards.
"""

import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from langfuse import observe
from pydantic import BaseModel, Field

from core.intelligence.pattern_detector import PatternDetector
from core.intelligence.roi_calculator import ROICalculator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class RecurringPattern(BaseModel):
    """A recurring issue pattern."""
    
    pattern: str
    frequency: int
    avg_confidence: float
    recommendation: str


class DocumentationGap(BaseModel):
    """A detected documentation gap."""
    
    topic: str
    frequency: int
    avg_confidence: float
    suggested_doc_title: str


class PotentialRegression(BaseModel):
    """A potential regression detection."""
    
    topic: str
    recent_count: int
    prior_count: int
    spike_factor: float


class PatternReportResponse(BaseModel):
    """Response containing pattern analysis."""
    
    report_date: str = Field(description="When the report was generated")
    insufficient_data: bool = Field(
        default=False,
        description="True if not enough data for analysis",
    )
    recurring_patterns: list[RecurringPattern] = Field(default_factory=list)
    documentation_gaps: list[DocumentationGap] = Field(default_factory=list)
    potential_regressions: list[PotentialRegression] = Field(default_factory=list)
    overall_kb_health_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall knowledge base health (0-1)",
    )
    top_recommendation: str = Field(
        default="",
        description="Top actionable recommendation",
    )


class ROIReportResponse(BaseModel):
    """Response containing ROI metrics."""
    
    report_date: str = Field(description="When the report was generated")
    insufficient_data: bool = Field(
        default=False,
        description="True if not enough data for analysis",
    )
    
    # Volume metrics
    tickets_handled: int = Field(description="Total tickets processed")
    escalation_rate: float = Field(description="Rate of escalations (0-1)")
    deflection_rate: float = Field(description="Rate of high-confidence auto-responses (0-1)")
    
    # Time metrics
    avg_resolution_seconds: float = Field(description="Average response time")
    manual_baseline_seconds: float = Field(description="Industry baseline for manual")
    time_reduction_percent: float = Field(description="Time saved vs manual baseline")
    
    # Cost metrics
    hours_saved_this_week: float = Field(description="Hours saved this week")
    cost_saved_usd_this_week: float = Field(description="Cost saved in USD")
    cost_per_query_usd: float = Field(description="Average cost per query")
    
    # Quality metrics
    avg_confidence_score: float = Field(description="Average confidence (0-1)")
    faithfulness_rate: float = Field(description="Rate without factual corrections (0-1)")


class AnalyticsErrorResponse(BaseModel):
    """Error response for analytics."""
    
    error: str = Field(description="Error message")
    code: str = Field(description="Error code")


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/roi",
    response_model=ROIReportResponse,
    responses={500: {"model": AnalyticsErrorResponse}},
    summary="Get ROI metrics",
    description="Get weekly ROI report with time and cost savings",
)
@observe(name="api_get_roi")
async def get_roi_report(request: Request) -> ROIReportResponse:
    """
    Get the latest ROI metrics.
    
    Calculates time savings, cost savings, and quality metrics
    from the past week of data.
    """
    request_id = getattr(request.state, "request_id", str(uuid4()))
    
    logger.info(f"Fetching ROI report | request_id={request_id}")
    
    try:
        # langfuse_context.update_current_trace(
        #     name="roi_report_fetch",
        #     metadata={"request_id": request_id},
        # )
        
        calculator = ROICalculator()
        report = await calculator.calculate_weekly_metrics()
        
        if report.insufficient_data:
            logger.info("ROI report: insufficient data")
            return ROIReportResponse(
                report_date=datetime.utcnow().isoformat(),
                insufficient_data=True,
                tickets_handled=0,
                escalation_rate=0.0,
                deflection_rate=0.0,
                avg_resolution_seconds=0.0,
                manual_baseline_seconds=510.0,  # 8.5 minutes
                time_reduction_percent=0.0,
                hours_saved_this_week=0.0,
                cost_saved_usd_this_week=0.0,
                cost_per_query_usd=0.0,
                avg_confidence_score=0.0,
                faithfulness_rate=0.0,
            )
        
        logger.info(
            f"ROI report generated | "
            f"tickets={report.tickets_handled} | "
            f"cost_saved=${report.cost_saved_usd_this_week:.2f}"
        )
        
        return ROIReportResponse(
            report_date=datetime.utcnow().isoformat(),
            insufficient_data=False,
            tickets_handled=report.tickets_handled,
            escalation_rate=report.escalation_rate,
            deflection_rate=report.deflection_rate,
            avg_resolution_seconds=report.avg_resolution_seconds,
            manual_baseline_seconds=report.manual_baseline_seconds,
            time_reduction_percent=report.time_reduction_percent,
            hours_saved_this_week=report.hours_saved_this_week,
            cost_saved_usd_this_week=report.cost_saved_usd_this_week,
            cost_per_query_usd=report.cost_per_query_usd,
            avg_confidence_score=report.avg_confidence_score,
            faithfulness_rate=report.faithfulness_rate,
        )
        
    except Exception as e:
        logger.error(f"Error fetching ROI report: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "code": "ROI_CALCULATION_ERROR",
            },
        )


@router.get(
    "/patterns",
    response_model=PatternReportResponse,
    responses={500: {"model": AnalyticsErrorResponse}},
    summary="Get pattern analysis",
    description="Get latest pattern report with recurring issues and documentation gaps",
)
@observe(name="api_get_patterns")
async def get_pattern_report(
    request: Request,
    force_refresh: bool = Query(
        default=False,
        description="Force regeneration instead of using cached report",
    ),
) -> PatternReportResponse:
    """
    Get the latest pattern analysis report.
    
    Returns cached weekly report or generates fresh if force_refresh=True.
    """
    request_id = getattr(request.state, "request_id", str(uuid4()))
    
    logger.info(
        f"Fetching pattern report | "
        f"request_id={request_id} | "
        f"force_refresh={force_refresh}"
    )
    
    try:
        # langfuse_context.update_current_trace(
        #     name="pattern_report_fetch",
        #     metadata={
        #         "request_id": request_id,
        #         "force_refresh": force_refresh,
        #     },
        # )
        
        detector = PatternDetector()
        
        if force_refresh:
            # Generate new report
            report = await detector.run_weekly_analysis()
        else:
            # Try to get cached report first
            report = await detector.get_latest_report()
            if report is None:
                report = await detector.run_weekly_analysis()
        
        if report.insufficient_data:
            logger.info("Pattern report: insufficient data")
            return PatternReportResponse(
                report_date=datetime.utcnow().isoformat(),
                insufficient_data=True,
                overall_kb_health_score=0.0,
            )
        
        logger.info(
            f"Pattern report generated | "
            f"patterns={len(report.recurring_patterns)} | "
            f"gaps={len(report.documentation_gaps)}"
        )
        
        return PatternReportResponse(
            report_date=datetime.utcnow().isoformat(),
            insufficient_data=False,
            recurring_patterns=[
                RecurringPattern(
                    pattern=p["pattern"],
                    frequency=p["frequency"],
                    avg_confidence=p["avg_confidence"],
                    recommendation=p["recommendation"],
                )
                for p in report.recurring_patterns
            ],
            documentation_gaps=[
                DocumentationGap(
                    topic=g["topic"],
                    frequency=g["frequency"],
                    avg_confidence=g["avg_confidence"],
                    suggested_doc_title=g["suggested_doc_title"],
                )
                for g in report.documentation_gaps
            ],
            potential_regressions=[
                PotentialRegression(
                    topic=r["topic"],
                    recent_count=r["recent_count"],
                    prior_count=r["prior_count"],
                    spike_factor=r["spike_factor"],
                )
                for r in report.potential_regressions
            ],
            overall_kb_health_score=report.overall_kb_health_score,
            top_recommendation=report.top_recommendation,
        )
        
    except Exception as e:
        logger.error(f"Error fetching pattern report: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "code": "PATTERN_ANALYSIS_ERROR",
            },
        )


@router.get(
    "/summary",
    summary="Get quick analytics summary",
    description="Get a quick summary of key metrics for dashboard",
)
@observe(name="api_get_summary")
async def get_analytics_summary(request: Request) -> dict:
    """
    Get a quick summary of key analytics.
    
    Lightweight endpoint for dashboard cards.
    """
    request_id = getattr(request.state, "request_id", str(uuid4()))
    
    logger.info(f"Fetching analytics summary | request_id={request_id}")
    
    try:
        calculator = ROICalculator()
        roi_report = await calculator.calculate_weekly_metrics()
        
        return {
            "tickets_this_week": roi_report.tickets_handled,
            "hours_saved": round(roi_report.hours_saved_this_week, 1),
            "cost_saved_usd": round(roi_report.cost_saved_usd_this_week, 2),
            "avg_confidence": round(roi_report.avg_confidence_score, 2),
            "escalation_rate": round(roi_report.escalation_rate, 2),
            "time_reduction_percent": round(roi_report.time_reduction_percent, 1),
        }
        
    except Exception as e:
        logger.error(f"Error fetching summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "code": "SUMMARY_FETCH_ERROR",
            },
        )
