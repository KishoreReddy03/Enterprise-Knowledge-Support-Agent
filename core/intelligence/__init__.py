"""Intelligence modules for pattern detection and analysis."""

from core.intelligence.pattern_detector import (
    PatternDetector,
    PatternReport,
    RecurringPattern,
    DocumentationGap,
    PotentialRegression,
    pattern_detector,
)
from core.intelligence.feedback_loop import (
    FeedbackProcessor,
    FeedbackRecord,
    EditAnalysis,
    EditType,
    EditSeverity,
    ImprovementInsights,
    feedback_processor,
)
from core.intelligence.roi_calculator import (
    ROICalculator,
    ROIReport,
    roi_calculator,
)

__all__ = [
    # Pattern Detection
    "PatternDetector",
    "PatternReport",
    "RecurringPattern",
    "DocumentationGap",
    "PotentialRegression",
    "pattern_detector",
    # Feedback Loop
    "FeedbackProcessor",
    "FeedbackRecord",
    "EditAnalysis",
    "EditType",
    "EditSeverity",
    "ImprovementInsights",
    "feedback_processor",
    # ROI Calculator
    "ROICalculator",
    "ROIReport",
    "roi_calculator",
]
