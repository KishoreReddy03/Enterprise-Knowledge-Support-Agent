"""
API routes module.

Contains all route handlers organized by domain:
- tickets: Ticket processing
- feedback: Rep feedback capture
- analytics: ROI and pattern reports
- health: System health checks
"""

from api.routes.analytics import router as analytics_router
from api.routes.feedback import router as feedback_router
from api.routes.health import router as health_router
from api.routes.tickets import router as tickets_router

__all__ = [
    "tickets_router",
    "feedback_router",
    "analytics_router",
    "health_router",
]
