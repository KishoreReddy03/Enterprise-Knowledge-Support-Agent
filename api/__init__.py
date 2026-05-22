"""
API module for Stripe Support Agent.

FastAPI application exposing ticket processing, feedback,
analytics, and health check endpoints.
"""

from api.main import app

__all__ = ["app"]
