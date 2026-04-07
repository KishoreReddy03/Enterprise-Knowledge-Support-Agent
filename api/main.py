"""
FastAPI application entry point.

Main application with middleware, CORS, and route registration.

Usage:
    uvicorn api.main:app --reload
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langfuse import Langfuse
from pydantic import ValidationError

from api.routes import (
    analytics_router,
    feedback_router,
    health_router,
    tickets_router,
)
from config import settings

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LIFESPAN MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting Stripe Support Agent API | env={settings.ENVIRONMENT}")
    
    # Initialize Langfuse
    try:
        langfuse = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        app.state.langfuse = langfuse
        logger.info("Langfuse initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse: {e}")
        app.state.langfuse = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down Stripe Support Agent API")
    
    # Flush Langfuse
    if hasattr(app.state, "langfuse") and app.state.langfuse:
        try:
            app.state.langfuse.flush()
        except Exception as e:
            logger.warning(f"Error flushing Langfuse: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Stripe Support Agent API",
    description="AI-powered support ticket processing with RAG and quality gates",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════════

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_middleware(request: Request, call_next) -> Response:
    """
    Request middleware for logging, timing, and request ID injection.
    """
    # Generate request ID
    request_id = str(uuid4())
    request.state.request_id = request_id
    
    # Start timing
    start_time = time.time()
    
    # Process request
    try:
        response = await call_next(request)
    except Exception as e:
        # Log unhandled exceptions
        logger.error(
            f"Unhandled exception | "
            f"request_id={request_id} | "
            f"path={request.url.path} | "
            f"error={str(e)}",
            exc_info=True,
        )
        raise
    
    # Calculate latency
    latency_ms = int((time.time() - start_time) * 1000)
    
    # Add response headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(latency_ms)
    
    # Get trace ID if available from Langfuse
    trace_id = getattr(request.state, "langfuse_trace_id", None)
    if trace_id:
        response.headers["X-Trace-ID"] = trace_id
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} | "
        f"status={response.status_code} | "
        f"latency_ms={latency_ms} | "
        f"request_id={request_id}"
    )
    
    return response


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTION HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(ValidationError)
async def validation_exception_handler(
    request: Request,
    exc: ValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    request_id = getattr(request.state, "request_id", str(uuid4()))
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation error",
            "code": "VALIDATION_ERROR",
            "details": exc.errors(),
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle all unhandled exceptions."""
    request_id = getattr(request.state, "request_id", str(uuid4()))
    
    logger.error(
        f"Unhandled exception | "
        f"request_id={request_id} | "
        f"error={str(exc)}",
        exc_info=True,
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "code": "INTERNAL_ERROR",
            "request_id": request_id,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# Register routers with /api/v1 prefix
app.include_router(tickets_router, prefix="/api/v1")
app.include_router(feedback_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")


# Root endpoint
@app.get("/", tags=["root"])
async def root() -> dict:
    """Root endpoint with API info."""
    return {
        "name": "Stripe Support Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level="info",
    )
