"""
Health check endpoints.

Provides system health status including external service connectivity.
"""

import logging
from datetime import datetime
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

ServiceStatus = Literal["connected", "error", "unknown"]
OverallStatus = Literal["healthy", "degraded", "unhealthy"]


class HealthResponse(BaseModel):
    """Response from health check."""
    
    status: OverallStatus = Field(description="Overall system status")
    qdrant: ServiceStatus = Field(description="Qdrant vector DB status")
    supabase: ServiceStatus = Field(description="Supabase DB status")
    redis: ServiceStatus = Field(description="Redis cache status")
    llm: ServiceStatus = Field(description="LLM API status")
    knowledge_base_chunks: int = Field(
        description="Total chunks in knowledge base",
    )
    last_ingestion: str | None = Field(
        default=None,
        description="Last successful ingestion timestamp",
    )
    environment: str = Field(description="Current environment")
    version: str = Field(default="1.0.0", description="API version")


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def check_qdrant() -> tuple[ServiceStatus, int]:
    """
    Check Qdrant connectivity and get collection stats.
    
    Returns:
        Tuple of (status, chunk_count).
    """
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=5.0,
        )
        
        # Check collections exist
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        total_chunks = 0
        for collection_name in [
            settings.QDRANT_DOCS_COLLECTION,
            settings.QDRANT_ISSUES_COLLECTION,
            settings.QDRANT_STACKOVERFLOW_COLLECTION,
        ]:
            if collection_name in collection_names:
                info = client.get_collection(collection_name)
                total_chunks += info.points_count
        
        return "connected", total_chunks
        
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        return "error", 0


async def check_supabase() -> ServiceStatus:
    """
    Check Supabase connectivity.
    
    Returns:
        Service status.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Simple REST API health check
            response = await client.get(
                f"{settings.SUPABASE_URL}/rest/v1/",
                headers={
                    "apikey": settings.SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
                },
            )
            
            if response.status_code in (200, 401, 404):
                # 401/404 are fine - means API is responding
                return "connected"
            return "error"
            
    except Exception as e:
        logger.warning(f"Supabase health check failed: {e}")
        return "error"


async def check_redis() -> ServiceStatus:
    """
    Check Redis/Upstash connectivity.
    
    Returns:
        Service status.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Use Upstash REST API for health check
            response = await client.get(
                f"{settings.UPSTASH_REDIS_URL}/ping",
                headers={
                    "Authorization": f"Bearer {settings.UPSTASH_REDIS_TOKEN}",
                },
            )
            
            if response.status_code == 200:
                return "connected"
            return "error"
            
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return "error"


async def check_llm() -> ServiceStatus:
    """
    Check LLM API connectivity via OpenRouter.
    
    Returns:
        Service status.
    """
    try:
        from core.llm_client import test_connection
        
        result = await test_connection()
        return "connected" if result else "error"
        
    except Exception as e:
        logger.warning(f"LLM health check failed: {e}")
        return "error"


async def get_last_ingestion() -> str | None:
    """
    Get timestamp of last successful ingestion.
    
    Returns:
        ISO timestamp or None.
    """
    try:
        # This would query Supabase for last ingestion record
        # For now, return placeholder
        return None
        
    except Exception as e:
        logger.warning(f"Failed to get last ingestion: {e}")
        return None


def determine_overall_status(
    qdrant: ServiceStatus,
    supabase: ServiceStatus,
    redis: ServiceStatus,
    llm: ServiceStatus,
) -> OverallStatus:
    """
    Determine overall system status from service statuses.
    
    Args:
        qdrant: Qdrant status.
        supabase: Supabase status.
        redis: Redis status.
        anthropic: Anthropic status.
        
    Returns:
        Overall status.
    """
    statuses = [qdrant, supabase, redis, anthropic]
    error_count = sum(1 for s in statuses if s == "error")
    
    # Critical services: Qdrant and LLM
    critical_down = qdrant == "error" or llm == "error"
    
    if error_count == 0:
        return "healthy"
    elif critical_down:
        return "unhealthy"
    else:
        return "degraded"


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get(
    "",
    response_model=HealthResponse,
    summary="System health check",
    description="Check connectivity to all external services",
)
async def health_check(request: Request) -> HealthResponse:
    """
    Comprehensive health check of all system dependencies.
    
    Returns status of each service and overall system health.
    """
    request_id = getattr(request.state, "request_id", str(uuid4()))
    
    logger.info(f"Health check | request_id={request_id}")
    
    # Run health checks
    qdrant_status, chunk_count = await check_qdrant()
    supabase_status = await check_supabase()
    redis_status = await check_redis()
    llm_status = await check_llm()
    
    last_ingestion = await get_last_ingestion()
    
    overall_status = determine_overall_status(
        qdrant_status,
        supabase_status,
        redis_status,
        llm_status,
    )
    
    logger.info(
        f"Health check complete | "
        f"status={overall_status} | "
        f"qdrant={qdrant_status} | "
        f"supabase={supabase_status} | "
        f"redis={redis_status} | "
        f"llm={llm_status}"
    )
    
    return HealthResponse(
        status=overall_status,
        qdrant=qdrant_status,
        supabase=supabase_status,
        redis=redis_status,
        llm=llm_status,
        knowledge_base_chunks=chunk_count,
        last_ingestion=last_ingestion,
        environment=settings.ENVIRONMENT,
    )


@router.get(
    "/ping",
    summary="Quick liveness check",
    description="Simple ping endpoint for load balancers",
)
async def ping() -> dict[str, str]:
    """
    Simple liveness check.
    
    Returns immediately without checking dependencies.
    """
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@router.get(
    "/ready",
    summary="Readiness check",
    description="Check if service is ready to accept traffic",
)
async def readiness() -> dict:
    """
    Readiness check for Kubernetes/container orchestration.
    
    Checks critical services only.
    """
    qdrant_status, _ = await check_qdrant()
    anthropic_status = await check_anthropic()
    
    ready = qdrant_status == "connected" and anthropic_status == "connected"
    
    return {
        "ready": ready,
        "qdrant": qdrant_status,
        "anthropic": anthropic_status,
    }
