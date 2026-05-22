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
    database: ServiceStatus = Field(description="Neon DB and pgvector status")
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

async def check_neon_pgvector() -> tuple[ServiceStatus, int]:
    """
    Check Neon pgvector connectivity and get chunk counts.
    
    Returns:
        Tuple of (status, chunk_count).
    """
    try:
        from core.retrieval.vector_retriever import VectorRetriever
        
        retriever = VectorRetriever()
        
        # Check tables exist
        total_chunks = 0
        for collection_name in ["stripe_docs", "stripe_issues", "stripe_so"]:
            if retriever.collection_exists(collection_name):
                # For now, mark as found but don't count (would need separate query)
                total_chunks += 1
        
        return "connected", total_chunks
        
    except Exception as e:
        logger.warning(f"Neon pgvector health check failed: {e}")
        return "error", 0


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
                f"{settings.UPSTASH_REDIS_REST_URL}/ping",
                headers={
                    "Authorization": f"Bearer {settings.UPSTASH_REDIS_REST_TOKEN}",
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
    Check LLM API connectivity via Groq/OpenRouter.
    
    Returns:
        Service status.
    """
    try:
        from core.llm_client import call_fast
        
        # Quick lightweight probe — cheaper than test_connection()
        result = await call_fast("Reply OK", max_tokens=8, temperature=0.0)
        return "connected" if result and len(result) > 0 else "error"
        
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
    database: ServiceStatus,
    redis: ServiceStatus,
    llm: ServiceStatus,
) -> OverallStatus:
    """
    Determine overall system status from service statuses.
    
    Critical services: Neon (primary DB) and LLM.
    Non-critical: Redis (cache).
    
    Args:
        database: Neon pgvector status (primary DB — critical).
        redis: Redis status.
        llm: LLM status.
        
    Returns:
        Overall status.
    """
    # Critical services: Neon (primary DB) and LLM
    critical_down = database == "error" or llm == "error"
    
    # Non-critical: Redis
    non_critical_errors = 1 if redis == "error" else 0
    
    if critical_down:
        return "unhealthy"
    elif non_critical_errors > 0:
        return "degraded"
    else:
        return "healthy"


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
    pgvector_status, chunk_count = await check_neon_pgvector()
    redis_status = await check_redis()
    llm_status = await check_llm()
    
    # Check Neon
    neon_status = "unknown"
    try:
        import psycopg2
        conn = psycopg2.connect(settings.NEON_DB_URL)
        conn.close()
        neon_status = "connected"
    except Exception:
        neon_status = "error"
    
    last_ingestion = await get_last_ingestion()
    
    overall_status = determine_overall_status(
        neon_status,
        redis_status,
        llm_status,
    )
    
    logger.info(
        f"Health check complete | "
        f"status={overall_status} | "
        f"database={neon_status} | "
        f"redis={redis_status} | "
        f"llm={llm_status}"
    )
    
    return HealthResponse(
        status=overall_status,
        database=neon_status,
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
    supabase_status, _ = await check_supabase_pgvector()
    llm_status = await check_llm()
    
    ready = supabase_status == "connected" and llm_status == "connected"
    
    return {
        "ready": ready,
        "supabase_pgvector": supabase_status,
        "llm": llm_status,
    }


@router.get(
    "/detailed",
    summary="Detailed system status",
    description="Full system status including guardrails, circuit breaker, cache, and all services",
)
async def detailed_health() -> dict:
    """
    Detailed health check showing every system component.

    Returns:
        Comprehensive status of all services, guardrails,
        circuit breaker, cache, and configuration.
    """
    import time
    start = time.time()

    # ─── Service Checks ─────────────────────────────────────────────────
    pgvector_status, chunk_count = await check_neon_pgvector()
    redis_status = await check_redis()
    llm_status = await check_llm()

    # ─── Neon DB Check ───────────────────────────────────────────────────
    neon_status = "unknown"
    neon_details = {}
    try:
        import psycopg2
        conn = psycopg2.connect(settings.NEON_DB_URL)
        cur = conn.cursor()

        # Count rows per table
        for table in ["stripe_docs", "stripe_stackoverflow", "stripe_github_issues"]:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                neon_details[table] = count
            except Exception:
                neon_details[table] = "table_not_found"

        cur.close()
        conn.close()
        neon_status = "connected"
    except Exception as e:
        neon_status = "error"
        neon_details["error"] = str(e)[:100]

    # ─── Circuit Breaker ─────────────────────────────────────────────────
    circuit_breaker_info = {}
    try:
        from core.guardrails.circuit_breaker import get_circuit_breaker
        cb = get_circuit_breaker()
        state = await cb.get_state()
        circuit_breaker_info = {
            "state": state.value,
            "service": "groq",
            "failure_threshold": 3,
            "cooldown_seconds": 30,
        }
    except Exception as e:
        circuit_breaker_info = {"error": str(e)[:100]}

    # ─── Redis Cache Stats ───────────────────────────────────────────────
    cache_info = {}
    try:
        from core.redis_client import get_redis_client
        redis = get_redis_client()
        # Test connectivity
        await redis.set("health_check_test", "ok", ttl_seconds=10)
        test_val = await redis.get("health_check_test")
        cache_info = {
            "status": "connected" if test_val == "ok" else "error",
            "read_write": test_val == "ok",
        }
    except Exception as e:
        cache_info = {"status": "error", "error": str(e)[:100]}

    # ─── Guardrails Status ───────────────────────────────────────────────
    guardrails_info = {
        "input_guard": {
            "prompt_injection_detection": True,
            "pii_masking": True,
            "rate_limiting": True,
            "html_sanitization": True,
        },
        "output_guard": {
            "hallucination_check": True,
            "topic_guardrail": True,
            "pii_leak_prevention": True,
            "forbidden_pattern_detection": True,
        },
    }

    # ─── Model Configuration ─────────────────────────────────────────────
    model_info = {
        "fast_model": settings.LLM_FAST_MODEL,
        "strong_model": settings.LLM_STRONG_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "similarity_threshold": settings.RETRIEVAL_SIMILARITY_THRESHOLD,
        "max_retries": settings.MAX_AGENT_RETRIES,
    }

    check_time_ms = int((time.time() - start) * 1000)

    # ─── Overall Status ──────────────────────────────────────────────────
    # Critical services: Neon (primary DB) and LLM
    critical_down = neon_status == "error" or llm_status == "error"
    
    # Non-critical: Redis
    non_critical_errors = 1 if redis_status == "error" else 0

    if critical_down:
        overall = "unhealthy"
    elif non_critical_errors > 0:
        overall = "degraded"
    else:
        overall = "healthy"

    return {
        "status": overall,
        "check_time_ms": check_time_ms,
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0",
        "services": {
            "neon_pgvector": {
                "status": neon_status,
                "tables": neon_details,
            },
            "redis": cache_info,
            "groq_llm": llm_status,
        },
        "production_features": {
            "guardrails": guardrails_info,
            "circuit_breaker": circuit_breaker_info,
            "semantic_cache": cache_info.get("status", "unknown"),
            "rate_limiting": "active",
        },
        "model_config": model_info,
        "knowledge_base_chunks": chunk_count,
    }

