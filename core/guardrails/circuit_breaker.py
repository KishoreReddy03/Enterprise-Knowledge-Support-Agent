"""
Circuit breaker for LLM API calls.

Prevents cascading failures when Groq or other external services go down.
Uses Redis to track failures across multiple API workers.

States:
    CLOSED  → Normal operation. Requests go through.
    OPEN    → Service is down. Return fallback immediately.
    HALF_OPEN → Testing recovery. Allow ONE request through.

⚠️ RESILIENCY & CIRCUIT BREAKER HARDENING AUDIT
=============================================
The current circuit breaker implementation relies on a static, heuristic-driven
threshold model (e.g. static failure thresholds and rigid cooldown limits). While
adequate for basic containment, this design poses operational risks under production load:

1. THE LIMITATIONS OF STATIC HEURISTICS:
   * No Adaptive Scaling: At high query volumes, a minor network hiccup can trigger 
     3 consecutive failures almost instantly, tripping the circuit breaker and 
     routing standard tickets directly to fallback escalation unnecessarily.
   * Provider-Blind Architecture: The circuit is locked onto a single provider ("groq")
     and lacks multi-provider health routing. If Groq degrades, the system escalates 
     directly to human agents instead of failing over to backup providers (e.g. OpenAI/Anthropic).
   * Latency-Blind Trips: The circuit only trips on raw API failure responses (5xx, timeouts).
     It does not detect latency degradation (e.g., requests succeeding but taking 30s),
     which can exhaust downstream client thread pools.

2. RESILIENCE HARDENING ROADMAP:
   To elevate system resilience under heavy loads, we plan to transition the circuit 
   breaker to an adaptive, intelligent reliability grid:
   * Layer A - Adaptive Thresholds: Dynamically adjust the failure trigger using a 
     sliding window error rate percentage (e.g. trip if > 15% of requests fail over a 
     5-minute window) rather than raw consecutive counts.
   * Layer B - Provider-Aware Reliability Scoring: Calculate a real-time Health Index
     Score per provider based on successful request latency and success rates. Use this 
     index to dynamically fail over to alternative API endpoints before tripping the circuit.
   * Layer C - Latency-Aware Circuit Logic: Trip the circuit (or degrade search depth/LLM complexity)
     if the sliding p95 latency exceeds 5.0 seconds, preserving system response speeds.
"""

import logging
import time
from enum import Enum

from core.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal — requests go through
    OPEN = "open"          # Service down — return fallback
    HALF_OPEN = "half_open"  # Testing — allow one request


# Thresholds
FAILURE_THRESHOLD = 3        # Open circuit after N consecutive failures
COOLDOWN_SECONDS = 30        # Wait N seconds before trying again (half-open)
FAILURE_WINDOW_SECONDS = 60  # Reset failure count if no failures for N seconds


class CircuitBreaker:
    """
    Circuit breaker for external service calls.

    Tracks consecutive failures via Redis. When failures exceed the
    threshold, the circuit opens and all requests are immediately
    rejected with a fallback response.

    After a cooldown period, the circuit enters half-open state
    and allows one test request. If it succeeds, the circuit closes.
    If it fails, the circuit opens again.
    """

    def __init__(self, service_name: str = "groq") -> None:
        """
        Initialize the circuit breaker.

        Args:
            service_name: Name of the service to protect.
        """
        self._service = service_name
        self._redis = get_redis_client()
        self._local_state = CircuitState.CLOSED
        self._last_failure_time: float = 0.0
        logger.info(f"CircuitBreaker initialized for service: {service_name}")

    async def get_state(self) -> CircuitState:
        """
        Get the current circuit state.

        Checks Redis for failure count and determines state.

        Returns:
            Current CircuitState.
        """
        failure_count = await self._redis.get_failure_count(self._service)

        if failure_count >= FAILURE_THRESHOLD:
            # Check if cooldown has passed
            elapsed = time.time() - self._last_failure_time
            if elapsed >= COOLDOWN_SECONDS:
                logger.info(
                    f"Circuit {self._service}: HALF_OPEN "
                    f"(cooldown {COOLDOWN_SECONDS}s elapsed)"
                )
                return CircuitState.HALF_OPEN
            else:
                return CircuitState.OPEN
        else:
            return CircuitState.CLOSED

    async def record_success(self) -> None:
        """
        Record a successful service call.

        Resets the failure counter and closes the circuit.
        """
        await self._redis.reset_failures(self._service)
        self._local_state = CircuitState.CLOSED
        logger.info(f"Circuit {self._service}: SUCCESS → CLOSED")

    async def record_failure(self) -> CircuitState:
        """
        Record a failed service call.

        Increments the failure counter. If threshold is exceeded,
        the circuit opens.

        Returns:
            New CircuitState after recording the failure.
        """
        self._last_failure_time = time.time()
        count = await self._redis.record_failure(self._service)

        if count >= FAILURE_THRESHOLD:
            self._local_state = CircuitState.OPEN
            logger.error(
                f"Circuit {self._service}: OPEN "
                f"({count}/{FAILURE_THRESHOLD} failures). "
                f"Returning fallback for next {COOLDOWN_SECONDS}s."
            )
            return CircuitState.OPEN
        else:
            logger.warning(
                f"Circuit {self._service}: failure {count}/{FAILURE_THRESHOLD}"
            )
            return CircuitState.CLOSED

    async def is_available(self) -> bool:
        """
        Check if the service is available (circuit not open).

        Returns:
            True if requests should be allowed through.
        """
        state = await self.get_state()
        return state != CircuitState.OPEN

    def get_fallback_response(self) -> dict:
        """
        Generate a fallback response when the circuit is open.

        Returns:
            A response dict that can be used as a pipeline result.
        """
        return {
            "final_response": {
                "reply_text": (
                    "Thanks for reaching out! We're experiencing high demand "
                    "right now and our AI assistant is temporarily unavailable. "
                    "A human support representative will follow up with you "
                    "within 2 hours.\n\n"
                    "In the meantime, you can check our documentation at "
                    "https://docs.stripe.com or our status page at "
                    "https://status.stripe.com for any known issues.\n\n"
                    "We apologize for the delay!"
                ),
                "confidence": 0.0,
                "sources": [],
                "needs_review": True,
                "review_reason": "AI service unavailable — circuit breaker active",
            },
            "confidence_score": 0.0,
            "escalated": True,
            "escalation_brief": (
                f"Circuit breaker OPEN for {self._service}. "
                f"Service has failed {FAILURE_THRESHOLD}+ times consecutively. "
                f"Auto-generated fallback response sent. "
                f"Manual response required."
            ),
            "agent_path": ["circuit_breaker_fallback"],
            "guardrail_warnings": [
                f"circuit_breaker_active: {self._service} service unavailable"
            ],
        }


# Module-level singleton
_breaker: CircuitBreaker | None = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get or create the CircuitBreaker singleton."""
    global _breaker
    if _breaker is None:
        _breaker = CircuitBreaker("groq")
    return _breaker
