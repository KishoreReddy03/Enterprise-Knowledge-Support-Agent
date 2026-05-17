"""
Redis client for caching, rate limiting, and circuit breaker state.

Uses Upstash Redis REST API — no persistent TCP connection needed.
This is the ONLY place in the codebase that talks to Redis.
"""

import json
import logging
import time
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Lightweight Redis client using Upstash REST API.

    Uses HTTP requests instead of TCP connections, making it
    ideal for serverless and stateless deployments.
    """

    def __init__(self) -> None:
        """Initialize the Redis client with Upstash credentials."""
        self._base_url = settings.UPSTASH_REDIS_REST_URL
        self._token = settings.UPSTASH_REDIS_REST_TOKEN
        self._headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(timeout=5.0)

    async def _execute(self, *args: str) -> Any:
        """
        Execute a Redis command via Upstash REST API.

        Args:
            *args: Redis command and arguments (e.g., "SET", "key", "value").

        Returns:
            The Redis response result, or None on error.
        """
        try:
            response = await self._client.post(
                self._base_url,
                headers=self._headers,
                json=list(args),
            )
            response.raise_for_status()
            data = response.json()
            return data.get("result")
        except Exception as e:
            logger.warning(f"Redis command failed: {args[0]} — {e}")
            return None

    # ─── Key-Value Operations ────────────────────────────────────────────────

    async def get(self, key: str) -> str | None:
        """Get a value by key."""
        return await self._execute("GET", key)

    async def set(
        self, key: str, value: str, ttl_seconds: int | None = None
    ) -> bool:
        """
        Set a key-value pair with optional TTL.

        Args:
            key: Redis key.
            value: Value to store.
            ttl_seconds: Time-to-live in seconds (None = no expiry).

        Returns:
            True if set successfully.
        """
        if ttl_seconds:
            result = await self._execute("SET", key, value, "EX", str(ttl_seconds))
        else:
            result = await self._execute("SET", key, value)
        return result == "OK"

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        result = await self._execute("DEL", key)
        return result is not None and result > 0

    # ─── Counter Operations (for rate limiting) ──────────────────────────────

    async def incr(self, key: str) -> int | None:
        """Increment a counter and return the new value."""
        return await self._execute("INCR", key)

    async def expire(self, key: str, seconds: int) -> bool:
        """Set TTL on an existing key."""
        result = await self._execute("EXPIRE", key, str(seconds))
        return result == 1

    async def ttl(self, key: str) -> int | None:
        """Get remaining TTL on a key (-1 = no expiry, -2 = key doesn't exist)."""
        return await self._execute("TTL", key)

    # ─── JSON Operations (for caching complex objects) ───────────────────────

    async def set_json(
        self, key: str, data: dict | list, ttl_seconds: int | None = None
    ) -> bool:
        """Store a JSON-serializable object."""
        return await self.set(key, json.dumps(data), ttl_seconds)

    async def get_json(self, key: str) -> dict | list | None:
        """Retrieve a stored JSON object."""
        value = await self.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    # ─── Rate Limiting ───────────────────────────────────────────────────────

    async def check_rate_limit(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """
        Check if an identifier has exceeded the rate limit.

        Uses a sliding window counter pattern.

        Args:
            identifier: The entity to rate limit (e.g., customer_id, IP).
            max_requests: Maximum requests allowed in the window.
            window_seconds: Time window in seconds.

        Returns:
            Tuple of (is_allowed, current_count).
        """
        key = f"ratelimit:{identifier}"

        current = await self.incr(key)
        if current is None:
            # Redis unavailable — fail open (allow the request)
            return True, 0

        # Set expiry on first request in window
        if current == 1:
            await self.expire(key, window_seconds)

        is_allowed = current <= max_requests
        if not is_allowed:
            logger.warning(
                f"Rate limit exceeded for {identifier}: "
                f"{current}/{max_requests} in {window_seconds}s"
            )

        return is_allowed, current

    # ─── Semantic Cache ──────────────────────────────────────────────────────

    async def cache_response(
        self,
        query_hash: str,
        response_data: dict,
        ttl_seconds: int = 3600,
    ) -> bool:
        """
        Cache a pipeline response for a query.

        Args:
            query_hash: Hash of the query for lookup.
            response_data: The full response to cache.
            ttl_seconds: Cache TTL (default 1 hour).

        Returns:
            True if cached successfully.
        """
        key = f"cache:response:{query_hash}"
        return await self.set_json(key, response_data, ttl_seconds)

    async def get_cached_response(self, query_hash: str) -> dict | None:
        """
        Retrieve a cached response for a query.

        Args:
            query_hash: Hash of the query.

        Returns:
            Cached response dict, or None if not found.
        """
        key = f"cache:response:{query_hash}"
        return await self.get_json(key)

    # ─── Circuit Breaker State ───────────────────────────────────────────────

    async def record_failure(self, service: str) -> int:
        """
        Record a service failure for circuit breaker tracking.

        Args:
            service: Service name (e.g., "groq", "neon").

        Returns:
            Current failure count.
        """
        key = f"circuit:{service}:failures"
        count = await self.incr(key)
        # Auto-reset after 60 seconds of no failures
        await self.expire(key, 60)
        return count or 0

    async def reset_failures(self, service: str) -> None:
        """Reset failure count for a service (circuit closed)."""
        await self.delete(f"circuit:{service}:failures")

    async def get_failure_count(self, service: str) -> int:
        """Get current failure count for a service."""
        result = await self.get(f"circuit:{service}:failures")
        return int(result) if result else 0


# Module-level singleton
_client: RedisClient | None = None


def get_redis_client() -> RedisClient:
    """Get or create the RedisClient singleton."""
    global _client
    if _client is None:
        _client = RedisClient()
    return _client
