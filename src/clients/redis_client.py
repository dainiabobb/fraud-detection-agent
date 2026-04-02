"""
RedisClient — thin wrapper around a redis.Redis instance.

Constructor injection keeps the class testable: pass a fakeredis.FakeRedis (or
any mock with the same interface) in tests instead of a live Redis connection.

Caching strategy:
  - Persona cache key:  "persona:<user_id>"
  - Pattern cache key:  "patterns:all"
  - Rate-limit key:     "ratelimit:<user_id>:<window_bucket>"
    Uses a sliding-window counter: a single INCR + EXPIRE per request within
    the current time window.  The window boundary is the floor-divided epoch
    second, so the counter resets cleanly every *window_seconds* seconds.
"""

import json
import logging
import time

from typing import Any

logger = logging.getLogger(__name__)

_PERSONA_KEY_PREFIX = "persona:"
_PATTERN_CACHE_KEY = "patterns:all"
_RATE_LIMIT_KEY_PREFIX = "ratelimit:"


class RedisClient:
    """Wraps a redis.Redis instance for fraud-detection caching and rate limiting."""

    def __init__(self, redis_client: Any) -> None:
        """
        Args:
            redis_client: An initialised redis.Redis (or compatible) instance.
                          Injected so tests can supply fakeredis or a mock.
        """
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Persona cache
    # ------------------------------------------------------------------

    def get_persona_cache(self, user_id: str) -> dict | None:
        """Return the cached persona for *user_id*, or None on cache miss.

        Handles JSON deserialisation and gracefully returns None if the cached
        value cannot be decoded (e.g. stale/corrupt entry).
        """
        key = f"{_PERSONA_KEY_PREFIX}{user_id}"
        try:
            raw = self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Corrupt persona cache entry — ignoring",
                extra={"user_id": user_id, "key": key},
            )
            return None
        except Exception:
            logger.exception(
                "Redis get_persona_cache failed",
                extra={"user_id": user_id},
            )
            raise

    def set_persona_cache(
        self,
        user_id: str,
        persona: dict,
        ttl: int = 3600,
    ) -> None:
        """Serialise *persona* as JSON and store it with a *ttl*-second expiry.

        Args:
            user_id: Cache key owner.
            persona: Persona dict to cache.
            ttl:     Time-to-live in seconds (default 1 hour).
        """
        key = f"{_PERSONA_KEY_PREFIX}{user_id}"
        try:
            self._redis.set(key, json.dumps(persona), ex=ttl)
        except Exception:
            logger.exception(
                "Redis set_persona_cache failed",
                extra={"user_id": user_id},
            )
            raise

    # ------------------------------------------------------------------
    # Rate limiting — sliding window counter
    # ------------------------------------------------------------------

    def check_rate_limit(
        self,
        user_id: str,
        window_seconds: int = 60,
        max_requests: int = 10,
    ) -> bool:
        """Increment the request counter for the current time window and check
        whether the caller is within the rate limit.

        Implementation uses a simple fixed-window counter keyed to the current
        time bucket (floor(epoch / window_seconds)).  The first INCR in a
        bucket sets the key's TTL to *window_seconds* so it expires atomically.

        Returns:
            True  — the request is within the limit and should proceed.
            False — the limit has been exceeded; the caller should reject the request.
        """
        # Compute the current window bucket so each bucket gets its own key.
        bucket = int(time.time()) // window_seconds
        key = f"{_RATE_LIMIT_KEY_PREFIX}{user_id}:{bucket}"
        try:
            count = self._redis.incr(key)
            if count == 1:
                # First request in this window — set TTL so the key self-cleans.
                self._redis.expire(key, window_seconds)
            within_limit: bool = count <= max_requests
            if not within_limit:
                logger.warning(
                    "Rate limit exceeded",
                    extra={
                        "user_id": user_id,
                        "count": count,
                        "max_requests": max_requests,
                        "window_seconds": window_seconds,
                    },
                )
            return within_limit
        except Exception:
            logger.exception(
                "Redis check_rate_limit failed — allowing request to proceed",
                extra={"user_id": user_id},
            )
            # Fail open: if Redis is unavailable, allow the request rather than
            # blocking all traffic.  Operators should alert on Redis errors separately.
            return True

    # ------------------------------------------------------------------
    # Pattern cache
    # ------------------------------------------------------------------

    def get_pattern_cache(self) -> list[dict] | None:
        """Return the cached list of fraud patterns, or None on cache miss.

        Returns None (rather than raising) if the cached value cannot be decoded
        so callers fall back to DynamoDB transparently.
        """
        try:
            raw = self._redis.get(_PATTERN_CACHE_KEY)
            if raw is None:
                return None
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "Corrupt pattern cache entry — ignoring",
                extra={"key": _PATTERN_CACHE_KEY},
            )
            return None
        except Exception:
            logger.exception("Redis get_pattern_cache failed")
            raise

    def set_pattern_cache(
        self,
        patterns: list[dict],
        ttl: int = 86400,
    ) -> None:
        """Serialise *patterns* as JSON and store them with a *ttl*-second expiry.

        Args:
            patterns: List of pattern dicts to cache.
            ttl:      Time-to-live in seconds (default 24 hours).
        """
        try:
            self._redis.set(_PATTERN_CACHE_KEY, json.dumps(patterns), ex=ttl)
        except Exception:
            logger.exception("Redis set_pattern_cache failed")
            raise
