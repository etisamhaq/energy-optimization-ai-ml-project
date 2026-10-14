"""
API authentication, rate limiting and request metrics.

Configuration (environment variables):
  API_KEYS               Comma-separated keys accepted in the X-API-Key header.
                         Unset means authentication is disabled (local development only).
  RATE_LIMIT_PER_MINUTE  Requests allowed per client per minute (default 60, 0 disables).
"""

import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Set

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"

# Paths that never require a key or count against the rate limit
PUBLIC_PATHS = {"/", "/health", "/ready", "/metrics", "/docs", "/redoc", "/openapi.json"}

_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

REQUEST_COUNT = Counter(
    "energy_api_requests_total", "HTTP requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "energy_api_request_duration_seconds", "HTTP request latency", ["method", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
)
PREDICTIONS = Counter(
    "energy_api_predicted_hours_total", "Hourly predictions served", ["model"]
)


def configured_api_keys() -> Set[str]:
    return {key.strip() for key in os.environ.get("API_KEYS", "").split(",") if key.strip()}


def require_api_key(request: Request, api_key: Optional[str] = Security(_api_key_header)) -> Optional[str]:
    """FastAPI dependency: reject requests without a valid key when keys are configured."""
    keys = configured_api_keys()
    if not keys or request.url.path in PUBLIC_PATHS:
        return None
    if api_key and any(secrets.compare_digest(api_key, key) for key in keys):
        return api_key
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": API_KEY_HEADER},
    )


class RateLimiter:
    """
    Sliding-window limit of requests per client per minute.

    State is kept in memory, so the limit applies per API process. Behind several
    replicas, enforce a global limit at the gateway or load balancer instead.
    """

    def __init__(self, limit_per_minute: int, window_seconds: float = 60.0):
        self.limit = limit_per_minute
        self.window = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str) -> Optional[float]:
        """Record a request; return seconds to wait if the client is over the limit."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return self.window - (now - hits[0])
            hits.append(now)
            if len(self._hits) > 10_000:
                # Drop idle clients so memory stays bounded
                for key in [k for k, v in self._hits.items() if not v]:
                    del self._hits[key]
            return None


def _client_id(request: Request) -> str:
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key:
        return f"key:{api_key}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Rate limiting, Prometheus metrics and an access log line per request."""

    def __init__(self, app, limit_per_minute: int):
        super().__init__(app)
        self.rate_limiter = RateLimiter(limit_per_minute) if limit_per_minute > 0 else None

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        start = time.perf_counter()

        if self.rate_limiter and path not in PUBLIC_PATHS:
            retry_after = self.rate_limiter.check(_client_id(request))
            if retry_after is not None:
                response = JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": str(int(retry_after) + 1)},
                )
                self._record(request, "rate_limited", response.status_code, start)
                return response

        response = await call_next(request)
        # Label by route template (e.g. /predict) rather than raw URL to keep metric cardinality low
        route = request.scope.get("route")
        self._record(request, getattr(route, "path", "unmatched"), response.status_code, start)
        return response

    @staticmethod
    def _record(request: Request, path: str, status: int, start: float):
        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(request.method, path, str(status)).inc()
        REQUEST_LATENCY.labels(request.method, path).observe(elapsed)
        logger.info(f"{request.method} {request.url.path} {status} {elapsed * 1000:.0f}ms")
