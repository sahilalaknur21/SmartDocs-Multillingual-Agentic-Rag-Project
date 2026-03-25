# smartdocs/api/middleware/user_context.py
"""
WHY THIS EXISTS:
Every DB query requires a user_id for RLS enforcement. LAW 6.
This middleware extracts user_id from the X-User-ID header and injects
it into request.state so every route handler can access it without
passing it through function arguments.

IN DEVELOPMENT: If X-User-ID header is missing, defaults to "dev_user".
IN PRODUCTION: Missing X-User-ID returns 401. Set ENVIRONMENT=production in .env.

WHY MIDDLEWARE AND NOT A DEPENDENCY:
A FastAPI Depends() function would require every route to declare it.
Middleware runs on every request unconditionally — user_id is always available.
One place to change auth logic. No route can accidentally skip it.
"""

from __future__ import annotations

import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Development fallback — NEVER use in production
_DEV_USER_ID = "dev_user_001"


class UserContextMiddleware(BaseHTTPMiddleware):
    """
    Extracts user_id from X-User-ID header.
    Injects into request.state.user_id for use in route handlers.

    Development mode (ENVIRONMENT=development):
      Missing header → default to "dev_user_001" + log warning
    Production mode (ENVIRONMENT=production):
      Missing header → 401 Unauthorized

    Usage in route handlers:
      user_id = request.state.user_id
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        user_id = request.headers.get("X-User-ID", "").strip()

        if not user_id:
            if settings.environment == "production":
                logger.warning(
                    "[middleware] Missing X-User-ID in production — returning 401"
                )
                return Response(
                    content='{"detail": "X-User-ID header is required"}',
                    status_code=401,
                    media_type="application/json",
                )
            else:
                # Development: allow requests without auth header
                user_id = _DEV_USER_ID
                logger.debug(
                    f"[middleware] No X-User-ID header — using dev default: {_DEV_USER_ID}"
                )

        request.state.user_id = user_id
        response = await call_next(request)
        return response