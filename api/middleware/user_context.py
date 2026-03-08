# api/middleware/user_context.py
# WHY THIS EXISTS: Injects user_id into every API request state.
# Missing X-User-ID header = 401 before retrieval runs. LAW 6.

import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class UserContextMiddleware(BaseHTTPMiddleware):
    """
    Extracts user_id from X-User-ID request header.
    Injects into request.state.user_id.
    All route handlers read from request.state.user_id — never hardcode.

    Production upgrade path: replace header extraction with JWT validation.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        user_id = request.headers.get("X-User-ID", "").strip()

        if not user_id:
            logger.warning(
                "Request rejected — missing X-User-ID",
                extra={"path": request.url.path},
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Missing X-User-ID header",
                    "detail": "All requests require X-User-ID header for per-user isolation.",
                },
            )

        request.state.user_id = user_id
        logger.debug("User context set", extra={"user_id": user_id})
        return await call_next(request)