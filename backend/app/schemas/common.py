"""
schemas/common.py — Shared Pydantic Schemas
============================================
Reusable response envelopes, pagination, and error schemas.
"""

from pydantic import BaseModel


class SuccessResponse(BaseModel):
    """Generic success response envelope."""
    success: bool = True
    message: str
    data: dict | None = None


class ErrorResponse(BaseModel):
    """Standard error response returned on validation/HTTP errors."""
    success: bool = False
    error: str
    detail: str | None = None
    request_id: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str           # "healthy" | "degraded" | "unhealthy"
    app_name: str
    version: str
    environment: str
    database: str         # "connected" | "disconnected"
    uptime_seconds: float
