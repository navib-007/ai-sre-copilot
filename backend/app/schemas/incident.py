"""
schemas/incident.py — Pydantic Schemas for Incident API
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class IncidentBase(BaseModel):
    title: str = Field(..., min_length=5, max_length=255)
    description: str = Field(..., min_length=10)
    severity: str = Field(default="P3", description="P1 | P2 | P3 | P4")
    affected_services: str | None = Field(
        default=None, description="Comma-separated service names, e.g. 'payment-api,auth-service'"
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        allowed = {"P1", "P2", "P3", "P4"}
        if v.upper() not in allowed:
            raise ValueError(f"Severity must be one of: {', '.join(allowed)}")
        return v.upper()


class IncidentCreate(IncidentBase):
    """Schema for creating a new incident report."""
    pass


class IncidentUpdate(BaseModel):
    """Schema for partial incident update."""
    title: str | None = None
    description: str | None = None
    severity: str | None = None
    status: str | None = None
    affected_services: str | None = None
    root_cause: str | None = None
    timeline_json: str | None = None
    ticket_id: int | None = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"detected", "investigating", "mitigated", "resolved"}
        if v.lower() not in allowed:
            raise ValueError(f"Status must be one of: {', '.join(allowed)}")
        return v.lower()


class IncidentResponse(IncidentBase):
    """Schema for incident API responses."""
    id: int
    status: str
    root_cause: str | None = None
    timeline_json: str | None = None
    evidence_json: str | None = None
    reported_by: int
    ticket_id: int | None = None
    detected_at: datetime
    mitigated_at: datetime | None = None
    resolved_at: datetime | None = None

    model_config = {"from_attributes": True}


class IncidentListResponse(BaseModel):
    """Paginated list of incidents."""
    incidents: list[IncidentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class IncidentFilter(BaseModel):
    """Query parameters for filtering incidents."""
    severity: str | None = None
    status: str | None = None
    reported_by: int | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
