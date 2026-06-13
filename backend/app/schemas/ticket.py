"""
schemas/ticket.py — Pydantic Schemas for Ticket API
====================================================
CONCEPT: Pydantic Schemas vs SQLAlchemy Models

  WHY TWO SEPARATE CLASSES?
  
  SQLAlchemy Model (models.py):
    - Represents a DATABASE ROW
    - Knows about DB relationships, foreign keys
    - Has fields like hashed_password (never expose!)
    - Used INTERNALLY by the application

  Pydantic Schema (schemas/):
    - Represents the API CONTRACT (what goes in/out of HTTP)
    - Validates incoming request data
    - Serializes outgoing response data
    - Controls EXACTLY what fields the client sees
    - Enables OpenAPI documentation generation

  PATTERN: Separate schemas for Create / Update / Response
    - TicketCreate: fields needed to create a ticket
    - TicketUpdate: fields allowed to change
    - TicketResponse: what we return to the client
    
  This is the "DTO pattern" (Data Transfer Object).
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


# ─── Base Schema ──────────────────────────────────────────────────────────────
class TicketBase(BaseModel):
    """
    Shared fields between Create, Update, and Response schemas.
    Avoids repeating field definitions.
    """
    title: str = Field(..., min_length=5, max_length=255, description="Short ticket title")
    description: str = Field(..., min_length=10, description="Detailed problem description")
    priority: str = Field(default="medium", description="low | medium | high | critical")
    category: str | None = Field(default=None, description="e.g. database, kubernetes, network")
    tags: str | None = Field(default=None, description="Comma-separated tags")

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        """Ensure priority is one of the allowed values."""
        allowed = {"low", "medium", "high", "critical"}
        if v.lower() not in allowed:
            raise ValueError(f"Priority must be one of: {', '.join(allowed)}")
        return v.lower()


# ─── Create Schema ────────────────────────────────────────────────────────────
class TicketCreate(TicketBase):
    """
    Schema for POST /tickets — creating a new ticket.
    
    The user provides: title, description, priority
    The system fills in: id, created_by, created_at, status
    """
    pass  # Inherits all fields from TicketBase


# ─── Update Schema ────────────────────────────────────────────────────────────
class TicketUpdate(BaseModel):
    """
    Schema for PATCH /tickets/{id} — partial update.
    
    ALL fields are optional — user can update just the status,
    or just the assignee, or any combination.
    
    This implements the PATCH semantic (partial update) vs
    PUT (full replacement of the resource).
    """
    title: str | None = Field(default=None, min_length=5, max_length=255)
    description: str | None = Field(default=None, min_length=10)
    status: str | None = Field(default=None)
    priority: str | None = Field(default=None)
    category: str | None = Field(default=None)
    assigned_to: int | None = Field(default=None)
    resolution: str | None = Field(default=None)
    tags: str | None = Field(default=None)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"open", "in_progress", "resolved", "closed"}
        if v.lower() not in allowed:
            raise ValueError(f"Status must be one of: {', '.join(allowed)}")
        return v.lower()

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"low", "medium", "high", "critical"}
        if v.lower() not in allowed:
            raise ValueError(f"Priority must be one of: {', '.join(allowed)}")
        return v.lower()


# ─── Response Schema ──────────────────────────────────────────────────────────
class TicketResponse(TicketBase):
    """
    Schema for API responses — what clients receive.
    
    Includes server-generated fields (id, timestamps, status).
    `model_config = ConfigDict(from_attributes=True)` enables
    creating this schema directly from a SQLAlchemy model instance.
    """
    id: int
    status: str
    created_by: int
    assigned_to: int | None = None
    resolution: str | None = None
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None = None

    model_config = {"from_attributes": True}  # Allows: TicketResponse.model_validate(db_ticket)


# ─── List Response ────────────────────────────────────────────────────────────
class TicketListResponse(BaseModel):
    """Paginated list of tickets."""
    tickets: list[TicketResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


# ─── Filter Schema ────────────────────────────────────────────────────────────
class TicketFilter(BaseModel):
    """Query parameters for filtering tickets."""
    status: str | None = None
    priority: str | None = None
    category: str | None = None
    assigned_to: int | None = None
    created_by: int | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
