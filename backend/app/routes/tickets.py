"""
routes/tickets.py — Ticket CRUD API Endpoints
=============================================
CONCEPT: RESTful API Design

  REST conventions we follow:
    GET    /tickets          → list all tickets (with filtering/pagination)
    POST   /tickets          → create a new ticket
    GET    /tickets/{id}     → get a specific ticket
    PATCH  /tickets/{id}     → partial update
    DELETE /tickets/{id}     → soft delete (set status=closed)

CONCEPT: FastAPI Router
  Instead of registering all routes on `app` directly, we use APIRouter.
  Benefits:
    - Group related routes together
    - Add prefix/tags to all routes at once
    - Easier to test in isolation
    - Clean separation of concerns

CONCEPT: Dependency Injection
  FastAPI's `Depends()` system:
    - `Depends(get_db)` → provides an AsyncSession
    - `Depends(get_current_user)` → provides the authenticated User
    - Dependencies are composable and testable
    - You can mock them in tests without changing route code
"""

import math
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.models import AuditLog, Ticket, User
from app.logging_config import get_logger
from app.schemas.ticket import (
    TicketCreate,
    TicketListResponse,
    TicketResponse,
    TicketUpdate,
)

logger = get_logger(__name__)

# ─── Router ───────────────────────────────────────────────────────────────────
# prefix="/tickets" means all routes here are at /api/tickets/...
# tags=["Tickets"] groups them in the Swagger UI
router = APIRouter(prefix="/tickets", tags=["Tickets"])


# ─── Helper: Audit Log ────────────────────────────────────────────────────────
async def create_audit_log(
    db: AsyncSession,
    action: str,
    resource_type: str,
    resource_id: int,
    user_id: int | None = None,
    details: dict | None = None,
) -> None:
    """Create an audit log entry for any significant action."""
    import json
    audit = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details_json=json.dumps(details) if details else None,
    )
    db.add(audit)
    # Note: we don't commit here — the get_db dependency commits on success


# ─── GET /tickets ─────────────────────────────────────────────────────────────
@router.get("/", response_model=TicketListResponse, summary="List all tickets")
async def list_tickets(
    # Query parameters with defaults (appear in Swagger UI automatically)
    status: str | None = Query(default=None, description="Filter by status"),
    priority: str | None = Query(default=None, description="Filter by priority"),
    category: str | None = Query(default=None, description="Filter by category"),
    assigned_to: int | None = Query(default=None, description="Filter by assignee user ID"),
    search: str | None = Query(default=None, description="Search in title and description"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
) -> TicketListResponse:
    """
    Retrieve a paginated list of tickets with optional filters.

    CONCEPT: Query Building with SQLAlchemy
      We build the WHERE clause dynamically based on which filters are provided.
      This avoids separate SQL for each filter combination.
    """
    logger.info(
        "list_tickets_request",
        status=status,
        priority=priority,
        search=search,
        page=page,
    )

    # ── Build dynamic WHERE clause ────────────────────────────────────────────
    filters = []

    if status:
        filters.append(Ticket.status == status.lower())
    if priority:
        filters.append(Ticket.priority == priority.lower())
    if category:
        filters.append(Ticket.category == category)
    if assigned_to:
        filters.append(Ticket.assigned_to == assigned_to)
    if search:
        # Search in both title and description using OR
        search_filter = or_(
            Ticket.title.ilike(f"%{search}%"),
            Ticket.description.ilike(f"%{search}%"),
        )
        filters.append(search_filter)

    # ── Count total matching records ──────────────────────────────────────────
    count_stmt = select(func.count(Ticket.id))
    if filters:
        count_stmt = count_stmt.where(*filters)
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()

    # ── Fetch paginated results ───────────────────────────────────────────────
    offset = (page - 1) * page_size
    stmt = (
        select(Ticket)
        .where(*filters)
        .order_by(Ticket.created_at.desc())  # Newest first
        .offset(offset)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    tickets = result.scalars().all()

    logger.info("list_tickets_result", count=len(tickets), total=total)

    return TicketListResponse(
        tickets=[TicketResponse.model_validate(t) for t in tickets],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


# ─── POST /tickets ────────────────────────────────────────────────────────────
@router.post(
    "/",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new ticket",
)
async def create_ticket(
    ticket_data: TicketCreate,         # Pydantic validates the request body
    db: AsyncSession = Depends(get_db),
    # TODO Phase 9: Replace hardcoded user_id with: user: User = Depends(get_current_user)
    # For now, we use a default user_id=1 (set up in seed.py)
) -> TicketResponse:
    """
    Create a new support ticket.

    CONCEPT: Request Validation
      Pydantic automatically validates ticket_data before this function runs.
      If title is missing or priority is invalid → 422 Unprocessable Entity
      is returned BEFORE this function is called. Zero manual validation needed.
    """
    logger.info(
        "create_ticket_request",
        title=ticket_data.title,
        priority=ticket_data.priority,
    )

    # Create the ORM model from Pydantic data
    new_ticket = Ticket(
        title=ticket_data.title,
        description=ticket_data.description,
        priority=ticket_data.priority,
        category=ticket_data.category,
        tags=ticket_data.tags,
        created_by=1,           # Hardcoded until Phase 9 auth
        status="open",          # Always starts as open
    )

    db.add(new_ticket)
    await db.flush()           # Flush to get the auto-generated ID without committing

    # Create audit log
    await create_audit_log(
        db=db,
        action="ticket_created",
        resource_type="ticket",
        resource_id=new_ticket.id,
        user_id=1,
        details={"title": new_ticket.title, "priority": new_ticket.priority},
    )

    logger.info("ticket_created", ticket_id=new_ticket.id, title=new_ticket.title)

    return TicketResponse.model_validate(new_ticket)


# ─── GET /tickets/{ticket_id} ─────────────────────────────────────────────────
@router.get("/{ticket_id}", response_model=TicketResponse, summary="Get ticket by ID")
async def get_ticket(
    ticket_id: int,
    db: AsyncSession = Depends(get_db),
) -> TicketResponse:
    """Retrieve a single ticket by its ID."""
    logger.debug("get_ticket_request", ticket_id=ticket_id)

    result = await db.execute(select(Ticket).where(Ticket.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if ticket is None:
        logger.warning("ticket_not_found", ticket_id=ticket_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket with ID {ticket_id} not found",
        )

    return TicketResponse.model_validate(ticket)


# ─── PATCH /tickets/{ticket_id} ───────────────────────────────────────────────
@router.patch("/{ticket_id}", response_model=TicketResponse, summary="Update a ticket")
async def update_ticket(
    ticket_id: int,
    update_data: TicketUpdate,
    db: AsyncSession = Depends(get_db),
) -> TicketResponse:
    """
    Partially update a ticket (PATCH = only provided fields are updated).

    CONCEPT: model_dump(exclude_unset=True)
      This is crucial for PATCH semantics.
      If the client sends: {"status": "in_progress"}
      Then update_data.model_dump(exclude_unset=True) = {"status": "in_progress"}
      Only "status" is updated — NOT title, description, etc.
      Without exclude_unset=True, ALL fields including None values would overwrite DB.
    """
    logger.info("update_ticket_request", ticket_id=ticket_id)

    result = await db.execute(select(Ticket).where(Ticket.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket with ID {ticket_id} not found",
        )

    # Get only the fields the client actually sent
    update_fields = update_data.model_dump(exclude_unset=True)

    # Auto-set resolved_at when status changes to resolved
    if update_fields.get("status") == "resolved" and ticket.resolved_at is None:
        update_fields["resolved_at"] = datetime.now(timezone.utc)

    # Apply updates to the ORM object
    for field, value in update_fields.items():
        setattr(ticket, field, value)

    await create_audit_log(
        db=db,
        action="ticket_updated",
        resource_type="ticket",
        resource_id=ticket_id,
        user_id=1,
        details=jsonable_encoder(update_fields),
    )

    logger.info("ticket_updated", ticket_id=ticket_id, fields=list(update_fields.keys()))

    return TicketResponse.model_validate(ticket)


# ─── DELETE /tickets/{ticket_id} ──────────────────────────────────────────────
@router.delete(
    "/{ticket_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Close/delete a ticket",
)
async def delete_ticket(
    ticket_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Soft-delete a ticket by setting its status to 'closed'.

    CONCEPT: Soft Delete vs Hard Delete
      Hard delete: DELETE FROM tickets WHERE id = ? (data is gone forever)
      Soft delete: UPDATE tickets SET status='closed' (data preserved for audit)

      For IT ops, we NEVER hard delete tickets — they're audit records.
      Even closed tickets show up in historical queries.
    """
    result = await db.execute(select(Ticket).where(Ticket.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket with ID {ticket_id} not found",
        )

    ticket.status = "closed"

    await create_audit_log(
        db=db,
        action="ticket_closed",
        resource_type="ticket",
        resource_id=ticket_id,
        user_id=1,
    )

    logger.info("ticket_closed", ticket_id=ticket_id)
