"""
routes/incidents.py — Incident CRUD API Endpoints
"""

import math

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.models import AuditLog, Incident, User
from app.logging_config import get_logger
from app.auth import require_viewer, require_engineer
from app.schemas.incident import (
    IncidentCreate,
    IncidentListResponse,
    IncidentResponse,
    IncidentUpdate,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/incidents", tags=["Incidents"])


async def _audit(db: AsyncSession, action: str, resource_id: int, user_id: int, details: dict | None = None) -> None:
    """Helper to record audit logs for incident operations."""
    import json
    db.add(AuditLog(
        user_id=user_id,
        action=action,
        resource_type="incident",
        resource_id=resource_id,
        details_json=json.dumps(details) if details else None,
    ))


# ─── GET /incidents ───────────────────────────────────────────────────────────
@router.get("/", response_model=IncidentListResponse, summary="List incidents")
async def list_incidents(
    severity: str | None = Query(default=None),
    status: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
) -> IncidentListResponse:
    """List incidents with optional severity and status filters."""
    logger.info("list_incidents_request", severity=severity, status=status, page=page)

    filters = []
    if severity:
        filters.append(Incident.severity == severity.upper())
    if status:
        filters.append(Incident.status == status.lower())

    count_stmt = select(func.count(Incident.id))
    if filters:
        count_stmt = count_stmt.where(*filters)
    total = (await db.execute(count_stmt)).scalar_one()

    offset = (page - 1) * page_size
    stmt = (
        select(Incident)
        .where(*filters)
        .order_by(Incident.detected_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    incidents = (await db.execute(stmt)).scalars().all()

    return IncidentListResponse(
        incidents=[IncidentResponse.model_validate(i) for i in incidents],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


# ─── POST /incidents ──────────────────────────────────────────────────────────
@router.post(
    "/",
    response_model=IncidentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Report a new incident",
)
async def create_incident(
    incident_data: IncidentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_engineer),
) -> IncidentResponse:
    """Report a new incident."""
    logger.info(
        "create_incident_request",
        severity=incident_data.severity,
        title=incident_data.title,
    )

    new_incident = Incident(
        title=incident_data.title,
        description=incident_data.description,
        severity=incident_data.severity,
        affected_services=incident_data.affected_services,
        reported_by=current_user.id,
        status="detected",
    )
    db.add(new_incident)
    await db.flush()

    await _audit(db, "incident_created", new_incident.id, current_user.id, {
        "title": new_incident.title,
        "severity": new_incident.severity,
    })

    logger.info("incident_created", incident_id=new_incident.id, severity=new_incident.severity)
    return IncidentResponse.model_validate(new_incident)


# ─── GET /incidents/{incident_id} ─────────────────────────────────────────────
@router.get("/{incident_id}", response_model=IncidentResponse, summary="Get incident by ID")
async def get_incident(
    incident_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
) -> IncidentResponse:
    """Retrieve a specific incident by ID."""
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()

    if incident is None:
        logger.warning("incident_not_found", incident_id=incident_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {incident_id} not found",
        )

    return IncidentResponse.model_validate(incident)


# ─── PATCH /incidents/{incident_id} ───────────────────────────────────────────
@router.patch("/{incident_id}", response_model=IncidentResponse, summary="Update an incident")
async def update_incident(
    incident_id: int,
    update_data: IncidentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_engineer),
) -> IncidentResponse:
    """Update incident status, root cause, or other fields."""
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()

    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {incident_id} not found",
        )

    update_fields = update_data.model_dump(exclude_unset=True)

    # Auto-set timestamps based on status change
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    if update_fields.get("status") == "mitigated" and incident.mitigated_at is None:
        update_fields["mitigated_at"] = now
    if update_fields.get("status") == "resolved" and incident.resolved_at is None:
        update_fields["resolved_at"] = now

    for field, value in update_fields.items():
        setattr(incident, field, value)

    await _audit(db, "incident_updated", incident_id, current_user.id, update_fields)
    logger.info("incident_updated", incident_id=incident_id, fields=list(update_fields.keys()))

    return IncidentResponse.model_validate(incident)
