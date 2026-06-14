"""
tools/incident_tool.py — Incident Management Agent Tools
=========================================================
CONCEPT: Incident Lifecycle in IT Operations

  An incident goes through these states:
    detected → investigating → mitigated → resolved

  The Incident Agent uses these tools to:
    1. CREATE a new incident record when a P1/P2 issue is reported
    2. QUERY incident history to find similar past incidents
    3. UPDATE status as investigation progresses
    4. SEARCH for patterns across historical incidents (RCA aid)

CONCEPT: Historical Pattern Matching
  One of the most valuable things an agent can do is ask:
  "Have we seen this before?"

  The `search_incident_history` tool lets the agent find similar
  past incidents, which:
    - Speeds up diagnosis (don't repeat work)
    - Surfaces known fixes (what worked before)
    - Identifies recurring patterns (systemic issues)

  This is different from RAG (searching documents) — here we're
  searching STRUCTURED incident records in SQLite.

CONCEPT: Incident Context for RCA
  When the Incident Agent calls these tools, it builds up a body
  of evidence: logs, metrics, similar incidents, runbook excerpts.
  All this evidence is passed to the RCA Agent to determine root cause.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ─── Input Schemas ────────────────────────────────────────────────────────────

class IncidentCreateInput(BaseModel):
    """Input schema for creating a new incident."""
    title: str = Field(
        description="Short incident title, e.g. 'P1: Payment service OOM crash'.",
    )
    description: str = Field(
        description="Full description of the incident: what happened, when, impact.",
    )
    severity: str = Field(
        description="Severity level: 'P1' (critical), 'P2' (high), 'P3' (medium), 'P4' (low).",
    )
    affected_services: Optional[str] = Field(
        default=None,
        description="Comma-separated list of affected services, e.g. 'payment-api,checkout-service'.",
    )


class IncidentSearchInput(BaseModel):
    """Input schema for searching incident history."""
    query: str = Field(
        description=(
            "Search query to find similar past incidents. "
            "Use service names, error types, or symptoms. "
            "Examples: 'payment service OOM', 'database connection pool', 'Kubernetes pod crash'"
        )
    )
    severity: Optional[str] = Field(
        default=None,
        description="Filter by severity: 'P1', 'P2', 'P3', 'P4'. Leave null for all.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of incidents to return.",
    )


class IncidentUpdateInput(BaseModel):
    """Input schema for updating an incident's status."""
    incident_id: int = Field(
        description="Numeric ID of the incident to update.",
    )
    status: Optional[str] = Field(
        default=None,
        description="New status: 'detected', 'investigating', 'mitigated', 'resolved'.",
    )
    root_cause: Optional[str] = Field(
        default=None,
        description="Root cause analysis finding. Set when root cause is determined.",
    )
    resolution_note: Optional[str] = Field(
        default=None,
        description="What was done to resolve the incident. Set when resolved.",
    )


class IncidentGetInput(BaseModel):
    """Input schema for getting a specific incident."""
    incident_id: int = Field(
        description="Numeric ID of the incident to retrieve.",
    )


# ─── Tool Factory ─────────────────────────────────────────────────────────────

def build_incident_tools(db) -> list[StructuredTool]:
    """
    Factory that returns all incident-related tools with a bound DB session.

    Returns tools for:
      - create_incident:         Create a new incident record
      - search_incident_history: Find similar past incidents
      - update_incident:         Update status / root cause
      - get_incident:            Get full incident details
    """

    # ── Tool 1: Create Incident ────────────────────────────────────────────────
    async def create_incident(
        title: str,
        description: str,
        severity: str,
        affected_services: Optional[str] = None,
    ) -> str:
        """
        Create a new incident record to track an ongoing IT operations issue.

        Use this when a user reports a new outage, performance degradation,
        or security event. This logs the incident so the team can track it.

        Returns the incident ID and confirmation.
        """
        from app.db.models import Incident, IncidentSeverity, IncidentStatus

        logger.info("incident_create_tool", title=title[:80], severity=severity)

        valid_severities = [s.value for s in IncidentSeverity]
        if severity.upper() not in valid_severities:
            return f"Invalid severity '{severity}'. Valid values: {', '.join(valid_severities)}"

        try:
            incident = Incident(
                title=title,
                description=description,
                severity=severity.upper(),
                status=IncidentStatus.DETECTED,
                affected_services=affected_services,
                reported_by=1,   # Placeholder until Phase 9 auth
            )
            db.add(incident)
            await db.flush()
            await db.commit()

            logger.info("incident_created_by_agent", incident_id=incident.id, severity=severity)

            return (
                f"🚨 Incident INC-{incident.id} created!\n"
                f"  Title:            {incident.title}\n"
                f"  Severity:         {incident.severity}\n"
                f"  Status:           {incident.status}\n"
                f"  Affected Services:{affected_services or 'N/A'}\n\n"
                f"Investigation is now underway. Use search_incident_history to find similar past incidents."
            )

        except Exception as e:
            await db.rollback()
            logger.error("incident_create_failed", error=str(e))
            return f"Failed to create incident: {str(e)}"

    # ── Tool 2: Search Incident History ───────────────────────────────────────
    async def search_incident_history(
        query: str,
        severity: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """
        Search historical incidents to find similar past issues and their resolutions.

        This is critical for Root Cause Analysis:
        - "Have we seen this before?" → find matching incidents
        - "How did we fix it?" → read resolution from past incidents
        - "Is this a recurring pattern?" → identify systemic issues

        Returns a list of matching past incidents with their root causes and resolutions.
        """
        from sqlalchemy import select, or_
        from app.db.models import Incident

        logger.info("incident_search_tool", query=query[:80], severity=severity)

        try:
            stmt = select(Incident).order_by(Incident.detected_at.desc())

            if query.strip():
                stmt = stmt.where(
                    or_(
                        Incident.title.ilike(f"%{query}%"),
                        Incident.description.ilike(f"%{query}%"),
                        Incident.affected_services.ilike(f"%{query}%"),
                        Incident.root_cause.ilike(f"%{query}%"),
                    )
                )

            if severity:
                stmt = stmt.where(Incident.severity == severity.upper())

            stmt = stmt.limit(limit)
            result = await db.execute(stmt)
            incidents = result.scalars().all()

            if not incidents:
                return f"No historical incidents found matching '{query}'."

            lines = [f"Found {len(incidents)} historical incident(s):\n"]
            for inc in incidents:
                lines.append(
                    f"• INC-{inc.id}: [{inc.severity}] {inc.title}\n"
                    f"  Status: {inc.status} | Detected: {inc.detected_at.strftime('%Y-%m-%d %H:%M') if inc.detected_at else 'N/A'}\n"
                    f"  Affected: {inc.affected_services or 'N/A'}\n"
                    + (f"  Root Cause: {inc.root_cause[:200]}\n" if inc.root_cause else "  Root Cause: Not determined yet\n")
                )
            return "\n".join(lines)

        except Exception as e:
            logger.error("incident_search_failed", error=str(e))
            return f"Incident search failed: {str(e)}"

    # ── Tool 3: Update Incident ────────────────────────────────────────────────
    async def update_incident(
        incident_id: int,
        status: Optional[str] = None,
        root_cause: Optional[str] = None,
        resolution_note: Optional[str] = None,
    ) -> str:
        """
        Update an incident's status, root cause finding, or resolution.

        Use this as the investigation progresses:
        - When starting investigation: status='investigating'
        - When issue is partially fixed: status='mitigated'
        - When fully resolved: status='resolved', add root_cause and resolution_note
        """
        from sqlalchemy import select
        from app.db.models import Incident
        from datetime import datetime, timezone

        logger.info("incident_update_tool", incident_id=incident_id, status=status)

        try:
            result = await db.execute(select(Incident).where(Incident.id == incident_id))
            incident = result.scalar_one_or_none()

            if not incident:
                return f"Incident INC-{incident_id} not found."

            changes = []
            now = datetime.now(timezone.utc)

            if status:
                old_status = incident.status
                incident.status = status.lower()
                changes.append(f"status: {old_status} → {status.lower()}")
                if status.lower() == "mitigated":
                    incident.mitigated_at = now
                elif status.lower() == "resolved":
                    incident.resolved_at = now

            if root_cause:
                incident.root_cause = root_cause
                changes.append("root_cause: updated")

            if resolution_note:
                # Store resolution in timeline_json or description
                existing = incident.timeline_json or ""
                incident.timeline_json = existing + f"\n[{now.isoformat()}] RESOLUTION: {resolution_note}"
                changes.append("resolution: added")

            if not changes:
                return f"No changes specified for INC-{incident_id}."

            await db.commit()
            logger.info("incident_updated_by_agent", incident_id=incident_id, changes=changes)

            return (
                f"✅ Incident INC-{incident_id} updated!\n"
                f"  Changes: {', '.join(changes)}\n"
                f"  Current status: {incident.status}\n"
                f"  Severity: {incident.severity}"
            )

        except Exception as e:
            await db.rollback()
            logger.error("incident_update_failed", incident_id=incident_id, error=str(e))
            return f"Failed to update incident INC-{incident_id}: {str(e)}"

    # ── Tool 4: Get Incident Details ───────────────────────────────────────────
    async def get_incident(incident_id: int) -> str:
        """
        Get full details of a specific incident by its numeric ID.

        Use this to get the current state of an ongoing incident
        or to review the full timeline of a past incident.
        """
        from sqlalchemy import select
        from app.db.models import Incident

        logger.info("incident_get_tool", incident_id=incident_id)

        try:
            result = await db.execute(select(Incident).where(Incident.id == incident_id))
            incident = result.scalar_one_or_none()

            if not incident:
                return f"Incident INC-{incident_id} not found."

            return (
                f"Incident INC-{incident_id} Details:\n"
                f"  Title:            {incident.title}\n"
                f"  Severity:         {incident.severity}\n"
                f"  Status:           {incident.status}\n"
                f"  Affected Services:{incident.affected_services or 'N/A'}\n"
                f"  Detected At:      {incident.detected_at}\n"
                f"  Description:      {incident.description}\n"
                + (f"  Root Cause:       {incident.root_cause}\n" if incident.root_cause else "")
                + (f"  Mitigated At:     {incident.mitigated_at}\n" if incident.mitigated_at else "")
                + (f"  Resolved At:      {incident.resolved_at}\n" if incident.resolved_at else "")
                + (f"  Timeline:\n{incident.timeline_json}\n" if incident.timeline_json else "")
            )

        except Exception as e:
            logger.error("incident_get_failed", incident_id=incident_id, error=str(e))
            return f"Failed to retrieve incident INC-{incident_id}: {str(e)}"

    # ── Build and return all tools ─────────────────────────────────────────────
    return [
        StructuredTool.from_function(
            coroutine=create_incident,
            name="create_incident",
            description=(
                "Create a new incident record for a production outage or critical issue. "
                "Use when a user reports a P1/P2 incident or system failure. "
                "Always create an incident record before starting investigation."
            ),
            args_schema=IncidentCreateInput,
        ),
        StructuredTool.from_function(
            coroutine=search_incident_history,
            name="search_incident_history",
            description=(
                "Search historical incidents to find similar past issues and their resolutions. "
                "Use this during investigation to check if this issue has happened before. "
                "Previous root causes and resolutions can dramatically speed up diagnosis."
            ),
            args_schema=IncidentSearchInput,
        ),
        StructuredTool.from_function(
            coroutine=update_incident,
            name="update_incident",
            description=(
                "Update an incident's status, root cause, or resolution notes. "
                "Use as investigation progresses: investigating → mitigated → resolved."
            ),
            args_schema=IncidentUpdateInput,
        ),
        StructuredTool.from_function(
            coroutine=get_incident,
            name="get_incident",
            description=(
                "Get full details of a specific incident by its numeric ID. "
                "Use when you need the current state or full timeline of an incident."
            ),
            args_schema=IncidentGetInput,
        ),
    ]
