"""
app/agents/approval_agent.py — Human-in-the-Loop Approval Operations
====================================================================
CONCEPT: Human-in-the-Loop (HITL) DB Manager
  Before high-risk actions are executed, a request is logged to the
  database. This module provides the backend functionality to create,
  query, and process approval decisions (approve/reject) in SQLite.
"""

from datetime import datetime, timezone
import json
from typing import Any, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ApprovalRequest, ApprovalStatus, Incident
from app.logging_config import get_logger

logger = get_logger(__name__)


async def create_approval(
    db: AsyncSession,
    incident_id: Optional[int],
    thread_id: str,
    action_type: str,
    action_details: dict[str, Any],
    requested_by: int = 1,
) -> ApprovalRequest:
    """
    Log a new approval request in the database.

    Args:
        db:             AsyncSession
        incident_id:    Optional Incident ID related to the request
        thread_id:      The LangGraph session/thread ID (session_id)
        action_type:    The type of action (e.g. 'restart_pod')
        action_details: JSON-serializable details about the action
        requested_by:   User ID requesting the action (default 1)

    Returns:
        The created ApprovalRequest ORM object.
    """
    logger.info(
        "creating_approval_request",
        incident_id=incident_id,
        thread_id=thread_id,
        action_type=action_type,
    )

    try:
        request = ApprovalRequest(
            incident_id=incident_id,
            langgraph_thread_id=thread_id,
            action_type=action_type,
            action_details_json=json.dumps(action_details),
            status=ApprovalStatus.PENDING,
            requested_by=requested_by,
            created_at=datetime.now(timezone.utc),
        )
        db.add(request)
        await db.flush()  # Populates request.id
        logger.info("approval_request_created", id=request.id)
        return request

    except Exception as e:
        logger.error("create_approval_failed", error=str(e), exc_info=True)
        raise e


async def get_pending_approvals(db: AsyncSession) -> list[ApprovalRequest]:
    """Retrieve all approval requests with status PENDING."""
    try:
        result = await db.execute(
            select(ApprovalRequest)
            .where(ApprovalRequest.status == ApprovalStatus.PENDING)
            .order_by(ApprovalRequest.created_at.desc())
        )
        return list(result.scalars().all())
    except Exception as e:
        logger.error("get_pending_approvals_failed", error=str(e), exc_info=True)
        return []


async def process_approval_decision(
    db: AsyncSession,
    request_id: int,
    status: str,
    reviewer_id: int = 1,
    comment: Optional[str] = None,
) -> ApprovalRequest:
    """
    Approve or reject a pending approval request.

    Args:
        db:          AsyncSession
        request_id:  ID of the ApprovalRequest to update
        status:      New status ('approved' or 'rejected')
        reviewer_id: ID of the reviewing user
        comment:     Optional review comment

    Returns:
        The updated ApprovalRequest ORM object.
    """
    logger.info("processing_approval_decision", id=request_id, status=status)

    try:
        result = await db.execute(
            select(ApprovalRequest).where(ApprovalRequest.id == request_id)
        )
        request = result.scalar_one_or_none()

        if not request:
            raise ValueError(f"Approval request with ID {request_id} not found.")

        if request.status != ApprovalStatus.PENDING:
            raise ValueError(
                f"Approval request {request_id} is already in state: {request.status}."
            )

        # Update status
        if status.lower() == "approved":
            request.status = ApprovalStatus.APPROVED
        elif status.lower() == "rejected":
            request.status = ApprovalStatus.REJECTED
        else:
            raise ValueError(f"Invalid status decision: {status}")

        request.reviewed_by = reviewer_id
        request.reviewed_at = datetime.now(timezone.utc)
        request.review_comment = comment

        # If an incident is linked, add an entry to its timeline (if incident exists)
        if request.incident_id:
            incident_res = await db.execute(
                select(Incident).where(Incident.id == request.incident_id)
            )
            incident = incident_res.scalar_one_or_none()
            if incident:
                # Append to incident timeline
                timeline = json.loads(incident.timeline_json) if incident.timeline_json else []
                timeline.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "event": f"Human-in-the-Loop decision: {status.upper()} by User {reviewer_id}. Comment: {comment or 'None'}",
                        "agent": "approval_agent",
                    }
                )
                incident.timeline_json = json.dumps(timeline)

        logger.info("approval_decision_processed", id=request_id, status=request.status)
        return request

    except Exception as e:
        logger.error("process_approval_decision_failed", id=request_id, error=str(e), exc_info=True)
        raise e
