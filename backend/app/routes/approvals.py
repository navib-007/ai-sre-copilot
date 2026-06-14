"""
routes/approvals.py — Human-in-the-Loop Approval Endpoints
==========================================================
CONCEPT: Action Gates (HTTP Interface)
  Exposes the API surface for humans to query pending requests
  and submit decisions.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.logging_config import get_logger
from app.agents.approval_agent import (
    get_pending_approvals,
    process_approval_decision,
)
from app.schemas.approval import ApprovalRequestResponse, ApprovalDecision

logger = get_logger(__name__)

router = APIRouter(prefix="/approvals", tags=["Human-in-the-Loop Approvals"])


async def _audit(
    db: AsyncSession,
    action: str,
    resource_id: int,
    details: dict | None = None,
) -> None:
    """Record audit logs for human-in-the-loop decisions."""
    from app.db.models import AuditLog
    import json

    db.add(
        AuditLog(
            user_id=1,  # Placeholder until Phase 9 auth
            action=action,
            resource_type="approval_request",
            resource_id=resource_id,
            details_json=json.dumps(details) if details else None,
        )
    )


@router.get(
    "/pending",
    response_model=list[ApprovalRequestResponse],
    summary="List all pending approval requests",
)
async def list_pending_approvals(
    db: AsyncSession = Depends(get_db),
) -> list[ApprovalRequestResponse]:
    """Retrieve all pending approval requests awaiting human review."""
    logger.info("list_pending_approvals_request")
    requests = await get_pending_approvals(db=db)
    return [ApprovalRequestResponse.model_validate(r) for r in requests]


@router.post(
    "/{id}/approve",
    response_model=ApprovalRequestResponse,
    summary="Approve a pending request",
)
async def approve_request(
    id: int,
    decision: ApprovalDecision,
    db: AsyncSession = Depends(get_db),
) -> ApprovalRequestResponse:
    """Approve a pending action so the agent can execute it on its next turn."""
    logger.info("approve_request_api", request_id=id)

    try:
        updated = await process_approval_decision(
            db=db,
            request_id=id,
            status="approved",
            reviewer_id=1,
            comment=decision.comment,
        )

        await _audit(db, "approval_granted", id, {"comment": decision.comment})
        # Note: get_db dependency will automatically commit this transaction.
        return ApprovalRequestResponse.model_validate(updated)

    except ValueError as val_err:
        logger.warning("approve_request_invalid", request_id=id, error=str(val_err))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as e:
        logger.error("approve_request_failed", request_id=id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to approve request: {str(e)}",
        )


@router.post(
    "/{id}/reject",
    response_model=ApprovalRequestResponse,
    summary="Reject a pending request",
)
async def reject_request(
    id: int,
    decision: ApprovalDecision,
    db: AsyncSession = Depends(get_db),
) -> ApprovalRequestResponse:
    """Reject a pending action, stopping the agent from executing it."""
    logger.info("reject_request_api", request_id=id)

    try:
        updated = await process_approval_decision(
            db=db,
            request_id=id,
            status="rejected",
            reviewer_id=1,
            comment=decision.comment,
        )

        await _audit(db, "approval_rejected", id, {"comment": decision.comment})
        return ApprovalRequestResponse.model_validate(updated)

    except ValueError as val_err:
        logger.warning("reject_request_invalid", request_id=id, error=str(val_err))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as e:
        logger.error("reject_request_failed", request_id=id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reject request: {str(e)}",
        )
