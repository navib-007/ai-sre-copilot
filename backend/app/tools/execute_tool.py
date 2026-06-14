"""
tools/execute_tool.py — Action Remediation & Execution Tool
===========================================================
CONCEPT: High-Risk Action Guardrail
  This tool executes remediation actions (like restarting pods or modifying configs).
  It enforces a safety gate: if the action has not been approved by a human
  in the `approval_requests` table, it pauses and creates a pending request.
  If it has been approved, it executes the simulated operation.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from app.logging_config import get_logger
from app.db.models import ApprovalRequest, ApprovalStatus, Incident
from app.agents.approval_agent import create_approval

logger = get_logger(__name__)


# ─── Input Schema ─────────────────────────────────────────────────────────────

class ExecuteActionInput(BaseModel):
    """Input schema for executing remediation actions."""
    action_type: str = Field(
        description="The type of action to execute. Examples: 'restart_pod', 'scale_replicas', 'rollback_config'."
    )
    target: str = Field(
        description="The target resource or service. Examples: 'payment-service', 'api-gateway', 'redis-db'."
    )
    justification: str = Field(
        description="Why this action is necessary, based on the incident evidence gathered."
    )


# ─── Tool Factory ─────────────────────────────────────────────────────────────

def build_execute_tools(db, session_id: str) -> list[StructuredTool]:
    """
    Build execution tools bound to the current DB session and session ID.

    Args:
        db:         AsyncSession bound to the request
        session_id: The session ID of the current chat thread

    Returns:
        List of StructuredTools containing execute_remediation_action.
    """
    user_id = 1  # Placeholder until Phase 9 auth

    async def execute_remediation_action(
        action_type: str,
        target: str,
        justification: str,
    ) -> str:
        """
        Execute a remediation action on a target service.
        This is a high-risk tool. It automatically checks for human approval.
        If not approved, it creates a pending request and returns an error instruction.
        If approved, it executes the action.
        """
        logger.info(
            "execute_remediation_action_tool",
            action_type=action_type,
            target=target,
            session_id=session_id,
        )

        from sqlalchemy import select, desc

        try:
            # Step 1: Find any active non-resolved incident in this database
            incident_res = await db.execute(
                select(Incident)
                .where(Incident.status != "resolved")
                .order_by(desc(Incident.detected_at))
                .limit(1)
            )
            active_incident = incident_res.scalar_one_or_none()
            incident_id = active_incident.id if active_incident else None

            # Step 2: Check if an approved request exists for this session and action_type
            app_res = await db.execute(
                select(ApprovalRequest).where(
                    ApprovalRequest.langgraph_thread_id == session_id,
                    ApprovalRequest.action_type == action_type,
                    ApprovalRequest.status == ApprovalStatus.APPROVED,
                )
            )
            approved_request = app_res.scalar_one_or_none()

            # Step 3: If not approved, create a pending approval request
            if not approved_request:
                action_details = {
                    "action_type": action_type,
                    "target": target,
                    "justification": justification,
                    "session_id": session_id,
                }
                req = await create_approval(
                    db=db,
                    incident_id=incident_id,
                    thread_id=session_id,
                    action_type=action_type,
                    action_details=action_details,
                    requested_by=user_id,
                )
                # Flush the database so the pending request is saved immediately.
                # The caller (chat router) will commit it.
                await db.flush()

                logger.warning(
                    "remediation_requires_approval",
                    request_id=req.id,
                    action=action_type,
                    target=target,
                )

                return (
                    f"⚠️ CRITICAL: The action '{action_type}' on target '{target}' requires human approval. "
                    f"A pending approval request has been created with ID: {req.id}.\n"
                    f"Instruction to Agent: You MUST pause and inform the user that you are waiting for them to "
                    f"approve Request ID {req.id} before you can execute this. Do not try calling this tool again "
                    f"until they confirm they have approved it."
                )

            # Step 4: If approved, execute the simulated action
            logger.info(
                "remediation_executing_approved",
                request_id=approved_request.id,
                action=action_type,
                target=target,
            )

            # Perform the mock remediation changes
            # In a real environment, this would call Kubernetes API, AWS CLI, Ansible etc.
            # Here we simulate the mitigation:
            if active_incident:
                # Update incident status to mitigated
                active_incident.status = "mitigated"
                active_incident.root_cause = f"Action {action_type} executed on {target}. Justification: {justification}"
                
                # Append to incident timeline
                import json
                from datetime import datetime, timezone
                timeline = json.loads(active_incident.timeline_json) if active_incident.timeline_json else []
                timeline.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": f"Remediation action '{action_type}' successfully executed on '{target}'.",
                    "agent": "incident_agent"
                })
                active_incident.timeline_json = json.dumps(timeline)
                await db.flush()

            # Mark approval request as resolved / closed by updating its status
            # to prevent replay attacks (using the same approval twice)
            approved_request.status = "closed"  # Custom closed state so it is no longer reusable
            await db.flush()

            return (
                f"✅ SUCCESS: Action '{action_type}' was successfully executed on target '{target}'.\n"
                f"Remediation logs: [target={target}] - connection established - sent rollout restart signal - rolling update complete.\n"
                f"Status: Service mitigated. Please update the user and resolve the incident record."
            )

        except Exception as e:
            logger.error("execute_remediation_failed", error=str(e), exc_info=True)
            return f"Error executing remediation action: {str(e)}"

    return [
        StructuredTool.from_function(
            coroutine=execute_remediation_action,
            name="execute_remediation_action",
            description=(
                "Execute a remediation action (like 'restart_pod', 'scale_replicas', 'rollback_config') "
                "on a target service. This is a high-risk tool that automatically checks for human approval. "
                "If not yet approved, it will automatically register a pending request and tell you to pause."
            ),
            args_schema=ExecuteActionInput,
        )
    ]
