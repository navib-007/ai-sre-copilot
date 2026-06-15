"""
a2a/protocol.py — A2A JSON-RPC 2.0 Message Dispatcher & Schemas
===============================================================
CONCEPT: JSON-RPC 2.0 Protocol Layer

  The A2A standard specifies JSON-RPC 2.0 over HTTP for message exchanges.
  This module parses incoming payloads, matches them against schemas,
  validates parameters, and dispatches to the corresponding handlers:
    - message/send  → Create task, start background SRE supervisor execution.
    - tasks/get     → Retrieve task status and results.
    - tasks/cancel  → Cancel execution and update task state.

  It handles standard JSON-RPC error codes (e.g. Method Not Found, Invalid Params).
"""

import asyncio
import uuid
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.a2a.task_manager import create_a2a_task, get_a2a_task, update_task_status, process_a2a_task_background
from app.db.models import A2ATask
from app.logging_config import get_logger

logger = get_logger(__name__)


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class JsonRpcRequest(BaseModel):
    jsonrpc: str = Field(default="2.0", pattern="^2.0$")
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[int, str]] = None


class MessageSendParams(BaseModel):
    messageId: str = Field(description="Unique client-supplied ID for deduplication")
    text: str = Field(description="The natural language task instruction for the agent")
    skillId: str = Field(description="The target skill to invoke (e.g. investigate_incident)")


class TaskGetParams(BaseModel):
    taskId: str = Field(description="The UUID of the task to retrieve")


class TaskCancelParams(BaseModel):
    taskId: str = Field(description="The UUID of the task to cancel")


# ─── JSON-RPC Error Builders ──────────────────────────────────────────────────

def make_jsonrpc_error(code: int, message: str, data: Any = None, req_id: Any = None) -> Dict[str, Any]:
    res = {
        "jsonrpc": "2.0",
        "error": {
            "code": code,
            "message": message,
        },
        "id": req_id
    }
    if data is not None:
        res["error"]["data"] = data
    return res


def make_jsonrpc_success(result: Any, req_id: Any) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "result": result,
        "id": req_id
    }


# ─── RPC Core Dispatcher ──────────────────────────────────────────────────────

async def handle_a2a_request(
    payload: Dict[str, Any],
    db: AsyncSession,
    session_factory: async_sessionmaker[AsyncSession]
) -> Dict[str, Any]:
    """
    Parse, validate, and execute an A2A JSON-RPC 2.0 request.
    Returns a standard JSON-RPC 2.0 response dictionary.
    """
    # 1. Parse and validate JSON-RPC structure
    try:
        req = JsonRpcRequest.model_validate(payload)
    except Exception as parse_err:
        logger.warning("a2a_rpc_parse_error", error=str(parse_err))
        return make_jsonrpc_error(-32600, "Invalid Request", data=str(parse_err))

    logger.info("a2a_rpc_request_received", method=req.method, request_id=req.id)

    # 2. Dispatch by method
    if req.method == "message/send":
        # Validate parameters
        if not req.params:
            return make_jsonrpc_error(-32602, "Invalid params: params object is required", req_id=req.id)
        try:
            params = MessageSendParams.model_validate(req.params)
        except Exception as param_err:
            logger.warning("a2a_rpc_invalid_params", method=req.method, error=str(param_err))
            return make_jsonrpc_error(-32602, "Invalid params", data=str(param_err), req_id=req.id)

        # Derive idempotent session ID from messageId
        derived_session_id = f"a2a-{params.messageId}"
        
        # Idempotency Check: check if task with this session ID already exists
        existing_res = await db.execute(
            select(A2ATask).where(A2ATask.session_id == derived_session_id)
        )
        existing_task = existing_res.scalar_one_or_none()
        if existing_task:
            logger.info("a2a_rpc_idempotent_hit", task_id=existing_task.task_id, session_id=derived_session_id)
            return make_jsonrpc_success({
                "taskId": existing_task.task_id,
                "status": existing_task.status,
                "skillId": existing_task.skill_id,
                "result": existing_task.result,
                "error": existing_task.error,
            }, req_id=req.id)

        # Create new task
        try:
            task = await create_a2a_task(
                db=db,
                skill_id=params.skillId,
                input_message=params.text,
                session_id=derived_session_id,
            )
            # Commit task creation immediately so background thread can fetch it
            await db.commit()
        except Exception as create_err:
            logger.error("a2a_rpc_task_creation_failed", error=str(create_err), exc_info=True)
            await db.rollback()
            return make_jsonrpc_error(-32603, "Internal error during task creation", req_id=req.id)

        # Spawn supervisor flow in background
        asyncio.create_task(process_a2a_task_background(task.task_id, session_factory))

        return make_jsonrpc_success({
            "taskId": task.task_id,
            "status": task.status,
            "skillId": task.skill_id,
        }, req_id=req.id)

    elif req.method == "tasks/get":
        if not req.params:
            return make_jsonrpc_error(-32602, "Invalid params: params object is required", req_id=req.id)
        try:
            params = TaskGetParams.model_validate(req.params)
        except Exception as param_err:
            logger.warning("a2a_rpc_invalid_params", method=req.method, error=str(param_err))
            return make_jsonrpc_error(-32602, "Invalid params", data=str(param_err), req_id=req.id)

        # Fetch task
        task = await get_a2a_task(db, params.taskId)
        if not task:
            logger.warning("a2a_rpc_task_not_found", task_id=params.taskId)
            return make_jsonrpc_error(-32001, f"Task with ID '{params.taskId}' not found", req_id=req.id)

        return make_jsonrpc_success({
            "taskId": task.task_id,
            "status": task.status,
            "skillId": task.skill_id,
            "result": task.result,
            "error": task.error,
        }, req_id=req.id)

    elif req.method == "tasks/cancel":
        if not req.params:
            return make_jsonrpc_error(-32602, "Invalid params: params object is required", req_id=req.id)
        try:
            params = TaskCancelParams.model_validate(req.params)
        except Exception as param_err:
            logger.warning("a2a_rpc_invalid_params", method=req.method, error=str(param_err))
            return make_jsonrpc_error(-32602, "Invalid params", data=str(param_err), req_id=req.id)

        # Fetch task
        task = await get_a2a_task(db, params.taskId)
        if not task:
            logger.warning("a2a_rpc_task_not_found", task_id=params.taskId)
            return make_jsonrpc_error(-32001, f"Task with ID '{params.taskId}' not found", req_id=req.id)

        # Cancel task if in active/runnable state
        if task.status in {"submitted", "working", "input-required"}:
            try:
                task = await update_task_status(db, task.task_id, "canceled")
            except Exception as cancel_err:
                logger.error("a2a_rpc_cancel_failed", task_id=task.task_id, error=str(cancel_err))
                return make_jsonrpc_error(-32603, f"Internal error during cancel: {str(cancel_err)}", req_id=req.id)

        return make_jsonrpc_success({
            "taskId": task.task_id,
            "status": task.status,
        }, req_id=req.id)

    else:
        logger.warning("a2a_rpc_method_not_found", method=req.method)
        return make_jsonrpc_error(-32601, "Method not found", req_id=req.id)
