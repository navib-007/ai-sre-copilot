"""
a2a/task_manager.py — A2A Task Lifecycle & Background Orchestration
===================================================================
CONCEPT: Task Lifecycle Management

  The A2A protocol requires a stateful lifecycle for delegated tasks:
    - submitted      → Task is registered in the database, waiting to start.
    - working        → Task execution has started in a background worker.
    - input-required → The task requires human intervention (approval needed).
    - completed      → Task completed successfully; result is ready.
    - failed         → Task failed due to an exception; error is logged.
    - canceled       → Task execution was canceled by the client.

  We use SQLite to persist task states and run the supervisor agent
  asynchronously in a background worker to keep the JSON-RPC HTTP request fast
  and non-blocking.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import A2ATask, ChatMessage, ChatSession
from app.logging_config import get_logger

logger = get_logger(__name__)


async def create_a2a_task(
    db: AsyncSession,
    skill_id: str,
    input_message: str,
    session_id: str,
    user_id: int = 1,
) -> A2ATask:
    """
    Create a new A2A task entry in the database.
    """
    task_id = str(uuid.uuid4())
    logger.info("create_a2a_task", task_id=task_id, skill_id=skill_id, session_id=session_id)

    task = A2ATask(
        task_id=task_id,
        status="submitted",
        skill_id=skill_id,
        input_message=input_message,
        session_id=session_id,
        user_id=user_id,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(task)
    await db.flush()
    return task


async def get_a2a_task(db: AsyncSession, task_id: str) -> Optional[A2ATask]:
    """
    Retrieve an A2A task by its public UUID.
    """
    result = await db.execute(select(A2ATask).where(A2ATask.task_id == task_id))
    return result.scalar_one_or_none()


async def update_task_status(
    db: AsyncSession,
    task_id: str,
    status: str,
    result: Optional[str] = None,
    error: Optional[str] = None,
) -> A2ATask:
    """
    Update the status, result, or error of an A2A task.
    """
    logger.info("update_task_status", task_id=task_id, status=status, has_result=result is not None, has_error=error is not None)
    db_task = await get_a2a_task(db, task_id)
    if not db_task:
        raise ValueError(f"Task with ID {task_id} not found.")

    db_task.status = status
    db_task.updated_at = datetime.now(timezone.utc)
    if result is not None:
        db_task.result = result
    if error is not None:
        db_task.error = error

    db.add(db_task)
    await db.commit()
    return db_task


async def process_a2a_task_background(
    task_id: str,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """
    Background worker that runs the multi-agent supervisor flow for the task.
    """
    logger.info("a2a_background_worker_start", task_id=task_id)

    # 1. Update task to 'working'
    async with session_factory() as db:
        try:
            task = await update_task_status(db, task_id, "working")
            input_message = task.input_message
            session_id = task.session_id
            user_id = task.user_id
        except Exception as e:
            logger.error("a2a_worker_init_failed", task_id=task_id, error=str(e), exc_info=True)
            return

    # 2. Run supervisor flow
    try:
        from app.mcp.client import mcp_tools_client
        from app.agents.supervisor import build_supervisor, run_supervisor
        from app.agents.state import create_initial_state
        from langchain_core.messages import SystemMessage, HumanMessage
        from app.memory.short_term import load_short_term_memory
        from app.memory.long_term import get_memory_entries
        from app.memory.semantic import search_semantic_memory
        from app.rag.qdrant_client import get_qdrant_client, get_retriever
        from app.rag.embedder import get_embedder
        from app.agents.memory_agent import extract_and_consolidate_memory

        qclient = get_qdrant_client()
        embedder = get_embedder()
        retriever = get_retriever()

        async with session_factory() as db:
            # Load short-term history, long-term facts, and semantic memory
            past_messages = await load_short_term_memory(db=db, session_id=session_id, limit=10)
            long_term_entries = await get_memory_entries(db=db, user_id=user_id)
            semantic_entries = await search_semantic_memory(
                qdrant_client=qclient,
                embedder=embedder,
                db=db,
                user_id=user_id,
                query=input_message,
                limit=3,
            )

            # Build System prompt formatting memory
            parts = []
            if long_term_entries:
                parts.append("### User Preferences & Configuration Facts (Long-Term Memory)")
                for entry in long_term_entries:
                    parts.append(f"- [{entry.memory_type.upper()}] {entry.key}: {entry.value}")
            if semantic_entries:
                parts.append("### Relevant Past Interactions (Semantic Memory)")
                for entry in semantic_entries:
                    parts.append(
                        f"- User asked: \"{entry['query']}\"\n"
                        f"  Resolution was: \"{entry['response']}\""
                    )
            memory_text = ""
            if parts:
                memory_text = (
                    "You have access to the following persistent memory context about the user "
                    "and environment. Use this context to personalize your responses, respect preferences, "
                    "and leverage past resolutions.\n\n" + "\n".join(parts)
                )

            initial_state = create_initial_state(
                user_message=input_message,
                session_id=session_id,
                user_id=user_id,
            )

            initial_messages = []
            if memory_text:
                initial_messages.append(SystemMessage(content=memory_text))
            if past_messages:
                initial_messages.extend(past_messages)
            initial_messages.append(HumanMessage(content=input_message))

            initial_state["messages"] = initial_messages
            initial_state["memory_context"] = {
                "long_term": [
                    {"memory_type": e.memory_type, "key": e.key, "value": e.value}
                    for e in long_term_entries
                ],
                "semantic": semantic_entries,
            }

            # Run supervisor inside MCP tools context manager
            async with mcp_tools_client(session_id=session_id) as mcp_tools:
                supervisor = build_supervisor(
                    db=db,
                    retriever=retriever,
                    session_id=session_id,
                    tools=mcp_tools,
                )
                result_state = await run_supervisor(supervisor=supervisor, state=initial_state)

            final_response = result_state.get("final_response", "")
            error_msg = result_state.get("error")
            needs_approval = result_state.get("needs_approval", False)
            current_agent = result_state.get("current_agent", "rag_agent")

            # Persist chat messages to SQLite for the thread
            # Upsert Chat Session
            session_result = await db.execute(
                select(ChatSession).where(ChatSession.session_id == session_id)
            )
            chat_session = session_result.scalar_one_or_none()
            now = datetime.utcnow()
            if not chat_session:
                chat_session = ChatSession(
                    session_id=session_id,
                    user_id=user_id,
                    started_at=now,
                    last_active=now,
                )
                db.add(chat_session)
                await db.flush()
            else:
                chat_session.last_active = now

            db.add(ChatMessage(
                session_id=chat_session.id,
                role="user",
                content=input_message,
                agent_name=None,
                timestamp=now,
            ))
            db.add(ChatMessage(
                session_id=chat_session.id,
                role="assistant",
                content=final_response,
                agent_name=current_agent,
                timestamp=now,
            ))

            # Consolidate memory
            await extract_and_consolidate_memory(
                db=db,
                qdrant_client=qclient,
                embedder=embedder,
                user_id=user_id,
                session_id=session_id,
                user_message=input_message,
                assistant_response=final_response,
            )
            await db.commit()

        # Update A2A task status based on supervisor output
        async with session_factory() as db:
            if error_msg:
                await update_task_status(db, task_id, "failed", error=error_msg)
            elif needs_approval:
                await update_task_status(db, task_id, "input-required", result=final_response)
            else:
                await update_task_status(db, task_id, "completed", result=final_response)

    except Exception as e:
        logger.error("a2a_worker_execution_failed", task_id=task_id, error=str(e), exc_info=True)
        async with session_factory() as db:
            try:
                await update_task_status(db, task_id, "failed", error=str(e))
            except Exception as inner_e:
                logger.error("a2a_worker_failed_to_update_error_status", task_id=task_id, error=str(inner_e))
