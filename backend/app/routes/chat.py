"""
routes/chat.py — Chat API Endpoint
====================================
CONCEPT: The Chat Endpoint as Agent Orchestrator

  This route is the main entry point for AI interactions.
  It acts as the bridge between the HTTP world (FastAPI) and
  the agent world (LangGraph).

  What happens on POST /api/chat:
    1. Validate the request (Pydantic)
    2. Load request-scoped resources (DB session, Qdrant client, retriever)
    3. Build the agent with those resources
    4. Create initial agent state from the request
    5. Invoke the agent and wait for a response
    6. Persist the conversation to SQLite
    7. Extract source citations from retrieved docs
    8. Return the formatted response

CONCEPT: Dependency Injection in FastAPI
  FastAPI's `Depends()` system creates request-scoped resources.
  Each request gets its own:
    - `db`: AsyncSession (fresh connection per request)
    - `qdrant_client`: AsyncQdrantClient (shared, injected per request)

  When the request finishes, FastAPI automatically calls cleanup
  (closes DB session, etc.) via the `yield` in get_db().

CONCEPT: Request Timing
  We measure total processing time with `time.perf_counter()`.
  This is important for:
    - Identifying slow requests (LLM latency, DB latency)
    - SLA monitoring (is the API within acceptable response time?)
    - LangSmith comparison (does tracing add overhead?)

CONCEPT: Chat History Persistence
  After the agent responds, we persist the exchange to SQLite:
    - User message → ChatMessage(role="user")
    - Assistant response → ChatMessage(role="assistant")
  This enables:
    - Conversation history (load previous messages in Phase 5)
    - Audit trail (who asked what, when)
    - Analytics (common questions, agent performance)
"""

import time
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.database import get_db
from app.logging_config import get_logger
from app.schemas.chat import ChatHistoryResponse, ChatRequest, ChatResponse, SourceCitation

settings = get_settings()
logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat — AI Agent"])


# ─── Dependency: Retriever ────────────────────────────────────────────────────

def get_retriever_dep():
    """
    Dependency that provides a configured RAGRetriever.

    Used by the supervisor to build the Incident Agent and RAG Agent.
    The same singleton retriever is reused across all specialist agents.
    """
    from app.rag.qdrant_client import get_retriever
    return get_retriever()


# ─── POST /api/chat ───────────────────────────────────────────────────────────

@router.post("/",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the AI agent",
    description=(
        "Submit a user message and get an AI-generated response. "
        "The agent searches the knowledge base and returns a grounded answer "
        "with source citations. "
        "Pass the same `session_id` across requests to maintain conversation context."
    ),)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    retriever=Depends(get_retriever_dep),) -> ChatResponse:
    """
    Main chat endpoint — routes through the Phase 4 Multi-Agent Supervisor.

    Flow:
      1. Supervisor detects intent (knowledge_query / ticket_operation / incident_investigation / general_chat)
      2. Routes to the matching specialist agent
      3. Specialist runs its ReAct loop with appropriate tools
      4. Response is persisted and returned with source citations
    """
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]

    logger.info(
        "chat_request_received",
        request_id=request_id,
        session_id=request.session_id,
        message_preview=request.message[:80],
        agent_override=request.agent_override,
    )

    try:
        # ── Step 1: Build the Supervisor ──────────────────────────────────────
        # CONCEPT: Per-request supervisor build
        #   We build the supervisor fresh per request to bind the request-scoped
        #   DB session. Each specialist agent inside the supervisor shares this
        #   DB session, ensuring all DB operations are in the same transaction.
        from app.agents.supervisor import build_supervisor, run_supervisor
        from app.agents.state import create_initial_state

        supervisor = build_supervisor(db=db, retriever=retriever, session_id=request.session_id)

        # ── Step 1b: Load Memory Context (Phase 5) ────────────────────────────
        from langchain_core.messages import SystemMessage, HumanMessage
        from app.memory.short_term import load_short_term_memory
        from app.memory.long_term import get_memory_entries
        from app.memory.semantic import search_semantic_memory
        from app.rag.qdrant_client import get_qdrant_client
        from app.rag.embedder import get_embedder

        qclient = get_qdrant_client()
        embedder = get_embedder()

        # Load short-term history, long-term facts, and semantically similar past Q&A
        past_messages = await load_short_term_memory(db=db, session_id=request.session_id, limit=10)
        long_term_entries = await get_memory_entries(db=db, user_id=1)
        semantic_entries = await search_semantic_memory(
            qdrant_client=qclient,
            embedder=embedder,
            db=db,
            user_id=1,
            query=request.message,
            limit=3,
        )

        # ── Step 2: Create initial state ──────────────────────────────────────
        initial_state = create_initial_state(
            user_message=request.message,
            session_id=request.session_id,
            user_id=1,  # Real user ID added in Phase 9 (JWT auth)
        )

        # Format memories and prepend to conversation messages
        memory_text = _format_memory_context(long_term_entries, semantic_entries)
        initial_messages = []
        if memory_text:
            initial_messages.append(SystemMessage(content=memory_text))
        if past_messages:
            initial_messages.extend(past_messages)
        initial_messages.append(HumanMessage(content=request.message))

        initial_state["messages"] = initial_messages
        initial_state["memory_context"] = {
            "long_term": [
                {"memory_type": e.memory_type, "key": e.key, "value": e.value}
                for e in long_term_entries
            ],
            "semantic": semantic_entries,
        }

        # ── Step 3: Run through Supervisor ────────────────────────────────────
        result_state = await run_supervisor(supervisor=supervisor, state=initial_state)

        # ── Step 4: Persist conversation to SQLite ────────────────────────────
        await _persist_chat_messages(
            db=db,
            session_id=request.session_id,
            user_message=request.message,
            assistant_response=result_state.get("final_response", ""),
            agent_name=result_state.get("current_agent", "rag_agent"),
        )

        # ── Step 4b: Consolidate Memory (Extract facts & save semantic QA) ─────
        from app.agents.memory_agent import extract_and_consolidate_memory
        await extract_and_consolidate_memory(
            db=db,
            qdrant_client=qclient,
            embedder=embedder,
            user_id=1,
            session_id=request.session_id,
            user_message=request.message,
            assistant_response=result_state.get("final_response", ""),
        )
        await db.commit()

        # ── Step 5: Extract source citations ──────────────────────────────────
        sources = _extract_sources(result_state)

        # ── Step 6: Calculate processing time ─────────────────────────────────
        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "chat_request_complete",
            request_id=request_id,
            session_id=request.session_id,
            agent_used=result_state.get("current_agent", "rag_agent"),
            processing_time_ms=processing_time_ms,
            response_length=len(result_state.get("final_response", "")),
            sources_count=len(sources),
        )

        return ChatResponse(
            message=result_state.get("final_response", "No response generated."),
            session_id=request.session_id,
            agent_used=result_state.get("current_agent", "rag_agent"),
            intent_detected=result_state.get("intent", "knowledge_query"),
            sources=sources,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow(),
            error=result_state.get("error"),
        )

    except Exception as e:
        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(
            "chat_request_failed",
            request_id=request_id,
            session_id=request.session_id,
            error=str(e),
            processing_time_ms=processing_time_ms,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Agent processing failed",
                "message": "The AI agent encountered an error. Please try again.",
                "request_id": request_id,
            },
        )


# ─── GET /api/chat/history/{session_id} ──────────────────────────────────────

@router.get(
    "/history/{session_id}",
    response_model=ChatHistoryResponse,
    summary="Get chat history for a session",
    description="Retrieve all messages for a given session ID, ordered chronologically.",
)
async def get_chat_history(
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> ChatHistoryResponse:
    """
    Retrieve the conversation history for a session.

    CONCEPT: Session-based history
      Each session is a logical container for a conversation thread.
      This endpoint lets the frontend reload past messages when the
      user navigates back to a previous chat.
    """
    from sqlalchemy import select, asc
    from app.db.models import ChatMessage, ChatSession

    logger.info("chat_history_requested", session_id=session_id)

    # Find the session
    session_result = await db.execute(
        select(ChatSession).where(ChatSession.session_id == session_id)
    )
    session = session_result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chat session found with ID: {session_id}",
        )

    # Get all messages for this session, ordered by timestamp
    messages_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(asc(ChatMessage.timestamp))
    )
    messages = messages_result.scalars().all()

    from app.schemas.chat import ChatMessageRecord
    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            ChatMessageRecord(
                role=msg.role,
                content=msg.content,
                agent_name=msg.agent_name,
                timestamp=msg.timestamp,
            )
            for msg in messages
        ],
        total_messages=len(messages),
    )


# ─── GET /api/chat/sessions ───────────────────────────────────────────────────

@router.get(
    "/sessions",
    summary="List all chat sessions",
    description="Returns a list of all chat sessions with their last activity timestamps.",
)
async def list_sessions(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """List recent chat sessions."""
    from sqlalchemy import select, desc
    from app.db.models import ChatSession

    result = await db.execute(
        select(ChatSession)
        .order_by(desc(ChatSession.last_active))
        .limit(limit)
    )
    sessions = result.scalars().all()

    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "started_at": s.started_at,
                "last_active": s.last_active,
            }
            for s in sessions
        ],
        "total": len(sessions),
    }


# ─── Private Helpers ──────────────────────────────────────────────────────────

async def _persist_chat_messages(
    db: AsyncSession,
    session_id: str,
    user_message: str,
    assistant_response: str,
    agent_name: str,) -> None:
    """
    Save user message and assistant response to SQLite.

    CONCEPT: Upsert Chat Session
      If the session doesn't exist yet, create it.
      If it does, update last_active timestamp.
      This is an "upsert" pattern (insert or update).
    """
    from sqlalchemy import select
    from app.db.models import ChatMessage, ChatSession

    try:
        # ── Find or create session ─────────────────────────────────────────────
        session_result = await db.execute(
            select(ChatSession).where(ChatSession.session_id == session_id)
        )
        session = session_result.scalar_one_or_none()

        now = datetime.utcnow()

        if not session:
            # Create new session (user_id=1 until Phase 9 auth)
            session = ChatSession(
                session_id=session_id,
                user_id=1,
                started_at=now,
                last_active=now,
            )
            db.add(session)
            await db.flush()  # Get session.id without committing
        else:
            session.last_active = now

        # ── Save user message ──────────────────────────────────────────────────
        db.add(ChatMessage(
            session_id=session.id,
            role="user",
            content=user_message,
            agent_name=None,
            timestamp=now,
        ))

        # ── Save assistant response ────────────────────────────────────────────
        db.add(ChatMessage(
            session_id=session.id,
            role="assistant",
            content=assistant_response,
            agent_name=agent_name,
            timestamp=now,
        ))

        await db.commit()
        logger.info(
            "chat_messages_persisted",
            session_id=session_id,
            session_db_id=session.id,
        )

    except Exception as e:
        logger.error(
            "chat_messages_persist_failed",
            session_id=session_id,
            error=str(e),
            exc_info=True,
        )
        await db.rollback()
        # Don't raise — a persistence failure shouldn't break the user's response


def _extract_sources(state: dict) -> list[SourceCitation]:
    """
    Extract source citations from the agent state.

    In Phase 3, we parse the retrieved_documents list.
    In Phase 4+, specialist agents will populate this list directly.

    CONCEPT: Source Attribution
      Showing where information came from is critical for:
        1. Trust: users can verify AI answers against source docs
        2. Debugging: identify if wrong/outdated docs are being retrieved
        3. RAG evaluation: measure retrieval quality
    """
    sources = []
    retrieved_docs = state.get("retrieved_documents", [])

    for doc in retrieved_docs:
        try:
            sources.append(SourceCitation(
                source=doc.get("source", "unknown"),
                score=round(doc.get("score", 0.0), 4),
                chunk_index=doc.get("chunk_index", 0),
                text_preview=doc.get("text", "")[:200],
            ))
        except Exception:
            continue  # Skip malformed source entries

    return sources


def _format_memory_context(long_term_entries, semantic_entries) -> str:
    """
    Format retrieved long-term and semantic memories into a structured system instructions block.
    """
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
    if parts:
        return (
            "You have access to the following persistent memory context about the user "
            "and environment. Use this context to personalize your responses, respect preferences, "
            "and leverage past resolutions.\n\n" + "\n".join(parts)
        )
    return ""

