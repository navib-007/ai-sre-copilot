"""
app/memory/short_term.py — Working/Episodic Memory (Short-Term Conversation History)
=============================================================================
CONCEPT: Short-Term Conversation Buffer
  To maintain flow and coherence in a conversation, the agent needs to know
  what has been said in the immediate past (within the current session).

  Instead of keeping the conversation history only in the client or relying
  solely on LangGraph checkpointers, we retrieve the last N messages for
  the current `session_id` from the SQLite `chat_messages` table on every
  request and inject them into the agent's initial state.
"""

from sqlalchemy import select, asc
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.db.models import ChatMessage, ChatSession
from app.logging_config import get_logger

logger = get_logger(__name__)


async def load_short_term_memory(
    db: AsyncSession,
    session_id: str,
    limit: int = 10,
) -> list[BaseMessage]:
    """
    Retrieve the last `limit` messages for a session from SQLite,
    ordered chronologically, and map them to LangChain Message objects.

    Args:
        db:         AsyncSession bound to the current request
        session_id: UUID of the chat session
        limit:      Maximum number of past messages to load (e.g. 10 messages = 5 rounds)

    Returns:
        List of LangChain messages (HumanMessage or AIMessage)
    """
    logger.info("loading_short_term_memory", session_id=session_id, limit=limit)

    try:
        # Step 1: Find the ChatSession ID
        session_result = await db.execute(
            select(ChatSession).where(ChatSession.session_id == session_id)
        )
        session = session_result.scalar_one_or_none()

        if not session:
            logger.info("short_term_memory_not_found", session_id=session_id)
            return []

        # Step 2: Query messages for this session ordered chronologically
        messages_result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session.id)
            .order_by(asc(ChatMessage.timestamp))
        )
        db_messages = messages_result.scalars().all()

        # Step 3: Slice to get the last N messages
        recent_db_messages = db_messages[-limit:] if len(db_messages) > limit else db_messages

        # Step 4: Map to LangChain Message objects
        langchain_messages = []
        for msg in recent_db_messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                # If agent_name is present, attach it to help differentiate agents
                langchain_messages.append(
                    AIMessage(
                        content=msg.content,
                        name=msg.agent_name if msg.agent_name else "assistant",
                    )
                )

        logger.info(
            "short_term_memory_loaded",
            session_id=session_id,
            total_db_messages=len(db_messages),
            retrieved_messages=len(langchain_messages),
        )
        return langchain_messages

    except Exception as e:
        logger.error(
            "short_term_memory_load_failed",
            session_id=session_id,
            error=str(e),
            exc_info=True,
        )
        return []
