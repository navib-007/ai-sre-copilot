"""
schemas/chat.py — Chat Request & Response Schemas
==================================================
CONCEPT: Request/Response Schema Separation

  We define separate schemas for:
    - Request (what comes IN from the client)
    - Response (what goes OUT to the client)

  This separation is important because:
    1. The request contains only what the user provides
    2. The response enriches the data with server-computed fields
    3. It's clear what the client must send vs what it receives back
    4. Pydantic validates both directions independently

CONCEPT: Session-Based Chat Architecture
  Each conversation belongs to a "session" — a UUID that groups
  messages together. This enables:
    - Conversation history (agent remembers previous turns)
    - Multi-turn dialogue (user can refer back to earlier messages)
    - Concurrent conversations (user can have multiple chat windows)

  In Phase 5 (Memory), we'll use session_id to load conversation
  history from SQLite before invoking the agent.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ─── Chat Request ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Incoming chat message from the client (Streamlit UI or API consumer).

    The client must provide at minimum the `message` field.
    All other fields have sensible defaults.
    """

    message: str = Field(
        min_length=1,
        max_length=4000,
        description="The user's message or question to the AI assistant.",
        examples=["How do I fix a CrashLoopBackOff in Kubernetes?"],
    )

    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "Chat session UUID. Send the same session_id to maintain "
            "conversation history across multiple messages. "
            "If not provided, a new session is automatically created."
        ),
    )

    # Optional: override which agent handles this (for debugging/testing)
    # In Phase 4 (Supervisor), the agent is auto-selected based on intent.
    # For Phase 3, this is always "rag_agent".
    agent_override: str | None = Field(
        default=None,
        description=(
            "Optional: Force routing to a specific agent. "
            "Values: 'rag_agent', 'ticket_agent', 'incident_agent'. "
            "Leave null for automatic intent-based routing (default)."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "How do I restart a Kubernetes pod that's stuck in CrashLoopBackOff?",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }


# ─── Source Citation ──────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    """
    A single document source cited in the agent's response.

    Used to show the user WHERE the information came from,
    building trust and allowing them to read the full document.
    """

    source: str = Field(description="Filename of the source document.")
    score: float = Field(description="Relevance score (0.0-1.0).")
    chunk_index: int = Field(description="Which chunk of the document was used.")
    text_preview: str = Field(description="First 200 chars of the retrieved text.")


# ─── Chat Response ────────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    """
    Response returned after the agent processes a chat message.

    Includes the agent's answer plus metadata about HOW it was generated
    (which agent, what sources, how long it took).

    This transparency is important for:
      - Debugging agent behavior
      - Building user trust (show your work)
      - Evaluating RAG quality (inspect source relevance)
    """

    # Core response
    message: str = Field(
        description="The agent's response to the user's message."
    )
    session_id: str = Field(
        description="The session ID for this conversation (echo back to client)."
    )

    # Agent metadata
    agent_used: str = Field(
        description="Which agent generated this response.",
        examples=["rag_agent", "ticket_agent", "supervisor"],
    )
    intent_detected: str = Field(
        description="The intent the supervisor detected in the user's message.",
        examples=["knowledge_query", "ticket_operation", "incident_investigation", "general_chat"],
    )

    # RAG-specific: sources used to generate the answer
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description=(
            "Document sources cited in the response. "
            "Empty list if no documents were retrieved."
        ),
    )

    # Performance metadata
    processing_time_ms: float = Field(
        description="Total time to process this request (in milliseconds)."
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the response was generated.",
    )

    # Error info (if something went wrong but we still returned a response)
    error: str | None = Field(
        default=None,
        description="Error message if the agent encountered an issue. Null on success.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "To fix a CrashLoopBackOff, first check the pod logs...",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "agent_used": "rag_agent",
                "intent_detected": "knowledge_query",
                "sources": [
                    {
                        "source": "kubernetes_troubleshooting.md",
                        "score": 0.87,
                        "chunk_index": 3,
                        "text_preview": "CrashLoopBackOff indicates the container...",
                    }
                ],
                "processing_time_ms": 1234.56,
                "timestamp": "2026-01-01T12:00:00",
                "error": None,
            }
        }


# ─── Session History ──────────────────────────────────────────────────────────

class ChatMessageRecord(BaseModel):
    """
    A single message record for session history retrieval.
    Used by GET /api/chat/history/{session_id}
    """

    role: str = Field(description="Message role: 'user' or 'assistant'.")
    content: str = Field(description="Message content.")
    agent_name: str | None = Field(default=None, description="Which agent produced this (for assistant messages).")
    timestamp: datetime = Field(description="When this message was sent.")


class ChatHistoryResponse(BaseModel):
    """Response for session history endpoint."""

    session_id: str
    messages: list[ChatMessageRecord]
    total_messages: int

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "messages": [
                    {
                        "role": "user",
                        "content": "How do I fix CrashLoopBackOff?",
                        "agent_name": None,
                        "timestamp": "2026-01-01T12:00:00",
                    },
                    {
                        "role": "assistant",
                        "content": "To fix CrashLoopBackOff...",
                        "agent_name": "rag_agent",
                        "timestamp": "2026-01-01T12:00:01",
                    },
                ],
                "total_messages": 2,
            }
        }
