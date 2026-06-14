"""
agents/state.py — Shared Agent State Definition
================================================
CONCEPT: LangGraph State Machine

  LangGraph models agent execution as a directed graph where:
    - NODES  = processing steps (supervisor, rag_agent, ticket_agent, etc.)
    - EDGES  = transitions between steps (conditional or direct)
    - STATE  = a shared data structure that flows through every node

  Every node receives the current state, does its work, and returns a
  PARTIAL state update (only the fields it changed). LangGraph merges
  these updates automatically.

  WHY TYPEDDICT?
    TypedDict gives us static typing (IDE autocomplete + mypy checks)
    without the overhead of a full Pydantic model. LangGraph uses
    type annotations to understand how state fields should be merged.

CONCEPT: Annotated Fields and Reducers

  The special `Annotated[list, add_messages]` syntax tells LangGraph:
    "When multiple nodes update this field, USE add_messages to merge them
    instead of overwriting."

  add_messages is a built-in reducer that appends new messages to the list.
  This is how conversation history accumulates correctly even when multiple
  agents contribute messages to the same thread.

  For all other fields (strings, ints, dicts), LangGraph uses LAST-WRITE-WINS:
  the most recent node's value replaces the previous one.

STATE DESIGN PRINCIPLE:
  Keep state minimal — only store what needs to persist across node transitions.
  Large data (like full document text) should be stored in databases and
  referenced by ID in the state.
"""

from typing import Annotated, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(dict):
    """
    Shared state that flows through every node in the LangGraph graph.

    This is the single source of truth for an ongoing agent execution.
    Each node reads from this state and returns partial updates.

    CONCEPT: Why inherit from dict instead of TypedDict?
      LangGraph works best with TypedDict or dict subclasses.
      We use dict subclass for flexibility — allows dynamic field access
      that TypedDict would reject at runtime.

    Field Groups:
      - CORE:     messages + session tracking
      - ROUTING:  intent detection + agent selection
      - CONTEXT:  user info + memory
      - RAG:      retrieved document chunks
      - INCIDENT: incident investigation data
      - APPROVAL: human-in-the-loop workflow data
      - OUTPUT:   final response to the user
    """

    # ── CORE: Conversation ────────────────────────────────────────────────────
    # CONCEPT: add_messages reducer
    #   When multiple nodes add messages, they're APPENDED (not overwritten).
    #   This is how the chat history grows correctly through the graph.
    messages: Annotated[list[BaseMessage], add_messages]

    # ── ROUTING: Intent & Agent Selection ────────────────────────────────────
    # Set by the supervisor, read by conditional edge functions
    intent: str                    # "knowledge_query" | "ticket_operation" | "incident_investigation" | "general_chat"
    current_agent: str             # Which agent is currently active

    # ── CONTEXT: User & Session ───────────────────────────────────────────────
    user_id: int                   # Authenticated user ID (from JWT in Phase 9)
    session_id: str                # Chat session UUID
    memory_context: dict           # Retrieved memories (short + long + semantic)

    # ── RAG: Document Retrieval ───────────────────────────────────────────────
    retrieved_documents: list[dict]  # List of SearchResult.to_dict() from retriever

    # ── INCIDENT: Investigation Data ──────────────────────────────────────────
    incident_id: int | None          # Active incident being investigated
    evidence: list[dict]             # Gathered evidence from tools
    root_cause: str | None           # RCA result
    recommended_action: dict | None  # Proposed remediation

    # ── APPROVAL: Human-in-the-Loop ───────────────────────────────────────────
    needs_approval: bool             # Whether HITL approval is required
    approval_status: str | None      # "pending" | "approved" | "rejected"

    # ── OUTPUT: Final Response ────────────────────────────────────────────────
    final_response: str              # The formatted response to send to the user
    error: str | None                # Error message if something went wrong


def create_initial_state(
    user_message: str,
    session_id: str,
    user_id: int = 1,
) -> dict[str, Any]:
    """
    Create the initial state for a new agent invocation.

    This is called at the start of every chat request. It sets up
    a clean state with the user's message and sensible defaults.

    CONCEPT: HumanMessage vs SystemMessage vs AIMessage
      LangChain uses typed message classes to distinguish speakers:
        - HumanMessage: user's input
        - AIMessage: model's response
        - SystemMessage: instructions to the LLM (not shown to user)
        - ToolMessage: result from a tool call

      These are stored in the `messages` list and passed to the LLM.
      The model uses them as conversation history for context.

    Args:
        user_message: The text the user typed
        session_id:   UUID of the chat session
        user_id:      ID of the authenticated user (default=1 until Phase 9)

    Returns:
        A dict matching the AgentState schema with defaults set.
    """
    from langchain_core.messages import HumanMessage

    return {
        # Start with the user's message as the first message in the graph
        "messages": [HumanMessage(content=user_message)],

        # Routing (set by supervisor in Phase 4; hardcoded to RAG for Phase 3)
        "intent": "knowledge_query",
        "current_agent": "rag_agent",

        # Context
        "user_id": user_id,
        "session_id": session_id,
        "memory_context": {},

        # RAG
        "retrieved_documents": [],

        # Incident (not used in Phase 3)
        "incident_id": None,
        "evidence": [],
        "root_cause": None,
        "recommended_action": None,

        # Approval (not used in Phase 3)
        "needs_approval": False,
        "approval_status": None,

        # Output
        "final_response": "",
        "error": None,
    }
