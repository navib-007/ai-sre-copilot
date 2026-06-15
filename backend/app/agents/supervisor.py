"""
agents/supervisor.py — Multi-Agent Supervisor with Intent Routing
=================================================================
CONCEPT: The Supervisor Pattern (Orchestration Function)

  Phase 4 implements the Supervisor Pattern as a plain async function,
  NOT as a LangGraph StateGraph. Here's why:

  WHY NOT A CUSTOM StateGraph FOR THE SUPERVISOR?
    Each specialist agent (rag_agent, ticket_agent, etc.) is already
    built with `create_react_agent`, which internally creates its own
    MessagesState + StateGraph. When you nest a StateGraph (the specialist)
    inside another StateGraph (the supervisor), LangGraph enforces schema
    compatibility between them — and `StateGraph(dict)` vs `MessagesState`
    causes the error: "Must write to at least one of ['messages']"

  THE CLEAN SOLUTION: Supervisor as an Orchestration Function
    Instead of wrapping everything in a StateGraph:
      1. Supervisor = async Python function
      2. Intent detection = direct LLM call (fast, no graph overhead)
      3. Dispatch = simple if/elif routing
      4. Specialist agents = their own LangGraph graphs (invoked directly)

  THIS IS STILL THE SUPERVISOR PATTERN:
    User → Supervisor (detects intent) → Specialist Agent (does the work)
    The difference is the supervisor uses Python control flow instead of
    LangGraph edges for routing. Same architecture, fewer complications.

CONCEPT: Intent Detection via Structured JSON Output
  We ask the LLM: "What does the user want?" and expect a JSON response:
    {"intent": "knowledge_query", "confidence": 0.95, "reason": "..."}

  Why JSON instead of free-form?
    - Deterministic: easier to parse and validate
    - Reliable: LLMs are good at producing valid JSON when asked
    - Fallback-safe: if parsing fails, default to knowledge_query (safest)

CONCEPT: Graceful Degradation
  If intent detection fails (LLM down, JSON parse error), we default to
  'knowledge_query' which calls the RAG agent. This is the safest fallback:
    - No side effects (no DB writes)
    - The RAG agent can handle most technical questions
    - Better than returning an error to the user
"""

import json
from typing import Optional, List, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ─── Intent Classification Prompt ────────────────────────────────────────────

INTENT_DETECTION_PROMPT = """You are an intent classification system for an IT Operations AI platform.

Classify the user's message into exactly ONE of these intents:

1. **knowledge_query** — User is asking a question about IT procedures, runbooks,
   troubleshooting steps, or any technical "how-to".
   Examples: "How do I restart a Kubernetes pod?", "What is the database backup schedule?"

2. **ticket_operation** — User wants to create, update, search, or query support tickets.
   Examples: "Create a ticket for the Redis issue", "What tickets are open?",
   "Update TKT-42 to resolved", "Show me high-priority tickets"

3. **incident_investigation** — User is reporting an active outage, performance issue,
   production incident that needs investigation, or is instructing the agent to execute/proceed
   with an approved remediation action.
   Examples: "The [service-name] is returning 500 errors", "P1 incident: [resource] is down",
   "I have approved Request ID [ID]. Go ahead and execute the action.", "Proceed with the approved action"

4. **general_chat** — Greetings, help requests, or questions not fitting above categories.
   Examples: "Hello", "What can you do?", "Help", "Thanks"

Respond with ONLY a JSON object, no other text:
{"intent": "<one of the four intents>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}
"""

# Direct response prompt (for general_chat)
SUPERVISOR_DIRECT_PROMPT = """You are an AI Operations Assistant for {app_name}.

You help IT operations teams with:
- 📚 Knowledge queries (runbooks, procedures, troubleshooting)
- 🎫 Ticket management (create, update, search tickets)
- 🔥 Incident investigation (outage analysis, RCA)

Respond helpfully and concisely. For greetings, introduce yourself and your capabilities.

Platform: {app_name} v{app_version}
"""


# ─── Intent Detection ─────────────────────────────────────────────────────────

async def _detect_intent(user_message: str, llm: ChatOpenAI) -> str:
    """
    Classify the user's message intent using an LLM JSON response.

    Returns one of: 'knowledge_query', 'ticket_operation',
                    'incident_investigation', 'general_chat'

    Falls back to 'knowledge_query' on any error (safest default).
    """
    logger.info("supervisor_intent_detection", message_preview=user_message[:80])

    try:
        messages = [
            SystemMessage(content=INTENT_DETECTION_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = await llm.ainvoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if the LLM wraps its output
        if "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        parsed = json.loads(raw)
        intent = parsed.get("intent", "knowledge_query")
        confidence = parsed.get("confidence", 0.0)
        reason = parsed.get("reason", "")

        valid = {"knowledge_query", "ticket_operation", "incident_investigation", "general_chat"}
        if intent not in valid:
            intent = "knowledge_query"

        # If confidence is below threshold, default to safe fallback
        if confidence < settings.supervisor_intent_confidence_threshold:
            logger.info(
                "supervisor_low_confidence_fallback",
                intent=intent,
                confidence=confidence,
                threshold=settings.supervisor_intent_confidence_threshold,
            )
            intent = "knowledge_query"

        logger.info(
            "supervisor_intent_detected",
            intent=intent,
            confidence=confidence,
            reason=reason[:100],
        )
        return intent

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(
            "supervisor_intent_detection_fallback",
            error=str(e),
        )
        return "knowledge_query"   # Safest default


# ─── Direct Response (for general_chat) ─────────────────────────────────────

async def _direct_response(user_message: str, state: dict, llm: ChatOpenAI) -> dict:
    """Generate a direct LLM response for general/greeting messages."""
    system_prompt = SUPERVISOR_DIRECT_PROMPT.format(
        app_name=settings.app_name,
        app_version=settings.app_version,
    )
    try:
        messages = [SystemMessage(content=system_prompt)] + state.get("messages", [])
        response = await llm.ainvoke(messages)
        final_response = response.content
        logger.info("supervisor_direct_response", length=len(final_response))
        return {
            **state,
            "messages": [*state.get("messages", []), AIMessage(content=final_response)],
            "final_response": final_response,
            "current_agent": "direct_response",
            "error": None,
        }
    except Exception as e:
        logger.error("supervisor_direct_response_failed", error=str(e))
        return {
            **state,
            "final_response": (
                f"Hello! I'm the {settings.app_name} AI assistant. I can help with:\n"
                "- 📚 IT knowledge queries (runbooks, procedures)\n"
                "- 🎫 Ticket management (create, update, search)\n"
                "- 🔥 Incident investigation (outage analysis)\n\n"
                "How can I help you today?"
            ),
            "current_agent": "direct_response",
            "error": None,
        }


# ─── Supervisor Builder ───────────────────────────────────────────────────────

def build_supervisor(db, retriever, session_id: str, tools: Optional[List[Any]] = None):
    """
    Build the supervisor as a pre-configured async callable.

    CONCEPT: Closure-Based Supervisor
      Instead of returning a LangGraph compiled graph, we return a simple
      async function (closure) that captures `db` and `retriever`.

      This avoids all LangGraph state schema conflicts while keeping
      the same interface: `await run_supervisor(supervisor, state)`

    Args:
        db:         AsyncSession bound to this request
        retriever:  RAGRetriever singleton
        session_id: The session ID of the current chat thread
        tools:      Optional pre-constructed list of tools (e.g. MCP client tools)

    Returns:
        An async callable with signature: (state: dict) -> dict
    """
    from app.agents.rag_agent import build_rag_agent, run_rag_agent
    from app.agents.ticket_agent import build_ticket_agent, run_ticket_agent
    from app.agents.incident_agent import build_incident_agent, run_incident_agent

    # ── Shared LLM for intent classification (low token budget) ───────────────
    intent_llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,          # Fully deterministic for routing
        api_key=settings.openai_api_key,
        max_tokens=128,           # Intent JSON is tiny
    )

    # ── Pre-build specialist agents ────────────────────────────────────────────
    # Built once per request (not once per agent call) for efficiency.
    # Each agent captures the same `db` and `retriever` from this closure.
    rag_agent     = build_rag_agent(retriever=retriever, db=db, tools=tools)
    ticket_agent  = build_ticket_agent(db=db, tools=tools) if settings.enable_ticket_agent else None
    incident_agent = build_incident_agent(db=db, retriever=retriever, session_id=session_id, tools=tools) if settings.enable_incident_agent else None

    logger.info(
        "supervisor_built",
        ticket_agent_enabled=settings.enable_ticket_agent,
        incident_agent_enabled=settings.enable_incident_agent,
    )

    # ── Supervisor dispatch function ───────────────────────────────────────────
    async def _dispatch(state: dict) -> dict:
        """
        Core supervisor logic:
          1. Extract the user's message from state
          2. Classify intent via LLM
          3. Route to the right specialist agent
          4. Return enriched state with final_response
        """
        # Extract the latest user message from state
        messages = state.get("messages", [])
        user_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        if not user_message:
            return {
                **state,
                "final_response": "I didn't receive a message. Please try again.",
                "current_agent": "supervisor",
                "intent": "general_chat",
                "error": None,
            }

        # Step 1: Detect intent
        intent = await _detect_intent(user_message, intent_llm)

        # Store intent in state so it's available in the response
        state_with_intent = {**state, "intent": intent}

        logger.info(
            "supervisor_routing",
            intent=intent,
            session_id=state.get("session_id", "unknown"),
        )

        # Step 2: Route to specialist agent
        if intent == "knowledge_query":
            return await run_rag_agent(agent=rag_agent, state=state_with_intent)

        elif intent == "ticket_operation":
            if ticket_agent and settings.enable_ticket_agent:
                return await run_ticket_agent(agent=ticket_agent, state=state_with_intent)
            else:
                # Fallback to RAG if ticket agent is disabled
                logger.info("supervisor_ticket_agent_disabled_fallback")
                return await run_rag_agent(agent=rag_agent, state=state_with_intent)

        elif intent == "incident_investigation":
            if incident_agent and settings.enable_incident_agent:
                return await run_incident_agent(agent=incident_agent, state=state_with_intent)
            else:
                # Fallback to RAG if incident agent is disabled
                logger.info("supervisor_incident_agent_disabled_fallback")
                return await run_rag_agent(agent=rag_agent, state=state_with_intent)

        else:
            # general_chat — direct LLM response, no tools
            return await _direct_response(user_message, state_with_intent, intent_llm)

    return _dispatch


# ─── Supervisor Runner ────────────────────────────────────────────────────────

async def run_supervisor(supervisor, state: dict) -> dict:
    """
    Execute the supervisor dispatch function.

    Maintains the same interface as the old LangGraph-based supervisor
    so the chat route doesn't need any changes.

    Args:
        supervisor: The async dispatch function returned by build_supervisor()
        state:      Initial AgentState dict

    Returns:
        Final state dict with final_response populated.
    """
    logger.info(
        "supervisor_invocation_started",
        session_id=state.get("session_id", "unknown"),
    )

    try:
        result_state = await supervisor(state)

        logger.info(
            "supervisor_invocation_complete",
            session_id=state.get("session_id", "unknown"),
            intent=result_state.get("intent", "unknown"),
            agent_used=result_state.get("current_agent", "unknown"),
        )
        return result_state

    except Exception as e:
        logger.error(
            "supervisor_invocation_failed",
            session_id=state.get("session_id", "unknown"),
            error=str(e),
            exc_info=True,
        )
        return {
            **state,
            "final_response": (
                "I encountered an error processing your request. "
                "Please try again or contact support."
            ),
            "current_agent": "supervisor",
            "error": str(e),
        }
