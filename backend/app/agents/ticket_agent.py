"""
agents/ticket_agent.py — Ticket Management Agent
=================================================
CONCEPT: Specialist Agents vs General Agents

  A GENERAL agent knows about everything but is mediocre at each task.
  A SPECIALIST agent is deeply focused on one domain and excels at it.

  The Ticket Agent specializes in:
    - Understanding ticket-related requests from natural language
    - Extracting structured fields (title, priority, category) from free text
    - Checking for duplicate tickets before creating new ones
    - Maintaining ticket lifecycle (open → in_progress → resolved)

  The Supervisor Agent (Phase 4) knows WHICH specialist to call.
  Each specialist knows HOW to do its specific job.

CONCEPT: Information Extraction as an Agent Task
  "Create a ticket: our Redis is timing out in the auth service after upgrade"

  The LLM must extract:
    title:    "Redis connection timeout in auth-service"
    priority: "medium" (implied — not explicitly stated)
    category: "database" (Redis is a database)
    description: full context from the message

  This is the agent's key skill — translating natural language intent
  into structured database operations.

CONCEPT: Duplicate Detection Strategy
  Before creating a new ticket, the agent:
    1. Searches for existing tickets matching the issue
    2. Presents the user with any matches found
    3. Creates a new ticket only if no match exists

  This prevents ticket sprawl — a common pain point in real IT ops.
"""

from typing import Optional, List, Any
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ─── System Prompt ────────────────────────────────────────────────────────────

TICKET_AGENT_SYSTEM_PROMPT = """You are a specialist Ticket Management Agent for {app_name}.

Your ONLY responsibility is managing support tickets.

## Your Capabilities
- Search existing tickets by keyword, status, or priority
- Create new support tickets from natural language descriptions
- Update ticket status, priority, and resolution
- Look up specific tickets by their ID

## Rules You Must Follow

### Always Check for Duplicates First
- Before creating ANY ticket, use search_tickets to find similar existing ones
- If a similar open ticket exists, report it to the user instead of creating a duplicate
- Only create a new ticket if no matching open ticket is found

### Accurate Field Extraction
Extract these fields from the user's message:
  - **title**: 5-10 words summarizing the issue
  - **description**: Full context including symptoms, impact, when it started
  - **priority**: one of low/medium/high/critical
    * critical: system down, data loss risk, revenue impact
    * high: significant degradation, many users affected
    * medium: partial degradation, workaround exists
    * low: minor issue, cosmetic, enhancement
  - **category**: database/kubernetes/networking/security/performance/deployment

### Priority Guidelines
- "production is down" → critical
- "high priority issue" → high
- "not urgent but..." → low
- When unsure → default to medium

### Response Format
After completing a ticket operation, summarize:
  1. What you did (created/found/updated)
  2. Ticket ID (e.g., TKT-42)
  3. Current status

Current Platform: {app_name} v{app_version}
"""


def build_ticket_agent(db, tools: Optional[List[Any]] = None):
    """
    Build the Ticket Management specialist agent.

    This agent has access to ticket CRUD tools and is focused
    exclusively on ticket-related operations.

    Args:
        db:    AsyncSession for database operations
        tools: Optional pre-constructed list of tools (e.g. MCP client tools)

    Returns:
        Compiled LangGraph ReAct agent graph.
    """
    # ── LLM ───────────────────────────────────────────────────────────────────
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.agent_temperature,
        api_key=settings.openai_api_key,
        max_tokens=settings.agent_max_tokens,
    )

    # ── Tools ─────────────────────────────────────────────────────────────────
    if tools is not None:
        # Filter dynamic tools for ticket agent specific needs
        allowed_names = {"search_tickets", "create_ticket", "update_ticket", "get_ticket", "save_user_preference", "save_environment_fact"}
        agent_tools = [t for t in tools if t.name in allowed_names]
    else:
        # Fall back to local tools builders
        from app.tools.ticket_tool import build_ticket_tools
        from app.tools.memory_tool import build_memory_tools
        agent_tools = build_ticket_tools(db=db) + build_memory_tools(db=db)

    logger.info(
        "ticket_agent_configured",
        model=settings.llm_model,
        tool_names=[t.name for t in agent_tools],
    )

    # ── System Prompt ─────────────────────────────────────────────────────────
    system_prompt = TICKET_AGENT_SYSTEM_PROMPT.format(
        app_name=settings.app_name,
        app_version=settings.app_version,
    )

    # ── Build ReAct Graph ─────────────────────────────────────────────────────
    agent = create_react_agent(
        model=llm,
        tools=agent_tools,
        state_modifier=SystemMessage(content=system_prompt),
    )

    logger.info("ticket_agent_compiled")
    return agent


async def run_ticket_agent(agent, state: dict) -> dict:
    """
    Run the Ticket Agent and return updated state.

    Args:
        agent: Compiled LangGraph agent graph
        state: Current AgentState dict

    Returns:
        Updated state dict with final_response set.
    """
    from langchain_core.messages import AIMessage

    logger.info(
        "ticket_agent_invocation_started",
        session_id=state.get("session_id", "unknown"),
    )

    try:
        config = {"recursion_limit": settings.agent_max_iterations * 2}
        result_state = await agent.ainvoke(state, config=config)

        messages = result_state.get("messages", [])
        final_response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content
                break

        if not final_response:
            final_response = "Ticket operation completed but no response was generated."

        logger.info(
            "ticket_agent_invocation_complete",
            session_id=state.get("session_id", "unknown"),
            response_length=len(final_response),
        )

        return {
            **state,
            "messages": messages,
            "final_response": final_response,
            "current_agent": "ticket_agent",
            "error": None,
        }

    except Exception as e:
        logger.error(
            "ticket_agent_invocation_failed",
            session_id=state.get("session_id", "unknown"),
            error=str(e),
            exc_info=True,
        )
        return {
            **state,
            "final_response": (
                "I encountered an error managing tickets. "
                "Please try again or use the ticket API directly."
            ),
            "current_agent": "ticket_agent",
            "error": str(e),
        }
