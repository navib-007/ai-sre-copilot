"""
agents/incident_agent.py — Incident Investigation Agent
========================================================
CONCEPT: Multi-Step Tool-Calling Agent

  The Incident Agent is the most complex specialist in Phase 4.
  Unlike the RAG or Ticket agents (which usually do 1-2 tool calls),
  the Incident Agent runs a MULTI-STEP investigation:

  Phase 1 — Create Incident Record
    → create_incident tool

  Phase 2 — Gather Evidence (3-5 tool calls)
    → search_logs (errors in the last hour)
    → query_metrics (CPU, memory, error rate)
    → search_incident_history (similar past incidents)
    → search_knowledge_base (runbook for this type of issue)

  Phase 3 — Root Cause Analysis (sub-agent call)
    → Passes all evidence to the RCA Agent
    → Gets structured RCA report back

  Phase 4 — Recommend Action
    → Creates/updates tickets
    → Proposes remediation
    → Sets needs_approval=True for high-risk actions (Phase 6)

CONCEPT: Evidence Accumulation
  The agent builds up evidence progressively:
    1. Each tool call adds more evidence to the context
    2. The LLM uses previous evidence to decide what to check next
    3. The final RCA uses ALL accumulated evidence

  This is similar to how a human SRE investigates:
  "Check logs → see OOM → check memory metrics → see 98% → 
   search runbook → find memory leak pattern → recommend restart"

CONCEPT: Incident Severity → Investigation Depth
  P1/P2 incidents trigger thorough investigation (all tools).
  P3/P4 incidents may need only basic investigation.
  The LLM decides based on the severity in the system prompt.
"""

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ─── System Prompt ────────────────────────────────────────────────────────────

INCIDENT_AGENT_SYSTEM_PROMPT = """You are a specialist Incident Investigation Agent for {app_name}.

You investigate IT operations incidents systematically using a structured methodology.

## Your Investigation Workflow

### Step 1: Create Incident Record
- ALWAYS start by creating an incident record using create_incident
- This logs the incident so the team can track it

### Step 2: Gather Evidence (in this order)
1. **Search logs** for the affected service (last 1h, filter ERROR level)
2. **Query metrics** for the affected service (metric='all', last 1h)
3. **Search incident history** for similar past incidents
4. **Search knowledge base** for runbooks related to this type of issue

### Step 3: Root Cause Analysis
After gathering evidence, analyze and provide:
- Root cause (the underlying technical reason)
- Contributing factors
- Impact assessment (how many users/services affected)

### Step 4: Recommend Action
Based on the RCA, provide:
- **Immediate remediation** (fix RIGHT NOW)
- **Risk assessment** for the proposed action
- Whether human approval is needed (P1/P2 actions always need approval)

## Tool Usage Rules

### Evidence Gathering Order
Always gather: logs → metrics → incident history → runbook context
Don't skip steps for P1/P2 incidents.

### Service Name Conventions
When the user mentions a service, map it to the tool-expected format:
  "payment service" → "payment-service"
  "auth" → "auth-service"
  "gateway" → "api-gateway"
  "database" or "postgres" → "postgres"

### When to Escalate
- P1 incidents: Always flag as needing_approval = True in your response
- Data-destructive actions: Always require approval
- Pod restarts in prod: Require approval
- Config changes: Require approval

## Response Format

Provide a structured incident report:

## 🚨 Incident Report

### Incident Created
[Incident ID and summary]

### Evidence Gathered
[Summary of what you found in logs, metrics, history]

### Root Cause
[Your RCA finding]

### Recommended Action
[What to do now — be specific with commands/steps]

### Risk Level
[Low/Medium/High] — [Explanation]
⚠️ [If high-risk: "This action requires approval before execution"]

Current Platform: {app_name} v{app_version}
"""


def build_incident_agent(db, retriever):
    """
    Build the Incident Investigation specialist agent.

    This agent has access to:
    - Incident tools (create, search, update)
    - Log search tool (simulated)
    - Metrics query tool (simulated)
    - RAG tool (for runbook lookup)

    Args:
        db:        AsyncSession for database operations
        retriever: RAGRetriever for knowledge base search

    Returns:
        Compiled LangGraph ReAct agent graph.
    """
    from app.tools.incident_tool import build_incident_tools
    from app.tools.logs_tool import build_logs_tool
    from app.tools.metrics_tool import build_metrics_tool
    from app.tools.rag_tool import build_rag_tool

    # ── LLM (slightly higher max_tokens for incident reports) ─────────────────
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.agent_temperature,
        api_key=settings.openai_api_key,
        max_tokens=settings.agent_max_tokens,
    )

    # ── Tools ─────────────────────────────────────────────────────────────────
    # CONCEPT: Tool Selection for Incident Investigation
    #   We give the incident agent a comprehensive tool set:
    #   - All incident CRUD tools (manage the incident record)
    #   - Log search (find error messages)
    #   - Metrics (find resource exhaustion)
    #   - RAG (look up runbooks for this type of issue)
    from app.tools.memory_tool import build_memory_tools
    tools = [
        *build_incident_tools(db=db),   # 4 incident tools
        build_logs_tool(),               # 1 log search tool
        build_metrics_tool(),            # 1 metrics tool
        build_rag_tool(retriever=retriever, db=db),  # 1 RAG tool
        *build_memory_tools(db=db),      # 4 memory tools
    ]

    logger.info(
        "incident_agent_configured",
        model=settings.llm_model,
        tool_count=len(tools),
        tool_names=[t.name for t in tools],
    )

    # ── System Prompt ─────────────────────────────────────────────────────────
    system_prompt = INCIDENT_AGENT_SYSTEM_PROMPT.format(
        app_name=settings.app_name,
        app_version=settings.app_version,
    )

    # ── Build ReAct Graph ─────────────────────────────────────────────────────
    # CONCEPT: Higher recursion limit for incident investigation
    #   Incident investigation requires more tool calls than other agents.
    #   We use agent_max_iterations from settings (configurable via .env).
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=SystemMessage(content=system_prompt),
    )

    logger.info("incident_agent_compiled")
    return agent


async def run_incident_agent(agent, state: dict) -> dict:
    """
    Run the Incident Agent and return updated state.

    CONCEPT: State Enrichment
      The Incident Agent enriches the state with:
        - needs_approval: True for P1/P2 remediation actions
        - evidence: Gathered evidence for RCA
        - root_cause: Determined root cause

      These fields will be used by the Approval Agent in Phase 6.

    Args:
        agent: Compiled LangGraph agent graph
        state: Current AgentState dict

    Returns:
        Updated state with final_response and incident fields set.
    """
    logger.info(
        "incident_agent_invocation_started",
        session_id=state.get("session_id", "unknown"),
    )

    try:
        # Incident investigation needs more iterations (evidence gathering loop)
        config = {
            "recursion_limit": settings.agent_max_iterations * 2,
        }

        result_state = await agent.ainvoke(state, config=config)

        messages = result_state.get("messages", [])
        final_response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content
                break

        if not final_response:
            final_response = "Incident investigation completed but no report was generated."

        # CONCEPT: Detecting approval need from response content
        # In Phase 6, this will use LangGraph interrupt() for true HITL.
        # For now, we detect approval need from keywords in the response.
        needs_approval = any(
            phrase in final_response.lower()
            for phrase in [
                "requires approval",
                "human approval",
                "need approval",
                "approval required",
                "⚠️",
            ]
        )

        logger.info(
            "incident_agent_invocation_complete",
            session_id=state.get("session_id", "unknown"),
            response_length=len(final_response),
            needs_approval=needs_approval,
        )

        return {
            **state,
            "messages": messages,
            "final_response": final_response,
            "current_agent": "incident_agent",
            "needs_approval": needs_approval,
            "error": None,
        }

    except Exception as e:
        logger.error(
            "incident_agent_invocation_failed",
            session_id=state.get("session_id", "unknown"),
            error=str(e),
            exc_info=True,
        )
        return {
            **state,
            "final_response": (
                "I encountered an error during incident investigation. "
                "Please check the server logs and escalate manually if needed."
            ),
            "current_agent": "incident_agent",
            "error": str(e),
        }
