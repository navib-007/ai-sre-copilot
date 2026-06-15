"""
agents/rag_agent.py — RAG Agent using LangGraph
================================================
CONCEPT: LangGraph ReAct Agent

  ReAct (Reasoning + Acting) is the most common agent pattern:
    1. REASON: LLM looks at the conversation and available tools
    2. ACT:    LLM decides to call a tool (or respond directly)
    3. OBSERVE: LLM reads the tool's output
    4. Repeat steps 1-3 until the LLM decides to give a final answer

  In LangGraph, this is implemented as a graph with two nodes:
    - "agent" node: calls the LLM (decides what to do next)
    - "tools" node: executes tool calls requested by the LLM

  The conditional edge between them looks at whether the last message
  contains tool calls — if yes → tools node; if no → END.

CONCEPT: create_react_agent vs custom graph

  LangGraph provides `create_react_agent()` as a pre-built ReAct agent.
  This is perfect for Phase 3 — it gives us a working agent with:
    - Automatic tool call loop
    - Message history management
    - Error handling

  In Phase 4, we'll build a custom graph (supervisor with routing) to
  understand what `create_react_agent` abstracts away.

CONCEPT: System Prompt Design for RAG Agents

  The system prompt is critical for RAG agents:
    1. ALWAYS instruct the agent to use the search tool
       (LLMs tend to answer from training data if not told otherwise)
    2. REQUIRE citations — "mention the source document"
    3. Set boundaries — "if the tool returns no results, say so"
    4. Tone/persona — "You are an IT ops assistant..."

  A well-designed system prompt = grounded, honest, useful answers.
  A poor system prompt = hallucinations presented as facts.

CONCEPT: LangSmith Tracing
  When LANGCHAIN_TRACING_V2=true, every LLM call, tool invocation,
  and agent step is automatically sent to LangSmith for visualization.
  This lets you inspect the full reasoning chain for debugging.
"""

from typing import Optional, List, Any
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ─── System Prompt ────────────────────────────────────────────────────────────
# CONCEPT: System Prompt Template (no hardcoded values)
#   The system prompt sets the agent's persona, capabilities, and rules.
#   We define it as a module-level constant so it's easy to find and modify.
#   All dynamic values come from settings (not hardcoded strings).

RAG_AGENT_SYSTEM_PROMPT = """You are an expert IT Operations AI Assistant for {app_name}.

Your primary role is to answer questions about IT infrastructure, operations, \
and incidents by searching the knowledge base.

## Your Capabilities
- Search and retrieve information from IT runbooks, SOPs, and documentation
- Answer questions about Kubernetes, databases, networking, and incident response
- Provide step-by-step troubleshooting guides based on documented procedures

## Rules You Must Follow

### ALWAYS use the search_knowledge_base tool
- Before answering ANY technical question, search the knowledge base
- Do not answer from memory alone — always verify against documentation
- If the search returns no results, say so explicitly

### ALWAYS cite your sources
- When you use information from the knowledge base, mention the source document
- Format: "According to [source filename]..." or "The [source] runbook states..."
- Include relevance scores when they help the user assess confidence

### Be Honest About Limitations
- If the knowledge base doesn't have an answer, say: "I don't have documentation
  on this topic. Consider uploading relevant runbooks or SOPs."
- Never invent procedures or commands — only cite documented ones

### Formatting
- Use markdown for structured answers (headers, code blocks, bullet lists)
- For multi-step procedures, always use numbered lists
- Highlight critical warnings with ⚠️

## Current Context
- Platform: {app_name} v{app_version}
- Environment: {environment}
"""


def build_rag_agent(retriever, db, tools: Optional[List[Any]] = None):
    """
    Build and return a configured RAG LangGraph agent.

    CONCEPT: Agent Construction
      Building an agent requires:
        1. An LLM (the "brain" that reasons and decides)
        2. Tools (the "hands" that take actions)
        3. A system prompt (the "instructions" that shape behavior)

      `create_react_agent()` wires these together into a LangGraph graph:

          START → agent_node ──[has tool calls]──→ tools_node ──→ agent_node
                       └───[no tool calls]──────→ END

    CONCEPT: Model Selection via Settings
      The LLM model is configured in settings (LLM_MODEL env var).
      Default: gpt-4o-mini (fast, cheap, good for most ops tasks)
      Override in .env: LLM_MODEL=gpt-4o (for complex reasoning)

    Args:
        retriever: RAGRetriever instance (wraps Qdrant + embedder)
        db:        AsyncSession for embedding cache
        tools:     Optional pre-constructed list of tools (e.g. MCP client tools)

    Returns:
        A compiled LangGraph CompiledGraph ready to be invoked.
    """
    # ── Step 1: Create the LLM ─────────────────────────────────────────────────
    # CONCEPT: Temperature = 0 for factual/retrieval agents
    #   - Temperature 0.0 = deterministic, consistent answers (good for RAG)
    #   - Temperature 0.7-1.0 = creative, varied (good for brainstorming)
    #   - For an IT ops assistant, we want consistent, accurate answers
    #   - Temperature is configurable via settings for flexibility
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.agent_temperature,
        api_key=settings.openai_api_key,
        # CONCEPT: max_tokens limits response length
        # Prevents runaway responses that cost too much or timeout
        max_tokens=settings.agent_max_tokens,
    )

    logger.info(
        "rag_agent_llm_configured",
        model=settings.llm_model,
        temperature=settings.agent_temperature,
        max_tokens=settings.agent_max_tokens,
    )

    # ── Step 2: Create or Filter tools ─────────────────────────────────────────
    if tools is not None:
        # Filter the dynamic tools list for RAG agent specific needs
        allowed_names = {"search_knowledge_base", "save_user_preference", "save_environment_fact"}
        agent_tools = [t for t in tools if t.name in allowed_names]
    else:
        # Fall back to local tools builders
        from app.tools.rag_tool import build_rag_tool
        from app.tools.memory_tool import build_memory_tools
        rag_tool = build_rag_tool(retriever=retriever, db=db)
        agent_tools = [rag_tool] + build_memory_tools(db=db)

    logger.info("rag_agent_tools_configured", tool_names=[t.name for t in agent_tools])

    # ── Step 3: Format the system prompt ──────────────────────────────────────
    # All values come from settings — no hardcoded strings in code
    system_prompt = RAG_AGENT_SYSTEM_PROMPT.format(
        app_name=settings.app_name,
        app_version=settings.app_version,
        environment=settings.environment,
    )

    # ── Step 4: Create the ReAct agent graph ──────────────────────────────────
    # CONCEPT: create_react_agent internals
    #   This builds a StateGraph with:
    #     - MessagesState (auto-manages messages list)
    #     - "agent" node: llm.bind_tools(tools) — LLM that knows about tools
    #     - "tools" node: ToolNode(tools) — executes tool calls
    #     - Conditional edge: check if last message has tool_calls
    #
    #   The `state_modifier` (system prompt) is injected as the first message
    #   in the LLM call, ensuring consistent behavior across all invocations.
    agent = create_react_agent(
        model=llm,
        tools=agent_tools,
        state_modifier=SystemMessage(content=system_prompt),
    )

    logger.info("rag_agent_compiled", agent_type="react")

    return agent


# ─── Agent Runner ─────────────────────────────────────────────────────────────

async def run_rag_agent(
    agent,
    state: dict,) -> dict:
    """
    Execute the RAG agent graph with the given state.

    CONCEPT: Agent Invocation
      When we call `agent.ainvoke(state)`, LangGraph:
        1. Passes the state to the first node ("agent" node)
        2. Runs the ReAct loop until no more tool calls
        3. Returns the final state (with all messages accumulated)

      `ainvoke` = async invoke (uses asyncio, doesn't block)

    CONCEPT: What's in the Response?
      The response state contains:
        - `messages`: All messages including LLM responses + tool results
        - The last AIMessage is the agent's final answer

      We extract the final answer and add it to the AgentState
      as `final_response` for the FastAPI route handler.

    Args:
        agent: Compiled LangGraph graph from build_rag_agent()
        state: Current AgentState dict with messages and context

    Returns:
        Updated state dict with final_response and retrieved_documents set.
    """
    logger.info(
        "rag_agent_invocation_started",
        session_id=state.get("session_id", "unknown"),
        message_count=len(state.get("messages", [])),
    )

    try:
        # ── Invoke the agent ───────────────────────────────────────────────────
        # CONCEPT: config["recursion_limit"]
        #   Prevents infinite loops in the ReAct cycle.
        #   If the agent calls tools more than max_iterations times,
        #   LangGraph raises GraphRecursionError instead of looping forever.
        #   Configured via settings, not hardcoded.
        config = {
            "recursion_limit": settings.agent_max_iterations * 2,  # ×2 because each iteration = 2 steps (agent + tools)
        }

        result_state = await agent.ainvoke(state, config=config)

        # ── Extract the final response ─────────────────────────────────────────
        # The last AIMessage in the messages list is the agent's final answer
        messages = result_state.get("messages", [])
        final_response = ""

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                # Filter out intermediate tool-calling messages (they have tool_calls but no content)
                final_response = msg.content
                break

        if not final_response:
            final_response = "I was unable to generate a response. Please try again."

        logger.info(
            "rag_agent_invocation_complete",
            session_id=state.get("session_id", "unknown"),
            response_length=len(final_response),
            total_messages=len(messages),
        )

        # ── Update and return state ────────────────────────────────────────────
        return {
            **state,                        # Preserve all existing state fields
            "messages": messages,           # Full message history
            "final_response": final_response,
            "current_agent": "rag_agent",
            "error": None,
        }

    except Exception as e:
        error_msg = f"RAG agent error: {str(e)}"
        logger.error(
            "rag_agent_invocation_failed",
            session_id=state.get("session_id", "unknown"),
            error=str(e),
            exc_info=True,
        )

        return {
            **state,
            "final_response": (
                "I encountered an error while processing your request. "
                "Please try again or contact support if the issue persists."
            ),
            "error": error_msg,
        }
