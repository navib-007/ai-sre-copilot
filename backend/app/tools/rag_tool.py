"""
tools/rag_tool.py — Knowledge Base Search Tool
===============================================
CONCEPT: LangChain @tool Decorator

  The @tool decorator transforms a regular Python function into a tool
  that LLMs can call. It does several things automatically:
    1. Creates a JSON schema from the function's type annotations
    2. Wraps the function to handle errors gracefully
    3. Makes the tool discoverable by LangChain agents

  The LLM reads the docstring to understand WHAT the tool does and WHEN
  to use it. Well-written docstrings = better tool selection by the LLM.

CONCEPT: Tool Design Principles
  Good tools are:
    1. FOCUSED: Does exactly one thing (single responsibility)
    2. DESCRIPTIVE: Clear docstring explaining purpose + use cases
    3. TYPED: Explicit input/output types (helps the LLM form correct calls)
    4. SAFE: Validates inputs, handles errors, never crashes silently
    5. OBSERVABLE: Logs what it does (for debugging agent behavior)

CONCEPT: Async Tools in LangGraph
  LangGraph supports both sync and async tools. We use async here
  because our retriever makes network calls (to Qdrant + OpenAI embeddings).
  Async tools don't block the event loop while waiting for I/O.

CONCEPT: ToolError vs RuntimeError
  Tools should return meaningful error strings (not raise exceptions)
  when something goes wrong. The LLM can then decide how to handle
  the error — retry with different parameters, apologize to the user,
  or try a different approach.

DEPENDENCY INJECTION PATTERN:
  Because tools need database sessions and Qdrant clients (which are
  request-scoped resources), we use a factory function pattern:
    - `build_rag_tool(retriever, db)` returns a configured tool instance
    - The agent creates the tool fresh for each request
    - This avoids global state and makes testing easy
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ─── Input Schema ─────────────────────────────────────────────────────────────
# CONCEPT: Pydantic Input Schema for Tools
#   Instead of relying on type annotations alone, we define an explicit
#   Pydantic schema. This gives the LLM a structured JSON schema to fill,
#   with field descriptions that guide what values to provide.

class RAGSearchInput(BaseModel):
    """Input schema for the knowledge base search tool."""

    query: str = Field(
        description=(
            "The search query to look up in the knowledge base. "
            "Use natural language — e.g., 'How do I restart a Kubernetes pod?' "
            "or 'database backup failure runbook'. "
            "Be specific for better results."
        )
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description=(
            "Number of document chunks to retrieve. "
            "Use 3-5 for focused answers, up to 10 for broad research."
        ),
    )
    score_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum similarity score (0.0-1.0). "
            "Higher values = more relevant but fewer results. "
            "Lower values = more results but possibly less relevant."
        ),
    )


# ─── Tool Factory ─────────────────────────────────────────────────────────────

def build_rag_tool(retriever, db) -> StructuredTool:
    """
    Factory function that creates a configured RAG search tool.

    WHY A FACTORY?
      Tools need access to the retriever and database session, which are
      request-scoped objects (created per HTTP request, closed after).
      A factory creates the tool with these dependencies already bound,
      avoiding global state.

    CONCEPT: StructuredTool vs @tool decorator
      `StructuredTool.from_function()` is equivalent to `@tool` but gives
      us more control over the tool's name, description, and args schema.
      We use it here to bind the async function with request-scoped deps.

    Args:
        retriever: RAGRetriever instance (has Qdrant client + embedder)
        db:        AsyncSession for database access (embedding cache)

    Returns:
        A configured StructuredTool that the RAG agent can use.
    """

    async def search_knowledge_base(
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.60,
    ) -> str:
        """
        Search the IT operations knowledge base for relevant information.

        Use this tool when the user asks questions about:
          - Kubernetes issues (pod failures, deployments, scaling)
          - Database problems (backups, connections, performance)
          - Incident response procedures (runbooks, SOPs)
          - Any technical IT operations topic

        Returns a formatted context string with relevant document excerpts
        and their source files. Returns a "no results" message if nothing
        relevant is found.

        Args:
            query:           Natural language search query
            top_k:           Number of results to retrieve (1-10)
            score_threshold: Minimum relevance score (0.0-1.0)
        """
        logger.info(
            "rag_tool_invoked",
            query=query[:100],
            top_k=top_k,
            score_threshold=score_threshold,
        )

        try:
            # ── Validate and cap parameters from config ────────────────────
            # Read limits from settings (not hardcoded) to keep it configurable
            effective_top_k = min(top_k, settings.rag_max_chunks)
            effective_threshold = max(score_threshold, settings.rag_min_score)

            # ── Call the retriever ─────────────────────────────────────────
            # CONCEPT: search_with_context formats results as a context string
            # ready to inject into the LLM prompt. Each result includes
            # the source filename and relevance score for transparency.
            context = await retriever.search_with_context(
                query=query,
                db=db,
                top_k=effective_top_k,
                score_threshold=effective_threshold,
            )

            logger.info(
                "rag_tool_complete",
                query=query[:100],
                context_length=len(context),
                found_results=(context != "No relevant documentation found for this query."),
            )

            return context

        except Exception as e:
            logger.error(
                "rag_tool_error",
                query=query[:100],
                error=str(e),
                exc_info=True,
            )
            # Return error as string — let the LLM handle it gracefully
            # (vs. raising exception which would crash the agent)
            if "gaierror" in str(e).lower() or "connect" in str(e).lower():
                return (
                    "⚠️ Knowledge base is currently unreachable (Qdrant connection failed). "
                    "I'll answer from my training knowledge instead, but cannot cite specific runbooks. "
                    "To fix: check QDRANT_URL in .env or use in-memory mode by removing QDRANT_URL."
                )
            return f"Knowledge base search failed: {str(e)}"


    # ── Build the StructuredTool ───────────────────────────────────────────────
    # CONCEPT: Tool metadata matters!
    #   The `name` and `description` are what the LLM sees in its system prompt.
    #   Agents choose tools based on these descriptions, so clarity = better
    #   tool selection = better answers.

    return StructuredTool.from_function(
        coroutine=search_knowledge_base,   # Use `coroutine` for async functions
        name="search_knowledge_base",
        description=(
            "Search the IT operations knowledge base for relevant documentation, "
            "runbooks, SOPs, and troubleshooting guides. "
            "Use this tool when you need information about specific technologies, "
            "procedures, or incident responses. "
            "Always use this tool before answering technical questions to ensure "
            "your answer is grounded in actual documentation."
        ),
        args_schema=RAGSearchInput,
    )
