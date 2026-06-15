"""
app/mcp/server.py — Model Context Protocol (MCP) Tool Server
============================================================
Exposes all system operational tools (RAG, logs, metrics, tickets, incidents,
memory, and remediation execution) via the standardized Model Context Protocol.
Supports stdio transport (process-based) and SSE transport (HTTP-based).
"""

import os
import sys
import logging
import structlog
from typing import Optional, Any
from contextlib import asynccontextmanager

# Configure logging and structlog to output exclusively to stderr BEFORE importing app modules.
# This prevents SRE operational/DB logs from polluting stdout and corrupting stdio JSON-RPC.
logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(colors=False),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

from mcp.server.fastmcp import FastMCP

# Ensure the parent directory is in python path to import app modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.config import get_settings
from app.db.database import AsyncSessionLocal as sessionmaker

logger = logging.getLogger("mcp_server")
settings = get_settings()

# Initialize FastMCP Server
mcp_server = FastMCP(name="AgenticOpsTools")


# ─── Helper Context Manager for SQLite DB Sessions ────────────────────────────
_db_initialized = False

@asynccontextmanager
async def get_mcp_db():
    """Context manager to provide a request-scoped AsyncSession for MCP tools."""
    global _db_initialized
    if not _db_initialized:
        from app.db.database import init_db
        await init_db()
        _db_initialized = True

    async with sessionmaker() as db:
        try:
            yield db
            await db.commit()
        except Exception as e:
            await db.rollback()
            raise e


# ─── Tool 1: Logs Search ──────────────────────────────────────────────────────
@mcp_server.tool()
def search_logs(
    service: str,
    time_window: str = "1h",
    log_level: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 10,
) -> str:
    """
    Search application logs for a specific service within a time window.
    Use this tool to find error messages, stack traces, identify when issues started,
    and inspect resource exhaustion messages (OOM, database saturation, connection resets).
    Key services: 'payment-service', 'auth-service', 'api-gateway', 'postgres', 'kubernetes'.
    """
    from app.tools.logs_tool import build_logs_tool
    tool = build_logs_tool()
    return tool.func(
        service=service,
        time_window=time_window,
        log_level=log_level,
        keyword=keyword,
        limit=limit,
    )


# ─── Tool 2: System Metrics ───────────────────────────────────────────────────
@mcp_server.tool()
def query_metrics(
    service: str,
    metric: str = "all",
    time_window: str = "1h",
) -> str:
    """
    Query system metrics (CPU, memory, latency, connection counts) for a service.
    Use this tool during incident investigations to check resource saturation or SLA breaches.
    """
    from app.tools.metrics_tool import build_metrics_tool
    tool = build_metrics_tool()
    return tool.func(
        service=service,
        metric=metric,
        time_window=time_window,
    )


# ─── Tool 3: Knowledge Base (RAG) ─────────────────────────────────────────────
@mcp_server.tool()
async def search_knowledge_base(
    query: str,
    top_k: int = 5,
) -> str:
    """
    Search the internal knowledge base for IT operational runbooks, troubleshooting steps, and documentation.
    Use this when diagnosing issues to see if a known runbook exists for the issue.
    """
    from app.rag.qdrant_client import get_retriever
    from app.tools.rag_tool import build_rag_tool
    
    retriever = get_retriever()
    async with get_mcp_db() as db:
        tool = build_rag_tool(retriever=retriever, db=db)
        return await tool.coroutine(
            query=query,
            top_k=top_k,
        )


# ─── Tools 4-7: Ticket Operations ─────────────────────────────────────────────
@mcp_server.tool()
async def search_tickets(
    query: str,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    Search for existing support tickets by keyword, status, and priority.
    Always search tickets before creating a new ticket to check for duplicates.
    """
    from app.tools.ticket_tool import build_ticket_tools
    async with get_mcp_db() as db:
        tools = build_ticket_tools(db=db)
        tool = next(t for t in tools if t.name == "search_tickets")
        return await tool.coroutine(
            query=query,
            status=status,
            priority=priority,
            limit=limit,
        )


@mcp_server.tool()
async def create_ticket(
    title: str,
    description: str,
    priority: str,
    category: Optional[str] = None,
) -> str:
    """
    Create a new support ticket in the registry.
    Search for duplicates first. priority values: 'low', 'medium', 'high', 'critical'.
    """
    from app.tools.ticket_tool import build_ticket_tools
    async with get_mcp_db() as db:
        tools = build_ticket_tools(db=db)
        tool = next(t for t in tools if t.name == "create_ticket")
        return await tool.coroutine(
            title=title,
            description=description,
            priority=priority,
            category=category,
        )


@mcp_server.tool()
async def update_ticket(
    ticket_id: int,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    resolution: Optional[str] = None,
) -> str:
    """
    Update an existing ticket's status, priority, or resolution notes.
    Use to progress tickets (status='in_progress') or resolve them (status='resolved', with resolution details).
    """
    from app.tools.ticket_tool import build_ticket_tools
    async with get_mcp_db() as db:
        tools = build_ticket_tools(db=db)
        tool = next(t for t in tools if t.name == "update_ticket")
        return await tool.coroutine(
            ticket_id=ticket_id,
            status=status,
            priority=priority,
            resolution=resolution,
        )


@mcp_server.tool()
async def get_ticket(ticket_id: int) -> str:
    """
    Get full details of a support ticket by its numeric ID.
    Use when checking ticket progress or reading ticket descriptions.
    """
    from app.tools.ticket_tool import build_ticket_tools
    async with get_mcp_db() as db:
        tools = build_ticket_tools(db=db)
        tool = next(t for t in tools if t.name == "get_ticket")
        return await tool.coroutine(ticket_id=ticket_id)


# ─── Tools 8-11: Incident Operations ──────────────────────────────────────────
@mcp_server.tool()
async def create_incident(
    title: str,
    description: str,
    severity: str,
    affected_services: Optional[str] = None,
) -> str:
    """
    Create a new incident record to track an ongoing outage or critical degradation.
    Always create an incident record at the start of any P1/P2 outage investigation.
    severity: 'P1' (critical), 'P2' (high), 'P3' (medium), 'P4' (low).
    """
    from app.tools.incident_tool import build_incident_tools
    async with get_mcp_db() as db:
        tools = build_incident_tools(db=db)
        tool = next(t for t in tools if t.name == "create_incident")
        return await tool.coroutine(
            title=title,
            description=description,
            severity=severity,
            affected_services=affected_services,
        )


@mcp_server.tool()
async def search_incident_history(
    query: str,
    severity: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    Search historical incidents to identify similar past outages and their resolutions.
    Crucial for Root Cause Analysis (RCA) to find known fixes.
    """
    from app.tools.incident_tool import build_incident_tools
    async with get_mcp_db() as db:
        tools = build_incident_tools(db=db)
        tool = next(t for t in tools if t.name == "search_incident_history")
        return await tool.coroutine(
            query=query,
            severity=severity,
            limit=limit,
        )


@mcp_server.tool()
async def update_incident(
    incident_id: int,
    status: Optional[str] = None,
    root_cause: Optional[str] = None,
    resolution_note: Optional[str] = None,
) -> str:
    """
    Update an ongoing incident's status ('investigating', 'mitigated', 'resolved'), root cause, or resolution.
    """
    from app.tools.incident_tool import build_incident_tools
    async with get_mcp_db() as db:
        tools = build_incident_tools(db=db)
        tool = next(t for t in tools if t.name == "update_incident")
        return await tool.coroutine(
            incident_id=incident_id,
            status=status,
            root_cause=root_cause,
            resolution_note=resolution_note,
        )


@mcp_server.tool()
async def get_incident(incident_id: int) -> str:
    """
    Retrieve full details and timeline log of an incident by its numeric ID.
    """
    from app.tools.incident_tool import build_incident_tools
    async with get_mcp_db() as db:
        tools = build_incident_tools(db=db)
        tool = next(t for t in tools if t.name == "get_incident")
        return await tool.coroutine(incident_id=incident_id)


# ─── Tools 12-13: Memory Management ───────────────────────────────────────────
@mcp_server.tool()
async def save_user_preference(key: str, value: str) -> str:
    """
    Save a user behavioral preference (e.g. 'brief responses') to long-term memory.
    """
    from app.tools.memory_tool import build_memory_tools
    async with get_mcp_db() as db:
        tools = build_memory_tools(db=db)
        tool = next(t for t in tools if t.name == "save_user_preference")
        return await tool.coroutine(key=key, value=value)


@mcp_server.tool()
async def save_environment_fact(key: str, value: str) -> str:
    """
    Save a configuration fact about the infrastructure (e.g. 'kubernetes_version = 1.28') to memory.
    """
    from app.tools.memory_tool import build_memory_tools
    async with get_mcp_db() as db:
        tools = build_memory_tools(db=db)
        tool = next(t for t in tools if t.name == "save_environment_fact")
        return await tool.coroutine(key=key, value=value)


# ─── Tool 14: Remediation Action Execution (HITL-Guarded) ─────────────────────
@mcp_server.tool()
async def execute_remediation_action(
    action_type: str,
    target: str,
    justification: str,
    session_id: str,
) -> str:
    """
    Execute a mutating remediation action (like 'restart_pod', 'scale_replicas', 'rollback_config') on a target resource.
    This high-risk action requires human approval. If not yet approved in database for this session/thread,
    it automatically logs a pending approval request and prompts the agent to pause.
    """
    from app.tools.execute_tool import build_execute_tools
    async with get_mcp_db() as db:
        tools = build_execute_tools(db=db, session_id=session_id)
        tool = next(t for t in tools if t.name == "execute_remediation_action")
        return await tool.coroutine(
            action_type=action_type,
            target=target,
            justification=justification,
        )


# ─── Runner CLI Entry Point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Model Context Protocol (MCP) Tool Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio", help="Transport protocol to use (default: stdio)")
    parser.add_argument("--host", default="0.0.0.0", help="SSE host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8010, help="SSE port to run on (default: 8010)")
    
    args = parser.parse_args()
    
    if args.transport == "stdio":
        logger.info("Starting FastMCP server over stdio transport")
        mcp_server.run(transport="stdio")
    else:
        logger.info(f"Starting FastMCP server over SSE transport at http://{args.host}:{args.port}/sse")
        mcp_server.settings.host = args.host
        mcp_server.settings.port = args.port
        mcp_server.run(transport="sse")
