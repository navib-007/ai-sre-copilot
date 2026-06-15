"""
app/mcp/client.py — Model Context Protocol (MCP) Client Manager
===============================================================
Spawns the MCP Tool Server as a stdio subprocess, discovers all tools,
translates their JSON schemas into dynamic Pydantic schemas, and exposes
them as LangChain StructuredTool objects.
"""

import sys
import os
import logging
import subprocess
import asyncio
import httpx
from typing import Any, Optional, List
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, create_model
from langchain_core.tools import StructuredTool

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from app.logging_config import get_logger

logger = get_logger(__name__)


# ─── Schema Converter ─────────────────────────────────────────────────────────
def _json_schema_to_pydantic(tool_name: str, schema: dict) -> type[BaseModel]:
    """
    Dynamically translate an MCP JSON schema definition into a Pydantic BaseModel class
    so that LangChain/OpenAI can recognize and validate tool arguments correctly.
    """
    fields = {}
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    for prop_name, prop_info in properties.items():
        # Map JSON schema types to Python types
        prop_type: Any = Any
        t = prop_info.get("type")
        if t == "string":
            prop_type = str
        elif t == "integer":
            prop_type = int
        elif t == "boolean":
            prop_type = bool
        elif t == "number":
            prop_type = float
        elif t == "array":
            prop_type = list
        elif t == "object":
            prop_type = dict

        is_required = prop_name in required_fields
        default_val = ... if is_required else prop_info.get("default", None)

        # For execute_remediation_action, session_id is auto-injected by the client,
        # so we make it optional in the schema exposed to the LLM agent.
        if tool_name == "execute_remediation_action" and prop_name == "session_id":
            is_required = False
            default_val = None
            prop_type = Optional[str]

        fields[prop_name] = (
            prop_type,
            Field(
                default=default_val,
                description=prop_info.get("description", ""),
            )
        )

    # Dynamic model generation
    model_name = "".join(x.title() for x in tool_name.split("_")) + "Input"
    return create_model(model_name, **fields)


# ─── Tool Wrapper ─────────────────────────────────────────────────────────────
def _wrap_mcp_tool(session: ClientSession, mcp_tool: Any, session_id: str) -> StructuredTool:
    """
    Wrap an MCP protocol tool object into a LangChain StructuredTool.
    """
    tool_name = mcp_tool.name
    tool_desc = mcp_tool.description
    input_schema = mcp_tool.inputSchema

    # Build Pydantic schema model
    args_schema = _json_schema_to_pydantic(tool_name, input_schema)

    async def run_mcp_tool(**kwargs) -> str:
        """Coroutine that sends the tool execution request over stdio to the MCP server."""
        logger.info("mcp_client_invoking_tool", tool_name=tool_name, arguments=kwargs)
        
        # Auto-inject the active conversation session ID if expected by the tool
        if tool_name == "execute_remediation_action" and "session_id" not in kwargs:
            kwargs["session_id"] = session_id

        try:
            response = await session.call_tool(tool_name, kwargs)
            
            # Extract and consolidate text content from response
            text_contents = []
            for item in response.content:
                if getattr(item, "type", None) == "text":
                    text_contents.append(item.text)
                else:
                    text_contents.append(str(item))
            
            return "\n".join(text_contents)

        except Exception as e:
            logger.error("mcp_client_tool_invocation_failed", tool_name=tool_name, error=str(e))
            return f"Error executing tool via MCP server: {str(e)}"

    return StructuredTool.from_function(
        coroutine=run_mcp_tool,
        name=tool_name,
        description=tool_desc,
        args_schema=args_schema,
    )


# ─── MCP Client Lifecycle Manager ─────────────────────────────────────────────
@asynccontextmanager
async def mcp_tools_client(session_id: str):
    """
    Async context manager that starts the MCP Tool Server,
    discovers its exposed tools, wraps them for LangChain, and clean exits.
    Supports automatic SSE fallback on Windows systems where stdio subprocesses
    are not supported due to SelectorEventLoop (e.g. running uvicorn with --reload).
    """
    # Use the same python binary running this app to execute the server module
    command = sys.executable
    args = ["-m", "app.mcp.server", "--transport", "stdio"]

    # Inherit current environment variables (crucial for API keys & settings)
    env = os.environ.copy()
    # Ensure backend folder is in PYTHONPATH for the subprocess imports
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
    env["PYTHONIOENCODING"] = "utf-8"

    params = StdioServerParameters(command=command, args=args, env=env)
    
    try:
        logger.info("mcp_client_spawning_server", command=command, args=args)
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                logger.info("mcp_client_initializing_session")
                await session.initialize()
                
                # Discover tools from server
                list_result = await session.list_tools()
                mcp_tools = list_result.tools
                
                # Wrap all tools
                langchain_tools = []
                for tool in mcp_tools:
                    langchain_tools.append(_wrap_mcp_tool(session, tool, session_id))

                logger.info("mcp_client_tools_discovered", count=len(langchain_tools), names=[t.name for t in langchain_tools])
                
                # Yield the tools to the caller (chat router)
                yield langchain_tools
                
                logger.info("mcp_client_shutting_down")
    except NotImplementedError:
        logger.warning(
            "mcp_client_stdio_not_implemented_falling_back_to_sse",
            detail="SelectorEventLoop active on Windows. Starting standalone SSE server subprocess."
        )
        
        from mcp.client.sse import sse_client

        # Start the server subprocess in SSE mode using subprocess.Popen (safe on SelectorEventLoop)
        port = 8010
        server_args = ["-m", "app.mcp.server", "--transport", "sse", "--port", str(port)]
        url = f"http://localhost:{port}/sse"
        
        # Check if an SSE server is already running on port 8010
        server_already_running = False
        async with httpx.AsyncClient() as http_client:
            try:
                # FastMCP bindings might return a 404 or 405 for GET /, but we just check if the port is open and listening
                await http_client.get(f"http://localhost:{port}/")
                server_already_running = True
                logger.info("mcp_client_found_existing_sse_server", url=url)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                pass

        process = None
        if not server_already_running:
            logger.info("mcp_client_spawning_sse_server", command=command, args=server_args)
            process = subprocess.Popen(
                [command] + server_args,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            
            # Poll the server until it starts responding
            logger.info("mcp_client_waiting_for_sse_server", url=url)
            async with httpx.AsyncClient() as http_client:
                for attempt in range(50):  # Try for up to 5 seconds
                    try:
                        await http_client.get(f"http://localhost:{port}/")
                        break
                    except (httpx.ConnectError, httpx.ConnectTimeout):
                        await asyncio.sleep(0.1)
                else:
                    raise RuntimeError("MCP SSE Server failed to start on port 8010 within 5 seconds")
        
        try:
            # Connect via SSE transport
            logger.info("mcp_client_connecting_via_sse", url=url)
            async with sse_client(url=url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    logger.info("mcp_client_initializing_sse_session")
                    await session.initialize()
                    
                    list_result = await session.list_tools()
                    mcp_tools = list_result.tools
                    
                    langchain_tools = []
                    for tool in mcp_tools:
                        langchain_tools.append(_wrap_mcp_tool(session, tool, session_id))

                    logger.info("mcp_client_sse_tools_discovered", count=len(langchain_tools), names=[t.name for t in langchain_tools])
                    
                    yield langchain_tools
                    
                    logger.info("mcp_client_sse_shutting_down")
        finally:
            if process is not None:
                logger.info("mcp_client_terminating_sse_server")
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
