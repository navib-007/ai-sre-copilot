"""
routes/a2a.py — A2A HTTP Routes
================================
CONCEPT: A2A Endpoint Expose

  Exposes the HTTP surface of the Agent-to-Agent (A2A) protocol:
    - GET  /api/a2a/.well-known/agent-card.json -> Retrieve Agent Card
    - POST /api/a2a/rpc                         -> JSON-RPC endpoint
"""

from fastapi import APIRouter, Depends, Request, Body
from sqlalchemy.ext.asyncio import AsyncSession

from app.a2a.agent_card import get_agent_card, AgentCardModel
from app.a2a.protocol import handle_a2a_request
from app.db.database import get_db, AsyncSessionLocal
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/a2a", tags=["A2A Protocol"])


@router.get("/.well-known/agent-card.json", response_model=AgentCardModel, summary="Retrieve A2A Agent Card")
async def get_a2a_agent_card(request: Request) -> AgentCardModel:
    """
    Returns the Agent Card describing the capabilities and skills of this SRE agent.
    """
    # Derive the requesting client host base URL dynamically (e.g. http://localhost:8000)
    base_url = str(request.base_url).rstrip("/")
    logger.info("retrieve_agent_card", base_url=base_url)
    return get_agent_card(base_url)


@router.get("/.well-known/agent.json", response_model=AgentCardModel, summary="Retrieve A2A Agent Card (Alias)")
async def get_a2a_agent_card_alias(request: Request) -> AgentCardModel:
    """
    Alias endpoint for get_a2a_agent_card.
    """
    base_url = str(request.base_url).rstrip("/")
    return get_agent_card(base_url)


@router.post("/rpc", summary="A2A JSON-RPC 2.0 Endpoint")
async def a2a_rpc(
    payload: dict = Body(
        ...,
        examples=[
            {
                "summary": "Send Message / Start Task",
                "description": "Submit SRE instructions using message/send method.",
                "value": {
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "messageId": "msg-example-001",
                        "text": "Investigate why the payment-service is down.",
                        "skillId": "investigate_incident"
                    },
                    "id": 1
                }
            },
            {
                "summary": "Get Task Status",
                "description": "Retrieve status and results of a task using tasks/get method.",
                "value": {
                    "jsonrpc": "2.0",
                    "method": "tasks/get",
                    "params": {
                        "taskId": "insert-task-uuid-here"
                    },
                    "id": 2
                }
            },
            {
                "summary": "Cancel Task",
                "description": "Cancel an active task using tasks/cancel method.",
                "value": {
                    "jsonrpc": "2.0",
                    "method": "tasks/cancel",
                    "params": {
                        "taskId": "insert-task-uuid-here"
                    },
                    "id": 3
                }
            }
        ]
    ),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Endpoint for A2A JSON-RPC communication (message/send, tasks/get, tasks/cancel).
    """
    response = await handle_a2a_request(
        payload=payload,
        db=db,
        session_factory=AsyncSessionLocal
    )
    return response
