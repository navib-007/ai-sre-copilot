"""
a2a/agent_card.py — Agent Card Configuration & Schemas
======================================================
CONCEPT: Agent Card (Discovery)

  The Agent Card is a machine-readable JSON profile describing the agent's:
    - Identity (name, description, version)
    - Protocol connectivity (url, protocolVersion)
    - Capabilities (streaming, webhooks)
    - Skills (supported operations, tags, natural language examples)

  This allows external parent/peer agents to discover our copilot and understand
  what inputs/tasks it accepts and how to send them.
"""

from typing import List
from pydantic import BaseModel, Field


class SkillModel(BaseModel):
    id: str = Field(description="Unique identifier for the skill")
    name: str = Field(description="Human-readable name of the skill")
    description: str = Field(description="Description of what the skill does")
    tags: List[str] = Field(default_factory=list, description="Keywords for discovery")
    examples: List[str] = Field(default_factory=list, description="Example natural language prompts for this skill")


class CapabilitiesModel(BaseModel):
    streaming: bool = Field(default=False, description="Supports real-time streaming updates")
    pushNotifications: bool = Field(default=False, description="Supports push notification webhooks")
    stateTransitionHistory: bool = Field(default=False, description="Maintains task state change records")


class AgentCardModel(BaseModel):
    name: str = Field(description="Name of the agent")
    description: str = Field(description="Description of the agent's overall purpose")
    version: str = Field(description="Version of the agent implementation")
    protocolVersion: str = Field(description="Version of the A2A protocol implemented")
    url: str = Field(description="The base service endpoint URL for the agent's A2A service")
    skills: List[SkillModel] = Field(description="List of skills the agent supports")
    capabilities: CapabilitiesModel = Field(default_factory=CapabilitiesModel)
    defaultInputModes: List[str] = Field(default_factory=lambda: ["text/plain"])
    defaultOutputModes: List[str] = Field(default_factory=lambda: ["text/plain"])


def get_agent_card(base_url: str) -> AgentCardModel:
    """
    Generate the SRE Copilot Agent Card dynamically.
    The base_url is derived from the requesting client host.
    """
    return AgentCardModel(
        name="AI SRE Copilot",
        description="An autonomous IT Operations and SRE agent that can investigate incidents, search runbooks, and manage support tickets.",
        version="1.0.0",
        protocolVersion="2024-06-20",
        url=f"{base_url}/api/a2a",
        skills=[
            SkillModel(
                id="investigate_incident",
                name="Investigate Incident",
                description="Analyze system behavior, logs, and metrics to find the root cause of an incident.",
                tags=["incident", "rca", "sre", "logs", "metrics"],
                examples=["Investigate why the payment service is down", "Investigate high CPU metrics"]
            ),
            SkillModel(
                id="manage_tickets",
                name="Manage Tickets",
                description="Create, retrieve, or update IT support tickets.",
                tags=["ticket", "support", "jira", "helpdesk"],
                examples=["Create a high priority ticket for database disk space", "Check status of ticket 42"]
            ),
            SkillModel(
                id="search_runbooks",
                name="Search Runbooks",
                description="Search the knowledge base for IT runbooks and standard operating procedures.",
                tags=["rag", "runbook", "docs", "kb"],
                examples=["How do I restart a Kubernetes pod?", "What is the database failover runbook?"]
            )
        ],
        capabilities=CapabilitiesModel(
            streaming=False,
            pushNotifications=False,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"]
    )
