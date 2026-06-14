"""
tools/memory_tool.py — Memory Management Tools for Agents
=========================================================
CONCEPT: Memory Tools in Agent Systems
  Allowing specialist agents to explicitly interact with the user's
  long-term memory. If the Incident Agent learns a new fact (e.g. "Kubernetes version is 1.28")
  or the Ticket Agent learns a user preference, they can store it
  permanently so that it is retrieved in future sessions.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from app.logging_config import get_logger
from app.memory.long_term import (
    save_memory_entry,
    get_memory_entries,
    delete_memory_entry,
)

logger = get_logger(__name__)


# ─── Input Schemas ────────────────────────────────────────────────────────────

class SavePreferenceInput(BaseModel):
    """Input schema for saving user preference."""
    key: str = Field(
        description="A unique key for the preference, lowercased with underscores. Example: 'response_style', 'timezone'."
    )
    value: str = Field(
        description="The preference value. Example: 'brief, bullet points', 'EST'."
    )


class SaveFactInput(BaseModel):
    """Input schema for saving environment fact."""
    key: str = Field(
        description="A unique key for the fact, lowercased with underscores. Example: 'k8s_version', 'primary_database'."
    )
    value: str = Field(
        description="The environment details. Example: 'v1.28', 'Postgres v15 on AWS RDS'."
    )


class GetMemoriesInput(BaseModel):
    """Input schema for listing memories."""
    memory_type: Optional[str] = Field(
        default=None,
        description="Optional filter: 'preference' or 'fact'. Leave null for all."
    )


class DeleteMemoryInput(BaseModel):
    """Input schema for deleting memory."""
    key: str = Field(
        description="The key of the memory entry to delete/forget."
    )


# ─── Tool Factory ─────────────────────────────────────────────────────────────

def build_memory_tools(db) -> list[StructuredTool]:
    """
    Factory that returns all memory-related tools bound to the request's DB session.

    Args:
        db: AsyncSession for database access

    Returns:
        List of configured StructuredTools.
    """
    user_id = 1  # Standard placeholder until Phase 9 auth

    # ── Tool 1: Save User Preference ──────────────────────────────────────────
    async def save_user_preference(key: str, value: str) -> str:
        """
        Save a user-specific preference to the long-term memory.
        Use this when the user explicitly mentions a personal preference.
        Example: "Please keep your explanations short" -> key='response_length', value='short'.
        """
        logger.info("save_user_preference_tool", key=key, value=value)
        try:
            await save_memory_entry(
                db=db,
                user_id=user_id,
                memory_type="preference",
                key=key,
                value=value,
            )
            # The session will commit at the end of the request
            return f"Saved user preference '{key}' = '{value}' to memory."
        except Exception as e:
            return f"Failed to save preference: {str(e)}"

    # ── Tool 2: Save Environment Fact ─────────────────────────────────────────
    async def save_environment_fact(key: str, value: str) -> str:
        """
        Save a technical fact about the IT environment or infrastructure.
        Use this when discovering system configs, versions, or constraints.
        Example: "We use Postgres 15" -> key='database_version', value='Postgres 15'.
        """
        logger.info("save_environment_fact_tool", key=key, value=value)
        try:
            await save_memory_entry(
                db=db,
                user_id=user_id,
                memory_type="fact",
                key=key,
                value=value,
            )
            return f"Saved environment fact '{key}' = '{value}' to memory."
        except Exception as e:
            return f"Failed to save environment fact: {str(e)}"

    # ── Tool 3: Get User Memories ─────────────────────────────────────────────
    async def get_user_memories(memory_type: Optional[str] = None) -> str:
        """
        List all saved facts and preferences about the user and their environment.
        Use this to inspect what the agent currently remembers.
        """
        logger.info("get_user_memories_tool", memory_type=memory_type)
        try:
            entries = await get_memory_entries(db=db, user_id=user_id, memory_type=memory_type)
            if not entries:
                return "No memories found."

            lines = ["Here are the stored memories:"]
            for e in entries:
                lines.append(f"• [{e.memory_type.upper()}] {e.key}: {e.value}")
            return "\n".join(lines)
        except Exception as e:
            return f"Failed to retrieve memories: {str(e)}"

    # ── Tool 4: Delete User Memory ────────────────────────────────────────────
    async def delete_user_memory(key: str) -> str:
        """
        Delete or forget a specific memory entry by key.
        Use this when the user says: "Forget my name" or "That is no longer true".
        """
        logger.info("delete_user_memory_tool", key=key)
        try:
            deleted = await delete_memory_entry(db=db, user_id=user_id, key=key)
            if deleted:
                return f"Successfully deleted memory for key '{key}'."
            return f"No memory entry found with key '{key}'."
        except Exception as e:
            return f"Failed to delete memory: {str(e)}"

    # ── Build and return all tools ─────────────────────────────────────────────
    return [
        StructuredTool.from_function(
            coroutine=save_user_preference,
            name="save_user_preference",
            description="Save a personal or behavior preference of the user (e.g. formatting, length).",
            args_schema=SavePreferenceInput,
        ),
        StructuredTool.from_function(
            coroutine=save_environment_fact,
            name="save_environment_fact",
            description="Save a factual configuration detail about the infrastructure or environment.",
            args_schema=SaveFactInput,
        ),
        StructuredTool.from_function(
            coroutine=get_user_memories,
            name="get_user_memories",
            description="List all currently remembered preferences and system configuration facts.",
            args_schema=GetMemoriesInput,
        ),
        StructuredTool.from_function(
            coroutine=delete_user_memory,
            name="delete_user_memory",
            description="Delete/forget a specific memory entry by key.",
            args_schema=DeleteMemoryInput,
        ),
    ]
