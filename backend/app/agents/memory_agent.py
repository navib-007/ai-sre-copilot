"""
agents/memory_agent.py — Memory Agent (Automated Memory Extraction & Management)
================================================================================
CONCEPT: Automated Fact & Preference Extraction (Memory Consolidation)
  While agents can explicitly call memory tools, the most seamless user experience
  comes from the agent *automatically* learning from conversation flow.

  The Memory Agent runs after each user turn. It:
    1. Analyzes the latest user message and assistant response via an LLM.
    2. Identifies any persistent facts or preferences.
    3. Performs upsert/delete operations on the SQLite long-term memory.
    4. Saves the overall turn to Qdrant semantic memory for semantic similarity lookup.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient

from app.config import get_settings
from app.logging_config import get_logger
from app.memory.long_term import save_memory_entry, delete_memory_entry
from app.memory.semantic import store_semantic_memory

settings = get_settings()
logger = get_logger(__name__)


MEMORY_EXTRACTOR_SYSTEM_PROMPT = """You are an automated memory extraction system for an IT Operations AI platform.

Your task is to analyze the conversation turn (User Message and Assistant Response) and extract any key user preferences or configuration/environment facts that should be remembered for FUTURE chat sessions.

## Definitions:
1. **Preference**: User's behavioral, language, or formatting preferences (e.g., "prefers concise answers", "likes bullet points", "prefers JSON formats").
2. **Fact**: Configuration details, infrastructure details, tool versions, service names, or user details (e.g., "runs Kubernetes v1.28", "uses Postgres 15 on AWS", "user name is John").

## Extraction Rules:
- Only extract persistent information that remains true across sessions. Do NOT store transient information about a single incident or ticket (e.g., do NOT store "TKT-42 status is resolved" or "the payment service is down right now").
- Format keys as lowercase with underscores (e.g., "k8s_version", "user_name", "db_provider"). Keep key names simple and descriptive.
- If a user changes their mind or updates a config (e.g., "We migrated to Postgres 16"), emit a memory with action "upsert" for the same key with the new value.
- If a user explicitly asks to forget something, emit a memory with action "delete" for that key.
- Be conservative. Do not extract trivial or transient facts. If nothing of persistent value is mentioned, output an empty list: []

Respond with ONLY a JSON array of objects with the following format. Do not include markdown fences, wrapping, or extra text:
[
  {
    "action": "upsert",
    "type": "preference" or "fact",
    "key": "unique_key_name",
    "value": "extracted memory value"
  }
]
"""


async def extract_and_consolidate_memory(
    db: AsyncSession,
    qdrant_client: AsyncQdrantClient,
    embedder,
    user_id: int,
    session_id: str,
    user_message: str,
    assistant_response: str,
) -> None:
    """
    Run the Memory Agent:
    1. Extract facts/preferences from the conversation turn and save to SQLite.
    2. Store the conversation turn in Qdrant semantic memory.

    Args:
        db:                 AsyncSession
        qdrant_client:      AsyncQdrantClient
        embedder:           CachedEmbedder
        user_id:            The owner of the memories
        session_id:         Current chat session
        user_message:       The user's latest query
        assistant_response: The agent's final response
    """
    logger.info("memory_consolidation_started", session_id=session_id, user_id=user_id)

    # Step 1: Store interaction in semantic memory (Qdrant)
    # This happens first, so if the LLM extraction fails, we still have the semantic trace.
    await store_semantic_memory(
        qdrant_client=qdrant_client,
        embedder=embedder,
        db=db,
        user_id=user_id,
        session_id=session_id,
        query=user_message,
        response=assistant_response,
    )

    # Step 2: Use LLM to extract facts & preferences
    try:
        extractor_llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.0,
            api_key=settings.openai_api_key,
            max_tokens=256,
        )

        turn_content = f"User Message:\n{user_message}\n\nAssistant Response:\n{assistant_response}"

        messages = [
            SystemMessage(content=MEMORY_EXTRACTOR_SYSTEM_PROMPT),
            HumanMessage(content=turn_content),
        ]

        response = await extractor_llm.ainvoke(messages)
        raw_output = response.content.strip()

        # Strip potential markdown code fences
        if "```" in raw_output:
            raw_output = raw_output.split("```")[1].split("```")[0].strip()
            if raw_output.startswith("json"):
                raw_output = raw_output[4:].strip()

        if not raw_output or raw_output == "[]":
            logger.info("no_new_memories_extracted", session_id=session_id)
            return

        extractions = json.loads(raw_output)
        if not isinstance(extractions, list):
            logger.warning("invalid_extraction_format_not_list", raw=raw_output)
            return

        logger.info("memories_extracted", count=len(extractions), session_id=session_id)

        # Step 3: Apply the extracted operations to SQLite
        for ext in extractions:
            action = ext.get("action", "upsert")
            mem_type = ext.get("type", "fact")
            key = ext.get("key")
            value = ext.get("value")

            if not key:
                continue

            key = key.lower().strip()

            if action == "delete":
                await delete_memory_entry(db=db, user_id=user_id, key=key)
            else:
                if not value:
                    continue
                await save_memory_entry(
                    db=db,
                    user_id=user_id,
                    memory_type=mem_type,
                    key=key,
                    value=value,
                )

        # Note: The db session is committed by the chat route calling db.commit() after this function completes.
        logger.info("memory_consolidation_complete", session_id=session_id)

    except json.JSONDecodeError as decode_err:
        logger.warning(
            "memory_extraction_json_parse_failed",
            session_id=session_id,
            error=str(decode_err),
            raw=raw_output if 'raw_output' in locals() else None,
        )
    except Exception as e:
        logger.error(
            "memory_consolidation_failed",
            session_id=session_id,
            error=str(e),
            exc_info=True,
        )
