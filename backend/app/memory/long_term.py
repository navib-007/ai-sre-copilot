"""
app/memory/long_term.py — Long-Term Memory Storage (SQLite Facts & Preferences)
=============================================================================
CONCEPT: Long-Term Memory (Persistent & Cross-Session)
  Short-term memory tracks conversation context *within* a session.
  Long-term memory tracks facts and preferences *across* all sessions for a user.

  Examples:
    - Fact: "User runs Kubernetes version 1.28"
    - Preference: "User prefers concise bulleted troubleshooting guides"

  This module interacts with the SQLite `memory_entries` table via SQLAlchemy
  to create, update, delete, and list memories for a given user.
"""

from datetime import datetime, timezone
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MemoryEntry, MemoryType
from app.logging_config import get_logger

logger = get_logger(__name__)


async def save_memory_entry(
    db: AsyncSession,
    user_id: int,
    memory_type: str,
    key: str,
    value: str,
    relevance_score: float = 1.0,
) -> MemoryEntry:
    """
    Save or update a persistent long-term memory entry for a user (Upsert pattern).

    If an entry with the same user_id and key already exists, update its value,
    relevance score, and last accessed timestamp. Otherwise, create a new one.

    Args:
        db:              AsyncSession bound to the request
        user_id:         ID of the user this memory belongs to
        memory_type:     MemoryType enum value ("fact" or "preference")
        key:             Unique string identifier for this memory (e.g. "k8s_version")
        value:           The actual content of the memory (e.g. "v1.28")
        relevance_score: Importance or confidence score (0.0 to 1.0)

    Returns:
        The created or updated MemoryEntry ORM object
    """
    logger.info(
        "saving_memory_entry",
        user_id=user_id,
        memory_type=memory_type,
        key=key,
        value_preview=value[:50],
    )

    try:
        # Check if the key already exists for this user
        result = await db.execute(
            select(MemoryEntry).where(
                MemoryEntry.user_id == user_id,
                MemoryEntry.key == key,
            )
        )
        existing_entry = result.scalar_one_or_none()

        now = datetime.now(timezone.utc)

        if existing_entry:
            # Update the existing entry
            existing_entry.value = value
            existing_entry.memory_type = memory_type
            existing_entry.relevance_score = relevance_score
            existing_entry.last_accessed = now
            logger.info("memory_entry_updated", id=existing_entry.id, key=key)
            # The session will commit this on request end
            return existing_entry
        else:
            # Create a new memory entry
            new_entry = MemoryEntry(
                user_id=user_id,
                memory_type=memory_type,
                key=key,
                value=value,
                relevance_score=relevance_score,
                created_at=now,
                last_accessed=now,
            )
            db.add(new_entry)
            await db.flush()  # Populates new_entry.id
            logger.info("memory_entry_created", id=new_entry.id, key=key)
            return new_entry

    except Exception as e:
        logger.error(
            "save_memory_entry_failed",
            user_id=user_id,
            key=key,
            error=str(e),
            exc_info=True,
        )
        raise e


async def get_memory_entries(
    db: AsyncSession,
    user_id: int,
    memory_type: str | None = None,
) -> list[MemoryEntry]:
    """
    Retrieve all long-term memory entries for a user, optionally filtered by type.

    Args:
        db:          AsyncSession
        user_id:     ID of the user
        memory_type: Optional filter (e.g. "fact", "preference")

    Returns:
        List of MemoryEntry ORM objects
    """
    logger.info("retrieving_memory_entries", user_id=user_id, filter_type=memory_type)

    try:
        query = select(MemoryEntry).where(MemoryEntry.user_id == user_id)
        if memory_type:
            query = query.where(MemoryEntry.memory_type == memory_type)

        result = await db.execute(query)
        entries = list(result.scalars().all())

        # Update last accessed timestamp for all retrieved memories (batch update)
        now = datetime.now(timezone.utc)
        for entry in entries:
            entry.last_accessed = now

        logger.info("memory_entries_retrieved", count=len(entries), user_id=user_id)
        return entries

    except Exception as e:
        logger.error(
            "get_memory_entries_failed",
            user_id=user_id,
            error=str(e),
            exc_info=True,
        )
        return []


async def delete_memory_entry(
    db: AsyncSession,
    user_id: int,
    key: str,
) -> bool:
    """
    Delete a specific long-term memory entry by key.

    Args:
        db:      AsyncSession
        user_id: User ID
        key:     Unique memory key to delete

    Returns:
        True if an entry was deleted, False otherwise
    """
    logger.info("deleting_memory_entry", user_id=user_id, key=key)

    try:
        result = await db.execute(
            delete(MemoryEntry).where(
                MemoryEntry.user_id == user_id,
                MemoryEntry.key == key,
            )
        )
        # result.rowcount indicates how many rows were affected
        deleted = result.rowcount > 0
        logger.info("memory_entry_delete_result", key=key, deleted=deleted)
        return deleted

    except Exception as e:
        logger.error(
            "delete_memory_entry_failed",
            user_id=user_id,
            key=key,
            error=str(e),
            exc_info=True,
        )
        return False


async def clear_user_memories(
    db: AsyncSession,
    user_id: int,
) -> int:
    """
    Delete all long-term memory entries for a specific user.

    Args:
        db:      AsyncSession
        user_id: User ID

    Returns:
        Number of entries deleted
    """
    logger.warning("clearing_all_user_memories", user_id=user_id)

    try:
        result = await db.execute(
            delete(MemoryEntry).where(MemoryEntry.user_id == user_id)
        )
        deleted_count = result.rowcount
        logger.info("user_memories_cleared", user_id=user_id, count=deleted_count)
        return deleted_count

    except Exception as e:
        logger.error(
            "clear_user_memories_failed",
            user_id=user_id,
            error=str(e),
            exc_info=True,
        )
        return 0
