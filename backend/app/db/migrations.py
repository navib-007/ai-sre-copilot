"""
db/migrations.py — Lightweight Schema Migration Utility
========================================================
CONCEPT: Why SQLAlchemy create_all() Doesn't Add Columns

  `Base.metadata.create_all()` uses this logic:
    → For each table in metadata:
        IF table doesn't exist in DB → CREATE TABLE (with all columns)
        IF table already exists     → DO NOTHING (skip entirely)

  It will NEVER issue ALTER TABLE to add missing columns.
  This is by design — altering tables can be dangerous (data loss, locks).

  In PRODUCTION, you use Alembic for versioned migrations:
    alembic revision --autogenerate -m "add file_hash to documents"
    alembic upgrade head

  For DEVELOPMENT / LEARNING, we write explicit ALTER TABLE statements.
  This script detects missing columns and adds them safely.

  KEY LESSON: Every time you add a new column to a model, you must
  also write a migration that adds it to existing databases.

Run this script manually when schema changes are needed:
    python -m app.db.migrations

Or it runs automatically at startup (called from lifespan in main.py).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import AsyncSessionLocal, engine, init_db
from app.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


# ─── Migration Registry ───────────────────────────────────────────────────────
# Each migration is a dict describing a column to add if missing.
# Format: {table, column, sql_type, default}
#
# IMPORTANT: Always use nullable=True or provide a DEFAULT for new columns,
# because existing rows need a value for the new column.
# Adding NOT NULL without a default fails if the table has any rows.

COLUMN_MIGRATIONS = [
    # Phase 2: Add file_hash to documents table
    {
        "table": "documents",
        "column": "file_hash",
        "sql_type": "VARCHAR(64)",
        "default": "NULL",          # Existing rows get NULL — acceptable
        "description": "SHA-256 hash of file content for duplicate detection",
    },
]

# ─── Table Migrations ─────────────────────────────────────────────────────────
# New tables added in Phase 2 — create_all() handles these since they're new.
# Listed here for documentation only.
NEW_TABLES_PHASE2 = [
    "embedding_cache",    # Two-level embedding cache
    "document_chunks",    # Chunk registry with Qdrant point IDs
]


async def get_existing_columns(table_name: str) -> set[str]:
    """
    Get the set of column names that currently exist in the database table.

    Uses SQLAlchemy's inspector — a tool for reflecting existing DB schema.

    CONCEPT: Schema Reflection
      SQLAlchemy can "reflect" an existing database schema — reading the actual
      table definitions from the DB and making them available as Python objects.
      This is the opposite of create_all() (which writes schema TO the DB).
    """
    async with engine.connect() as conn:
        # run_sync lets us use synchronous inspector in an async context
        def _inspect(sync_conn):
            inspector = inspect(sync_conn)
            if not inspector.has_table(table_name):
                return set()
            columns = inspector.get_columns(table_name)
            return {col["name"] for col in columns}

        return await conn.run_sync(_inspect)


async def run_column_migrations() -> list[str]:
    """
    Add any missing columns to existing tables.

    For each migration in COLUMN_MIGRATIONS:
      1. Check if the column already exists (idempotent — safe to run multiple times)
      2. If missing → run ALTER TABLE ... ADD COLUMN ...
      3. If exists → skip (log a debug message)

    Returns list of applied migration descriptions.
    """
    applied: list[str] = []

    for migration in COLUMN_MIGRATIONS:
        table = migration["table"]
        column = migration["column"]
        sql_type = migration["sql_type"]
        default = migration["default"]
        description = migration["description"]

        # Check if column already exists
        existing_columns = await get_existing_columns(table)

        if column in existing_columns:
            logger.debug(
                "migration_column_exists_skip",
                table=table,
                column=column,
            )
            continue

        # Column is missing — add it
        logger.info(
            "migration_adding_column",
            table=table,
            column=column,
            sql_type=sql_type,
            description=description,
        )

        # SQLite ALTER TABLE ADD COLUMN syntax:
        #   ALTER TABLE <table> ADD COLUMN <name> <type> DEFAULT <value>
        #
        # SQLite limitations on ALTER TABLE:
        #   - Cannot DROP COLUMN (SQLite < 3.35)
        #   - Cannot ADD CONSTRAINT after table creation
        #   - Cannot change column type
        #   - CAN add a new nullable column or column with DEFAULT ✅
        alter_sql = f"ALTER TABLE {table} ADD COLUMN {column} {sql_type} DEFAULT {default}"

        async with engine.begin() as conn:
            await conn.execute(text(alter_sql))

        logger.info(
            "migration_column_added",
            table=table,
            column=column,
        )
        applied.append(f"ALTER {table}.{column} ({description})")

    return applied


async def run_migrations() -> None:
    """
    Run all pending migrations.

    Call order:
      1. init_db()         → creates any missing tables (new tables from Phase 2)
      2. run_column_migrations() → adds any missing columns to existing tables

    This is idempotent — safe to call on every startup.
    """
    logger.info("migrations_starting")

    # Step 1: Create any new tables (embedding_cache, document_chunks)
    await init_db()
    logger.info("migrations_tables_synced")

    # Step 2: Add missing columns to existing tables
    applied = await run_column_migrations()

    if applied:
        logger.info("migrations_applied", count=len(applied), migrations=applied)
    else:
        logger.info("migrations_no_changes_needed")

    logger.info("migrations_complete")


# ─── Standalone runner ────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup_logging()
    print("\n🔄 Running schema migrations...\n")
    asyncio.run(run_migrations())
    print("\n✅ Migrations complete!\n")
