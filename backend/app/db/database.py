"""
database.py — SQLAlchemy Database Engine & Session Management
=============================================================
CONCEPT: SQLAlchemy Async ORM

  SQLAlchemy provides two ways to talk to databases:
  1. Core    — Write SQL directly, but with Python (low-level)
  2. ORM     — Map Python classes to tables (what we use)

  We use the ASYNC version (AsyncSession) because FastAPI is async.
  Async means: while waiting for a DB query, Python can handle OTHER requests.
  This is critical for performance when you have many concurrent users.

CONCEPT: Session per Request Pattern
  Each HTTP request gets its own DB session.
  The session is:
    - Created at request start  (via get_db dependency)
    - Used throughout the request
    - Committed on success
    - Rolled back on error
    - Closed when request ends
  This prevents data corruption from multiple requests sharing a session.
"""

from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ─── Declarative Base ─────────────────────────────────────────────────────────
# All ORM models inherit from Base.
# SQLAlchemy uses this to track all mapped table classes.
class Base(DeclarativeBase):
    """
    Base class for all database models.

    Every model we create will look like:
        class User(Base):
            __tablename__ = "users"
            id: Mapped[int] = mapped_column(primary_key=True)
            ...
    """
    pass


# ─── Engine ───────────────────────────────────────────────────────────────────
def create_engine() -> AsyncEngine:
    """
    Create the async database engine.

    Engine = connection pool to the database.
    One engine per application (singleton pattern).

    For SQLite specifically:
      - check_same_thread=False: SQLite default restricts threads.
        We disable this because asyncio uses a single thread anyway.
    """
    # Ensure the data directory exists before SQLite tries to create the file
    db_url = settings.database_url
    if "sqlite" in db_url:
        # Extract file path from: sqlite+aiosqlite:///./data/agentic_ops.db
        db_path_str = db_url.replace("sqlite+aiosqlite:///", "")
        db_path = Path(db_path_str)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("database_path", path=str(db_path.resolve()))

    engine = create_async_engine(
        db_url,
        # ── SQLite-specific settings ──────────────────────────────────────────
        connect_args={
            "check_same_thread": False,
            "timeout": 30,  # 30 seconds busy timeout to wait for locks to clear
        } if "sqlite" in db_url else {},
        # ── Connection pool settings ──────────────────────────────────────────
        # pool_pre_ping: Test connection before using it (handles dropped connections)
        pool_pre_ping=True,
        # ── Debugging ────────────────────────────────────────────────────────
        # echo=False explicitly to prevent stdout corruption during MCP server execution
        echo=False,
    )

    # Listen to connection events on the sync engine underneath the async engine to set WAL mode
    if "sqlite" in db_url:
        from sqlalchemy import event
        @event.listens_for(engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.close()
        logger.info("database_sqlite_pragmas_registered")

    logger.info("database_engine_created", url=db_url.split("///")[0])
    return engine


# ─── Session Factory ──────────────────────────────────────────────────────────
# async_sessionmaker creates new AsyncSession instances.
# expire_on_commit=False: Don't expire objects after commit.
#   Without this, accessing obj.id AFTER commit would trigger another DB query.
#   Setting False means we can still use the object after committing.
def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to the given engine."""
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


# ─── Module-level singletons ──────────────────────────────────────────────────
# Created once when this module is first imported
engine: AsyncEngine = create_engine()
AsyncSessionLocal: async_sessionmaker[AsyncSession] = create_session_factory(engine)


# ─── Dependency: get_db ───────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session per request.

    CONCEPT: FastAPI Dependency Injection
      FastAPI's `Depends(get_db)` system automatically:
      1. Calls this function before the route handler
      2. Injects the yielded session into the handler
      3. Continues execution after the `yield` to close the session

    Usage in routes:
        @router.get("/tickets")
        async def get_tickets(db: AsyncSession = Depends(get_db)):
            tickets = await db.execute(select(Ticket))
            return tickets.scalars().all()

    The try/finally ensures the session is ALWAYS closed, even on errors.
    """
    async with AsyncSessionLocal() as session:
        try:
            logger.debug("db_session_opened")
            yield session
            await session.commit()
            logger.debug("db_session_committed")
        except Exception as e:
            await session.rollback()
            logger.error("db_session_rolled_back", error=str(e))
            raise
        finally:
            await session.close()
            logger.debug("db_session_closed")


# ─── Database Initialization ──────────────────────────────────────────────────
async def init_db() -> None:
    """
    Create all database tables if they don't exist.

    Called once at application startup via the lifespan event in main.py.

    IMPORTANT: In production, use Alembic migrations instead of create_all().
    create_all() is safe for development (won't drop existing tables),
    but Alembic gives you proper versioned schema migrations.
    """
    # Import models so SQLAlchemy's metadata knows about them
    from app.db import models  # noqa: F401 (import for side effect)

    async with engine.begin() as conn:
        # Create all tables defined in Base.metadata
        await conn.run_sync(Base.metadata.create_all)

    logger.info("database_tables_created")


# ─── LangGraph Checkpointer Persistence ──────────────────────────────────────────
# We use two variables:
# 1. _checkpointer_context: Stores the async context manager returned by from_conn_string().
# 2. _checkpointer_instance: Stores the actual active AsyncSqliteSaver instance yielded
#    when the context manager is entered.
_checkpointer_context = None
_checkpointer_instance = None


async def get_checkpointer():
    """
    Get the LangGraph AsyncSqliteSaver checkpointer connection instance singleton.
    This manages agent state checkpointer persistence in data/checkpoints.db.
    Entering the async context manager yields the actual saver instance which is stored
    locally and returned for agent runtime execution.
    """
    global _checkpointer_context, _checkpointer_instance
    if _checkpointer_instance is None:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        import os

        db_dir = settings.data_dir
        os.makedirs(db_dir, exist_ok=True)
        checkpoint_db_path = db_dir / "checkpoints.db"
        logger.info("checkpointer_database_path", path=str(checkpoint_db_path.resolve()))
        
        # Instantiate the async generator context manager
        _checkpointer_context = AsyncSqliteSaver.from_conn_string(str(checkpoint_db_path.resolve()))
        # Enter the context manager to obtain the actual AsyncSqliteSaver database connection instance
        _checkpointer_instance = await _checkpointer_context.__aenter__()
        
        # Configure SQLite pragmas to prevent database locks
        try:
            await _checkpointer_instance.conn.execute("PRAGMA journal_mode=WAL;")
            await _checkpointer_instance.conn.execute("PRAGMA synchronous=NORMAL;")
            await _checkpointer_instance.conn.execute("PRAGMA busy_timeout=30000;")
            await _checkpointer_instance.conn.commit()
            logger.info("checkpointer_pragmas_configured")
        except Exception as pe:
            logger.warning("checkpointer_pragmas_failed", error=str(pe))
            
        logger.info("checkpointer_entered_successfully", instance_id=id(_checkpointer_instance))
        
    return _checkpointer_instance


def get_checkpointer_sync():
    """
    Get the LangGraph checkpointer synchronously.
    Requires checkpointer to have been initialized during startup lifespan.
    This returns the actual AsyncSqliteSaver database connection instance.
    """
    global _checkpointer_instance
    return _checkpointer_instance


async def close_db() -> None:
    """Dispose the engine connection pool. Called at application shutdown."""
    await engine.dispose()
    logger.info("database_engine_disposed")

    global _checkpointer_context, _checkpointer_instance
    if _checkpointer_context is not None:
        try:
            # Properly exit the context manager to close the SQLite connection cleanly
            await _checkpointer_context.__aexit__(None, None, None)
            logger.info("checkpointer_connection_closed")
        except Exception as e:
            logger.warning("checkpointer_connection_close_failed", error=str(e))
        _checkpointer_context = None
        _checkpointer_instance = None

