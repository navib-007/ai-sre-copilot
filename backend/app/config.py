"""
config.py — Centralized Application Settings
=============================================
CONCEPT: The Settings class uses pydantic-settings to:
  1. Read from environment variables
  2. Read from a .env file (auto-discovered)
  3. Provide type validation and defaults
  4. Give you one place to change any configuration

WHY THIS MATTERS:
  - Avoid hardcoded values scattered across files
  - Easy environment switching (dev → staging → prod)
  - Type-safe access to config values (settings.DEBUG vs os.environ["DEBUG"])
  - Secrets never go in code
"""

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Base Directory ───────────────────────────────────────────────────────────
# __file__ = this file's path
# .parent   = app/ directory
# .parent   = backend/ directory
BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    Pydantic-settings reads values in this priority order:
      1. Environment variables (highest priority)
      2. .env file
      3. Default values defined here (lowest priority)
    """

    # ── Model config ──────────────────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",          # Look for .env in backend/ folder
        env_file_encoding="utf-8",
        case_sensitive=False,                # APP_NAME == app_name
        extra="ignore",                      # Ignore unknown env vars
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = Field(default="Agentic AI Ops Platform")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    environment: str = Field(default="development")

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default=f"sqlite+aiosqlite:///{BASE_DIR}/data/agentic_ops.db"
    )

    # ── Authentication ────────────────────────────────────────────────────────
    secret_key: str = Field(default="change-this-secret-key-in-production")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=60)
    refresh_token_expire_days: int = Field(default=7)

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="")
    llm_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimension: int = Field(default=1536)

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = Field(default="")
    qdrant_api_key: str = Field(default="")
    qdrant_docs_collection: str = Field(default="knowledge_base")
    qdrant_timeout_seconds: int = Field(
        default=30,
        description=(
            "Qdrant client request timeout in seconds. "
            "Default 5s is too short for cross-region cloud clusters (e.g., India → us-west-1). "
            "Set to 30s or higher for cloud deployments with network latency."
        ),
    )
    qdrant_memory_collection: str = Field(default="semantic_memory")

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="agentic-ai-ops")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com")

    # ── Agent Configuration (Phase 3+) ────────────────────────────────────────
    # LLM generation parameters
    agent_temperature: float = Field(
        default=0.0,
        description="LLM temperature for agents (0.0=deterministic, 1.0=creative)",
    )
    agent_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens in agent LLM response",
    )
    agent_max_iterations: int = Field(
        default=3,
        description="Maximum ReAct iterations before agent stops (prevents infinite loops)",
    )

    # RAG retrieval parameters
    rag_max_chunks: int = Field(
        default=5,
        description="Maximum document chunks to retrieve per RAG search",
    )
    rag_min_score: float = Field(
        default=0.30,
        description="Minimum cosine similarity score for RAG results (0.0-1.0)",
    )

    # ── Supervisor Configuration (Phase 4+) ──────────────────────────────────
    # Intent routing thresholds
    supervisor_intent_confidence_threshold: float = Field(
        default=0.70,
        description=(
            "Minimum confidence for intent routing. "
            "Below this threshold, defaults to knowledge_query (safest fallback)."
        ),
    )
    # Enable/disable specific agents (useful for testing individual agents)
    enable_ticket_agent: bool = Field(
        default=True,
        description="Enable the Ticket Management specialist agent.",
    )
    enable_incident_agent: bool = Field(
        default=True,
        description="Enable the Incident Investigation specialist agent.",
    )


    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="DEBUG")
    log_format: str = Field(default="console")  # "json" | "console"
    log_file: str = Field(default="")            # Empty = stdout only

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_per_minute: int = Field(default=60)
    rate_limit_chat_per_minute: int = Field(default=20)

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: str = Field(default="http://localhost:8501,http://localhost:3000")

    # ── Computed Properties ───────────────────────────────────────────────────
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def data_dir(self) -> Path:
        """Path to the data directory (auto-created on startup)."""
        return BASE_DIR / "data"

    @property
    def logs_dir(self) -> Path:
        """Path to the logs directory (auto-created on startup)."""
        return BASE_DIR / "logs"

    @property
    def is_production(self) -> bool:
        """True when running in production environment."""
        return self.environment == "production"


# ─── Singleton Pattern ────────────────────────────────────────────────────────
# @lru_cache ensures Settings() is only instantiated ONCE.
# This is important because reading from .env is a disk I/O operation.
# Every part of the app calls get_settings() to get the same instance.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the cached Settings singleton.

    CONCEPT: .env vs System Environment Variable Priority
      pydantic-settings priority (highest → lowest):
        1. System/process environment variables  ← stale old keys win here!
        2. .env file values
        3. Field defaults

      PROBLEM: If OPENAI_API_KEY was set as a Windows system env var
      from a previous project, it overrides the value in your .env file.

      SOLUTION: We use dotenv_values() to read .env directly as a plain
      dict (without touching os.environ), then inject those values back
      into os.environ for the keys most likely to have stale values.
      This makes .env always win for our sensitive API keys.

    Usage:
        from app.config import get_settings
        settings = get_settings()
        print(settings.openai_api_key)
    """
    import os
    from dotenv import dotenv_values

    env_file = BASE_DIR / ".env"

    # Keys most likely to have stale values from other projects in system env
    FORCE_FROM_DOTENV = {
        "OPENAI_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_ENDPOINT",
        "QDRANT_URL",
        "QDRANT_API_KEY",
    }

    if env_file.exists():
        # dotenv_values reads the file as a plain dict — does NOT modify os.environ
        dotenv_vals = dotenv_values(env_file)
        for key in FORCE_FROM_DOTENV:
            val = dotenv_vals.get(key)
            if val:  # Only override if .env has a non-empty value
                os.environ[key] = val

    return Settings()


# ─── Module-level convenience ─────────────────────────────────────────────────
# Some modules import `settings` directly for brevity
settings = get_settings()
