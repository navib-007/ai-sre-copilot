"""
main.py — FastAPI Application Entry Point
==========================================
CONCEPT: FastAPI Application Lifecycle

  FastAPI apps go through this lifecycle:
    1. STARTUP  → Initialize DB, set up logging, seed data, connect external services
    2. RUNNING  → Handle incoming HTTP requests  
    3. SHUTDOWN → Close DB connections, flush logs, release resources

  We use `@asynccontextmanager` for the lifespan — modern FastAPI approach.
  This replaced the old @app.on_event("startup") pattern.

CONCEPT: Middleware Stack
  Middleware runs AROUND every request (before and after your route handler).
  Our middleware stack (applied in reverse order — last added = first run):
    
    Request comes in ↓
    ┌─────────────────────────┐
    │  Request ID Middleware  │  ← adds X-Request-ID header
    ├─────────────────────────┤
    │  CORS Middleware        │  ← handles Cross-Origin requests
    ├─────────────────────────┤
    │  Logging Middleware     │  ← logs every request + response time
    ├─────────────────────────┤
    │  Route Handler          │  ← your actual endpoint code
    └─────────────────────────┘
    Response goes out ↑

  Each middleware can modify the request (going down) or response (going up).
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db.database import close_db, init_db
from app.db.migrations import run_migrations
from app.logging_config import get_logger, setup_logging
from app.routes import incidents, tickets
from app.schemas.common import HealthResponse

settings = get_settings()
logger = get_logger(__name__)

# Track startup time for uptime calculation
_startup_time: float = 0.0


# ─── Lifespan Manager ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application lifecycle: startup and shutdown events.

    Everything BEFORE yield = startup code
    Everything AFTER yield  = shutdown code

    This is called once when the server starts and once when it stops.
    """
    global _startup_time
    _startup_time = time.time()

    # ── STARTUP ──────────────────────────────────────────────────────────────
    # Step 1: Initialize structured logging FIRST (so all startup logs are structured)
    setup_logging()
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        debug=settings.debug,
    )

    # Step 2: Run schema migrations
    # This replaces the bare init_db() call.
    # run_migrations() does BOTH:
    #   a) create_all() → creates new tables (embedding_cache, document_chunks)
    #   b) ALTER TABLE → adds new columns to existing tables (documents.file_hash)
    # It is idempotent — safe to call on every restart.
    logger.info("database_migrations_starting")
    await run_migrations()
    logger.info("database_migrations_complete")

    # Step 2b: Ensure Qdrant collection and payload indexes exist
    from app.rag.qdrant_client import get_ingestion_pipeline
    pipeline = get_ingestion_pipeline()
    await pipeline.ensure_collection_exists()
    logger.info("qdrant_collection_ready")

    # Step 2c: Ensure Qdrant semantic memory collection exists
    from app.rag.qdrant_client import get_qdrant_client
    from app.memory.semantic import init_semantic_memory_collection
    qclient = get_qdrant_client()
    await init_semantic_memory_collection(qclient)
    logger.info("qdrant_semantic_memory_ready")

    # Step 3: Seed sample data (idempotent — safe to call every time)
    try:
        from app.db.seed import run_seed
        await run_seed()
    except Exception as e:
        logger.warning("seed_failed_continuing", error=str(e))

    # Step 4: Configure LangSmith tracing (if enabled)
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
        logger.info("langsmith_tracing_enabled", project=settings.langchain_project)
    else:
        logger.info("langsmith_tracing_disabled")

    logger.info("application_started", host=settings.host, port=settings.port)

    # ── YIELD (app is running) ────────────────────────────────────────────────
    yield

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    logger.info("application_shutting_down")
    await close_db()
    logger.info("application_shutdown_complete")


# ─── FastAPI Application ──────────────────────────────────────────────────────
def create_app() -> FastAPI:
    """
    Application factory pattern.

    WHY FACTORY PATTERN?
      Instead of creating `app` at module level, we use a factory function.
      Benefits:
        - Easy to create test instances with different settings
        - Cleaner to configure different middleware per environment
        - Avoids circular import issues
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
## Agentic AI Ops Platform API

A production-grade IT Operations AI assistant powered by LangGraph multi-agent system.

### Features
- **RAG Pipeline** — Query knowledge base documents
- **Ticket Management** — Create and manage support tickets
- **Incident Management** — Report and investigate incidents
- **Multi-Agent Orchestration** — LangGraph supervisor with specialized agents
- **Human-in-the-Loop** — Approval workflows for high-risk actions

### Authentication
Use `POST /api/auth/login` to get a JWT token, then include it as:
`Authorization: Bearer <token>`

*(Auth is added in Phase 9 — currently open for development)*
        """,
        docs_url="/docs",          # Swagger UI at http://localhost:8000/docs
        redoc_url="/redoc",        # ReDoc UI at http://localhost:8000/redoc
        openapi_url="/openapi.json",
        lifespan=lifespan,
        # Return more details in validation error responses
        # This helps during development to understand what went wrong
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    _add_middleware(app)

    # ── Exception Handlers ────────────────────────────────────────────────────
    _add_exception_handlers(app)

    # ── Routers ───────────────────────────────────────────────────────────────
    _add_routers(app)

    return app


def _add_middleware(app: FastAPI) -> None:
    """Register all middleware with the application."""

    # 1. CORS (Cross-Origin Resource Sharing)
    # CONCEPT: CORS lets your Streamlit frontend (localhost:8501)
    # call this API (localhost:8000). Without CORS, browsers block
    # cross-origin requests for security.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],      # Allow GET, POST, PATCH, DELETE, OPTIONS, etc.
        allow_headers=["*"],      # Allow all headers including Authorization
    )

    # 2. Request Logging Middleware
    # CONCEPT: We add a middleware that:
    #   - Generates a unique Request ID (for log correlation)
    #   - Times the request duration
    #   - Logs request in / response out with structured fields
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        """Log every HTTP request with timing and correlation ID."""
        # Generate unique request ID for tracing this request across logs
        request_id = str(uuid.uuid4())[:8]  # Short 8-char ID for readability

        # Store request_id in request state (accessible in route handlers)
        request.state.request_id = request_id

        start_time = time.perf_counter()

        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        try:
            response: Response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "request_completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

            # Add request ID to response headers (useful for debugging)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = str(duration_ms)
            return response

        except Exception as e:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "request_failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
                error=str(e),
                exc_info=True,
            )
            raise


def _add_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""

    # CONCEPT: Global exception handlers catch unhandled errors and
    # return structured JSON responses instead of HTML error pages.
    # This ensures the API always returns JSON (not HTML) on errors.

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "Not Found",
                "detail": f"The endpoint {request.url.path} does not exist",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred. Check logs for details.",
                "request_id": getattr(request.state, "request_id", None),
            },
        )


def _add_routers(app: FastAPI) -> None:
    """Register all API routers with the /api prefix."""

    # All API routes live under /api/
    # This makes it easy to:
    #   - Version the API: /api/v1/, /api/v2/
    #   - Set up a reverse proxy (nginx routes /api/ to FastAPI)
    #   - Serve static files at / without conflicting with API

    API_PREFIX = "/api"

    # Phase 1 routes
    app.include_router(tickets.router, prefix=API_PREFIX)
    app.include_router(incidents.router, prefix=API_PREFIX)

    # Phase 2 routes
    from app.routes import documents
    app.include_router(documents.router, prefix=API_PREFIX)

    # Phase 3 routes — AI Agent Chat
    from app.routes import chat
    app.include_router(chat.router, prefix=API_PREFIX)

    # ── Root endpoint ─────────────────────────────────────────────────────────
    @app.get("/", tags=["Root"], summary="API welcome")
    async def root() -> dict:
        """Welcome endpoint — confirms the API is running."""
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
        }

    # ── Health Check ──────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["System"], summary="Health check")
    async def health_check() -> HealthResponse:
        """
        Health check endpoint.

        CONCEPT: Health Checks
          Load balancers and orchestration systems (Kubernetes) ping /health
          to know if the service is alive and ready to handle traffic.

          Returns:
            - 200 OK  = Service is healthy
            - 503     = Service is degraded or unhealthy

          We check: database connectivity + any other critical dependencies.
        """
        from sqlalchemy import text
        from app.db.database import engine

        # Test DB connection
        db_status = "disconnected"
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            db_status = "connected"
        except Exception as e:
            logger.error("health_check_db_failed", error=str(e))

        uptime = time.time() - _startup_time if _startup_time > 0 else 0

        return HealthResponse(
            status="healthy" if db_status == "connected" else "degraded",
            app_name=settings.app_name,
            version=settings.app_version,
            environment=settings.environment,
            database=db_status,
            uptime_seconds=round(uptime, 2),
        )


# ─── Application Instance ─────────────────────────────────────────────────────
# Module-level `app` is what uvicorn imports to run the server.
app = create_app()


# ─── Development Runner ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,      # Auto-reload on code changes in development
        log_level=settings.log_level.lower(),
    )
