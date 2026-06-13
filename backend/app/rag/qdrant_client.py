"""
rag/qdrant_client.py — Qdrant Client Factory & Dependency
==========================================================
CONCEPT: Dependency Injection for External Clients

  Just like `get_db` provides a DB session, `get_qdrant_client`
  provides a Qdrant client via FastAPI's Depends() system.

  We use a singleton client (one per process) because:
    - Opening a new connection per request is expensive
    - AsyncQdrantClient has its own connection pool
    - The client is stateless — safe to share across requests
"""

from functools import lru_cache

from qdrant_client import AsyncQdrantClient

from app.config import get_settings
from app.logging_config import get_logger
from app.rag.embedder import CachedEmbedder, get_embedder
from app.rag.ingestion import IngestionPipeline
from app.rag.retriever import RAGRetriever

settings = get_settings()
logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_qdrant_client() -> AsyncQdrantClient:
    """
    Create and cache the Qdrant client singleton.

    Connects to Qdrant Cloud using URL + API key from settings.
    Falls back to in-memory mode if URL is not configured (for tests).
    """
    if settings.qdrant_url and settings.qdrant_api_key:
        logger.info("qdrant_client_cloud", url=settings.qdrant_url[:30])
        return AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    else:
        # In-memory Qdrant — data lost on restart, perfect for testing
        logger.warning("qdrant_client_in_memory")
        return AsyncQdrantClient(":memory:")


def get_ingestion_pipeline() -> IngestionPipeline:
    """FastAPI dependency: get IngestionPipeline with shared clients."""
    return IngestionPipeline(
        qdrant_client=get_qdrant_client(),
        embedder=get_embedder(),
    )


def get_retriever() -> RAGRetriever:
    """FastAPI dependency: get RAGRetriever with shared clients."""
    return RAGRetriever(
        qdrant_client=get_qdrant_client(),
        embedder=get_embedder(),
    )
