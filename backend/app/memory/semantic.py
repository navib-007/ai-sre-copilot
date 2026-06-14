"""
app/memory/semantic.py — Semantic Memory Storage (Qdrant Vector DB recall)
==========================================================================
CONCEPT: Semantic Memory (Episodic Recall)
  Short-term memory remembers the recent messages.
  Long-term memory stores structured facts and preferences.
  Semantic memory stores past user-assistant interactions (Q&A pairs) in Qdrant,
  allowing us to perform semantic vector search on new user queries to retrieve
  relevant past resolutions and operational history.

  This enables:
    - Context-aware recall: "I solved this Redis issue last week by rolling back v3.2 config"
    - Cross-session memory search.
"""

import uuid
from datetime import datetime, timezone
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


async def init_semantic_memory_collection(qdrant_client: AsyncQdrantClient) -> None:
    """
    Ensure the semantic memory collection exists in Qdrant and has payload indexes.

    This is called on application startup or lazily before the first write.
    """
    collection_name = settings.qdrant_memory_collection
    dimension = settings.embedding_dimension

    logger.info("initializing_semantic_memory_collection", collection=collection_name)

    try:
        collections = await qdrant_client.get_collections()
        existing_names = [c.name for c in collections.collections]

        if collection_name not in existing_names:
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "qdrant_memory_collection_created",
                collection=collection_name,
                dimension=dimension,
            )
        else:
            logger.debug("qdrant_memory_collection_exists", collection=collection_name)

        # Create payload indexes for pre-filtering
        indexes_to_create = [
            ("user_id", PayloadSchemaType.INTEGER),
            ("session_id", PayloadSchemaType.KEYWORD),
        ]

        for field_name, field_schema in indexes_to_create:
            try:
                await qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception as index_err:
                logger.debug(
                    "qdrant_memory_index_exists_or_failed",
                    field=field_name,
                    error=str(index_err),
                )

    except Exception as e:
        logger.error("qdrant_memory_init_failed", collection=collection_name, error=str(e))


async def store_semantic_memory(
    qdrant_client: AsyncQdrantClient,
    embedder,
    db: AsyncSession,
    user_id: int,
    session_id: str,
    query: str,
    response: str,
) -> None:
    """
    Embed the user query and store the QA pair as a semantic memory in Qdrant.

    Args:
        qdrant_client: The AsyncQdrantClient
        embedder:      CachedEmbedder instance
        db:            AsyncSession for embedder cache lookup
        user_id:       The user ID owner of this memory
        session_id:    The current session ID
        query:         The user's input query
        response:      The agent's generated response
    """
    collection_name = settings.qdrant_memory_collection
    logger.info(
        "storing_semantic_memory",
        user_id=user_id,
        session_id=session_id,
        query_preview=query[:80],
    )

    try:
        # Step 1: Embed the query to capture semantic meaning
        vector = await embedder.embed_text(query, db)

        # Step 2: Prepare the point structure
        point_id = str(uuid.uuid4())
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Step 3: Upsert the vector point to Qdrant
        await qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        logger.info("semantic_memory_stored", point_id=point_id, user_id=user_id)

    except Exception as e:
        logger.error(
            "store_semantic_memory_failed",
            user_id=user_id,
            query=query[:80],
            error=str(e),
            exc_info=True,
        )


async def search_semantic_memory(
    qdrant_client: AsyncQdrantClient,
    embedder,
    db: AsyncSession,
    user_id: int,
    query: str,
    limit: int = 3,
    min_score: float = 0.3,
) -> list[dict]:
    """
    Search Qdrant for semantically similar past query/response interactions of this user.

    Args:
        qdrant_client: AsyncQdrantClient
        embedder:      CachedEmbedder
        db:            AsyncSession
        user_id:       Filter memories belonging only to this user
        query:         The query to find similar interactions for
        limit:         Number of memories to return
        min_score:     Cosine similarity threshold for matches

    Returns:
        List of matching memory payload dicts (with similar past queries and their responses)
    """
    collection_name = settings.qdrant_memory_collection
    logger.info(
        "searching_semantic_memory",
        user_id=user_id,
        query_preview=query[:80],
    )

    try:
        # Step 1: Embed the search query
        query_vector = await embedder.embed_text(query, db)

        # Step 2: Build the filter condition for user_id scoping
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            ]
        )

        # Step 3: Query Qdrant
        search_results = await qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=min_score,
            query_filter=user_filter,
            with_payload=True,
            with_vectors=False,
        )

        # Step 4: Parse matches
        memories = []
        for res in search_results:
            if res.payload:
                memories.append(
                    {
                        "query": res.payload.get("query"),
                        "response": res.payload.get("response"),
                        "score": res.score,
                        "timestamp": res.payload.get("timestamp"),
                    }
                )

        logger.info(
            "semantic_memory_results",
            query_preview=query[:80],
            hits_found=len(memories),
        )
        return memories

    except Exception as e:
        logger.error(
            "semantic_memory_search_failed",
            user_id=user_id,
            error=str(e),
            exc_info=True,
        )
        return []
