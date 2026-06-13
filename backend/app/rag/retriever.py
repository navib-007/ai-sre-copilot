"""
rag/retriever.py — Semantic Search & Retrieval
===============================================
CONCEPT: Retrieval Strategies

  Given a user query ("How do I fix a CrashLoopBackOff?"), we need to find
  the most relevant chunks from our knowledge base.

  NAIVE APPROACH (what we'll start with):
    1. Embed the query
    2. Find the N most similar vectors in Qdrant (by cosine similarity)
    3. Return those chunks

  IMPROVED APPROACHES (we implement progressively):
    1. Score filtering: only return chunks with similarity > threshold
    2. Query expansion: rephrase the query multiple ways, search all of them
    3. Hybrid search: combine dense (embedding) + sparse (BM25) retrieval
    4. Contextual compression: trim retrieved chunks to relevant sentences only
    5. Re-ranking: use a cross-encoder to re-score the initial results

CONCEPT: Cosine Similarity Score
  Qdrant returns a "score" (0.0 to 1.0) for each result.
  - Score 1.0 = identical vectors (exact duplicate)
  - Score 0.9+ = very similar (high confidence)
  - Score 0.7-0.9 = related (medium confidence)
  - Score < 0.7 = loosely related (use with caution)
  - Score < 0.5 = probably irrelevant

  We filter out low-score results to avoid hallucinations:
  "I found nothing relevant" is better than wrong confident answers.

CONCEPT: Metadata Filtering
  Qdrant can filter by payload metadata BEFORE vector similarity.
  Example: "search only within documents of type 'runbook'"
  This is much faster than embedding-based filtering.
"""

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.logging_config import get_logger
from app.rag.embedder import CachedEmbedder

settings = get_settings()
logger = get_logger(__name__)


class SearchResult:
    """A single retrieved document chunk with its similarity score."""

    def __init__(
        self,
        text: str,
        score: float,
        source: str,
        document_id: int,
        chunk_index: int,
        metadata: dict,
    ):
        self.text = text
        self.score = score          # Cosine similarity (0.0 - 1.0)
        self.source = source        # Original filename
        self.document_id = document_id
        self.chunk_index = chunk_index
        self.metadata = metadata

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "source": self.source,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"<SearchResult score={self.score:.3f} source={self.source!r}>"


class RAGRetriever:
    """
    Retrieves relevant document chunks for a given query.

    Usage:
        retriever = RAGRetriever(qdrant_client, embedder)

        results = await retriever.search(
            query="How do I fix CrashLoopBackOff in Kubernetes?",
            db=session,
            top_k=5,
            score_threshold=0.65,
        )
        for result in results:
            print(f"[{result.score:.2f}] {result.source}: {result.text[:100]}")
    """

    def __init__(self, qdrant_client: AsyncQdrantClient, embedder: CachedEmbedder):
        self._qdrant = qdrant_client
        self._embedder = embedder
        self._collection = settings.qdrant_docs_collection

    async def search(
        self,
        query: str,
        db: AsyncSession,
        top_k: int = 5,
        score_threshold: float = 0.60,
        document_id: int | None = None,     # Filter by specific document
        file_type: str | None = None,        # Filter by file type (pdf, md, txt)
    ) -> list[SearchResult]:
        """
        Semantic search: embed query, find similar chunks in Qdrant.

        Args:
            query:           User's search query
            db:              DB session (for embedding cache)
            top_k:           Max number of results to return
            score_threshold: Minimum similarity score (filter noise)
            document_id:     If set, only search within this document
            file_type:       If set, only search documents of this type

        Returns:
            List of SearchResult, sorted by score descending (best first).
        """
        logger.info(
            "rag_search_started",
            query=query[:80],
            top_k=top_k,
            score_threshold=score_threshold,
        )

        # ── Step 1: Embed the query (uses cache) ──────────────────────────────
        # IMPORTANT: The query is embedded with the SAME model as the documents.
        # Using different models = incompatible vector spaces = bad results.
        query_vector = await self._embedder.embed_text(query, db)

        # ── Step 2: Build optional metadata filter ────────────────────────────
        qdrant_filter = self._build_filter(document_id=document_id, file_type=file_type)

        # ── Step 3: Search Qdrant ─────────────────────────────────────────────
        # CONCEPT: Qdrant search returns the top_k nearest vectors
        # by cosine similarity. It uses an HNSW index (Hierarchical
        # Navigable Small World graph) for approximate nearest-neighbor search.
        # This is much faster than brute-force comparison — O(log n) vs O(n).
        search_results = await self._qdrant.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,   # Include the stored metadata
            with_vectors=False,  # Don't return vectors (saves bandwidth)
        )

        # ── Step 4: Convert to SearchResult objects ───────────────────────────
        results: list[SearchResult] = []
        for hit in search_results:
            payload = hit.payload or {}
            results.append(SearchResult(
                text=payload.get("text", ""),
                score=hit.score,
                source=payload.get("source", "unknown"),
                document_id=payload.get("document_id", 0),
                chunk_index=payload.get("chunk_index", 0),
                metadata={
                    k: v for k, v in payload.items()
                    if k not in ("text",)  # Exclude text from metadata (already a field)
                },
            ))

        logger.info(
            "rag_search_complete",
            query=query[:80],
            results_found=len(results),
            top_score=results[0].score if results else 0,
            filtered_by_threshold=score_threshold,
        )

        return results

    async def search_with_context(
        self,
        query: str,
        db: AsyncSession,
        top_k: int = 5,
        score_threshold: float = 0.60,
    ) -> str:
        """
        Search and format results as a context string for the LLM.

        This is what the RAG Agent uses to inject context into the LLM prompt.

        Returns a formatted string like:
            [Source: kubernetes_runbook.md | Relevance: 0.87]
            To restart a Kubernetes pod...

            [Source: kubernetes_runbook.md | Relevance: 0.82]
            If the pod is stuck in CrashLoopBackOff...
        """
        results = await self.search(
            query=query, db=db, top_k=top_k, score_threshold=score_threshold
        )

        if not results:
            return "No relevant documentation found for this query."

        context_parts = []
        for result in results:
            context_parts.append(
                f"[Source: {result.source} | Relevance: {result.score:.2f}]\n"
                f"{result.text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _build_filter(
        self,
        document_id: int | None = None,
        file_type: str | None = None,
    ) -> Filter | None:
        """
        Build a Qdrant metadata filter for narrowing search scope.

        CONCEPT: Pre-filtering vs Post-filtering
          Qdrant filters are applied BEFORE vector similarity search.
          This is efficient because irrelevant vectors are never compared.
          Post-filtering (filter after retrieval) would waste computation.
        """
        conditions = []

        if document_id is not None:
            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            )

        if file_type is not None:
            conditions.append(
                FieldCondition(
                    key="file_type",
                    match=MatchValue(value=file_type.lower().lstrip(".")),
                )
            )

        if not conditions:
            return None

        return Filter(must=conditions)
