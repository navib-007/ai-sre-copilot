"""
rag/embedder.py — OpenAI Embeddings with Two-Level Caching
===========================================================
CONCEPT: What are Embeddings?

  An embedding is a list of numbers (a vector) that represents the MEANING
  of a piece of text in a high-dimensional space.

  text-embedding-3-small produces 1,536-dimensional vectors.
  That means each chunk → a list of 1,536 floats.

  KEY INSIGHT: Texts with similar meanings have vectors that are CLOSE TOGETHER
  in this 1,536-dimensional space. This is called "semantic similarity".

  "How do I restart a pod?" ≈ "kubectl restart pod procedure"
  (they'll have high cosine similarity even though the words differ)

  This is fundamentally different from keyword search (Elasticsearch):
  - Keyword search: must match exact words
  - Semantic search: matches meaning, even with different words

CONCEPT: Two-Level Caching Strategy

  Level 1: In-Memory Cache (Python dict in this process)
    - Ultra-fast: O(1) dictionary lookup
    - Lives as long as the server is running
    - Lost on server restart
    - Use for: hot chunks that are searched repeatedly in one session

  Level 2: SQLite Persistent Cache (EmbeddingCache table)
    - Survives server restarts
    - Shared across all server processes (if you run multiple workers)
    - Slower than RAM but faster than OpenAI API call
    - Use for: all chunks, forever

  Cache lookup order:
    1. Check in-memory dict → HIT: return immediately (< 1ms)
    2. Check SQLite table   → HIT: load into memory, return (< 5ms)
    3. Call OpenAI API      → MISS: store in SQLite + memory (500-2000ms + cost)

  WHY TWO LEVELS?
    Opening the SQLite DB for every request adds ~2-5ms overhead.
    For repeated queries (e.g. same chunk retrieved many times), the
    in-memory cache eliminates even that overhead.
"""

import hashlib
import json
from datetime import datetime, timezone

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import EmbeddingCache
from app.logging_config import get_logger

settings = get_settings()
logger = get_logger(__name__)


class CachedEmbedder:
    """
    Generates text embeddings using OpenAI, with a two-level cache.

    Usage:
        embedder = CachedEmbedder()

        # Single text
        vector = await embedder.embed_text("How to restart Kubernetes pods?", db)

        # Batch (more efficient — fewer API calls)
        results = await embedder.embed_batch(["text1", "text2", ...], db)
    """

    def __init__(self):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._dimension = settings.embedding_dimension

        # ── Level 1: In-Memory Cache ──────────────────────────────────────────
        # Dict[text_hash → embedding_vector]
        # Using a plain dict. In production you'd use an LRU cache with max size.
        self._memory_cache: dict[str, list[float]] = {}

        # Track cache stats for the /cache/stats endpoint
        self._stats = {
            "memory_hits": 0,
            "db_hits": 0,
            "api_calls": 0,
            "total_tokens_saved": 0,
        }

        logger.info(
            "embedder_initialized",
            model=self._model,
            dimension=self._dimension,
        )

    @staticmethod
    def compute_hash(text: str) -> str:
        """
        Compute the SHA-256 hash of text — this is the cache key.

        CONCEPT: Content-Addressable Storage
          The KEY is derived FROM the content, not assigned arbitrarily.
          Same text → same hash → same cache entry.
          This makes deduplication automatic.

          sha256("restart pod") = "a3f5..." (always, forever)
          sha256("restart pod") = "a3f5..." (same!)
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def embed_text(self, text: str, db: AsyncSession) -> list[float]:
        """
        Embed a single text string, using cache if available.

        Args:
            text: Text to embed (a document chunk)
            db:   Database session for SQLite cache lookups

        Returns:
            Embedding vector as list of floats
        """
        results = await self.embed_batch([text], db)
        return results[0]

    async def embed_batch(
        self,
        texts: list[str],
        db: AsyncSession,
    ) -> list[list[float]]:
        """
        Embed multiple texts efficiently, using cache for already-embedded texts.

        CONCEPT: Batch Embedding (cost optimization)
          OpenAI charges per TOKEN, not per API call.
          Batching doesn't reduce cost, but reduces API call overhead
          (fewer round trips = faster processing).

          We batch only the UNCACHED texts.
          If 80% of chunks are cached, we call OpenAI for only 20%.

        Algorithm:
          1. Check each text against both caches
          2. Collect uncached texts into a batch
          3. Call OpenAI ONCE for all uncached texts
          4. Update both caches with results
          5. Return all embeddings in original order

        Args:
            texts: List of texts to embed
            db:    Database session

        Returns:
            List of embedding vectors, same order as input texts
        """
        if not texts:
            return []

        # ── Step 1: Hash all texts and check caches ───────────────────────────
        hashes = [self.compute_hash(text) for text in texts]
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        for i, (text, text_hash) in enumerate(zip(texts, hashes)):
            # Level 1: In-Memory Cache
            if text_hash in self._memory_cache:
                results[i] = self._memory_cache[text_hash]
                self._stats["memory_hits"] += 1
                logger.debug("embedding_cache_l1_hit", hash=text_hash[:12])
                continue

            # Level 2: SQLite Cache
            db_entry = await self._get_from_db_cache(text_hash, db)
            if db_entry is not None:
                results[i] = db_entry
                # Promote to L1 cache (next access will be L1 hit)
                self._memory_cache[text_hash] = db_entry
                self._stats["db_hits"] += 1
                logger.debug("embedding_cache_l2_hit", hash=text_hash[:12])
                continue

            # Cache MISS — needs OpenAI API call
            uncached_indices.append(i)

        # ── Step 2: Batch-embed all uncached texts ────────────────────────────
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            uncached_hashes = [hashes[i] for i in uncached_indices]

            logger.info(
                "embedding_api_call",
                count=len(uncached_texts),
                total=len(texts),
                cached=len(texts) - len(uncached_texts),
            )

            api_embeddings = await self._call_openai(uncached_texts)
            self._stats["api_calls"] += 1

            # ── Step 3: Store results in both caches ──────────────────────────
            for idx, embedding, text_hash, text in zip(
                uncached_indices, api_embeddings, uncached_hashes, uncached_texts
            ):
                results[idx] = embedding

                # Store in SQLite (persistent)
                await self._store_in_db_cache(text_hash, text, embedding, db)

                # Store in memory (fast)
                self._memory_cache[text_hash] = embedding

        # All results should be filled
        assert all(r is not None for r in results), "Some embeddings are None!"
        return results  # type: ignore[return-value]

    async def _call_openai(self, texts: list[str]) -> list[list[float]]:
        """
        Call OpenAI Embeddings API for a batch of texts.

        OpenAI's batch endpoint accepts up to 2,048 texts per call.
        We split into batches of 100 to be conservative.

        Cost: text-embedding-3-small = $0.02 per 1M tokens
        """
        all_embeddings: list[list[float]] = []

        # Process in batches of 100 (safe limit)
        BATCH_SIZE = 100
        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch = texts[batch_start:batch_start + BATCH_SIZE]

            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
                encoding_format="float",  # Return as list of floats (not base64)
            )

            # Sort by index (API returns them in order, but let's be safe)
            sorted_embeddings = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_embeddings]
            all_embeddings.extend(batch_embeddings)

            usage = response.usage
            logger.info(
                "openai_embedding_call",
                batch_size=len(batch),
                prompt_tokens=usage.prompt_tokens,
                total_tokens=usage.total_tokens,
                model=self._model,
            )

        return all_embeddings

    async def _get_from_db_cache(
        self, text_hash: str, db: AsyncSession
    ) -> list[float] | None:
        """Look up an embedding vector from the SQLite cache."""
        try:
            result = await db.execute(
                select(EmbeddingCache).where(
                    EmbeddingCache.text_hash == text_hash,
                    EmbeddingCache.model == self._model,
                )
            )
            entry = result.scalar_one_or_none()

            if entry is None:
                return None

            # Update cache hit stats
            entry.hit_count += 1
            entry.last_used_at = datetime.now(timezone.utc)

            return json.loads(entry.embedding_json)

        except Exception as e:
            # Cache read failure should never crash the application
            logger.warning("db_cache_read_failed", error=str(e), hash=text_hash[:12])
            return None

    async def _store_in_db_cache(
        self,
        text_hash: str,
        text: str,
        embedding: list[float],
        db: AsyncSession,
    ) -> None:
        """Store an embedding vector in the SQLite cache."""
        try:
            # Check if already exists (race condition protection)
            existing = await db.execute(
                select(EmbeddingCache).where(EmbeddingCache.text_hash == text_hash)
            )
            if existing.scalar_one_or_none() is not None:
                return  # Already cached (by another concurrent request)

            cache_entry = EmbeddingCache(
                text_hash=text_hash,
                model=self._model,
                embedding_json=json.dumps(embedding),
                dimension=len(embedding),
                hit_count=0,
            )
            db.add(cache_entry)
            # Note: The db session auto-commits when the request completes.
            # We use flush() here so the entry is visible within this transaction.
            await db.flush()

            logger.debug(
                "embedding_cached",
                hash=text_hash[:12],
                dimension=len(embedding),
                model=self._model,
            )

        except Exception as e:
            # Cache write failure is non-fatal — just means future calls won't benefit
            logger.warning("db_cache_write_failed", error=str(e), hash=text_hash[:12])

    def get_cache_stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self._stats["memory_hits"] + self._stats["db_hits"] + self._stats["api_calls"]
        hit_rate = (
            (self._stats["memory_hits"] + self._stats["db_hits"]) / total
            if total > 0 else 0
        )
        return {
            "memory_cache_size": len(self._memory_cache),
            "memory_hits": self._stats["memory_hits"],
            "db_hits": self._stats["db_hits"],
            "api_calls": self._stats["api_calls"],
            "total_requests": total,
            "hit_rate_percent": round(hit_rate * 100, 1),
        }

    def clear_memory_cache(self) -> int:
        """Clear the in-memory cache. Returns number of entries cleared."""
        count = len(self._memory_cache)
        self._memory_cache.clear()
        logger.info("memory_cache_cleared", entries_removed=count)
        return count


# ─── Singleton Instance ───────────────────────────────────────────────────────
# One embedder per application process.
# The memory cache is shared across all requests in the same process.
_embedder_instance: CachedEmbedder | None = None


def get_embedder() -> CachedEmbedder:
    """
    Get the singleton CachedEmbedder instance.

    WHY SINGLETON?
      - The in-memory cache (self._memory_cache) must be shared across requests.
      - Creating a new CachedEmbedder per request would lose the L1 cache.
      - One instance per process = shared memory cache across all requests.
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = CachedEmbedder()
    return _embedder_instance
