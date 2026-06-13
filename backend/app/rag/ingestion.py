"""
rag/ingestion.py — Document Ingestion Pipeline
===============================================
CONCEPT: Ingestion Pipeline (ETL for RAG)

  ETL = Extract, Transform, Load (a classic data engineering pattern).

  In our RAG context:
    EXTRACT   → Read the uploaded file bytes
    TRANSFORM → Parse text → chunk → embed (with caching)
    LOAD      → Store chunks in Qdrant + metadata in SQLite

  This pipeline is triggered when a user uploads a document.
  It runs asynchronously so the HTTP response returns quickly
  while processing happens in the background.

  DOCUMENT-LEVEL DEDUPLICATION:
    Before processing, we compute SHA-256 of the entire file.
    If a document with the same hash already exists → skip processing.
    This handles: same file re-uploaded, same file renamed and re-uploaded.

  CHUNK-LEVEL DEDUPLICATION:
    Handled by the EmbeddingCache (see embedder.py).
    Even for new documents, individual chunks that exist in other documents
    won't be re-embedded.

  QDRANT COLLECTION:
    We store all document chunks in the "knowledge_base" collection.
    Each point in Qdrant has:
      - vector: the 1536-float embedding
      - payload: metadata (doc_id, chunk_index, text, source, etc.)
    The payload lets us do filtered searches (e.g. "only search doc #5").
"""

import hashlib
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Document, DocumentChunk
from app.logging_config import get_logger
from app.rag.chunker import DocumentChunker
from app.rag.embedder import CachedEmbedder

settings = get_settings()
logger = get_logger(__name__)


class IngestionPipeline:
    """
    Orchestrates the full document → chunks → embeddings → Qdrant pipeline.

    Usage:
        pipeline = IngestionPipeline(qdrant_client, embedder)

        result = await pipeline.ingest_document(
            file_bytes=b"...",
            filename="kubernetes_runbook.md",
            file_type="md",
            uploaded_by=1,
            db=session,
        )
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        embedder: CachedEmbedder,
        chunker: DocumentChunker | None = None,
    ):
        self._qdrant = qdrant_client
        self._embedder = embedder
        self._chunker = chunker or DocumentChunker()
        self._collection = settings.qdrant_docs_collection

    async def ensure_collection_exists(self) -> None:
        """
        Create the Qdrant collection (if missing) and ensure payload indexes exist.

        CONCEPT: Qdrant Payload Indexes
          In Qdrant, payload fields are just JSON metadata stored alongside each vector.
          By default you CANNOT filter by them — Qdrant would have to scan ALL points.

          A payload index is like a B-tree index in PostgreSQL:
          Without index: O(n) scan through all points to find document_id == 5
          With index:    O(log n) lookup — instant even with millions of points

          CRITICAL: If you try to filter by a field without an index, Qdrant
          throws: "Index required but not found for <field>"

          Index types:
            - PayloadSchemaType.INTEGER  → for numeric fields (document_id, chunk_index)
            - PayloadSchemaType.KEYWORD  → for exact string match (file_type, source)
            - PayloadSchemaType.FLOAT    → for numeric range filters
            - PayloadSchemaType.TEXT     → for full-text search within the payload field

          We index: document_id (integer) and file_type (keyword).
          We do NOT index: text, chunk_hash (too large, not used in filters).

          create_payload_index() is idempotent — safe to call even if index exists.
        """
        from qdrant_client.models import PayloadSchemaType

        collections = await self._qdrant.get_collections()
        existing_names = [c.name for c in collections.collections]

        if self._collection not in existing_names:
            await self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "qdrant_collection_created",
                collection=self._collection,
                dimension=settings.embedding_dimension,
            )
        else:
            logger.debug("qdrant_collection_exists", collection=self._collection)

        # ── Create payload indexes (idempotent — safe on existing collections) ─
        # These are required for filtering. Must exist BEFORE any filtered search.
        indexes_to_create = [
            ("document_id", PayloadSchemaType.INTEGER),   # filter by specific document
            ("file_type", PayloadSchemaType.KEYWORD),      # filter by md / pdf / txt
            ("source", PayloadSchemaType.KEYWORD),         # filter by filename
            ("chunk_index", PayloadSchemaType.INTEGER),    # for future ordered retrieval
        ]

        for field_name, field_schema in indexes_to_create:
            try:
                await self._qdrant.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=field_schema,
                )
                logger.debug(
                    "qdrant_payload_index_ready",
                    field=field_name,
                    schema=field_schema,
                )
            except Exception as e:
                # Qdrant raises if index already exists — that's fine, just log it
                logger.debug(
                    "qdrant_payload_index_skipped",
                    field=field_name,
                    reason=str(e)[:80],
                )


    async def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        file_type: str,
        uploaded_by: int,
        db: AsyncSession,
        extra_metadata: dict | None = None,
    ) -> dict:
        """
        Full ingestion pipeline: file bytes → chunks → embeddings → Qdrant.

        Returns a summary dict with ingestion stats.
        """
        logger.info(
            "ingestion_started",
            filename=filename,
            file_type=file_type,
            size_bytes=len(file_bytes),
        )

        # ── Step 1: Document-Level Deduplication ──────────────────────────────
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        existing_doc = await self._check_duplicate_document(file_hash, db)
        if existing_doc:
            logger.info(
                "document_duplicate_skipped",
                filename=filename,
                existing_doc_id=existing_doc.id,
                file_hash=file_hash[:12],
            )
            return {
                "status": "duplicate",
                "message": f"Document already exists (uploaded as '{existing_doc.filename}')",
                "existing_document_id": existing_doc.id,
                "file_hash": file_hash[:12],
                "chunks_processed": 0,
                "cache_hits": 0,
                "api_calls": 0,
            }

        # ── Step 2: Ensure Qdrant collection exists ───────────────────────────
        await self.ensure_collection_exists()

        # ── Step 3: Create Document record in SQLite ──────────────────────────
        document = Document(
            filename=filename,
            file_type=file_type,
            file_size_bytes=len(file_bytes),
            file_hash=file_hash,
            collection_name=self._collection,
            uploaded_by=uploaded_by,
            chunk_count=0,  # Will update after chunking
            metadata_json=None,
        )
        db.add(document)
        await db.flush()  # Get the auto-generated document.id

        logger.info("document_record_created", doc_id=document.id, filename=filename)

        # ── Step 4: Parse + Chunk the document ───────────────────────────────
        chunks = self._chunker.chunk_file_content(
            file_bytes=file_bytes,
            filename=filename,
            file_type=file_type,
            extra_metadata=extra_metadata,
        )

        if not chunks:
            logger.warning("no_chunks_produced", filename=filename)
            return {
                "status": "error",
                "message": "No text could be extracted from the document",
                "document_id": document.id,
                "chunks_processed": 0,
            }

        # ── Step 5: Embed all chunks (with two-level caching) ─────────────────
        logger.info("embedding_chunks", count=len(chunks), filename=filename)

        stats_before = self._embedder.get_cache_stats()
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = await self._embedder.embed_batch(chunk_texts, db)
        stats_after = self._embedder.get_cache_stats()

        new_api_calls = stats_after["api_calls"] - stats_before["api_calls"]
        new_cache_hits = (
            (stats_after["memory_hits"] + stats_after["db_hits"])
            - (stats_before["memory_hits"] + stats_before["db_hits"])
        )

        # ── Step 6: Store vectors in Qdrant ───────────────────────────────────
        logger.info("storing_vectors_qdrant", count=len(chunks), collection=self._collection)

        qdrant_points: list[PointStruct] = []
        chunk_records: list[DocumentChunk] = []

        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            chunk_hash = self._embedder.compute_hash(chunk.text)

            # Build Qdrant point
            # CONCEPT: Qdrant Payload
            #   Every vector point can have a "payload" — arbitrary JSON metadata.
            #   This metadata is stored alongside the vector and returned with results.
            #   We store enough info to reconstruct the original chunk without
            #   querying SQLite: source, text, doc_id, chunk_index.
            qdrant_point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "document_id": document.id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "source": filename,
                    "file_type": file_type,
                    "token_count": chunk.token_count,
                    "chunk_hash": chunk_hash,
                    **(extra_metadata or {}),
                },
            )
            qdrant_points.append(qdrant_point)

            # Track chunk in SQLite
            chunk_record = DocumentChunk(
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.text,
                chunk_hash=chunk_hash,
                qdrant_point_id=point_id,
                token_count=chunk.token_count,
                was_cached=(chunk_hash in self._embedder._memory_cache),
            )
            chunk_records.append(chunk_record)

        # Batch upsert to Qdrant (more efficient than one-by-one)
        # CONCEPT: Upsert = Insert or Update
        #   If a point with that ID already exists, replace it.
        #   If not, insert it. This makes the operation idempotent.
        await self._qdrant.upsert(
            collection_name=self._collection,
            points=qdrant_points,
        )

        # ── Step 7: Save chunk records to SQLite ──────────────────────────────
        for chunk_record in chunk_records:
            db.add(chunk_record)

        # Update document chunk count
        document.chunk_count = len(chunks)

        # Flush to DB (auto-committed by get_db dependency)
        await db.flush()

        logger.info(
            "ingestion_complete",
            doc_id=document.id,
            filename=filename,
            chunks=len(chunks),
            cache_hits=new_cache_hits,
            api_calls=new_api_calls,
        )

        return {
            "status": "success",
            "document_id": document.id,
            "filename": filename,
            "file_hash": file_hash[:12],
            "chunks_processed": len(chunks),
            "total_tokens": sum(c.token_count for c in chunks),
            "cache_hits": new_cache_hits,
            "api_calls_made": new_api_calls,
            "qdrant_collection": self._collection,
        }

    async def delete_document(self, document_id: int, db: AsyncSession) -> dict:
        """
        Delete a document and all its Qdrant vectors.

        CONCEPT: Vector Deletion
          Qdrant stores vectors by their point IDs.
          We stored the point IDs in DocumentChunk.qdrant_point_id
          so we can delete the exact vectors belonging to this document.
        """
        # Load document and its chunks
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        if not document:
            return {"status": "not_found"}

        # Get all Qdrant point IDs for this document
        chunk_result = await db.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        chunks = chunk_result.scalars().all()
        point_ids = [chunk.qdrant_point_id for chunk in chunks]

        if point_ids:
            # Delete vectors from Qdrant
            from qdrant_client.models import PointIdsList
            await self._qdrant.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=point_ids),
            )
            logger.info(
                "qdrant_vectors_deleted",
                doc_id=document_id,
                count=len(point_ids),
            )

        # Delete chunk records from SQLite
        for chunk in chunks:
            await db.delete(chunk)

        # Delete document record
        await db.delete(document)
        await db.flush()

        logger.info("document_deleted", doc_id=document_id, filename=document.filename)
        return {
            "status": "deleted",
            "document_id": document_id,
            "chunks_deleted": len(chunks),
            "vectors_deleted": len(point_ids),
        }

    async def _check_duplicate_document(
        self, file_hash: str, db: AsyncSession
    ) -> Document | None:
        """Check if a document with the same file hash already exists."""
        result = await db.execute(
            select(Document).where(Document.file_hash == file_hash)
        )
        return result.scalar_one_or_none()
