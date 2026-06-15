"""
routes/documents.py — Document Upload, Search & Management Endpoints
=====================================================================
CONCEPT: File Upload in FastAPI

  FastAPI handles file uploads via:
    - UploadFile: the actual file (stream with metadata)
    - File(...)  or Form(...): for form fields alongside the file

  The file is uploaded as multipart/form-data (standard HTML form encoding).
  FastAPI reads it into memory (for small files) or streams it (for large files).

  Our approach:
    - Read file into bytes (fine for documents up to ~10MB)
    - Pass bytes to IngestionPipeline for processing

CONCEPT: Background Tasks in FastAPI
  For long-running operations (like PDF processing + embedding), we have options:
    1. Synchronous: Process in the request, block until done (simple but slow)
    2. BackgroundTask: FastAPI runs task AFTER responding to client (medium)
    3. Celery/RQ queue: Proper job queue for production (complex but scalable)

  For Phase 2, we use option 1 (synchronous) to keep things simple and visible.
  We'll see the progress in logs. Phase 9+ can add proper background processing.

ENDPOINTS:
  POST   /documents/upload          → upload + ingest a document
  GET    /documents/                → list all documents
  GET    /documents/{id}            → get document metadata
  DELETE /documents/{id}            → delete document + its Qdrant vectors
  POST   /documents/search          → semantic search
  GET    /documents/cache/stats     → embedding cache statistics
  DELETE /documents/cache/clear     → clear in-memory cache (admin)
"""

import time
import math
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.models import Document, User
from app.logging_config import get_logger
from app.auth import require_viewer, require_admin
from app.rag.embedder import get_embedder
from app.rag.ingestion import IngestionPipeline
from app.rag.qdrant_client import get_ingestion_pipeline, get_retriever
from app.rag.retriever import RAGRetriever
from app.schemas.document import (
    CacheStatsResponse,
    DocumentListResponse,
    DocumentResponse,
    IngestionResult,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents & RAG"])

# Allowed file types (security: reject unexpected file types)
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
MAX_FILE_SIZE_MB = 50


# ─── POST /documents/upload ───────────────────────────────────────────────────
@router.post(
    "/upload",
    response_model=IngestionResult,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document into the knowledge base",
)
async def upload_document(
    file: UploadFile = File(..., description="Document to upload (txt, md, pdf, docx)"),
    db: AsyncSession = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    current_user: User = Depends(require_admin),
) -> IngestionResult:
    """
    Upload a document and ingest it into the RAG knowledge base.

    The pipeline:
    1. Validates file type and size
    2. Checks for duplicate (SHA-256 hash comparison)
    3. Parses text from the document
    4. Chunks the text into overlapping pieces
    5. Embeds each chunk (using cache — no API call for already-seen chunks)
    6. Stores vectors in Qdrant Cloud
    7. Records metadata in SQLite

    Returns ingestion statistics including cache hit rate.
    """
    # ── Validate file ──────────────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file_ext}' not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file bytes
    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB",
        )

    logger.info(
        "document_upload_received",
        filename=file.filename,
        size_mb=round(file_size_mb, 2),
        content_type=file.content_type,
    )

    # ── Run ingestion pipeline ─────────────────────────────────────────────────
    result = await pipeline.ingest_document(
        file_bytes=file_bytes,
        filename=file.filename,
        file_type=file_ext.lstrip("."),
        uploaded_by=current_user.id,
        db=db,
    )

    return IngestionResult(**result)


# ─── GET /documents/ ──────────────────────────────────────────────────────────
@router.get("/", response_model=DocumentListResponse, summary="List all documents")
async def list_documents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
) -> DocumentListResponse:
    """List all ingested documents with their metadata."""
    total = (await db.execute(select(func.count(Document.id)))).scalar_one()
    offset = (page - 1) * page_size

    docs = (
        await db.execute(
            select(Document).order_by(Document.uploaded_at.desc())
            .offset(offset).limit(page_size)
        )
    ).scalars().all()

    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in docs],
        total=total,
    )


# ─── GET /documents/{document_id} ─────────────────────────────────────────────
@router.get("/{document_id}", response_model=DocumentResponse, summary="Get document metadata")
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
) -> DocumentResponse:
    """Retrieve metadata for a specific document."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return DocumentResponse.model_validate(doc)


# ─── DELETE /documents/{document_id} ──────────────────────────────────────────
@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its vectors from Qdrant",
)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    current_user: User = Depends(require_admin),
) -> None:
    """
    Delete a document and permanently remove its vectors from Qdrant.

    Note: This does NOT delete the embedding cache entries.
    The cached embeddings remain usable if the same text appears in other documents.
    """
    result = await pipeline.delete_document(document_id, db)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    logger.info(
        "document_deleted_via_api",
        doc_id=document_id,
        chunks_deleted=result.get("chunks_deleted", 0),
    )


# ─── POST /documents/search ───────────────────────────────────────────────────
@router.post("/search", response_model=SearchResponse, summary="Semantic search across documents")
async def search_documents(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    retriever: RAGRetriever = Depends(get_retriever),
    current_user: User = Depends(require_viewer),
) -> SearchResponse:
    """
    Search the knowledge base using semantic similarity.

    The query is embedded with the same model as the documents,
    then Qdrant finds the most similar chunks by cosine similarity.

    Use `score_threshold` to control result quality:
    - 0.8+ = very strict (only highly relevant results)
    - 0.6  = balanced (default, good for most queries)
    - 0.4  = lenient (more results, possibly less relevant)
    """
    start_time = time.perf_counter()

    logger.info(
        "semantic_search_request",
        query=request.query[:80],
        top_k=request.top_k,
        threshold=request.score_threshold,
    )

    results = await retriever.search(
        query=request.query,
        db=db,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        document_id=request.document_id,
        file_type=request.file_type,
    )

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return SearchResponse(
        query=request.query,
        results=[SearchResultItem(**r.to_dict()) for r in results],
        total_results=len(results),
        search_time_ms=duration_ms,
    )


# ─── GET /documents/cache/stats ───────────────────────────────────────────────
@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Embedding cache statistics",
)
async def get_cache_stats(current_user: User = Depends(require_admin)) -> CacheStatsResponse:
    """
    Get embedding cache statistics.

    Shows:
    - How many embeddings are in the in-memory (L1) cache
    - L1 hit rate (served from memory)
    - L2 hit rate (served from SQLite)
    - API call count (actual OpenAI calls made)
    - Overall hit rate (L1 + L2 combined)

    A high hit rate means you're saving money!
    """
    embedder = get_embedder()
    stats = embedder.get_cache_stats()
    return CacheStatsResponse(**stats)


# ─── DELETE /documents/cache/clear ────────────────────────────────────────────
@router.delete(
    "/cache/clear",
    summary="Clear in-memory embedding cache (L1)",
)
async def clear_memory_cache(current_user: User = Depends(require_admin)) -> dict:
    """
    Clear the in-memory (L1) embedding cache.

    Use this after:
    - Changing the embedding model (all cached vectors are invalid)
    - Troubleshooting cache-related issues
    - Testing to simulate a cold start

    Note: This only clears the in-memory cache (L1).
    The SQLite cache (L2) is preserved — it will be repopulated automatically.
    """
    embedder = get_embedder()
    cleared = embedder.clear_memory_cache()
    return {"cleared_entries": cleared, "message": "In-memory cache cleared"}
