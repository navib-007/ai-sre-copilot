"""
schemas/document.py — Pydantic Schemas for Document API
"""

from datetime import datetime
from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Schema returned after document upload or when listing documents."""
    id: int
    filename: str
    file_type: str
    file_size_bytes: int | None = None
    chunk_count: int
    collection_name: str
    uploaded_by: int
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""
    documents: list[DocumentResponse]
    total: int


class IngestionResult(BaseModel):
    """Result of a document upload/ingestion."""
    status: str                              # "success" | "duplicate" | "error"
    message: str | None = None
    document_id: int | None = None
    existing_document_id: int | None = None
    filename: str | None = None
    file_hash: str | None = None
    chunks_processed: int = 0
    total_tokens: int = 0
    cache_hits: int = 0
    api_calls_made: int = 0
    qdrant_collection: str | None = None


class SearchRequest(BaseModel):
    """Request body for semantic search."""
    query: str = Field(..., min_length=3, description="Search query text")
    top_k: int = Field(default=5, ge=1, le=20, description="Max number of results")
    score_threshold: float = Field(
        default=0.60, ge=0.0, le=1.0,
        description="Minimum similarity score (0.0-1.0). Higher = more strict."
    )
    document_id: int | None = Field(
        default=None, description="Filter results to a specific document"
    )
    file_type: str | None = Field(
        default=None, description="Filter by file type (pdf, md, txt)"
    )


class SearchResultItem(BaseModel):
    """A single search result chunk."""
    text: str
    score: float
    source: str
    document_id: int
    chunk_index: int
    metadata: dict


class SearchResponse(BaseModel):
    """Response from semantic search."""
    query: str
    results: list[SearchResultItem]
    total_results: int
    search_time_ms: float


class CacheStatsResponse(BaseModel):
    """Embedding cache statistics."""
    memory_cache_size: int
    memory_hits: int
    db_hits: int
    api_calls: int
    total_requests: int
    hit_rate_percent: float
