"""
models.py — SQLAlchemy ORM Models (Database Tables)
====================================================
CONCEPT: ORM (Object Relational Mapper)

  Without ORM:
    cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))

  With ORM (SQLAlchemy v2 style):
    user = User(name=name, email=email)
    session.add(user)
    await session.commit()

  ORM benefits:
    - Work with Python objects instead of raw SQL
    - Type checking (Mapped[int] = int column, cannot insert string)
    - Relationships (user.tickets loads related tickets automatically)
    - Database-agnostic (change SQLite → PostgreSQL by changing one URL)

CONCEPT: SQLAlchemy v2 Mapped Annotations
  SQLAlchemy v2 uses Python type hints for column definitions:
    Mapped[int]          → NOT NULL INTEGER column
    Mapped[str]          → NOT NULL VARCHAR column
    Mapped[str | None]   → NULLABLE VARCHAR column
    Mapped[datetime]     → NOT NULL DATETIME column

  This gives you full IDE autocompletion and type checking.
"""

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


# ─── Enums ────────────────────────────────────────────────────────────────────
# Using Python Enum for type safety. SQLAlchemy stores these as strings.
class UserRole(str, Enum):
    ADMIN = "admin"
    ENGINEER = "engineer"
    VIEWER = "viewer"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentSeverity(str, Enum):
    P1 = "P1"   # Critical — all hands on deck
    P2 = "P2"   # High — immediate attention needed
    P3 = "P3"   # Medium — fix within hours
    P4 = "P4"   # Low — fix within days


class IncidentStatus(str, Enum):
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MemoryType(str, Enum):
    FACT = "fact"           # Extracted facts ("User uses Kubernetes")
    PREFERENCE = "preference"  # User preferences ("Prefers brief answers")
    CONTEXT = "context"     # Contextual history summaries


# ─── Helper: UTC timestamp ────────────────────────────────────────────────────
def utcnow() -> datetime:
    """Return timezone-aware UTC datetime (best practice for DB storage)."""
    return datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE: users
# ═══════════════════════════════════════════════════════════════════════════════
class User(Base):
    """
    Represents an authenticated user of the platform.

    Relationships:
      - tickets_created   → tickets this user created
      - tickets_assigned  → tickets assigned to this user
      - incidents         → incidents reported by this user
      - chat_sessions     → all chat sessions
      - memory_entries    → this user's long-term memory
      - audit_logs        → actions this user took
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(20), default=UserRole.ENGINEER, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    # back_populates creates a bidirectional relationship:
    #   user.tickets_created  → list of Ticket objects
    #   ticket.creator        → the User object
    tickets_created: Mapped[list["Ticket"]] = relationship(
        "Ticket", back_populates="creator", foreign_keys="Ticket.created_by"
    )
    tickets_assigned: Mapped[list["Ticket"]] = relationship(
        "Ticket", back_populates="assignee", foreign_keys="Ticket.assigned_to"
    )
    incidents: Mapped[list["Incident"]] = relationship(
        "Incident", back_populates="reporter"
    )
    chat_sessions: Mapped[list["ChatSession"]] = relationship(
        "ChatSession", back_populates="user"
    )
    memory_entries: Mapped[list["MemoryEntry"]] = relationship(
        "MemoryEntry", back_populates="user"
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        "AuditLog", back_populates="user"
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="uploader"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r} role={self.role!r}>"


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE: tickets
# ═══════════════════════════════════════════════════════════════════════════════
class Ticket(Base):
    """
    IT support tickets created by users or agents.

    In our agentic system, the Ticket Agent can:
      - CREATE tickets from natural language
      - UPDATE ticket status during incident resolution
      - QUERY tickets to check on pending issues
    """
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default=TicketStatus.OPEN, nullable=False, index=True
    )
    priority: Mapped[str] = mapped_column(
        String(20), default=TicketPriority.MEDIUM, nullable=False, index=True
    )
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # ── Foreign Keys ──────────────────────────────────────────────────────────
    created_by: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    assigned_to: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )

    # ── Content ───────────────────────────────────────────────────────────────
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[str | None] = mapped_column(String(500), nullable=True)  # JSON array string

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    creator: Mapped["User"] = relationship(
        "User", back_populates="tickets_created", foreign_keys=[created_by]
    )
    assignee: Mapped["User | None"] = relationship(
        "User", back_populates="tickets_assigned", foreign_keys=[assigned_to]
    )

    # ── Indexes for fast querying ─────────────────────────────────────────────
    __table_args__ = (
        Index("ix_tickets_status_priority", "status", "priority"),
        Index("ix_tickets_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Ticket id={self.id} title={self.title!r} status={self.status!r}>"


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE: incidents
# ═══════════════════════════════════════════════════════════════════════════════
class Incident(Base):
    """
    IT incidents (outages, performance issues, security events).

    In our agentic system:
      - Incident Agent investigates incidents
      - RCA Agent finds root cause
      - Approval Agent handles remediation approval
      - Timeline is stored as JSON for flexibility
    """
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(
        String(5), default=IncidentSeverity.P3, nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(20), default=IncidentStatus.DETECTED, nullable=False, index=True
    )
    affected_services: Mapped[str | None] = mapped_column(
        String(500), nullable=True  # Comma-separated service names
    )

    # ── Root Cause Analysis ───────────────────────────────────────────────────
    root_cause: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON: list of {timestamp, event, agent} objects
    timeline_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON: {cpu: ..., memory: ..., logs: [...]} evidence gathered by agent
    evidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Foreign Key ───────────────────────────────────────────────────────────
    reported_by: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    # Link to the ticket created for this incident
    ticket_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("tickets.id"), nullable=True
    )

    # ── Timestamps ────────────────────────────────────────────────────────────
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    mitigated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    reporter: Mapped["User"] = relationship("User", back_populates="incidents")
    approval_requests: Mapped[list["ApprovalRequest"]] = relationship(
        "ApprovalRequest", back_populates="incident"
    )

    def __repr__(self) -> str:
        return f"<Incident id={self.id} severity={self.severity!r} status={self.status!r}>"


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE: chat_sessions
# ═══════════════════════════════════════════════════════════════════════════════
class ChatSession(Base):
    """
    A conversation session between a user and the AI system.

    CONCEPT: Session Management
      Each session has a unique session_id (UUID).
      LangGraph uses this as the thread_id for its checkpointer,
      which means it can resume the conversation from where it left off.
      Multiple sessions = separate conversation histories.
    """
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    # UUID string — used as LangGraph thread_id for state persistence
    session_id: Mapped[str] = mapped_column(
        String(36), unique=True, index=True, nullable=False
    )
    title: Mapped[str | None] = mapped_column(
        String(255), nullable=True  # Auto-generated from first message
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    last_active: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="chat_sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="session", order_by="ChatMessage.timestamp"
    )

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id} session_id={self.session_id!r}>"


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE: chat_messages
# ═══════════════════════════════════════════════════════════════════════════════
class ChatMessage(Base):
    """
    Individual messages within a chat session.

    Stores the full conversation history including:
      - user messages
      - assistant responses
      - tool calls and results (for agent transparency)
      - which agent produced the response
    """
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chat_sessions.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Which agent generated this message (supervisor, rag_agent, incident_agent, etc.)
    agent_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # JSON: list of tool call objects [{tool: "...", args: {...}, result: "..."}]
    tool_calls_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Token count for cost tracking
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    session: Mapped["ChatSession"] = relationship(
        "ChatSession", back_populates="messages"
    )

    def __repr__(self) -> str:
        return f"<ChatMessage id={self.id} role={self.role!r} agent={self.agent_name!r}>"


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE: documents
# ═══════════════════════════════════════════════════════════════════════════════
class Document(Base):
    """
    Metadata for documents uploaded to the knowledge base.

    CONCEPT: RAG Document Lifecycle
      1. User uploads a document (PDF, MD, TXT)
      2. We chunk it into smaller pieces
      3. We embed each chunk with OpenAI
      4. We store embeddings in Qdrant (referenced by doc_id + chunk_index)
      5. We store metadata here (filename, chunk count, etc.)
      6. When searching, Qdrant returns chunk references → we know which doc
    """
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)  # pdf, md, txt
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    # SHA-256 of the file bytes — used to detect duplicate uploads
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # Which Qdrant collection this document's chunks live in
    collection_name: Mapped[str] = mapped_column(String(100), nullable=False)
    # JSON: additional metadata (source, author, version, etc.)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_by: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    uploader: Mapped["User"] = relationship("User", back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document"
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename!r} chunks={self.chunk_count}>"


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE: approval_requests
# ═══════════════════════════════════════════════════════════════════════════════
class ApprovalRequest(Base):
    """
    Human-in-the-loop approval requests for high-risk agent actions.

    CONCEPT: Human-in-the-Loop (HITL)
      When an agent wants to take a potentially dangerous action
      (restart a production service, scale down pods, delete data),
      it creates an ApprovalRequest and PAUSES execution.
      A human reviews it and approves/rejects.
      The agent then RESUMES from where it paused.

      This is implemented using LangGraph's interrupt() + Command(resume=...).
    """
    __tablename__ = "approval_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # Which incident triggered this approval request
    incident_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("incidents.id"), nullable=True
    )
    # LangGraph thread_id — used to resume the paused graph
    langgraph_thread_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    action_type: Mapped[str] = mapped_column(String(100), nullable=False)
    # JSON: full details of the proposed action
    action_details_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default=ApprovalStatus.PENDING, nullable=False, index=True
    )
    requested_by: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    reviewed_by: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )
    review_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    incident: Mapped["Incident | None"] = relationship(
        "Incident", back_populates="approval_requests"
    )

    def __repr__(self) -> str:
        return f"<ApprovalRequest id={self.id} action={self.action_type!r} status={self.status!r}>"


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE: memory_entries
# ═══════════════════════════════════════════════════════════════════════════════
class MemoryEntry(Base):
    """
    Long-term memory storage for the Memory Agent.

    CONCEPT: Long-Term Memory
      Short-term memory = current conversation (LangGraph state)
      Long-term memory  = knowledge that persists ACROSS sessions

      Examples of long-term memory:
        - "User prefers concise answers"
        - "User's team uses Kubernetes v1.28"
        - "User escalated ticket TKT-42 to CTO"

      The Memory Agent extracts these facts from conversations
      and stores them here. Future conversations load relevant
      memories to personalize the agent's responses.
    """
    __tablename__ = "memory_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    memory_type: Mapped[str] = mapped_column(String(20), nullable=False)  # MemoryType enum
    key: Mapped[str] = mapped_column(String(255), nullable=False)   # "kubernetes_version"
    value: Mapped[str] = mapped_column(Text, nullable=False)         # "v1.28"
    # Relevance score (0.0 - 1.0) updated by retrieval frequency
    relevance_score: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    last_accessed: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="memory_entries")

    # ── Index for fast per-user memory retrieval ──────────────────────────────
    __table_args__ = (
        Index("ix_memory_user_type", "user_id", "memory_type"),
    )

    def __repr__(self) -> str:
        return f"<MemoryEntry user={self.user_id} key={self.key!r} type={self.memory_type!r}>"


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE: audit_logs
# ═══════════════════════════════════════════════════════════════════════════════
class AuditLog(Base):
    """
    Immutable audit trail of all significant actions.

    CONCEPT: Audit Logging
      Every important action in the system is recorded here:
        - Who did it (user_id)
        - What they did (action: "ticket_created", "agent_action", "approval_granted")
        - What resource (resource_type: "ticket", resource_id: 42)
        - When (timestamp)
        - Extra context (details_json)

      This is critical for:
        - Security compliance
        - Debugging agent behavior
        - Understanding what actions the agents took
        - Post-incident review

      IMPORTANT: Audit logs should NEVER be deleted or modified.
    """
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True  # None = system action
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    resource_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # JSON: additional context (IP address, agent name, tool args, etc.)
    details_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Which agent performed this action (if it was an agent)
    agent_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User | None"] = relationship("User", back_populates="audit_logs")

    # ── Index for efficient log queries ───────────────────────────────────────
    __table_args__ = (
        Index("ix_audit_user_timestamp", "user_id", "timestamp"),
        Index("ix_audit_action_timestamp", "action", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<AuditLog id={self.id} action={self.action!r} user={self.user_id}>"


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE: embedding_cache
# ═══════════════════════════════════════════════════════════════════════════════
class EmbeddingCache(Base):
    """
    SQLite-backed embedding cache to avoid re-calling OpenAI for identical text.

    CONCEPT: Content-Addressable Caching
      Instead of caching by URL or filename (which changes), we cache by the
      SHA-256 HASH of the actual text content.

      Why SHA-256?
        - SHA-256("How to restart a Kubernetes pod?") → always the same 64-char hex string
        - If two documents have the same paragraph, we embed it ONLY ONCE
        - If you re-upload a document, ALL its chunks are already cached
        - Deterministic: same input → same key, always

      Flow:
        1. compute hash = sha256(chunk_text)
        2. look up hash in this table
        3a. CACHE HIT  → return stored vector, skip OpenAI call ✅ (free!)
        3b. CACHE MISS → call OpenAI, store result in this table 💰 (costs money)

      Cost Example (text-embedding-3-small @ $0.02/1M tokens):
        100 documents × 50 chunks each = 5,000 chunks
        Average chunk ≈ 200 tokens → 1,000,000 tokens = $0.02 total
        Without cache: $0.02 per upload × 10 re-uploads = $0.20
        With cache:    $0.02 ONCE regardless of re-uploads ✅

      Cache invalidation: Manual only (admin can clear specific entries).
      We never auto-expire because embeddings don't change unless the
      model changes. If you switch models, run a cache clear.
    """
    __tablename__ = "embedding_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # SHA-256 hash of the chunk text (64 hex chars) — the cache KEY
    text_hash: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )

    # The embedding model used (e.g. "text-embedding-3-small")
    # IMPORTANT: Different models produce different-dimensional vectors.
    # We store the model so we know if cached embeddings are still valid.
    model: Mapped[str] = mapped_column(String(100), nullable=False)

    # The embedding vector stored as a JSON array of floats
    # Example: "[0.023, -0.145, 0.891, ...]" (1536 floats for text-embedding-3-small)
    # We use Text (not a special vector type) because SQLite doesn't have vector type.
    # Qdrant stores the actual searchable vector — this is just our cache.
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Dimension of the vector (useful for validation)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)

    # Track cache usage for analytics
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # Composite index: hash + model (different models = different cache entries)
    __table_args__ = (
        Index("ix_embedding_cache_hash_model", "text_hash", "model"),
    )

    def __repr__(self) -> str:
        return (
            f"<EmbeddingCache hash={self.text_hash[:12]}... "
            f"model={self.model!r} hits={self.hit_count}>"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE: document_chunks
# ═══════════════════════════════════════════════════════════════════════════════
class DocumentChunk(Base):
    """
    Tracks individual chunks created from a document.

    CONCEPT: Chunk Deduplication
      When a document is uploaded:
        1. We compute file_hash = sha256(file_bytes) → document-level dedup
        2. For each chunk: chunk_hash = sha256(chunk_text) → chunk-level dedup
        3. chunk_hash links back to EmbeddingCache

      This table is a registry of: which chunk came from which document,
      and what is its Qdrant vector ID (so we can delete it later).

      Without this table, we couldn't:
        - Delete a document's vectors from Qdrant
        - Know which chunks were already indexed
        - Track chunk → document relationships
    """
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # Parent document
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id"), nullable=False, index=True
    )

    # Chunk position within the document (0-based)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # The actual chunk text (stored for cache lookup + debugging)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)

    # SHA-256 of chunk_text — links to EmbeddingCache.text_hash
    chunk_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Qdrant point ID (UUID string) — used to delete the vector from Qdrant
    qdrant_point_id: Mapped[str] = mapped_column(String(36), nullable=False)

    # Token count of this chunk (for cost estimation)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Was this chunk's embedding served from cache? (analytics)
    was_cached: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_doc_chunk_doc_index", "document_id", "chunk_index"),
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk doc={self.document_id} idx={self.chunk_index} hash={self.chunk_hash[:8]}...>"
