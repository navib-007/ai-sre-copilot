"""
rag/chunker.py — Text Chunking Strategies
==========================================
CONCEPT: Why do we chunk documents?

  LLMs have a context window limit (e.g. GPT-4o-mini: 128K tokens).
  A large runbook might be 50,000 tokens — too big to embed as one piece.
  Even if it fit, a single vector for an entire document would be too generic
  to capture the meaning of any specific section.

  Chunking breaks documents into smaller, semantically meaningful pieces.
  Each chunk gets its OWN embedding vector → precise retrieval.

  BAD: "Kubernetes Runbook" → 1 vector → matches everything vaguely
  GOOD: "Kubernetes Runbook, Section: Restart Pod" → 1 vector → precise match

CONCEPT: Chunking Strategies

  1. FIXED SIZE (naive):
     Split every 512 characters. Simple but breaks sentences mid-word.

  2. RECURSIVE CHARACTER SPLITTING (what we use):
     LangChain's RecursiveCharacterTextSplitter tries to split on:
       "\n\n" → paragraph boundaries (preferred)
       "\n"   → line boundaries (fallback)
       " "    → word boundaries (last resort)
       ""     → character split (never should reach here)
     This keeps sentences and paragraphs intact.

  3. SEMANTIC CHUNKING (advanced, Phase 2+):
     Split where embedding similarity drops — keeps related content together.

  4. OVERLAP:
     Each chunk overlaps with neighbors by `chunk_overlap` tokens.
     WHY? A concept that spans two chunks would be missed if chunks don't overlap.
     Example:
       Chunk 1: "...The pod crashed because of OOM. The memory limit was"
       Chunk 2: "set to 512Mi. Increase it to 1Gi to fix."
       Without overlap → "OOM" and "fix" are in separate chunks → bad retrieval
       With overlap → both chunks contain "OOM" and "fix" → both retrieved
"""

from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """
    A single text chunk ready for embedding.

    Dataclass gives us: automatic __repr__, __eq__, type hints.
    No database involvement — this is a pure in-memory data structure.
    """
    text: str               # The actual text content
    chunk_index: int        # Position in the original document (0-based)
    token_count: int        # Approximate token count (for cost estimation)
    metadata: dict          # Source file info, section headers, page numbers, etc.


class DocumentChunker:
    """
    Chunks documents into smaller pieces optimized for embedding and retrieval.

    Supports: plain text (.txt), markdown (.md), PDF text, Word text.

    Usage:
        chunker = DocumentChunker()
        chunks = chunker.chunk_text(text, filename="kubernetes_runbook.md")
        # Returns list[Chunk] ready for embedding
    """

    def __init__(
        self,
        chunk_size: int = 800,      # Target token count per chunk
        chunk_overlap: int = 150,   # Overlap tokens between adjacent chunks
        model_name: str = "cl100k_base",  # Tokenizer for GPT-4/OpenAI models
    ):
        """
        Args:
            chunk_size:    Target size of each chunk in TOKENS (not characters).
                           800 tokens ≈ 600 words ≈ 4KB text — a good balance.
                           Too small: loses context. Too large: loses precision.
            chunk_overlap: How many tokens overlap between consecutive chunks.
                           150 ≈ 1-2 paragraphs of overlap. Prevents context loss.
            model_name:    Tiktoken tokenizer (cl100k_base = GPT-4/text-embedding-3).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Tiktoken is OpenAI's tokenizer — counts tokens the same way the API does.
        # This lets us estimate embedding costs accurately.
        self._tokenizer = tiktoken.get_encoding(model_name)

        # RecursiveCharacterTextSplitter works in TWO steps:
        # 1. Convert chunk_size (tokens) to rough character estimate
        #    Using 4 chars/token (English average)
        # 2. But we provide a custom length_function that counts TOKENS,
        #    so the actual splitting is token-accurate.
        char_chunk_size = chunk_size * 4
        char_overlap = chunk_overlap * 4

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=char_chunk_size,
            chunk_overlap=char_overlap,
            length_function=self._count_tokens,  # Use token count, not char count
            separators=[
                "\n\n\n",   # Multiple blank lines (section breaks)
                "\n\n",     # Paragraph boundaries
                "\n",       # Line boundaries
                ". ",       # Sentence boundaries
                ", ",       # Clause boundaries
                " ",        # Word boundaries
                "",         # Character (last resort)
            ],
            is_separator_regex=False,
        )

        logger.info(
            "chunker_initialized",
            chunk_size_tokens=chunk_size,
            chunk_overlap_tokens=chunk_overlap,
        )

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string using tiktoken."""
        return len(self._tokenizer.encode(text, disallowed_special=()))

    def chunk_text(
        self,
        text: str,
        filename: str = "unknown",
        extra_metadata: dict | None = None,
    ) -> list[Chunk]:
        """
        Split a text string into chunks optimized for embedding.

        Args:
            text:           Full text to chunk (already extracted from PDF/MD/TXT)
            filename:       Source filename (stored in chunk metadata)
            extra_metadata: Additional metadata to attach to every chunk

        Returns:
            List of Chunk objects, ordered by their position in the document.
        """
        if not text or not text.strip():
            logger.warning("chunker_empty_text", filename=filename)
            return []

        # Pre-process: normalize whitespace
        text = text.strip()

        # Split into raw text pieces
        raw_chunks = self._splitter.split_text(text)

        chunks: list[Chunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue  # Skip empty chunks (can happen after stripping)

            token_count = self._count_tokens(chunk_text)

            chunk = Chunk(
                text=chunk_text,
                chunk_index=idx,
                token_count=token_count,
                metadata={
                    "source": filename,
                    "chunk_index": idx,
                    "total_chunks": len(raw_chunks),
                    "token_count": token_count,
                    **(extra_metadata or {}),
                },
            )
            chunks.append(chunk)

        total_tokens = sum(c.token_count for c in chunks)
        logger.info(
            "chunking_complete",
            filename=filename,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            avg_tokens_per_chunk=total_tokens // len(chunks) if chunks else 0,
        )

        return chunks

    def chunk_file_content(
        self,
        file_bytes: bytes,
        filename: str,
        file_type: str,
        extra_metadata: dict | None = None,
    ) -> list[Chunk]:
        """
        Extract text from a file and chunk it.

        Supports: .txt, .md, .pdf, .docx

        Args:
            file_bytes: Raw file content
            filename:   Original filename
            file_type:  "txt", "md", "pdf", "docx"
        """
        text = self._extract_text(file_bytes, filename, file_type)
        return self.chunk_text(text, filename=filename, extra_metadata=extra_metadata)

    def _extract_text(self, file_bytes: bytes, filename: str, file_type: str) -> str:
        """
        Extract plain text from various file formats.

        CONCEPT: Document Parsing
          Different file formats store text differently:
          - TXT/MD: bytes → decode to UTF-8 string (trivial)
          - PDF: binary format with fonts/layout → pypdf extracts page text
          - DOCX: ZIP file containing XML → python-docx parses the XML
        """
        file_type = file_type.lower().lstrip(".")

        if file_type in ("txt", "md", "markdown"):
            # Simple decode — markdown/text is already human-readable
            return file_bytes.decode("utf-8", errors="replace")

        elif file_type == "pdf":
            try:
                import io
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(file_bytes))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n\n".join(pages)
                logger.info("pdf_parsed", filename=filename, pages=len(reader.pages))
                return text
            except Exception as e:
                logger.error("pdf_parse_failed", filename=filename, error=str(e))
                return ""

        elif file_type in ("docx", "doc"):
            try:
                import io
                from docx import Document as DocxDocument
                doc = DocxDocument(io.BytesIO(file_bytes))
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                text = "\n\n".join(paragraphs)
                logger.info("docx_parsed", filename=filename, paragraphs=len(paragraphs))
                return text
            except Exception as e:
                logger.error("docx_parse_failed", filename=filename, error=str(e))
                return ""

        else:
            logger.warning("unsupported_file_type", file_type=file_type, filename=filename)
            # Try UTF-8 decode as fallback
            return file_bytes.decode("utf-8", errors="replace")
