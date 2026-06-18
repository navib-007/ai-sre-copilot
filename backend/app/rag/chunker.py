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
        chunk_size: int = 250,      # Target token count per chunk
        chunk_overlap: int = 50,    # Overlap tokens between adjacent chunks
        model_name: str = "cl100k_base",  # Tokenizer for GPT-4/OpenAI models
    ):
        """
        Args:
            chunk_size:    Target size of each chunk in TOKENS (not characters).
                           250 tokens — a good balance.
                           Too small: loses context. Too large: loses precision.
            chunk_overlap: How many tokens overlap between consecutive chunks.
                           50 tokens of overlap. Prevents context loss.
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

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text into sentence-aligned chunks of at most chunk_size tokens."""
        import re
        sentence_ends = re.compile(r'(?<=[.?!])\s+|\n+')
        sentences = [s.strip() for s in sentence_ends.split(text) if s.strip()]
        
        raw_chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            s_tokens = self._count_tokens(sentence)
            if current_tokens + s_tokens > self.chunk_size and current_chunk:
                raw_chunks.append(" ".join(current_chunk))
                # Apply overlap: prepend sentences from end of current chunk
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    overlap_s_tokens = self._count_tokens(s)
                    if overlap_tokens + overlap_s_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += overlap_s_tokens
                    else:
                        break
                current_chunk = overlap_sentences
                current_tokens = overlap_tokens
            
            current_chunk.append(sentence)
            current_tokens += s_tokens
            
        if current_chunk:
            raw_chunks.append(" ".join(current_chunk))
            
        return raw_chunks

    async def chunk_text(
        self,
        text: str,
        filename: str = "unknown",
        extra_metadata: dict | None = None,
        db = None,
    ) -> list[Chunk]:
        """
        Split a text string into chunks optimized for embedding.
        Uses Semantic Chunking if db session and CachedEmbedder are available.
        Otherwise falls back to standard RecursiveCharacterTextSplitter.
        """
        if not text or not text.strip():
            logger.warning("chunker_empty_text", filename=filename)
            return []

        text = text.strip()
        raw_chunks = []

        # Try Semantic Chunking
        if db is not None:
            try:
                from langchain_experimental.text_splitter import SemanticChunker
                from langchain_core.embeddings import Embeddings
                import asyncio
                import threading

                class LangChainEmbeddingsWrapper(Embeddings):
                    def __init__(self, cached_embedder, db_session):
                        self.cached_embedder = cached_embedder
                        self.db_session = db_session

                    def embed_documents(self, texts: list[str]) -> list[list[float]]:
                        if not texts:
                            return []
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)

                        if loop.is_running():
                            result = []
                            exception = None
                            def run():
                                nonlocal result, exception
                                try:
                                    new_loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(new_loop)
                                    result = new_loop.run_until_complete(
                                        self.cached_embedder.embed_batch(texts, self.db_session)
                                    )
                                except Exception as e:
                                    exception = e
                            t = threading.Thread(target=run)
                            t.start()
                            t.join()
                            if exception:
                                raise exception
                            return result
                        else:
                            return loop.run_until_complete(
                                self.cached_embedder.embed_batch(texts, self.db_session)
                            )

                    def embed_query(self, text: str) -> list[float]:
                        return self.embed_documents([text])[0]

                from app.rag.embedder import get_embedder
                embedder = get_embedder()
                embeddings_wrapper = LangChainEmbeddingsWrapper(embedder, db)

                # Instantiate langchain_experimental SemanticChunker
                splitter = SemanticChunker(
                    embeddings=embeddings_wrapper,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=0.7,
                )

                # Semantic chunking from LangChain
                semantic_raw_chunks = splitter.split_text(text)

                # Check if it grouped everything into a single chunk despite being large
                total_tokens = self._count_tokens(text)
                if len(semantic_raw_chunks) <= 1 and total_tokens > self.chunk_size:
                    logger.info("semantic_chunker_single_chunk_fallback", filename=filename)
                    raw_chunks = self._split_by_sentences(text)
                else:
                    # Post-process: pack small semantic chunks together and split oversized ones
                    raw_chunks = []
                    current_chunk_txts = []
                    current_chunk_tokens = 0
                    
                    for chunk_txt in semantic_raw_chunks:
                        tokens = self._count_tokens(chunk_txt)
                        
                        if tokens > self.chunk_size:
                            # Oversized chunk: flush any accumulated small chunks first
                            if current_chunk_txts:
                                raw_chunks.append(" ".join(current_chunk_txts))
                                current_chunk_txts = []
                                current_chunk_tokens = 0
                            # Split the oversized chunk
                            sub_chunks = self._split_by_sentences(chunk_txt)
                            raw_chunks.extend(sub_chunks)
                        elif current_chunk_tokens + tokens <= self.chunk_size:
                            # Fits in the current chunk, accumulate it
                            current_chunk_txts.append(chunk_txt)
                            current_chunk_tokens += tokens
                        else:
                            # Exceeds size: flush current accumulated chunk and start new one with overlap
                            raw_chunks.append(" ".join(current_chunk_txts))
                            
                            # Keep trailing sentences/sub-chunks for overlap
                            overlap_txts = []
                            overlap_tokens = 0
                            for t in reversed(current_chunk_txts):
                                t_tokens = self._count_tokens(t)
                                if overlap_tokens + t_tokens <= self.chunk_overlap:
                                    overlap_txts.insert(0, t)
                                    overlap_tokens += t_tokens
                                else:
                                    break
                            current_chunk_txts = overlap_txts + [chunk_txt]
                            current_chunk_tokens = overlap_tokens + tokens
                            
                    if current_chunk_txts:
                        raw_chunks.append(" ".join(current_chunk_txts))

                logger.info(
                    "semantic_chunking_success",
                    filename=filename,
                    chunks_count=len(raw_chunks),
                )
            except Exception as e:
                logger.warning(
                    "semantic_chunking_failed_fallback",
                    filename=filename,
                    error=str(e),
                )
                raw_chunks = []

        # Fallback to Recursive Character Splitting if Semantic Chunking was not run or failed
        if not raw_chunks:
            logger.info("using_recursive_character_splitter", filename=filename)
            raw_chunks = self._splitter.split_text(text)

        chunks: list[Chunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

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

    async def chunk_file_content(
        self,
        file_bytes: bytes,
        filename: str,
        file_type: str,
        extra_metadata: dict | None = None,
        db = None,
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
        return await self.chunk_text(text, filename=filename, extra_metadata=extra_metadata, db=db)

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
