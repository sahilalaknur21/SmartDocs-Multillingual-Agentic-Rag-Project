# ingestion/chunker.py
# WHY THIS EXISTS: Splits extracted PDF text into chunks for embedding.
# Devanagari = 400-token chunks (denser script).
# Latin = 500-token chunks.
# indic-nlp sentence tokenizer runs FIRST for Indic text.
# Parent-child architecture: child=256 tokens (retrieval), parent=1024 (context).
# LAW 3.

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ingestion.indic_preprocessing import (
    indic_preprocessing_pipeline,
    detect_script_type,
    ALL_INDIC_LANGUAGES,
)


@dataclass
class TextChunk:
    """A single text chunk ready for embedding."""
    chunk_id: str
    text: str
    page_number: int
    chunk_index: int
    script_type: str
    lang_code: str
    parent_chunk_id: Optional[str]
    is_parent: bool
    char_count: int
    token_estimate: int


def estimate_tokens(text: str) -> int:
    """
    Estimates token count without loading a tokenizer.
    Rule of thumb: 1 token ≈ 4 chars for Latin, ≈ 2 chars for Devanagari.
    Used for chunk size decisions — not for billing.
    """
    script = detect_script_type(text)
    if script == "devanagari":
        return len(text) // 2
    elif script == "dravidian":
        return len(text) // 2
    else:
        return len(text) // 4


def split_text_by_sentences(
    text: str,
    lang_code: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Splits text into chunks respecting sentence boundaries.
    Uses indic sentence tokenizer for Indic languages.
    Uses standard sentence splitting for Latin text.

    This is the core of LAW 3 — never chunk mid-sentence.

    Args:
        text: Preprocessed text
        lang_code: ISO language code
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks

    Returns:
        List of text chunks, no chunk ends mid-sentence
    """
    # Get sentences from indic preprocessing pipeline
    result = indic_preprocessing_pipeline(
        text=text,
        lang_code=lang_code,
        return_sentences=True,
    )

    sentences = result["sentences"]
    cleaned_text = result["cleaned_text"]

    # If sentence tokenization failed or returned nothing
    if not sentences:
        sentences = re.split(r"(?<=[.!?।॥])\s+", cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [cleaned_text] if cleaned_text else []

    # Build chunks by accumulating sentences up to max_tokens
    chunks = []
    current_sentences = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        # If adding this sentence exceeds limit, save current chunk
        if current_tokens + sentence_tokens > max_tokens and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(chunk_text)

            # Overlap: keep last N tokens worth of sentences
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_tokens = estimate_tokens(s)
                if overlap_count + s_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_count += s_tokens
                else:
                    break

            current_sentences = overlap_sentences
            current_tokens = overlap_count

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Don't forget the last chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return [c for c in chunks if c.strip()]


def chunk_page_text(
    text: str,
    page_number: int,
    lang_code: str,
    script_type: str,
    chunk_size_devanagari: int = 400,
    chunk_overlap_devanagari: int = 40,
    chunk_size_latin: int = 500,
    chunk_overlap_latin: int = 50,
) -> list[TextChunk]:
    """
    Chunks a single page's text.
    Automatically selects chunk size based on script type.

    Args:
        text: Extracted page text
        page_number: Page number for metadata
        lang_code: ISO language code
        script_type: "devanagari" | "latin" | "dravidian" | "mixed"
        chunk_size_devanagari: Token limit for Devanagari (default: 400)
        chunk_overlap_devanagari: Overlap for Devanagari (default: 40)
        chunk_size_latin: Token limit for Latin (default: 500)
        chunk_overlap_latin: Overlap for Latin (default: 50)

    Returns:
        List of TextChunk objects
    """
    if not text or not text.strip():
        return []

    # Select chunk parameters based on script
    if script_type in ("devanagari", "dravidian"):
        max_tokens = chunk_size_devanagari
        overlap_tokens = chunk_overlap_devanagari
    else:
        max_tokens = chunk_size_latin
        overlap_tokens = chunk_overlap_latin

    # Split into sentence-boundary-respecting chunks
    raw_chunks = split_text_by_sentences(
        text=text,
        lang_code=lang_code,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunk = TextChunk(
            chunk_id=str(uuid.uuid4()),
            text=chunk_text,
            page_number=page_number,
            chunk_index=idx,
            script_type=script_type,
            lang_code=lang_code,
            parent_chunk_id=None,
            is_parent=False,
            char_count=len(chunk_text),
            token_estimate=estimate_tokens(chunk_text),
        )
        chunks.append(chunk)

    return chunks


def build_parent_child_chunks(
    page_text: str,
    page_number: int,
    lang_code: str,
    script_type: str,
    parent_size: int = 1024,
    child_size: int = 256,
    child_overlap: int = 25,
) -> tuple[list[TextChunk], list[TextChunk]]:
    """
    Builds parent-child chunk pairs.
    Child chunks (256 tokens) → used for retrieval precision.
    Parent chunks (1024 tokens) → passed to Sarvam-30B for full context.

    This architecture gives you precise retrieval AND rich context
    for generation — not a tradeoff between the two.

    Args:
        page_text: Full page text
        page_number: Page number
        lang_code: ISO language code
        script_type: Script type string
        parent_size: Parent chunk token size (default: 1024)
        child_size: Child chunk token size (default: 256)
        child_overlap: Child chunk overlap (default: 25)

    Returns:
        tuple: (parent_chunks, child_chunks)
    """
    # Build parent chunks (large, for generation context)
    parent_chunks = chunk_page_text(
        text=page_text,
        page_number=page_number,
        lang_code=lang_code,
        script_type=script_type,
        chunk_size_devanagari=parent_size,
        chunk_overlap_devanagari=parent_size // 10,
        chunk_size_latin=parent_size,
        chunk_overlap_latin=parent_size // 10,
    )

    # Mark parent chunks
    for parent in parent_chunks:
        parent.is_parent = True

    # Build child chunks from each parent (for precise retrieval)
    all_child_chunks = []
    for parent in parent_chunks:
        child_raw = split_text_by_sentences(
            text=parent.text,
            lang_code=lang_code,
            max_tokens=child_size,
            overlap_tokens=child_overlap,
        )
        for idx, child_text in enumerate(child_raw):
            child = TextChunk(
                chunk_id=str(uuid.uuid4()),
                text=child_text,
                page_number=page_number,
                chunk_index=idx,
                script_type=script_type,
                lang_code=lang_code,
                parent_chunk_id=parent.chunk_id,
                is_parent=False,
                char_count=len(child_text),
                token_estimate=estimate_tokens(child_text),
            )
            all_child_chunks.append(child)

    return parent_chunks, all_child_chunks