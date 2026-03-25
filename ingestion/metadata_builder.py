# ingestion/metadata_builder.py
# WHY THIS EXISTS: Builds the complete metadata dict for every chunk.
# Every field is required — no field is optional except section_heading.
# Missing user_id = cross-user data access = product over. LAW 6.

import uuid
from datetime import datetime, timezone
from typing import Optional

from langdetect import detect, LangDetectException

from ingestion.indic_preprocessing import detect_script_type
from ingestion.chunker import TextChunk
from ingestion.injection_scanner import scan_chunk_for_injection
from ingestion.pii_detector import detect_pii_in_chunk


# Document type classification keywords
DOC_TYPE_KEYWORDS = {
    "gst_notice": [
        "gst", "gstin", "goods and services tax", "tax invoice",
        "जीएसटी", "कर चालान", "माल और सेवा कर",
    ],
    "legal": [
        "court", "tribunal", "plaintiff", "defendant", "order",
        "न्यायालय", "वादी", "प्रतिवादी", "आदेश",
    ],
    "insurance": [
        "insurance", "policy", "premium", "claim", "insured",
        "बीमा", "पॉलिसी", "प्रीमियम", "दावा",
    ],
    "circular": [
        "circular", "notification", "gazette", "परिपत्र", "अधिसूचना",
    ],
}


def classify_doc_type(text_sample: str) -> str:
    """
    Classifies document type from text content.
    Used in chunk metadata.doc_type field.

    Returns:
        "gst_notice" | "legal" | "insurance" | "circular" | "other"
    """
    text_lower = text_sample.lower()
    scores = {}

    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[doc_type] = score

    best_type = max(scores, key=scores.get)
    return best_type if scores[best_type] > 0 else "other"


def detect_chunk_language(text: str) -> str:
    """Detects language of individual chunk."""
    try:
        return detect(text[:500])
    except LangDetectException:
        return "en"


def build_chunk_metadata(
    chunk: TextChunk,
    doc_id: str,
    user_id: str,
    source_file_path: str,
    doc_title: str,
    doc_primary_language: str,
    doc_hash: str,
    total_chunks: int,
    section_heading: Optional[str] = None,
    embedding_model_version: str = "1.0.0",
) -> dict:
    """
    Builds complete metadata dict for a single chunk.
    Every chunk stored in pgvector carries all these fields.

    This is the single source of truth for chunk metadata.
    All fields match the spec exactly.

    Args:
        chunk: TextChunk object from chunker
        doc_id: UUID of parent document
        user_id: User who uploaded the document — NEVER NULL
        source_file_path: File path of source PDF
        doc_title: Document title
        doc_primary_language: Primary language of full document
        doc_hash: SHA-256 hash of source PDF
        total_chunks: Total chunks in this document
        section_heading: Section heading if detectable
        embedding_model_version: Version string for model tracking

    Returns:
        Complete metadata dict — all fields populated
    """
    assert user_id, "user_id MUST NOT be null — per-user isolation requires it"
    assert doc_id, "doc_id MUST NOT be null"

    # Run injection scan on this chunk
    injection_result = scan_chunk_for_injection(chunk.text)

    # Run PII detection on this chunk
    pii_result = detect_pii_in_chunk(chunk.text, redact=False)

    # Detect chunk-level language (may differ from doc language)
    chunk_language = detect_chunk_language(chunk.text)

    return {
        # Identity
        "chunk_id": chunk.chunk_id,
        "doc_id": doc_id,
        "user_id": user_id,                        # NEVER NULL

        # Source
        "source": source_file_path,
        "title": doc_title,

        # Language
        "doc_primary_language": doc_primary_language,
        "chunk_language": chunk_language,
        "script_type": chunk.script_type,

        # Document Classification
        "doc_type": classify_doc_type(chunk.text),

        # Position
        "page_number": chunk.page_number,
        "section_heading": section_heading or "",
        "chunk_index": chunk.chunk_index,
        "total_chunks": total_chunks,

        # Parent-child
        "parent_chunk_id": chunk.parent_chunk_id or "",
        "is_parent": chunk.is_parent,

        # Embedding
        "embedding_model": "intfloat/multilingual-e5-large",
        "embedding_model_version": embedding_model_version,

        # Security
        "injection_risk": injection_result.injection_risk,
        "pii_detected": pii_result.pii_detected,

        # Idempotency
        "doc_hash": doc_hash,

        # Timestamps
        "created_at": datetime.now(timezone.utc),
    }
