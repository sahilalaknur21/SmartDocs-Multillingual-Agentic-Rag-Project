# ingestion/ingestion_worker.py
# WHY THIS EXISTS: Full ingestion pipeline from PDF file to stored vectors.
# Orchestrates: load → preprocess → scan → chunk → embed → store.
# doc_hash idempotency prevents duplicate ingestion. LAW 6.

import uuid
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from ingestion.pdf_loader import load_pdf, ExtractedDocument
from ingestion.indic_preprocessing import detect_script_type
from ingestion.injection_scanner import scan_chunk_for_injection
from ingestion.pii_detector import detect_pii_in_chunk
from ingestion.chunker import build_parent_child_chunks
from ingestion.metadata_builder import build_chunk_metadata
from embeddings.dense_embedder import get_embedder
from vectorstore.pgvector_client import get_pgvector_client


@dataclass
class IngestionResult:
    """Result of a full document ingestion."""
    doc_id: str
    doc_hash: str
    file_name: str
    total_chunks: int
    chunks_stored: int
    primary_language: str
    ingestion_skipped: bool  # True if doc_hash already exists
    success: bool
    error_message: Optional[str] = None


async def ingest_document(
    file_path: str,
    user_id: str,
    doc_title: Optional[str] = None,
) -> IngestionResult:
    """
    Full ingestion pipeline for a single PDF document.

    Pipeline:
        1. Load PDF (pdfplumber)
        2. For each page: indic preprocessing
        3. Injection scan
        4. PII detection
        5. Build parent-child chunks
        6. Build metadata per chunk
        7. Embed all child chunks (multilingual-e5-large)
        8. Store in pgvector with RLS

    Args:
        file_path: Absolute path to PDF file
        user_id: Uploading user's ID — NEVER NULL
        doc_title: Optional title override

    Returns:
        IngestionResult with stats
    """
    assert user_id, "user_id required for per-user isolation"

    path = Path(file_path)
    title = doc_title or path.stem
    doc_id = str(uuid.uuid4())

    # ── Step 1: Load PDF ─────────────────────────────────────────────────────
    doc = load_pdf(file_path)

    if not doc.extraction_success and not doc.full_text:
        return IngestionResult(
            doc_id=doc_id,
            doc_hash=doc.doc_hash,
            file_name=doc.file_name,
            total_chunks=0,
            chunks_stored=0,
            primary_language="en",
            ingestion_skipped=False,
            success=False,
            error_message=f"PDF extraction failed: {doc.error_message}",
        )

    # ── Step 2: Check idempotency ────────────────────────────────────────────
    client = await get_pgvector_client()
    already_stored = await client.store_document(
        doc_id=doc_id,
        user_id=user_id,
        file_name=doc.file_name,
        file_path=file_path,
        doc_hash=doc.doc_hash,
        title=title,
        primary_language=doc.primary_language,
        doc_type="other",
        total_pages=doc.total_pages,
        total_chunks=0,
    )

    if not already_stored:
        return IngestionResult(
            doc_id=doc_id,
            doc_hash=doc.doc_hash,
            file_name=doc.file_name,
            total_chunks=0,
            chunks_stored=0,
            primary_language=doc.primary_language,
            ingestion_skipped=True,
            success=True,
        )

    # ── Step 3: Process each page ────────────────────────────────────────────
    all_child_chunks = []
    all_child_metadata = []

    for page in doc.pages:
        if not page.text.strip():
            continue

        script_type = detect_script_type(page.text)

        # Build parent + child chunks
        parent_chunks, child_chunks = build_parent_child_chunks(
            page_text=page.text,
            page_number=page.page_number,
            lang_code=doc.primary_language,
            script_type=script_type,
        )

        for child in child_chunks:
            # Build metadata for each child chunk
            metadata = build_chunk_metadata(
                chunk=child,
                doc_id=doc_id,
                user_id=user_id,
                source_file_path=file_path,
                doc_title=title,
                doc_primary_language=doc.primary_language,
                doc_hash=doc.doc_hash,
                total_chunks=0,  # Updated after counting all chunks
            )
            # Add chunk text to metadata for storage
            metadata["chunk_text"] = child.text
            all_child_chunks.append(child)
            all_child_metadata.append(metadata)

    if not all_child_chunks:
        return IngestionResult(
            doc_id=doc_id,
            doc_hash=doc.doc_hash,
            file_name=doc.file_name,
            total_chunks=0,
            chunks_stored=0,
            primary_language=doc.primary_language,
            ingestion_skipped=False,
            success=False,
            error_message="No chunks extracted from document",
        )

    # Update total_chunks in all metadata
    total = len(all_child_chunks)
    for meta in all_child_metadata:
        meta["total_chunks"] = total

    # ── Step 4: Embed all child chunks ───────────────────────────────────────
    embedder = get_embedder()
    chunk_texts = [c.text for c in all_child_chunks]
    embeddings = embedder.embed_passages(
        chunk_texts,
        show_progress=True,
    )

    # ── Step 5: Store in pgvector ────────────────────────────────────────────
    stored = await client.store_chunks(
        chunks_with_metadata=all_child_metadata,
        embeddings=list(embeddings),
        user_id=user_id,
    )

    return IngestionResult(
        doc_id=doc_id,
        doc_hash=doc.doc_hash,
        file_name=doc.file_name,
        total_chunks=total,
        chunks_stored=stored,
        primary_language=doc.primary_language,
        ingestion_skipped=False,
        success=True,
    )