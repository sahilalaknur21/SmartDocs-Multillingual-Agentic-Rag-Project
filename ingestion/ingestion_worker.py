# smartdocs/ingestion/ingestion_worker.py
"""
WHY THIS EXISTS:
Orchestrates the full ingestion pipeline: PDF → chunks → embeddings → pgvector.

FIX v2 — Root cause of test_duplicate_upload failure:
  Old order: load → check pages → idempotency → chunk
  New order: load → idempotency (via document_exists) → check pages → chunk

  A duplicate must short-circuit IMMEDIATELY after doc_hash is known,
  before any page validation. The original upload already validated pages.
  The old order hit the `if not load_result.pages` guard before reaching
  store_document, making the idempotency test impossible to write correctly.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from vectorstore.pgvector_client import get_pgvector_client
from ingestion.pdf_loader import load_pdf
from ingestion.indic_preprocessing import indic_preprocessing_pipeline, detect_script_type
from ingestion.chunker import build_parent_child_chunks
from ingestion.metadata_builder import build_chunk_metadata
from ingestion.document_classifier import classify_document
from embeddings.dense_embedder import DenseEmbedder

logger = logging.getLogger(__name__)


# ── Embedder Singleton ────────────────────────────────────────────────────────

_embedder: Optional[DenseEmbedder] = None


def get_embedder() -> DenseEmbedder:
    """
    Returns the shared DenseEmbedder singleton.
    multilingual-e5-large loads once on first call (~3-5s, ~1.8GB VRAM).
    All subsequent calls reuse the loaded model — zero reload cost.
    """
    global _embedder
    if _embedder is None:
        logger.info("[worker] Loading DenseEmbedder singleton (first call only)...")
        _embedder = DenseEmbedder()
        logger.info("[worker] DenseEmbedder loaded")
    return _embedder


# ── Result Dataclass ──────────────────────────────────────────────────────────

@dataclass
class IngestionResult:
    doc_id: str
    doc_hash: str
    file_name: str
    total_chunks: int
    chunks_stored: int
    primary_language: str
    ingestion_skipped: bool   # True = duplicate, doc_id is existing ID
    success: bool
    error_message: Optional[str] = None
    page_count: int = 0
    doc_type: str = "other"
    skipped_pages: int = 0


# ── Main Pipeline ─────────────────────────────────────────────────────────────

async def ingest_document(
    file_path: str,
    user_id: str,
    doc_title: Optional[str] = None,
) -> IngestionResult:
    """
    Full ingestion pipeline. Stage order is deliberate:

      STAGE 1: PDF load (gets doc_hash)
      STAGE 2: Idempotency check via document_exists — SHORT-CIRCUITS if duplicate
      STAGE 3: Zero-pages guard + document record registration
      STAGE 4: Indic preprocessing + chunking
      STAGE 5: Metadata build
      STAGE 6: Embedding (multilingual-e5-large, raw text — DenseEmbedder adds prefix)
      STAGE 7: Store chunks
      STAGE 8: Update status to 'completed'

    Idempotency (STAGE 2) runs before page validation (STAGE 3) because:
      - A duplicate upload may come from a re-try after a network timeout
      - The pages were already validated on first upload
      - Failing on pages=[] before reaching the duplicate check is wrong behavior
    """
    assert user_id, "user_id must not be empty (LAW 6)"

    path = Path(file_path)
    file_name = path.name
    title = doc_title or path.stem

    # Pre-initialize — error handler references these directly (no locals())
    doc_id: str = ""
    doc_hash: str = ""
    primary_language: str = "en"
    doc_type: str = "other"
    total_chunks: int = 0

    pgclient = get_pgvector_client()
    if pgclient._pool is None:
        logger.info("[worker] Connecting pgvector pool...")
        await pgclient.connect()

    try:
        # ── STAGE 1: Load PDF ─────────────────────────────────────────────────
        logger.info(f"[worker] STAGE 1 — Loading: {file_name}")
        load_result = load_pdf(file_path)

        if not load_result.extraction_success:
            raise ValueError(f"PDF extraction failed: {load_result.error_message}")

        doc_hash = load_result.doc_hash
        primary_language = load_result.primary_language
        doc_type = classify_document(load_result.full_text, file_name)

        logger.info(
            f"[worker] STAGE 1 DONE — pages={load_result.total_pages} "
            f"lang={primary_language} type={doc_type}"
        )

        # ── STAGE 2: Idempotency — runs before page validation ────────────────
        logger.info(f"[worker] STAGE 2 — Idempotency check")
        existing_id = await pgclient.document_exists(doc_hash=doc_hash, user_id=user_id)

        if existing_id:
            logger.info(f"[worker] STAGE 2 — Duplicate. Returning existing doc_id={existing_id}")
            return IngestionResult(
                doc_id=existing_id,
                doc_hash=doc_hash,
                file_name=file_name,
                total_chunks=0,
                chunks_stored=0,
                primary_language=primary_language,
                ingestion_skipped=True,
                success=True,
                doc_type=doc_type,
                page_count=load_result.total_pages,
            )

        # ── STAGE 3: Page validation + document registration ──────────────────
        if not load_result.pages:
            raise ValueError(f"PDF produced zero pages: {file_name}")

        doc_id = str(uuid.uuid4())
        logger.info(f"[worker] STAGE 3 — Registering doc_id={doc_id}")

        store_result = await pgclient.store_document(
            doc_id=doc_id,
            user_id=user_id,
            doc_hash=doc_hash,
            title=title,
            file_name=file_name,
            file_path=str(path),
            primary_language=primary_language,
            doc_type=doc_type,
        )

        # Race condition guard (two uploads of same file simultaneously)
        if not store_result.is_new:
            logger.info(f"[worker] STAGE 3 — Race condition duplicate: {store_result.doc_id}")
            return IngestionResult(
                doc_id=store_result.doc_id,
                doc_hash=doc_hash,
                file_name=file_name,
                total_chunks=0,
                chunks_stored=0,
                primary_language=primary_language,
                ingestion_skipped=True,
                success=True,
                doc_type=doc_type,
                page_count=load_result.total_pages,
            )

        doc_id = store_result.doc_id
        logger.info(f"[worker] STAGE 3 DONE — registered (status=pending)")

        # ── STAGE 4: Indic preprocessing + chunking ───────────────────────────
        logger.info(f"[worker] STAGE 4 — Chunking {load_result.total_pages} pages")
        all_chunks = []
        skipped_pages = 0

        for page in load_result.pages:
            prep_result = indic_preprocessing_pipeline(
                text=page.text,
                lang_code=primary_language,
            )
            clean_text = prep_result["cleaned_text"]

            if prep_result["is_low_quality"] or not clean_text.strip():
                skipped_pages += 1
                continue

            script_type = detect_script_type(clean_text)
            parents, children = build_parent_child_chunks(
                page_text=clean_text,
                page_number=page.page_number,
                lang_code=primary_language,
                script_type=script_type,
            )
            all_chunks.extend(parents)
            all_chunks.extend(children)

        total_chunks = len(all_chunks)
        logger.info(f"[worker] STAGE 4 DONE — {total_chunks} chunks ({skipped_pages} pages skipped)")

        if total_chunks == 0:
            error_msg = (
                "No text could be extracted. "
                "If this is a scanned PDF, OCR support is required."
            )
            await pgclient.update_document_status(
                doc_id=doc_id, user_id=user_id,
                status="failed", error_msg=error_msg, total_chunks=0,
            )
            return IngestionResult(
                doc_id=doc_id, doc_hash=doc_hash, file_name=file_name,
                total_chunks=0, chunks_stored=0, primary_language=primary_language,
                ingestion_skipped=False, success=False, error_message=error_msg,
                doc_type=doc_type, page_count=load_result.total_pages, skipped_pages=skipped_pages,
            )

        # ── STAGE 5: Metadata ─────────────────────────────────────────────────
        logger.info(f"[worker] STAGE 5 — Building metadata")
        chunks_with_metadata = []
        for chunk in all_chunks:
            meta = build_chunk_metadata(
                chunk=chunk, doc_id=doc_id, user_id=user_id,
                source_file_path=file_name, doc_title=title,
                doc_primary_language=primary_language, doc_hash=doc_hash,
                total_chunks=total_chunks,
            )
            meta["content"] = chunk.text
            chunks_with_metadata.append(meta)

        # ── STAGE 6: Embed ────────────────────────────────────────────────────
        # LAW 4: embed_passages adds "passage: " prefix internally.
        # Pass raw text — do NOT add prefix here. Double-prefix breaks retrieval.
        logger.info(f"[worker] STAGE 6 — Embedding {total_chunks} chunks")
        embedder = get_embedder()
        raw_texts = [c["content"] for c in chunks_with_metadata]

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, embedder.embed_passages, raw_texts)

        # Convert numpy array to list of lists — asyncpg pgvector requires Python lists
        # numpy ndarray triggers: 'truth value of array is ambiguous' inside asyncpg
        embeddings = [emb.tolist() for emb in embeddings]

        if len(embeddings) != total_chunks:
            raise ValueError(f"Embedding mismatch: {len(embeddings)} != {total_chunks}")

        logger.info(f"[worker] STAGE 6 DONE — dim={len(embeddings[0]) if embeddings else 'N/A'}")

        # ── STAGE 7: Store ────────────────────────────────────────────────────
        logger.info(f"[worker] STAGE 7 — Storing chunks")
        chunks_stored = await pgclient.store_chunks(
            chunks=chunks_with_metadata, embeddings=embeddings, user_id=user_id,
        )
        logger.info(f"[worker] STAGE 7 DONE — {chunks_stored} stored")

        # ── STAGE 8: Update status ────────────────────────────────────────────
        await pgclient.update_document_status(
            doc_id=doc_id, user_id=user_id,
            status="completed", total_chunks=chunks_stored,
        )

        logger.info(
            f"[worker] ✓ COMPLETE — {file_name} | doc_id={doc_id} | "
            f"chunks={chunks_stored} | lang={primary_language}"
        )

        return IngestionResult(
            doc_id=doc_id, doc_hash=doc_hash, file_name=file_name,
            total_chunks=total_chunks, chunks_stored=chunks_stored,
            primary_language=primary_language, ingestion_skipped=False,
            success=True, page_count=load_result.total_pages,
            doc_type=doc_type, skipped_pages=skipped_pages,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(
            f"[worker] ✗ FAILED — {file_name}\n"
            f"Error: {error_msg}\n"
            f"{traceback.format_exc()}"
        )

        if doc_id:
            await pgclient.update_document_status(
                doc_id=doc_id, user_id=user_id,
                status="failed", error_msg=error_msg[:500], total_chunks=0,
            )
        else:
            logger.warning("[worker] Failure before registration — no status update")

        return IngestionResult(
            doc_id=doc_id, doc_hash=doc_hash, file_name=file_name,
            total_chunks=0, chunks_stored=0, primary_language=primary_language,
            ingestion_skipped=False, success=False, error_message=error_msg,
            doc_type=doc_type,
        )