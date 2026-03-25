# smartdocs/api/routes/ingest.py
"""
WHY THIS EXISTS:
POST /ingest accepts a PDF file upload and runs the full ingestion pipeline.

KEY DESIGN DECISIONS:
  1. UploadFile gives bytes, not a path. ingest_document() needs a path.
     We write to a named temp file, pass the path, then delete.
     The temp file is cleaned up in a finally block — always deleted.

  2. Idempotency is handled by ingest_document() itself (doc_hash check).
     The route returns the existing doc_id if the file was already uploaded.
     HTTP 200 for both new and duplicate uploads — idempotency is not an error.

  3. File size limit: 50MB enforced in main.py via app config.
     This route trusts that limit was already applied.

  4. PDF-only validation: content-type check + .pdf extension check.
     Not bulletproof (spoofable) but catches accidental wrong uploads.

curl test (replace gst_notice.pdf with any real PDF):
  curl -X POST http://localhost:8000/ingest \
    -H "X-User-ID: user_123" \
    -F "file=@/path/to/gst_notice.pdf" \
    -F "doc_title=GST Notice March 2024"

Expected response:
  {
    "success": true,
    "doc_id": "<uuid>",
    "doc_hash": "<sha256>",
    "file_name": "gst_notice.pdf",
    "primary_language": "hi",
    "doc_type": "gst_notice",
    "total_chunks": 42,
    "chunks_stored": 42,
    "page_count": 3,
    "ingestion_skipped": false,
    "message": "Document ingested successfully"
  }
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

from ingestion.ingestion_worker import ingest_document

logger = logging.getLogger(__name__)
router = APIRouter()

# Allowed MIME types for upload
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "binary/octet-stream",  # Some browsers send this for PDFs
}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB


@router.post("/ingest")
async def ingest_endpoint(
    request: Request,
    file: UploadFile = File(..., description="PDF file to ingest"),
    doc_title: str = Form(default="", description="Optional document title"),
) -> JSONResponse:
    """
    Accepts a PDF upload and runs the full ingestion pipeline.
    Returns doc_id on success — store this for query calls.

    Args:
        file: PDF file (multipart/form-data)
        doc_title: Optional display title (defaults to filename stem)

    Returns:
        JSON with doc_id, chunks_stored, language, doc_type
    """
    user_id: str = request.state.user_id

    # ── Validate file ─────────────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext != ".pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are accepted. Got: {file_ext or 'no extension'}",
        )

    # Content-type check (advisory — some clients send wrong MIME)
    content_type = file.content_type or ""
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(
            f"[ingest] Unusual content-type: {content_type} for file: {file.filename}"
        )
        # Log but don't block — content-type is unreliable from browser uploads

    # ── Read file bytes ───────────────────────────────────────────────────────
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(file_bytes) / 1024 / 1024:.1f}MB. Max: 50MB",
        )

    logger.info(
        f"[ingest] Received upload: file={file.filename} "
        f"size={len(file_bytes) / 1024:.1f}KB user={user_id}"
    )

    # ── Write to temp file ────────────────────────────────────────────────────
    # ingest_document() needs a file path, not bytes.
    # NamedTemporaryFile with delete=False so we control cleanup explicitly.
    # suffix=".pdf" so pdfplumber recognizes the file type.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
            prefix=f"smartdocs_{user_id}_",
        ) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        logger.debug(f"[ingest] Temp file written: {tmp_path}")

        # ── Run ingestion pipeline ─────────────────────────────────────────────
        title = doc_title.strip() or Path(file.filename).stem

        result = await ingest_document(
            file_path=tmp_path,
            user_id=user_id,
            doc_title=title,
        )

        # ── Build response ────────────────────────────────────────────────────
        if result.success or result.ingestion_skipped:
            message = (
                "Document already ingested — returning existing record"
                if result.ingestion_skipped
                else "Document ingested successfully"
            )
            logger.info(
                f"[ingest] ✓ {message}: doc_id={result.doc_id} "
                f"chunks={result.chunks_stored} lang={result.primary_language}"
            )
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "doc_id": result.doc_id,
                    "doc_hash": result.doc_hash,
                    "file_name": result.file_name,
                    "primary_language": result.primary_language,
                    "doc_type": result.doc_type,
                    "total_chunks": result.total_chunks,
                    "chunks_stored": result.chunks_stored,
                    "page_count": result.page_count,
                    "ingestion_skipped": result.ingestion_skipped,
                    "message": message,
                },
            )
        else:
            # Ingestion ran but produced an error (scanned PDF, extraction failure, etc.)
            logger.error(
                f"[ingest] ✗ Ingestion failed: {result.error_message} "
                f"file={file.filename} user={user_id}"
            )
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "doc_id": result.doc_id or None,
                    "file_name": result.file_name,
                    "error": result.error_message,
                    "message": "Ingestion failed — see error field for details",
                },
            )

    except HTTPException:
        raise  # Re-raise FastAPI validation errors unchanged

    except Exception as e:
        logger.exception(f"[ingest] Unexpected error: {e} | file={file.filename}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during ingestion: {str(e)[:200]}",
        )

    finally:
        # Always clean up the temp file — regardless of success or failure
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.debug(f"[ingest] Temp file cleaned up: {tmp_path}")
            except OSError as e:
                logger.warning(f"[ingest] Failed to delete temp file {tmp_path}: {e}")


@router.get("/ingest/status/{doc_id}")
async def ingest_status(doc_id: str, request: Request) -> JSONResponse:
    """
    Returns ingestion status for a specific document.
    Useful for polling from the UI while waiting for large PDFs.

    curl test:
      curl http://localhost:8000/ingest/status/<doc_id> \
        -H "X-User-ID: user_123"
    """
    user_id: str = request.state.user_id

    try:
        from vectorstore.pgvector_client import get_pgvector_client
        client = get_pgvector_client()
        doc = await client.get_document(doc_id=doc_id, user_id=user_id)

        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id} not found for user {user_id}",
            )

        return JSONResponse(
            status_code=200,
            content={
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "file_name": doc["file_name"],
                "primary_language": doc["primary_language"],
                "doc_type": doc["doc_type"],
                "total_chunks": doc["total_chunks"],
                "ingestion_status": doc["ingestion_status"],
                "error_message": doc.get("error_message"),
                "created_at": str(doc.get("created_at", "")),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[ingest] Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/ingest/documents")
async def list_documents(request: Request) -> JSONResponse:
    """
    Returns all documents uploaded by the current user.
    Used by Streamlit UI to populate document selector.

    curl test:
      curl http://localhost:8000/ingest/documents \
        -H "X-User-ID: user_123"
    """
    user_id: str = request.state.user_id

    try:
        from vectorstore.pgvector_client import get_pgvector_client
        client = get_pgvector_client()
        docs = await client.list_user_documents(user_id=user_id)

        return JSONResponse(
            status_code=200,
            content={
                "documents": [
                    {
                        "doc_id": d["doc_id"],
                        "title": d["title"],
                        "file_name": d["file_name"],
                        "primary_language": d["primary_language"],
                        "doc_type": d["doc_type"],
                        "total_chunks": d["total_chunks"],
                        "ingestion_status": d["ingestion_status"],
                        "created_at": str(d.get("created_at", "")),
                    }
                    for d in docs
                ],
                "count": len(docs),
            },
        )
    except Exception as e:
        logger.exception(f"[ingest] List documents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])