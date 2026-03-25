# smartdocs/vectorstore/pgvector_client.py
"""
WHY THIS EXISTS:
All vector embeddings and document records live in Supabase pgvector.
This client wraps every DB operation with per-user RLS context enforcement.

FIXES IN THIS VERSION:
  1. store_document now returns StoreDocumentResult(is_new, doc_id) — never ambiguous
  2. update_document_status wrapped in transaction — set_config(is_local=true)
     only works inside a transaction. Without it, RLS context was leaking.
  3. exclude_injection_risk actually applied in similarity_search WHERE clause
  4. document_exists added as an explicit, typed method
  5. get_document added for idempotency checks in the API layer
  6. store_chunks batch size lowered to 50 to avoid Supabase payload limits
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Return type for store_document ───────────────────────────────────────────

@dataclass
class StoreDocumentResult:
    """
    Unambiguous result from store_document.
    is_new=True  → document was just inserted, doc_id is the new UUID
    is_new=False → document already existed, doc_id is the existing UUID
    doc_id is ALWAYS the correct UUID to use downstream — never empty string.
    """
    is_new: bool
    doc_id: str


# ── Client ───────────────────────────────────────────────────────────────────

class PgVectorClient:
    """
    Async PostgreSQL client for SmartDocs vector operations.
    All queries enforce per-user Row Level Security via set_config.
    Every method that writes or reads user data must call _set_user_context
    inside an explicit transaction.
    """

    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Opens the connection pool. Call once at app startup."""
        try:
            self._pool = await asyncpg.create_pool(
                dsn=settings.database_url,
                min_size=2,
                max_size=10,
                ssl="require",
                # statement_cache_size=0 required for pgvector prepared statements
                statement_cache_size=0,
                server_settings={
                    "search_path": "public",
                    "application_name": "smartdocs",
                },
                init=self._init_connection,
            )
            logger.info("[pgvector] Connection pool opened (min=2, max=10)")
        except Exception as e:
            logger.exception(f"[pgvector] Failed to open connection pool: {e}")
            raise

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        """Registers the pgvector type on every new connection."""
        await register_vector(conn)

    async def close(self) -> None:
        """Closes the connection pool. Call at app shutdown."""
        if self._pool:
            await self._pool.close()
            logger.info("[pgvector] Connection pool closed")

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(
                "PgVectorClient.connect() was never called. "
                "Call await client.connect() before any query."
            )
        return self._pool

    @staticmethod
    async def _set_user_context(conn: asyncpg.Connection, user_id: str) -> None:
        """
        Sets RLS context for the current transaction.
        MUST be called inside an explicit transaction (async with conn.transaction()).
        set_config with is_local=true is scoped to the current transaction.
        Outside a transaction, it only applies to the SET statement itself —
        the subsequent query runs without user context = silent RLS block.
        """
        await conn.execute(
            "SELECT set_config('app.current_user_id', $1, true)",
            str(user_id),
        )

    # ── Documents ─────────────────────────────────────────────────────────────

    async def store_document(
        self,
        doc_id: str,
        user_id: str,
        doc_hash: str,
        title: str,
        file_name: str,
        file_path: str,
        primary_language: str,
        doc_type: str,
    ) -> StoreDocumentResult:
        """
        Inserts a document record or returns the existing one.

        Returns StoreDocumentResult:
          is_new=True  → freshly inserted, doc_id is the UUID we just stored
          is_new=False → duplicate found (same doc_hash + user_id),
                         doc_id is the EXISTING UUID from the database

        The caller must always use result.doc_id, not the doc_id they passed in,
        because on duplicates the existing UUID is different from the new one generated.
        """
        pool = self._require_pool()
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await self._set_user_context(conn, user_id)

                    # Check for existing document by content hash + user
                    existing_id: Optional[str] = await conn.fetchval(
                        """
                        SELECT doc_id::text
                        FROM documents
                        WHERE doc_hash = $1
                          AND user_id = $2
                        LIMIT 1
                        """,
                        doc_hash,
                        str(user_id),
                    )

                    if existing_id:
                        logger.debug(
                            f"[pgvector] Duplicate doc detected: hash={doc_hash[:8]}... "
                            f"existing_id={existing_id}"
                        )
                        return StoreDocumentResult(is_new=False, doc_id=existing_id)

                    await conn.execute(
                        """
                        INSERT INTO documents (
                            doc_id, user_id, doc_hash, title, file_name,
                            file_path, primary_language, doc_type,
                            total_chunks, ingestion_status, created_at, updated_at
                        ) VALUES (
                            $1::uuid, $2, $3, $4, $5,
                            $6, $7, $8,
                            0, 'pending', NOW(), NOW()
                        )
                        """,
                        doc_id,
                        str(user_id),
                        doc_hash,
                        title,
                        file_name,
                        file_path,
                        primary_language,
                        doc_type,
                    )

                    logger.debug(
                        f"[pgvector] Document inserted: doc_id={doc_id} user={user_id}"
                    )
                    return StoreDocumentResult(is_new=True, doc_id=doc_id)

        except Exception as e:
            logger.exception(f"[pgvector] store_document failed for doc_id={doc_id}: {e}")
            raise

    async def document_exists(self, doc_hash: str, user_id: str) -> Optional[str]:
        """
        Checks if a document with this hash already exists for this user.

        Returns:
          str  → the existing doc_id (UUID as text) if found
          None → document does not exist
        """
        pool = self._require_pool()
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await self._set_user_context(conn, user_id)
                    result: Optional[str] = await conn.fetchval(
                        """
                        SELECT doc_id::text
                        FROM documents
                        WHERE doc_hash = $1
                          AND user_id = $2
                        LIMIT 1
                        """,
                        doc_hash,
                        str(user_id),
                    )
                    return result
        except Exception as e:
            logger.exception(f"[pgvector] document_exists check failed: {e}")
            return None

    async def get_document(self, doc_id: str, user_id: str) -> Optional[dict]:
        """
        Fetches a single document record by doc_id.
        Used by the API layer to return document metadata after upload.
        """
        pool = self._require_pool()
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await self._set_user_context(conn, user_id)
                    row = await conn.fetchrow(
                        """
                        SELECT
                            doc_id::text,
                            user_id,
                            title,
                            file_name,
                            primary_language,
                            doc_type,
                            total_chunks,
                            ingestion_status,
                            error_message,
                            created_at
                        FROM documents
                        WHERE doc_id::uuid = $1::uuid
                          AND user_id = $2
                        """,
                        doc_id,
                        str(user_id),
                    )
                    return dict(row) if row else None
        except Exception as e:
            logger.exception(f"[pgvector] get_document failed for doc_id={doc_id}: {e}")
            return None

    async def update_document_status(
        self,
        doc_id: str,
        user_id: str,
        status: str,
        error_msg: Optional[str] = None,
        total_chunks: int = 0,
    ) -> None:
        """
        Updates ingestion_status and total_chunks on a document record.

        Called at two points:
          1. After successful chunk storage → status='completed', total_chunks=N
          2. In the error handler → status='failed', error_msg=str(e)

        BUG FIXED: Wrapped in explicit transaction so _set_user_context works.
        Without the transaction, set_config(is_local=true) only applied to the
        SET statement, not the UPDATE — the UPDATE ran without RLS context.
        """
        if not doc_id:
            logger.warning("[pgvector] update_document_status called with empty doc_id — skipping")
            return

        pool = self._require_pool()
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():            # ← CRITICAL: must be inside transaction
                    await self._set_user_context(conn, user_id)
                    await conn.execute(
                        """
                        UPDATE documents
                        SET
                            ingestion_status = $1,
                            error_message    = $2,
                            total_chunks     = $3,
                            updated_at       = NOW()
                        WHERE doc_id::uuid = $4::uuid
                          AND user_id = $5
                        """,
                        status,
                        error_msg,
                        total_chunks,
                        doc_id,
                        str(user_id),
                    )
                    logger.debug(
                        f"[pgvector] Document status updated: doc_id={doc_id} "
                        f"status={status} chunks={total_chunks}"
                    )
        except Exception as e:
            logger.exception(
                f"[pgvector] update_document_status FAILED for doc_id={doc_id}: {e}"
            )
            # Do NOT re-raise — this is called from error handlers.
            # A failure here must never suppress the original exception.

    # ── Chunks ────────────────────────────────────────────────────────────────

    async def store_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        user_id: str,
    ) -> int:
        """
        Stores chunk records with their embeddings in document_chunks.

        Column mapping is strict — matches schema.sql exactly.
        ON CONFLICT (chunk_id) DO NOTHING is safe because:
          - chunk_ids are UUIDs generated fresh per ingestion call
          - duplicate documents are caught by doc_hash BEFORE reaching this method
          - the only conflict case is a crash-and-retry, which is correct to skip

        Returns:
          int → number of chunk records passed to the DB
                (ON CONFLICT skips are extremely rare — count is effectively accurate)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have equal length"
            )

        if not chunks:
            logger.warning("[pgvector] store_chunks called with empty chunk list")
            return 0

        pool = self._require_pool()
        stored = 0
        # 50 per batch — keeps Supabase payload under limits and aids debugging
        BATCH_SIZE = 50

        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await self._set_user_context(conn, user_id)

                    for batch_start in range(0, len(chunks), BATCH_SIZE):
                        batch_chunks = chunks[batch_start : batch_start + BATCH_SIZE]
                        batch_embeddings = embeddings[batch_start : batch_start + BATCH_SIZE]

                        records = []
                        for chunk, emb in zip(batch_chunks, batch_embeddings):
                            records.append((
                                str(chunk.get("chunk_id", "")),                      # $1  TEXT
                                str(chunk.get("doc_id", "")),                        # $2  UUID
                                str(user_id),                                        # $3  TEXT
                                str(chunk.get("source", "unknown")),                 # $4  TEXT
                                str(chunk.get("title", "document")),                 # $5  TEXT
                                str(chunk.get("doc_primary_language", "en")),        # $6  TEXT
                                str(chunk.get("chunk_language", "en")),              # $7  TEXT
                                str(chunk.get("script_type", "latin")),              # $8  TEXT
                                str(chunk.get("doc_type", "other")),                 # $9  TEXT
                                int(chunk.get("page_number", 0)),                    # $10 INTEGER
                                str(chunk.get("section_heading", "")),               # $11 TEXT
                                int(chunk.get("chunk_index", 0)),                    # $12 INTEGER
                                int(chunk.get("total_chunks", 0)),                   # $13 INTEGER
                                str(chunk.get("parent_chunk_id", "")),               # $14 TEXT
                                bool(chunk.get("is_parent", False)),                 # $15 BOOLEAN
                                emb.tolist() if hasattr(emb, 'tolist') else list(emb),   # $16 vector(1024)                                                # $16 vector(1024)
                                str(chunk.get("content", chunk.get("text", ""))),   # $17 TEXT chunk_text
                                bool(chunk.get("injection_risk", False)),            # $18 BOOLEAN
                                bool(chunk.get("pii_detected", False)),              # $19 BOOLEAN
                                str(chunk.get("doc_hash", "")),                      # $20 TEXT
                            ))

                        await conn.executemany(
                            """
                            INSERT INTO document_chunks (
                                chunk_id, doc_id, user_id, source, title,
                                doc_primary_language, chunk_language, script_type, doc_type,
                                page_number, section_heading, chunk_index, total_chunks,
                                parent_chunk_id, is_parent, embedding, chunk_text,
                                injection_risk, pii_detected, doc_hash, created_at
                            ) VALUES (
                                $1, $2::uuid, $3, $4, $5,
                                $6, $7, $8, $9,
                                $10, $11, $12, $13,
                                $14, $15, $16::vector, $17,
                                $18, $19, $20, NOW()
                            )
                            ON CONFLICT (chunk_id) DO NOTHING
                            """,
                            records,
                        )
                        stored += len(batch_chunks)
                        logger.debug(
                            f"[pgvector] Stored batch {batch_start}–{batch_start + len(batch_chunks)}"
                        )

            logger.info(f"[pgvector] store_chunks complete: {stored} chunks stored for user={user_id}")
            return stored

        except Exception as e:
            logger.exception(f"[pgvector] store_chunks FAILED at batch starting {stored}: {e}")
            raise

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def similarity_search(
        self,
        query_embedding: list[float],
        user_id: str,
        doc_id: Optional[str] = None,
        top_k: int = 20,
        exclude_injection_risk: bool = True,
    ) -> list[dict]:
        """
        Cosine similarity search within a user's document chunks.

        BUG FIXED: exclude_injection_risk is now actually applied.
        Previously the parameter was declared but the WHERE clause never used it.
        Injection-flagged chunks were being returned and passed to Sarvam-30B.
        """
        pool = self._require_pool()
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await self._set_user_context(conn, user_id)

                    conditions = ["user_id = $2"]
                    params: list = [query_embedding, str(user_id)]

                    if doc_id:
                        params.append(str(doc_id))
                        conditions.append(f"doc_id::uuid = ${len(params)}::uuid")

                    if exclude_injection_risk:
                        conditions.append("injection_risk = FALSE")

                    where_clause = " AND ".join(conditions)
                    params.append(top_k)
                    limit_param = f"${len(params)}"

                    query = f"""
                        SELECT
                            chunk_id::text            AS chunk_id,
                            doc_id::text              AS doc_id,
                            chunk_text,
                            title,
                            page_number,
                            chunk_index,
                            chunk_language,
                            script_type,
                            doc_primary_language,
                            doc_type,
                            section_heading,
                            parent_chunk_id,
                            injection_risk,
                            pii_detected,
                            1 - (embedding <=> $1::vector) AS similarity
                        FROM document_chunks
                        WHERE {where_clause}
                        ORDER BY embedding <=> $1::vector
                        LIMIT {limit_param}
                    """

                    rows = await conn.fetch(query, *params)
                    results = [dict(r) for r in rows]

                    logger.debug(
                        f"[pgvector] similarity_search: returned {len(results)} chunks "
                        f"(top_k={top_k}, doc_id={doc_id}, exclude_injection={exclude_injection_risk})"
                    )
                    return results

        except Exception as e:
            logger.exception(f"[pgvector] similarity_search FAILED: {e}")
            return []

    async def list_user_documents(self, user_id: str) -> list[dict]:
        """Returns all documents for a user, ordered by upload time desc."""
        pool = self._require_pool()
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await self._set_user_context(conn, user_id)
                    rows = await conn.fetch(
                        """
                        SELECT
                            doc_id::text,
                            title,
                            file_name,
                            primary_language,
                            doc_type,
                            total_chunks,
                            ingestion_status,
                            error_message,
                            created_at
                        FROM documents
                        WHERE user_id = $1
                        ORDER BY created_at DESC
                        """,
                        str(user_id),
                    )
                    return [dict(r) for r in rows]
        except Exception as e:
            logger.exception(f"[pgvector] list_user_documents FAILED: {e}")
            return []

    async def delete_document(self, doc_id: str, user_id: str) -> bool:
        """
        Deletes a document and all its chunks (CASCADE in schema).
        Returns True if deleted, False if not found.
        """
        pool = self._require_pool()
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await self._set_user_context(conn, user_id)
                    result = await conn.execute(
                        """
                        DELETE FROM documents
                        WHERE doc_id::uuid = $1::uuid
                          AND user_id = $2
                        """,
                        doc_id,
                        str(user_id),
                    )
                    deleted = result == "DELETE 1"
                    if deleted:
                        logger.info(f"[pgvector] Deleted document doc_id={doc_id}")
                    return deleted
        except Exception as e:
            logger.exception(f"[pgvector] delete_document FAILED: {e}")
            return False


# ── Singleton ────────────────────────────────────────────────────────────────

_client: Optional[PgVectorClient] = None


def get_pgvector_client() -> PgVectorClient:
    """
    Returns the shared PgVectorClient singleton.
    The pool is created lazily on first connect() call.
    Call await get_pgvector_client().connect() once at startup.
    """
    global _client
    if _client is None:
        _client = PgVectorClient()
    return _client