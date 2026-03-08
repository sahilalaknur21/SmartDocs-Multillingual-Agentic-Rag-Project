# vectorstore/pgvector_client.py
# WHY THIS EXISTS: All pgvector database operations in one place.
# Every query sets user context (RLS) before executing.
# Missing user context = cross-user data access. LAW 6.

import asyncio
import json
import numpy as np
from typing import Optional
import asyncpg
from pgvector.asyncpg import register_vector

from config.settings import get_settings

settings = get_settings()


class PgVectorClient:
    """
    Async pgvector client with per-user Row Level Security.

    Usage:
        client = PgVectorClient()
        await client.connect()
        await client.store_chunks(chunks, embeddings, user_id, doc_id)
        results = await client.similarity_search(query_embedding, user_id, doc_id)
        await client.close()
    """

    def __init__(self):
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Creates asyncpg connection pool with pgvector support."""
        self.pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=2,
            max_size=10,
            setup=self._setup_connection,
        )

    async def _setup_connection(self, conn: asyncpg.Connection) -> None:
        """Registers pgvector type on each new connection."""
        await register_vector(conn)

    async def close(self) -> None:
        """Closes connection pool."""
        if self.pool:
            await self.pool.close()

    async def _set_user_context(
        self,
        conn: asyncpg.Connection,
        user_id: str,
    ) -> None:
        """
        Sets RLS user context before any query.
        This is what makes per-user isolation work.
        NEVER skip this. LAW 6.

        Args:
            conn: Database connection
            user_id: Current user ID
        """
        await conn.execute(
            "SELECT set_config('app.current_user_id', $1, TRUE)",
            user_id,
        )

    async def store_document(
        self,
        doc_id: str,
        user_id: str,
        file_name: str,
        file_path: str,
        doc_hash: str,
        title: str,
        primary_language: str,
        doc_type: str,
        total_pages: int,
        total_chunks: int,
    ) -> bool:
        """
        Stores document record in documents table.
        Checks doc_hash for idempotency — skips if already exists.

        Returns:
            True if inserted, False if already exists
        """
        async with self.pool.acquire() as conn:
            await self._set_user_context(conn, user_id)

            existing = await conn.fetchval(
                "SELECT doc_id FROM documents WHERE doc_hash = $1",
                doc_hash,
            )
            if existing:
                return False

            await conn.execute(
                """
                INSERT INTO documents (
                    doc_id, user_id, file_name, file_path, doc_hash,
                    title, primary_language, doc_type,
                    total_pages, total_chunks, ingestion_status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'completed')
                """,
                doc_id, user_id, file_name, file_path, doc_hash,
                title, primary_language, doc_type,
                total_pages, total_chunks,
            )
            return True

    async def store_chunks(
        self,
        chunks_with_metadata: list[dict],
        embeddings: list[np.ndarray],
        user_id: str,
    ) -> int:
        """
        Stores chunks with embeddings in document_chunks table.
        Bulk insert for performance.

        Args:
            chunks_with_metadata: List of metadata dicts from metadata_builder
            embeddings: List of embedding vectors (one per chunk)
            user_id: Current user ID for RLS

        Returns:
            Number of chunks stored
        """
        if not chunks_with_metadata or not embeddings:
            return 0

        assert len(chunks_with_metadata) == len(embeddings), (
            f"Chunks ({len(chunks_with_metadata)}) and "
            f"embeddings ({len(embeddings)}) count mismatch"
        )

        async with self.pool.acquire() as conn:
            await self._set_user_context(conn, user_id)

            stored = 0
            for metadata, embedding in zip(chunks_with_metadata, embeddings):
                await conn.execute(
                    """
                    INSERT INTO document_chunks (
                        chunk_id, doc_id, user_id, source, title,
                        doc_primary_language, chunk_language, script_type,
                        doc_type, page_number, section_heading,
                        chunk_index, total_chunks, parent_chunk_id, is_parent,
                        embedding, chunk_text, embedding_model,
                        embedding_model_version, injection_risk,
                        pii_detected, doc_hash, created_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18,
                        $19, $20, $21, $22, $23
                    )
                    ON CONFLICT (chunk_id) DO NOTHING
                    """,
                    metadata["chunk_id"],
                    metadata["doc_id"],
                    metadata["user_id"],
                    metadata["source"],
                    metadata["title"],
                    metadata["doc_primary_language"],
                    metadata["chunk_language"],
                    metadata["script_type"],
                    metadata["doc_type"],
                    metadata["page_number"],
                    metadata["section_heading"],
                    metadata["chunk_index"],
                    metadata["total_chunks"],
                    metadata["parent_chunk_id"],
                    metadata["is_parent"],
                    embedding.tolist(),
                    metadata.get("chunk_text", ""),
                    metadata["embedding_model"],
                    metadata["embedding_model_version"],
                    metadata["injection_risk"],
                    metadata["pii_detected"],
                    metadata["doc_hash"],
                    metadata["created_at"],
                )
                stored += 1

            return stored

    async def similarity_search(
        self,
        query_embedding: np.ndarray,
        user_id: str,
        doc_id: Optional[str] = None,
        top_k: int = 20,
        exclude_injection_risk: bool = True,
    ) -> list[dict]:
        """
        Performs cosine similarity search in pgvector.
        Returns top_k most similar chunks for this user.

        Args:
            query_embedding: Query vector from DenseEmbedder.embed_query()
            user_id: Current user — only their chunks searched
            doc_id: Optional — restrict search to specific document
            top_k: Number of results to return (default: 20, reranked to 5)
            exclude_injection_risk: Skip chunks flagged as injection risk

        Returns:
            List of chunk dicts with similarity scores
        """
        async with self.pool.acquire() as conn:
            await self._set_user_context(conn, user_id)

            query_vec = query_embedding.tolist()

            where_clauses = ["user_id = $2"]
            params = [query_vec, user_id]
            param_count = 2

            if doc_id:
                param_count += 1
                where_clauses.append(f"doc_id = ${param_count}")
                params.append(doc_id)

            if exclude_injection_risk:
                where_clauses.append("injection_risk = FALSE")

            where_sql = " AND ".join(where_clauses)
            param_count += 1
            params.append(top_k)

            rows = await conn.fetch(
                f"""
                SELECT
                    chunk_id,
                    doc_id,
                    chunk_text,
                    page_number,
                    title,
                    doc_primary_language,
                    chunk_language,
                    script_type,
                    doc_type,
                    section_heading,
                    parent_chunk_id,
                    pii_detected,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM document_chunks
                WHERE {where_sql}
                ORDER BY embedding <=> $1::vector
                LIMIT ${param_count}
                """,
                *params,
            )

            return [dict(row) for row in rows]

    async def get_parent_chunk(
        self,
        parent_chunk_id: str,
        user_id: str,
    ) -> Optional[dict]:
        """
        Retrieves parent chunk by ID for full context assembly.
        Child chunks are retrieved for precision, parent for generation.

        Args:
            parent_chunk_id: chunk_id of parent chunk
            user_id: Current user for RLS

        Returns:
            Parent chunk dict or None
        """
        async with self.pool.acquire() as conn:
            await self._set_user_context(conn, user_id)

            row = await conn.fetchrow(
                """
                SELECT chunk_id, chunk_text, page_number, title,
                       doc_primary_language, section_heading
                FROM document_chunks
                WHERE chunk_id = $1 AND user_id = $2 AND is_parent = TRUE
                """,
                parent_chunk_id,
                user_id,
            )

            return dict(row) if row else None

    async def delete_document(
        self,
        doc_id: str,
        user_id: str,
    ) -> int:
        """
        Deletes document and all its chunks.
        Cascades via foreign key constraint.

        Returns:
            Number of chunks deleted
        """
        async with self.pool.acquire() as conn:
            await self._set_user_context(conn, user_id)

            chunk_count = await conn.fetchval(
                "SELECT COUNT(*) FROM document_chunks WHERE doc_id = $1 AND user_id = $2",
                doc_id,
                user_id,
            )

            await conn.execute(
                "DELETE FROM documents WHERE doc_id = $1 AND user_id = $2",
                doc_id,
                user_id,
            )

            return chunk_count or 0


# Module-level singleton
_client_instance: PgVectorClient | None = None


async def get_pgvector_client() -> PgVectorClient:
    """
    Returns singleton PgVectorClient with active connection pool.
    Call this anywhere you need database access.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = PgVectorClient()
        await _client_instance.connect()
    return _client_instance