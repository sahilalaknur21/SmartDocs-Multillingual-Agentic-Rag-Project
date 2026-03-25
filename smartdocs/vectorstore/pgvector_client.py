"""Production-grade async PostgreSQL client with pgvector extension."""

import asyncpg
import logging
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class PGVectorClient:
    """Async pgvector client with RLS enforcement."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
        self.dimension = 1024  # multilingual-e5-large
    
    async def connect(self) -> None:
        """Initialize connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60,
                ssl="require",
                statement_cache_size=0
            )
            logger.info("PGVectorClient: Connection pool initialized")
        except Exception:
            logger.exception("Failed to initialize connection pool")
            raise
    
    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PGVectorClient: Connection pool closed")
    
    async def _set_user_context(self, conn: asyncpg.Connection, user_id: str) -> None:
        """Set user context for RLS."""
        await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
    
    async def health_check(self) -> bool:
        """Basic connectivity check."""
        if not self.pool:
            return False
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception:
            logger.exception("Health check failed")
            return False
    
    async def document_exists(self, doc_id: str, user_id: str) -> bool:
        """Check if document exists."""
        if not self.pool:
            await self.connect()
        try:
            async with self.pool.acquire() as conn:
                await self._set_user_context(conn, user_id)
                row = await conn.fetchrow(
                    """
                    SELECT doc_id 
                    FROM documents 
                    WHERE doc_id::text = $1 AND user_id::text = $2
                    """,
                    doc_id,
                    user_id
                )
                return row is not None
        except Exception:
            logger.exception(f"document_exists failed: doc_id={doc_id}")
            raise
    
    async def store_document(
        self,
        user_id: str,
        doc_hash: str,
        source: str,
        title: str,
        doc_primary_language: str,
        doc_type: str,
        total_chunks: int,
        metadata: Dict[str, Any]
    ) -> Union[str, bool]:
        """Store document. Returns str (existing doc_id) or True (new). Never False."""
        if not self.pool:
            await self.connect()
        try:
            async with self.pool.acquire() as conn:
                await self._set_user_context(conn, user_id)
                
                existing = await conn.fetchrow(
                    """
                    SELECT doc_id::text as doc_id 
                    FROM documents 
                    WHERE doc_hash = $1 AND user_id::text = $2
                    """,
                    doc_hash,
                    user_id
                )
                
                if existing:
                    logger.info(f"Document exists, returning doc_id: {existing['doc_id']}")
                    return existing['doc_id']
                
                new_id = await conn.fetchval(
                    """
                    INSERT INTO documents (
                        user_id, file_name, file_path, doc_hash, title,
                        primary_language, doc_type, total_chunks,
                        created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW()
                    )
                    RETURNING doc_id::text
                    """,
                    user_id,
                    source,
                    source,
                    doc_hash,
                    title,
                    doc_primary_language,
                    doc_type,
                    total_chunks
                )
                
                logger.info(f"New document stored: doc_id={new_id}")
                return True
                
        except Exception:
            logger.exception("store_document failed")
            raise
    
    def _embedding_to_string(self, embedding: List[float]) -> str:
        """Convert embedding list to pgvector string format."""
        return "[" + ",".join(str(f) for f in embedding) + "]"
    
    async def store_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]], 
        user_id: str
    ) -> int:
        """Store chunks with embeddings. Updates total_chunks."""
        if not self.pool:
            await self.connect()
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
        
        if not chunks:
            return 0
        
        doc_id = chunks[0].get("doc_id")
        stored_count = 0
        
        try:
            async with self.pool.acquire() as conn:
                await self._set_user_context(conn, user_id)
                
                for chunk, embedding in zip(chunks, embeddings):
                    # Convert embedding to string format for pgvector
                    embedding_str = self._embedding_to_string(embedding)
                    
                    await conn.execute(
                        """
                        INSERT INTO document_chunks (
                            chunk_id, doc_id, user_id, source, title,
                            doc_primary_language, chunk_language, script_type,
                            doc_type, page_number, section_heading, chunk_index,
                            total_chunks, is_parent, embedding, chunk_text,
                            embedding_model, embedding_model_version,
                            injection_risk, pii_detected, doc_hash, created_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15::vector, $16, $17, $18, $19, $20, $21, NOW()
                        )
                        """,
                        chunk.get("chunk_id", str(hash(chunk["content"]))),
                        doc_id,
                        user_id,
                        chunk.get("source", ""),
                        chunk.get("title", ""),
                        chunk.get("doc_primary_language", "en"),
                        chunk.get("chunk_language", chunk.get("doc_primary_language", "en")),
                        chunk.get("script_type", "latin"),
                        chunk.get("doc_type", "other"),
                        chunk.get("page_number", 0),
                        chunk.get("section_heading", ""),
                        chunk["chunk_index"],
                        len(chunks),
                        chunk.get("is_parent", False),
                        embedding_str,  # FIXED: String format for pgvector
                        chunk["content"],
                        chunk.get("embedding_model", "intfloat/multilingual-e5-large"),
                        chunk.get("embedding_model_version", "1.0.0"),
                        chunk.get("injection_risk", False),
                        chunk.get("pii_detected", False),
                        chunk.get("doc_hash", "")
                    )
                    stored_count += 1
                
                # Update total_chunks
                await conn.execute(
                    """
                    UPDATE documents 
                    SET total_chunks = $1, updated_at = NOW()
                    WHERE doc_id::text = $2 AND user_id::text = $3
                    """,
                    stored_count,
                    doc_id,
                    user_id
                )
                
            logger.info("Ingestion complete: %d chunks", stored_count)
            return stored_count
            
        except Exception:
            logger.exception(f"store_chunks failed for doc_id={doc_id}")
            raise
    
    async def similarity_search(
        self, 
        query_embedding: List[float], 
        user_id: str, 
        doc_id: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Vector similarity search."""
        if not self.pool:
            await self.connect()
        
        try:
            async with self.pool.acquire() as conn:
                await self._set_user_context(conn, user_id)
                
                # Convert query embedding to string
                query_str = self._embedding_to_string(query_embedding)
                
                rows = await conn.fetch(
                    """
                    SELECT 
                        chunk_id,
                        doc_id::text as doc_id,
                        chunk_index,
                        chunk_text as content,
                        section_heading,
                        page_number,
                        1 - (embedding <=> $1::vector) as similarity
                    FROM document_chunks
                    WHERE doc_id::text = $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    query_str,
                    doc_id,
                    top_k
                )
                
                return [dict(row) for row in rows]
        except Exception:
            logger.exception("similarity_search failed")
            raise
    
    async def list_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """List user documents."""
        if not self.pool:
            await self.connect()
        
        try:
            async with self.pool.acquire() as conn:
                await self._set_user_context(conn, user_id)
                
                rows = await conn.fetch(
                    """
                    SELECT 
                        doc_id::text as doc_id,
                        file_name as source,
                        title,
                        primary_language as doc_primary_language,
                        doc_type,
                        total_chunks,
                        doc_hash,
                        created_at,
                        updated_at
                    FROM documents
                    WHERE user_id::text = $1
                    ORDER BY created_at DESC
                    """,
                    user_id
                )
                
                return [dict(row) for row in rows]
        except Exception:
            logger.exception(f"list_user_documents failed")
            raise
    
    async def delete_document(self, doc_id: str, user_id: str) -> int:
        """Delete document. Returns chunk count."""
        if not self.pool:
            await self.connect()
        
        try:
            async with self.pool.acquire() as conn:
                await self._set_user_context(conn, user_id)
                
                chunk_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) 
                    FROM document_chunks 
                    WHERE doc_id::text = $1 AND user_id::text = $2
                    """,
                    doc_id,
                    user_id
                )
                
                await conn.execute(
                    "DELETE FROM document_chunks WHERE doc_id::text = $1 AND user_id::text = $2",
                    doc_id,
                    user_id
                )
                
                await conn.execute(
                    "DELETE FROM documents WHERE doc_id::text = $1 AND user_id::text = $2",
                    doc_id,
                    user_id
                )
                
                logger.info(f"Deleted document {doc_id} with {chunk_count} chunks")
                return chunk_count
                
        except Exception:
            logger.exception(f"delete_document failed")
            raise
# Global client instance - ADD THIS AT THE BOTTOM of vectorstore/pgvector_client.py

from config.settings import settings
pgvector_client = PGVectorClient(settings.database_url)
