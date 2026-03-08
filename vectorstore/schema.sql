-- vectorstore/schema.sql
-- WHY THIS EXISTS: Creates pgvector tables with RLS enforced.
-- Per-user document isolation at database level. LAW 6.
-- Run this ONCE in Supabase SQL editor before any ingestion.

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Documents table ──────────────────────────────────────────────────────────
-- One row per uploaded PDF document

CREATE TABLE IF NOT EXISTS documents (
    doc_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         TEXT NOT NULL,
    file_name       TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    doc_hash        TEXT NOT NULL UNIQUE,  -- SHA-256, prevents duplicate ingestion
    title           TEXT NOT NULL,
    primary_language TEXT NOT NULL DEFAULT 'en',
    doc_type        TEXT NOT NULL DEFAULT 'other',
    total_pages     INTEGER NOT NULL DEFAULT 0,
    total_chunks    INTEGER NOT NULL DEFAULT 0,
    ingestion_status TEXT NOT NULL DEFAULT 'pending',
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Chunks table ─────────────────────────────────────────────────────────────
-- One row per text chunk with embedding vector

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id                TEXT PRIMARY KEY,
    doc_id                  UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    user_id                 TEXT NOT NULL,               -- NEVER NULL — per-user isolation
    source                  TEXT NOT NULL,
    title                   TEXT NOT NULL,
    doc_primary_language    TEXT NOT NULL,
    chunk_language          TEXT NOT NULL,
    script_type             TEXT NOT NULL,
    doc_type                TEXT NOT NULL DEFAULT 'other',
    page_number             INTEGER NOT NULL,
    section_heading         TEXT DEFAULT '',
    chunk_index             INTEGER NOT NULL,
    total_chunks            INTEGER NOT NULL,
    parent_chunk_id         TEXT DEFAULT '',
    is_parent               BOOLEAN NOT NULL DEFAULT FALSE,
    embedding               vector(1024),                -- multilingual-e5-large dim
    chunk_text              TEXT NOT NULL,
    embedding_model         TEXT NOT NULL DEFAULT 'intfloat/multilingual-e5-large',
    embedding_model_version TEXT NOT NULL DEFAULT '1.0.0',
    injection_risk          BOOLEAN NOT NULL DEFAULT FALSE,
    pii_detected            BOOLEAN NOT NULL DEFAULT FALSE,
    doc_hash                TEXT NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Indexes ──────────────────────────────────────────────────────────────────

-- Vector similarity search index (IVFFlat for production)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Per-user query performance
CREATE INDEX IF NOT EXISTS idx_chunks_user_id
    ON document_chunks (user_id);

-- Per-document query performance
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
    ON document_chunks (doc_id);

-- Document deduplication
CREATE INDEX IF NOT EXISTS idx_documents_hash
    ON documents (doc_hash);

-- Per-user document listing
CREATE INDEX IF NOT EXISTS idx_documents_user_id
    ON documents (user_id);

-- ── Row Level Security ───────────────────────────────────────────────────────
-- Users can ONLY see their own documents. Enforced at DB level.

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

-- Documents RLS policies
CREATE POLICY "Users can only see their own documents"
    ON documents FOR SELECT
    USING (user_id = current_setting('app.current_user_id', TRUE));

CREATE POLICY "Users can only insert their own documents"
    ON documents FOR INSERT
    WITH CHECK (user_id = current_setting('app.current_user_id', TRUE));

CREATE POLICY "Users can only delete their own documents"
    ON documents FOR DELETE
    USING (user_id = current_setting('app.current_user_id', TRUE));

-- Chunks RLS policies
CREATE POLICY "Users can only see their own chunks"
    ON document_chunks FOR SELECT
    USING (user_id = current_setting('app.current_user_id', TRUE));

CREATE POLICY "Users can only insert their own chunks"
    ON document_chunks FOR INSERT
    WITH CHECK (user_id = current_setting('app.current_user_id', TRUE));

CREATE POLICY "Users can only delete their own chunks"
    ON document_chunks FOR DELETE
    USING (user_id = current_setting('app.current_user_id', TRUE));

-- ── Service role bypass (for server-side operations) ─────────────────────────
-- The service role key bypasses RLS — only use server-side, never in client

CREATE POLICY "Service role bypass documents"
    ON documents FOR ALL
    USING (TRUE)
    WITH CHECK (TRUE);

CREATE POLICY "Service role bypass chunks"
    ON document_chunks FOR ALL
    USING (TRUE)
    WITH CHECK (TRUE);