# smartdocs/tests/test_5a_ingestion_pipeline.py
"""
WHY THIS EXISTS:
Part 5A validation. Tests every fix in pgvector_client and ingestion_worker.

FIX SUMMARY (what was wrong in v1 tests):
  1. test_store_document_result_is_new:
       conn.execute.assert_called_once() failed because _set_user_context
       also calls conn.execute("SELECT set_config..."). Two calls total.
       Fix: assert on the INSERT SQL content, not exact call count.

  2. test_store_document_result_duplicate:
       conn.execute.assert_not_called() failed because set_config IS called
       (it always is — it's the RLS enforcement step). INSERT was correctly skipped.
       Fix: assert INSERT SQL was NOT called, not that execute was never called.

  3. test_update_status_uses_transaction:
       Same set_config issue — execute called twice. Fix: check call count >= 2
       and verify UPDATE SQL is in the calls.

  4. test_injection_risk_not_filtered_when_disabled:
       "injection_risk" appears in the SELECT column list. Test checked
       assert "injection_risk" not in sql_query — wrong, the column IS there.
       Fix: check that "injection_risk = FALSE" (the WHERE filter) is absent.

  5. test_duplicate_upload_returns_existing_doc_id:
       Worker checked `if not load_result.pages` BEFORE the idempotency check.
       Test had pages=[] → zero-pages guard fired before duplicate detection.
       Fix: worker now calls document_exists before page validation.
       Test now uses mock_pgclient.document_exists instead of store_document.

HOW TO RUN:
  Unit tests (no DB, no GPU):
    uv run pytest tests/test_5a_ingestion_pipeline.py -v -m "not integration"
  
  Integration test (real DB + GPU):
    uv run pytest tests/test_5a_ingestion_pipeline.py -v -m integration -s

EXPECTED UNIT TEST OUTPUT:
  10 passed in <3s
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


# ── Mock factory ──────────────────────────────────────────────────────────────

def _make_fake_pool(fetchval_result=None):
    """
    Creates a mock asyncpg pool + connection pair.
    conn.execute handles both:
      - set_config (called by _set_user_context)
      - INSERT / UPDATE statements
    Tests must account for set_config being an execute call.
    """
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=fetchval_result)
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.executemany = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)

    tx_cm = MagicMock()
    tx_cm.__aenter__ = AsyncMock(return_value=None)
    tx_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx_cm)

    pool_cm = MagicMock()
    pool_cm.__aenter__ = AsyncMock(return_value=conn)
    pool_cm.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=pool_cm)
    pool.close = AsyncMock()

    return pool, conn


def _make_doc_id() -> str:
    return str(uuid.uuid4())


def _get_execute_sqls(conn) -> list[str]:
    """Extracts all SQL strings passed to conn.execute across all calls."""
    return [str(c.args[0]) for c in conn.execute.call_args_list]


# ── TestStoreDocument ─────────────────────────────────────────────────────────

class TestStoreDocument:

    @pytest.mark.asyncio
    async def test_store_document_result_is_new(self):
        """
        New document (no duplicate): must return StoreDocumentResult(is_new=True, doc_id=our_uuid).
        
        execute is called TWICE: once for set_config, once for INSERT.
        Assert on the INSERT SQL content, not exact call count.
        """
        from vectorstore.pgvector_client import PgVectorClient

        pool, conn = _make_fake_pool(fetchval_result=None)
        client = PgVectorClient()
        client._pool = pool

        doc_id = _make_doc_id()
        result = await client.store_document(
            doc_id=doc_id, user_id="user_test_123", doc_hash="abc123",
            title="GST Notice", file_name="gst.pdf", file_path="/tmp/gst.pdf",
            primary_language="hi", doc_type="gst_notice",
        )

        assert result.is_new is True, "is_new must be True for a new document"
        assert result.doc_id == doc_id, f"doc_id mismatch: {result.doc_id} != {doc_id}"
        assert len(result.doc_id) > 0

        # execute called at least twice: set_config + INSERT
        assert conn.execute.call_count >= 2, (
            f"Expected at least 2 execute calls (set_config + INSERT). "
            f"Got {conn.execute.call_count}"
        )

        # The INSERT must be present in the calls
        sqls = _get_execute_sqls(conn)
        insert_calls = [s for s in sqls if "INSERT INTO documents" in s]
        assert len(insert_calls) == 1, (
            f"Expected exactly 1 INSERT INTO documents call. "
            f"Found {len(insert_calls)} in: {sqls}"
        )

        # set_config must be present (RLS enforcement)
        setconfig_calls = [s for s in sqls if "set_config" in s]
        assert len(setconfig_calls) >= 1, "set_config (RLS context) must be called"

        print(f"\n✓ store_document(is_new=True): doc_id={result.doc_id[:8]}... INSERT confirmed")

    @pytest.mark.asyncio
    async def test_store_document_result_duplicate(self):
        """
        Duplicate document (fetchval returns existing id):
        must return StoreDocumentResult(is_new=False, doc_id=existing_id).
        INSERT must NOT be called. set_config IS still called (RLS).
        """
        from vectorstore.pgvector_client import PgVectorClient

        existing_id = _make_doc_id()
        pool, conn = _make_fake_pool(fetchval_result=existing_id)
        client = PgVectorClient()
        client._pool = pool

        new_uuid = _make_doc_id()
        result = await client.store_document(
            doc_id=new_uuid, user_id="user_test_123", doc_hash="same_hash",
            title="Duplicate", file_name="gst.pdf", file_path="/tmp/gst.pdf",
            primary_language="hi", doc_type="gst_notice",
        )

        assert result.is_new is False, "is_new must be False for duplicate"
        assert result.doc_id == existing_id, (
            f"Must return existing DB id ({existing_id[:8]}...), "
            f"not our new UUID ({new_uuid[:8]}...)"
        )
        assert result.doc_id != new_uuid

        # INSERT must NOT have been called — only set_config was
        sqls = _get_execute_sqls(conn)
        insert_calls = [s for s in sqls if "INSERT INTO documents" in s]
        assert len(insert_calls) == 0, (
            f"INSERT must NOT be called on duplicate. Found: {insert_calls}"
        )

        # set_config IS still called (RLS context always set)
        setconfig_calls = [s for s in sqls if "set_config" in s]
        assert len(setconfig_calls) >= 1, "set_config must always be called"

        print(f"\n✓ store_document(is_new=False): INSERT skipped, existing_id={result.doc_id[:8]}...")


# ── TestUpdateDocumentStatus ──────────────────────────────────────────────────

class TestUpdateDocumentStatus:

    @pytest.mark.asyncio
    async def test_update_status_uses_transaction(self):
        """
        update_document_status must use conn.transaction().
        execute is called TWICE: set_config + UPDATE.
        Assert on the UPDATE SQL content, not exact call count.
        """
        from vectorstore.pgvector_client import PgVectorClient

        pool, conn = _make_fake_pool()
        client = PgVectorClient()
        client._pool = pool

        await client.update_document_status(
            doc_id=_make_doc_id(), user_id="user_test_123",
            status="completed", total_chunks=42,
        )

        # transaction() must be called — this is the bug fix
        conn.transaction.assert_called_once()

        # UPDATE SQL must be present
        sqls = _get_execute_sqls(conn)
        update_calls = [s for s in sqls if "UPDATE documents" in s]
        assert len(update_calls) == 1, (
            f"Expected exactly 1 UPDATE documents call. Found: {update_calls}"
        )

        update_sql = update_calls[0]
        assert "ingestion_status" in update_sql
        assert "total_chunks" in update_sql

        # Verify correct values were passed
        all_args = [str(c.args) for c in conn.execute.call_args_list]
        assert any("completed" in a for a in all_args), (
            f"'completed' not found in execute args: {all_args}"
        )
        assert any("42" in a for a in all_args), (
            f"total_chunks=42 not found in execute args: {all_args}"
        )

        print("\n✓ update_document_status: transaction() called, UPDATE SQL confirmed")

    @pytest.mark.asyncio
    async def test_update_status_empty_doc_id_skips_gracefully(self):
        """Empty doc_id must skip without touching the DB."""
        from vectorstore.pgvector_client import PgVectorClient

        pool, conn = _make_fake_pool()
        client = PgVectorClient()
        client._pool = pool

        await client.update_document_status(
            doc_id="", user_id="user_123", status="failed",
        )

        conn.execute.assert_not_called()
        print("\n✓ update_document_status(doc_id=''): skipped, no DB call")


# ── TestSimilaritySearch ──────────────────────────────────────────────────────

class TestSimilaritySearch:

    @pytest.mark.asyncio
    async def test_exclude_injection_risk_filter_applied(self):
        """
        exclude_injection_risk=True must add 'injection_risk = FALSE' to WHERE clause.
        Previously this parameter was declared but never used — poisoned chunks returned.
        """
        from vectorstore.pgvector_client import PgVectorClient

        pool, conn = _make_fake_pool()
        client = PgVectorClient()
        client._pool = pool

        await client.similarity_search(
            query_embedding=[0.1] * 1024,
            user_id="user_test_123",
            exclude_injection_risk=True,
        )

        sql_query = conn.fetch.call_args[0][0]
        assert "injection_risk = FALSE" in sql_query, (
            f"WHERE clause must contain 'injection_risk = FALSE'. SQL:\n{sql_query}"
        )

        print("\n✓ similarity_search: 'injection_risk = FALSE' in WHERE clause")

    @pytest.mark.asyncio
    async def test_injection_risk_not_filtered_when_disabled(self):
        """
        exclude_injection_risk=False: the FILTER 'injection_risk = FALSE' must be absent.
        NOTE: 'injection_risk' the COLUMN NAME still appears in SELECT — that is correct.
        We check for the WHERE filter specifically, not the column name.
        """
        from vectorstore.pgvector_client import PgVectorClient

        pool, conn = _make_fake_pool()
        client = PgVectorClient()
        client._pool = pool

        await client.similarity_search(
            query_embedding=[0.1] * 1024,
            user_id="user_test_123",
            exclude_injection_risk=False,
        )

        sql_query = conn.fetch.call_args[0][0]

        # The WHERE filter must NOT be present
        assert "injection_risk = FALSE" not in sql_query, (
            f"'injection_risk = FALSE' WHERE filter must be absent when disabled. "
            f"SQL:\n{sql_query}"
        )

        # The SELECT column IS expected — it's always returned for metadata
        assert "injection_risk" in sql_query, (
            "injection_risk column should still appear in SELECT list"
        )

        print("\n✓ similarity_search(exclude=False): WHERE filter absent, SELECT column present")


# ── TestIngestionWorkerErrorHandling ─────────────────────────────────────────

class TestIngestionWorkerErrorHandling:

    @pytest.mark.asyncio
    async def test_error_before_registration_returns_safe_result(self):
        """
        PDF load failure (before doc_id set) must return success=False
        without crashing. update_document_status must NOT be called.
        """
        from ingestion.ingestion_worker import ingest_document

        with patch("ingestion.ingestion_worker.load_pdf") as mock_load:
            mock_load.side_effect = OSError("File not found: /fake/path.pdf")

            with patch("ingestion.ingestion_worker.get_pgvector_client") as mock_fn:
                mock_pg = AsyncMock()
                mock_pg._pool = MagicMock()
                mock_fn.return_value = mock_pg

                result = await ingest_document(
                    file_path="/fake/path.pdf", user_id="user_test_123"
                )

        assert result.success is False
        assert result.doc_id == ""
        assert result.ingestion_skipped is False
        assert "File not found" in result.error_message
        mock_pg.update_document_status.assert_not_called()

        print(f"\n✓ Pre-registration failure: success=False, update_status NOT called")

    @pytest.mark.asyncio
    async def test_error_after_registration_updates_status(self):
        """
        Failure during embedding (after doc_id registered):
        update_document_status must be called with status='failed'.
        """
        from ingestion.ingestion_worker import ingest_document

        registered_doc_id = _make_doc_id()
        from vectorstore.pgvector_client import StoreDocumentResult
        mock_store_result = StoreDocumentResult(is_new=True, doc_id=registered_doc_id)

        mock_load_result = MagicMock()
        mock_load_result.extraction_success = True
        mock_load_result.doc_hash = "abc123"
        mock_load_result.primary_language = "hi"
        mock_load_result.total_pages = 1
        mock_load_result.full_text = "GST Notice content"
        mock_page = MagicMock()
        mock_page.text = "भूमि अधिग्रहण नोटिस"
        mock_page.page_number = 1
        mock_load_result.pages = [mock_page]

        fake_chunk = MagicMock()
        fake_chunk.text = "भूमि अधिग्रहण नोटिस"
        fake_chunk.chunk_id = str(uuid.uuid4())
        fake_chunk.page_number = 1
        fake_chunk.chunk_index = 0
        fake_chunk.is_parent = False
        fake_chunk.parent_chunk_id = None
        fake_chunk.script_type = "devanagari"

        with patch("ingestion.ingestion_worker.load_pdf", return_value=mock_load_result):
            with patch("ingestion.ingestion_worker.classify_document", return_value="gst_notice"):
                with patch("ingestion.ingestion_worker.get_pgvector_client") as mock_fn:
                    mock_pg = AsyncMock()
                    mock_pg._pool = MagicMock()
                    mock_pg.document_exists = AsyncMock(return_value=None)  # Not duplicate
                    mock_pg.store_document = AsyncMock(return_value=mock_store_result)
                    mock_pg.update_document_status = AsyncMock()
                    mock_fn.return_value = mock_pg

                    with patch("ingestion.ingestion_worker.indic_preprocessing_pipeline") as mock_prep:
                        mock_prep.return_value = {
                            "cleaned_text": "भूमि अधिग्रहण नोटिस",
                            "is_low_quality": False,
                            "sentences": [],
                            "lang_code": "hi",
                            "steps_applied": [],
                        }
                        with patch("ingestion.ingestion_worker.detect_script_type", return_value="devanagari"):
                            with patch("ingestion.ingestion_worker.build_parent_child_chunks",
                                       return_value=([fake_chunk], [fake_chunk])):
                                with patch("ingestion.ingestion_worker.build_chunk_metadata",
                                           return_value={"chunk_id": fake_chunk.chunk_id,
                                                         "doc_id": registered_doc_id,
                                                         "user_id": "user_test_123",
                                                         "content": fake_chunk.text}):
                                    with patch("ingestion.ingestion_worker.get_embedder") as mock_emb_fn:
                                        mock_embedder = MagicMock()
                                        mock_embedder.embed_passages.side_effect = RuntimeError("CUDA OOM")
                                        mock_emb_fn.return_value = mock_embedder

                                        result = await ingest_document(
                                            file_path="/fake/gst.pdf",
                                            user_id="user_test_123",
                                        )

        assert result.success is False
        assert "CUDA OOM" in result.error_message

        mock_pg.update_document_status.assert_called_once()
        call_args = mock_pg.update_document_status.call_args
        # Check positional and keyword args
        all_call_str = str(call_args)
        assert "failed" in all_call_str, f"status='failed' expected. Got: {all_call_str}"

        print("\n✓ Post-registration failure: update_document_status called with 'failed'")

    @pytest.mark.asyncio
    async def test_zero_chunk_guard(self):
        """
        All pages low-quality (scanned PDF): must fail with clear error,
        status='failed', NOT 'completed'.
        """
        from ingestion.ingestion_worker import ingest_document
        from vectorstore.pgvector_client import StoreDocumentResult

        registered_doc_id = _make_doc_id()
        mock_store_result = StoreDocumentResult(is_new=True, doc_id=registered_doc_id)

        mock_load_result = MagicMock()
        mock_load_result.extraction_success = True
        mock_load_result.doc_hash = "scan_hash"
        mock_load_result.primary_language = "en"
        mock_load_result.total_pages = 3
        mock_load_result.full_text = ""
        mock_page = MagicMock()
        mock_page.text = "   "
        mock_page.page_number = 1
        mock_load_result.pages = [mock_page, mock_page, mock_page]

        with patch("ingestion.ingestion_worker.load_pdf", return_value=mock_load_result):
            with patch("ingestion.ingestion_worker.classify_document", return_value="other"):
                with patch("ingestion.ingestion_worker.get_pgvector_client") as mock_fn:
                    mock_pg = AsyncMock()
                    mock_pg._pool = MagicMock()
                    mock_pg.document_exists = AsyncMock(return_value=None)
                    mock_pg.store_document = AsyncMock(return_value=mock_store_result)
                    mock_pg.update_document_status = AsyncMock()
                    mock_fn.return_value = mock_pg

                    with patch("ingestion.ingestion_worker.indic_preprocessing_pipeline") as mock_prep:
                        mock_prep.return_value = {
                            "cleaned_text": "",
                            "is_low_quality": True,
                            "sentences": [],
                            "lang_code": "en",
                            "steps_applied": [],
                        }
                        result = await ingest_document(
                            file_path="/fake/scanned.pdf", user_id="user_test_123",
                        )

        assert result.success is False
        assert result.total_chunks == 0
        assert "No text could be extracted" in result.error_message

        mock_pg.update_document_status.assert_called_once()
        assert "failed" in str(mock_pg.update_document_status.call_args)

        print("\n✓ Zero-chunk guard: scanned PDF → status='failed', clear error message")

    @pytest.mark.asyncio
    async def test_duplicate_upload_returns_existing_doc_id(self):
        """
        Same PDF uploaded twice: second upload returns existing doc_id.
        
        FIX: Worker now calls document_exists BEFORE page validation.
        So this test correctly sets pages=[] (the duplicate has no new pages to process)
        and mocks document_exists to return the existing id.
        """
        from ingestion.ingestion_worker import ingest_document

        original_doc_id = _make_doc_id()

        mock_load_result = MagicMock()
        mock_load_result.extraction_success = True
        mock_load_result.doc_hash = "same_hash_both_times"
        mock_load_result.primary_language = "en"
        mock_load_result.total_pages = 2
        mock_load_result.full_text = "Insurance Policy"
        mock_load_result.pages = []  # Doesn't matter — duplicate exits before page check

        with patch("ingestion.ingestion_worker.load_pdf", return_value=mock_load_result):
            with patch("ingestion.ingestion_worker.classify_document", return_value="insurance"):
                with patch("ingestion.ingestion_worker.get_pgvector_client") as mock_fn:
                    mock_pg = AsyncMock()
                    mock_pg._pool = MagicMock()
                    # document_exists returns existing id → triggers early return
                    mock_pg.document_exists = AsyncMock(return_value=original_doc_id)
                    mock_fn.return_value = mock_pg

                    result = await ingest_document(
                        file_path="/fake/insurance.pdf", user_id="user_test_123",
                    )

        assert result.success is True, f"Duplicate upload must succeed. Error: {result.error_message}"
        assert result.ingestion_skipped is True, "ingestion_skipped must be True for duplicate"
        assert result.doc_id == original_doc_id, (
            f"Must return original_doc_id={original_doc_id[:8]}..., "
            f"got={result.doc_id}"
        )

        # store_document must NOT have been called (short-circuit before registration)
        mock_pg.store_document.assert_not_called()

        print(f"\n✓ Duplicate upload: store_document NOT called, existing doc_id returned")


# ── Integration Test ──────────────────────────────────────────────────────────

@pytest.mark.integration
class TestFullIngestionPipeline:
    """
    End-to-end: real PDF → real DB → real embeddings → query → idempotency.

    REQUIREMENTS:
      1. .env with DATABASE_URL, SUPABASE_URL, SUPABASE_SERVICE_KEY
      2. schema.sql + schema_patch.sql run in Supabase SQL Editor
      3. multilingual-e5-large accessible (HuggingFace or local cache)
      4. fpdf2: uv add fpdf2 --dev (already done if you ran the previous step)

    RUN:
      uv run pytest tests/test_5a_ingestion_pipeline.py -v -m integration -s

    EXPECTED:
      Ingestion complete | chunks_stored > 0 | status=completed
      Similarity search returns results with similarity > 0.5
      Second upload: ingestion_skipped=True, same doc_id
    """

    @pytest.mark.asyncio
    async def test_full_ingestion_pipeline(self, tmp_path):
        try:
            from fpdf import FPDF
        except ImportError:
            pytest.skip("fpdf2 not installed. Run: uv add fpdf2 --dev")

        # Create realistic GST notice PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        lines = [
            "GST NOTICE - DEMAND FOR TAX",
            "GSTIN: 27AAPFU0939F1ZV",
            "Assessment Year: 2023-24",
            "Demand Amount: Rs. 1,45,000",
            "This notice is issued under Section 73 of the CGST Act 2017.",
            "You are required to pay the above tax demand within 30 days.",
            "Failure to pay may result in recovery proceedings under Section 79.",
        ]
        for line in lines:
            pdf.cell(0, 10, txt=line, new_x="LMARGIN", new_y="NEXT")
        pdf.add_page()
        pdf.cell(0, 10, txt="Input Tax Credit disallowed: Rs. 45,000", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, txt="Penalty under Section 122: Rs. 29,000", new_x="LMARGIN", new_y="NEXT")

        pdf_path = tmp_path / "test_gst.pdf"
        pdf.output(str(pdf_path))
        print(f"\n[test] Created PDF: {pdf_path} ({pdf_path.stat().st_size} bytes)")

        from vectorstore.pgvector_client import get_pgvector_client
        pgclient = get_pgvector_client()
        await pgclient.connect()

        test_user = f"test_user_{uuid.uuid4().hex[:8]}"
        result_doc_id = None

        try:
            from ingestion.ingestion_worker import ingest_document
            result = await ingest_document(
                file_path=str(pdf_path),
                user_id=test_user,
                doc_title="Integration Test GST Notice",
            )

            print(f"\n[test] Ingestion result:")
            print(f"  success        : {result.success}")
            print(f"  doc_id         : {result.doc_id}")
            print(f"  chunks_stored  : {result.chunks_stored}")
            print(f"  primary_lang   : {result.primary_language}")
            print(f"  doc_type       : {result.doc_type}")

            assert result.success is True, f"Failed: {result.error_message}"
            assert result.doc_id, "doc_id must not be empty"
            assert result.chunks_stored > 0, (
                f"chunks_stored={result.chunks_stored}. Check PDF extraction and chunking."
            )
            result_doc_id = result.doc_id

            # Verify DB record
            doc = await pgclient.get_document(result.doc_id, test_user)
            print(f"\n[test] DB record:")
            print(f"  status         : {doc.get('ingestion_status') if doc else 'NOT FOUND'}")
            print(f"  total_chunks   : {doc.get('total_chunks') if doc else 'N/A'}")

            assert doc is not None, "Document record not found in DB"
            assert doc["ingestion_status"] == "completed", (
                f"Expected 'completed', got '{doc['ingestion_status']}'. "
                "Check update_document_status transaction fix + schema_patch.sql UPDATE policy."
            )
            assert doc["total_chunks"] == result.chunks_stored, (
                f"DB total_chunks ({doc['total_chunks']}) != chunks_stored ({result.chunks_stored})"
            )

            # Similarity search
            from embeddings.dense_embedder import DenseEmbedder
            embedder = DenseEmbedder()
            loop = asyncio.get_running_loop()
            query_emb = await loop.run_in_executor(
                None, lambda: embedder.embed_queries(["GST demand notice section 73"])
            )
            results = await pgclient.similarity_search(
                query_embedding=query_emb[0],
                user_id=test_user,
                doc_id=result.doc_id,
                top_k=5,
            )
            print(f"  similarity hits : {len(results)}")
            assert len(results) > 0, "Similarity search returned 0 results"
            assert results[0]["similarity"] > 0.5, (
                f"Top similarity {results[0]['similarity']:.3f} too low. Check embeddings."
            )

            # Idempotency
            result2 = await ingest_document(
                file_path=str(pdf_path), user_id=test_user, doc_title="Duplicate"
            )
            print(f"\n[test] Idempotency:")
            print(f"  skipped        : {result2.ingestion_skipped}")
            print(f"  same doc_id    : {result2.doc_id == result.doc_id}")

            assert result2.ingestion_skipped is True
            assert result2.doc_id == result.doc_id

            print("\n✓ INTEGRATION TEST PASSED")

        finally:
            if result_doc_id:
                deleted = await pgclient.delete_document(result_doc_id, test_user)
                print(f"\n[test] Cleanup: deleted={deleted}")
            await pgclient.close()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests requiring real DB + GPU (skip with -m 'not integration')"
    )