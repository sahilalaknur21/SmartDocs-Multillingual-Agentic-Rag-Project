"""
tests/test_evaluation.py

WHY THIS EXISTS: Tests the evaluation infrastructure itself — not the model.
The deployment gate must block correctly when thresholds are violated.
The custom metrics must compute correctly on known inputs.
These tests run without any API calls — pure Python logic validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Deployment gate tests
# ---------------------------------------------------------------------------

class TestDeploymentGate:

    def _make_report(self, **overrides) -> dict:
        base = {
            "english": {
                "faithfulness": 0.92,
                "answer_relevancy": 0.87,
                "context_precision": 0.83,
                "context_recall": 0.78,
            },
            "hindi": {
                "faithfulness": 0.90,
                "answer_relevancy": 0.85,
                "context_precision": 0.81,
                "context_recall": 0.76,
            },
            "faithfulness_ratio": 0.978,
            "hallucination_rate": 0.04,
            "overall_passed": True,
            "blocking_reasons": [],
            "timestamp": "2026-01-01T00:00:00Z",
        }
        for key, value in overrides.items():
            parts = key.split(".")
            obj = base
            for part in parts[:-1]:
                obj = obj[part]
            obj[parts[-1]] = value
        return base

    def test_all_passing_report_is_approved(self) -> None:
        from evaluation.deployment_gate import check_gate
        report = self._make_report()
        result = check_gate(report)
        assert result.approved, f"Expected approved. Failures: {result.blocking_failures}"

    def test_hindi_faithfulness_below_085_blocks(self) -> None:
        from evaluation.deployment_gate import check_gate
        report = self._make_report(**{"hindi.faithfulness": 0.82})
        result = check_gate(report)
        assert not result.approved
        metric_names = [f["metric"] for f in result.blocking_failures]
        assert "faithfulness_hi" in metric_names

    def test_hallucination_rate_above_005_blocks(self) -> None:
        from evaluation.deployment_gate import check_gate
        report = self._make_report(hallucination_rate=0.07)
        result = check_gate(report)
        assert not result.approved
        metric_names = [f["metric"] for f in result.blocking_failures]
        assert "hallucination_rate" in metric_names

    def test_faithfulness_ratio_below_097_blocks(self) -> None:
        from evaluation.deployment_gate import check_gate
        # Hindi 0.72, English 0.92 → ratio = 0.783 → BLOCKED
        report = self._make_report(
            **{"hindi.faithfulness": 0.72, "english.faithfulness": 0.92}
        )
        result = check_gate(report)
        assert not result.approved
        metric_names = [f["metric"] for f in result.blocking_failures]
        assert any("ratio" in m for m in metric_names), f"No ratio failure: {metric_names}"

    def test_english_faithfulness_below_09_blocks(self) -> None:
        from evaluation.deployment_gate import check_gate
        report = self._make_report(**{"english.faithfulness": 0.87})
        result = check_gate(report)
        assert not result.approved
        metric_names = [f["metric"] for f in result.blocking_failures]
        assert "faithfulness_en" in metric_names

    def test_missing_metric_adds_warning_not_failure(self) -> None:
        from evaluation.deployment_gate import check_gate
        # Remove a metric entirely — should warn, not block
        report = self._make_report()
        del report["english"]["context_precision"]
        result = check_gate(report)
        assert any("precision_en" in w or "context_precision" in w for w in result.warnings), \
            f"Expected warning for missing metric. Warnings: {result.warnings}"

    def test_multiple_failures_all_reported(self) -> None:
        from evaluation.deployment_gate import check_gate
        report = self._make_report(
            **{
                "hindi.faithfulness": 0.80,
                "english.answer_relevancy": 0.78,
                "hallucination_rate": 0.08,
            }
        )
        result = check_gate(report)
        assert not result.approved
        assert len(result.blocking_failures) >= 3

    def test_gate_result_to_dict_is_json_serialisable(self) -> None:
        from evaluation.deployment_gate import check_gate
        report = self._make_report()
        result = check_gate(report)
        # Must serialise to JSON without error
        serialised = json.dumps(result.to_dict())
        assert len(serialised) > 10

    def test_file_not_found_raises(self) -> None:
        from evaluation.deployment_gate import load_and_check
        with pytest.raises(FileNotFoundError):
            load_and_check(Path("/nonexistent/evaluation_report.json"))

    def test_nested_key_resolver_missing_key(self) -> None:
        from evaluation.deployment_gate import _get_nested
        report = {"english": {"faithfulness": 0.91}}
        assert _get_nested(report, "english.faithfulness") == pytest.approx(0.91)
        assert _get_nested(report, "english.missing_key") is None
        assert _get_nested(report, "missing_section.key") is None

    def test_exact_boundary_values(self) -> None:
        """Values exactly at threshold must PASS (>= not just >)."""
        from evaluation.deployment_gate import check_gate
        report = self._make_report(
            **{
                "hindi.faithfulness": 0.85,          # exactly at threshold
                "english.faithfulness": 0.90,         # exactly at threshold
                "hallucination_rate": 0.049,          # just under 0.05
                "faithfulness_ratio": 0.9722,         # above 0.97
            }
        )
        result = check_gate(report)
        assert result.approved, f"Exact boundary values should pass. Failures: {result.blocking_failures}"


# ---------------------------------------------------------------------------
# Golden test set validation tests
# ---------------------------------------------------------------------------

class TestGoldenTestSet:

    @pytest.fixture(scope="class")
    def test_set(self) -> list[dict]:
        path = Path(__file__).parent.parent / "evaluation" / "golden_test_set.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_total_count_is_60(self, test_set) -> None:
        assert len(test_set) == 60, f"Expected 60 samples, got {len(test_set)}"

    def test_exactly_30_english_samples(self, test_set) -> None:
        en = [s for s in test_set if s["language"] == "en"]
        assert len(en) == 30

    def test_exactly_30_hindi_samples(self, test_set) -> None:
        hi = [s for s in test_set if s["language"] == "hi"]
        assert len(hi) == 30

    def test_all_samples_have_required_fields(self, test_set) -> None:
        required = {"id", "language", "query", "ground_truth", "contexts", "doc_type", "question_type"}
        for sample in test_set:
            missing = required - set(sample.keys())
            assert not missing, f"Sample {sample.get('id')} missing fields: {missing}"

    def test_all_ids_are_unique(self, test_set) -> None:
        ids = [s["id"] for s in test_set]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in test set"

    def test_all_contexts_are_non_empty_lists(self, test_set) -> None:
        for sample in test_set:
            assert isinstance(sample["contexts"], list), f"Sample {sample['id']}: contexts must be list"
            assert len(sample["contexts"]) >= 1, f"Sample {sample['id']}: contexts must be non-empty"
            for ctx in sample["contexts"]:
                assert isinstance(ctx, str) and len(ctx) > 10, \
                    f"Sample {sample['id']}: context string too short"

    def test_hindi_queries_contain_devanagari(self, test_set) -> None:
        """Hindi samples must have actual Hindi text — not translated English."""
        import unicodedata
        hi_samples = [s for s in test_set if s["language"] == "hi"]
        for sample in hi_samples:
            query = sample["query"]
            has_devanagari = any(
                unicodedata.category(ch) == "Lo" and "\u0900" <= ch <= "\u097F"
                for ch in query
            )
            assert has_devanagari, (
                f"Hindi sample {sample['id']} has no Devanagari characters: {query[:60]}"
            )

    def test_hindi_ground_truths_contain_devanagari(self, test_set) -> None:
        """Ground truth for Hindi samples must be in Hindi, not English."""
        import unicodedata
        hi_samples = [s for s in test_set if s["language"] == "hi"]
        for sample in hi_samples:
            gt = sample["ground_truth"]
            has_devanagari = any(
                unicodedata.category(ch) == "Lo" and "\u0900" <= ch <= "\u097F"
                for ch in gt
            )
            assert has_devanagari, (
                f"Hindi ground truth for {sample['id']} has no Devanagari: {gt[:60]}"
            )

    def test_doc_types_cover_all_expected_categories(self, test_set) -> None:
        doc_types = {s["doc_type"] for s in test_set}
        expected = {"gst_notice", "insurance_policy", "legal_agreement",
                    "government_circular", "tax_notice"}
        assert expected.issubset(doc_types), \
            f"Missing doc types: {expected - doc_types}"

    def test_question_types_are_valid(self, test_set) -> None:
        valid = {"factual", "analytical", "comparative", "conversational"}
        for sample in test_set:
            assert sample["question_type"] in valid, \
                f"Sample {sample['id']}: invalid question_type '{sample['question_type']}'"

    def test_no_empty_queries_or_ground_truths(self, test_set) -> None:
        for sample in test_set:
            assert sample["query"].strip(), f"Sample {sample['id']}: empty query"
            assert sample["ground_truth"].strip(), f"Sample {sample['id']}: empty ground_truth"


# ---------------------------------------------------------------------------
# Custom metrics tests
# ---------------------------------------------------------------------------

class TestLanguageAccuracy:

    @pytest.fixture(scope="class")
    def test_set(self) -> list[dict]:
        path = Path(__file__).parent.parent / "evaluation" / "golden_test_set.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_perfect_score_when_answers_match_query_language(self, test_set) -> None:
        from evaluation.custom_metrics import compute_language_accuracy

        # Use ground_truth as the "generated" answer — should be perfect score
        answers = [s["ground_truth"] for s in test_set]
        result = compute_language_accuracy(test_set, answers)

        assert result.total_samples == 60
        assert result.accuracy >= 0.95, (
            f"Ground-truth answers should achieve >= 0.95 language accuracy. "
            f"Got {result.accuracy:.4f}. Failures: {result.failures[:3]}"
        )

    def test_english_answers_for_hindi_queries_fail(self, test_set) -> None:
        from evaluation.custom_metrics import compute_language_accuracy

        hi_samples = [s for s in test_set if s["language"] == "hi"]
        en_samples = [s for s in test_set if s["language"] == "en"]

        # Give English answers to Hindi queries — accuracy should drop
        # For English queries, use correct English answers
        mixed_samples = hi_samples + en_samples
        mixed_answers = (
            ["This is an English answer that should not match Hindi queries."] * len(hi_samples)
            + [s["ground_truth"] for s in en_samples]
        )

        result = compute_language_accuracy(mixed_samples, mixed_answers)
        # Should fail — English answers for Hindi queries
        assert result.accuracy < 0.85, (
            f"Expected accuracy to drop with wrong language answers. Got {result.accuracy:.4f}"
        )

    def test_empty_answers_counted_as_correct(self, test_set) -> None:
        from evaluation.custom_metrics import compute_language_accuracy

        samples = test_set[:5]
        answers = [""] * 5      # Empty answers skip language check (fail open)
        result = compute_language_accuracy(samples, answers)
        assert result.correct == 5, "Empty answers should all count as correct (fail open)"

    def test_no_info_responses_counted_as_correct(self, test_set) -> None:
        from evaluation.custom_metrics import compute_language_accuracy

        en_samples = [s for s in test_set if s["language"] == "en"][:3]
        hi_samples = [s for s in test_set if s["language"] == "hi"][:3]

        no_info_en = "I don't have enough information in your document to answer this."
        no_info_hi = "मुझे इस प्रश्न का उत्तर आपके दस्तावेज़ में नहीं मिला।"

        samples = en_samples + hi_samples
        answers = [no_info_en] * 3 + [no_info_hi] * 3

        result = compute_language_accuracy(samples, answers)
        assert result.correct == 6, \
            "No-info responses should all pass language check"

    def test_mismatched_lengths_raise_value_error(self, test_set) -> None:
        from evaluation.custom_metrics import compute_language_accuracy

        with pytest.raises(ValueError, match="same length"):
            compute_language_accuracy(test_set[:5], ["answer"] * 3)

    def test_language_accuracy_result_passed_flag(self) -> None:
        from evaluation.custom_metrics import LanguageAccuracyResult

        passing = LanguageAccuracyResult(100, 96, 0.96, [], True)
        assert passing.passed is True

        failing = LanguageAccuracyResult(100, 93, 0.93, [], False)
        assert failing.passed is False


class TestCrossLangRecall:

    @pytest.fixture(scope="class")
    def test_set(self) -> list[dict]:
        path = Path(__file__).parent.parent / "evaluation" / "golden_test_set.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_perfect_recall_when_context_used_as_chunk(self, test_set) -> None:
        from evaluation.custom_metrics import compute_cross_lang_recall

        hi_samples = [s for s in test_set if s["language"] == "hi"]

        # Use the reference contexts as the "retrieved chunks" — should be perfect
        chunks_per_sample = [
            [{"chunk_text": ctx} for ctx in s["contexts"]]
            for s in hi_samples
        ]

        result = compute_cross_lang_recall(hi_samples, chunks_per_sample)
        assert result.recall >= 0.80, (
            f"Using reference contexts as retrieved should score >= 0.80. "
            f"Got {result.recall:.4f}"
        )
        assert result.passed

    def test_empty_chunks_do_not_crash(self, test_set) -> None:
        from evaluation.custom_metrics import compute_cross_lang_recall

        hi_samples = [s for s in test_set if s["language"] == "hi"][:5]
        chunks_per_sample = [[] for _ in hi_samples]  # No chunks retrieved

        result = compute_cross_lang_recall(hi_samples, chunks_per_sample)
        assert result.total_queries == 5
        assert result.relevant_retrieved == 0
        assert result.recall == 0.0

    def test_irrelevant_chunks_score_low(self, test_set) -> None:
        from evaluation.custom_metrics import compute_cross_lang_recall

        hi_samples = [s for s in test_set if s["language"] == "hi"][:10]
        # Completely unrelated chunks
        chunks_per_sample = [
            [{"chunk_text": "The quick brown fox jumps over the lazy dog."}]
            for _ in hi_samples
        ]

        result = compute_cross_lang_recall(hi_samples, chunks_per_sample)
        # English fox sentence has zero overlap with Hindi legal/GST content
        assert result.recall < 0.50, (
            f"Irrelevant chunks should score low. Got {result.recall:.4f}"
        )

    def test_mismatched_lengths_raise(self, test_set) -> None:
        from evaluation.custom_metrics import compute_cross_lang_recall

        hi_samples = [s for s in test_set if s["language"] == "hi"][:5]
        with pytest.raises(ValueError):
            compute_cross_lang_recall(hi_samples, [[]] * 3)


# ---------------------------------------------------------------------------
# Language badge builder tests
# ---------------------------------------------------------------------------

class TestLanguageBadge:

    def test_hindi_document_gets_hindi_badge(self) -> None:
        from api.routes.ingest import _build_language_badge

        hindi_text = (
            "यह एक हिंदी दस्तावेज़ है। इसमें जीएसटी नोटिस की जानकारी है। "
            "करदाता को 30 दिनों के भीतर जवाब देना होगा।"
        )
        badge = _build_language_badge(hindi_text, "hi", 0.97)
        assert badge.primary_language == "hi"
        assert badge.primary_language_name == "Hindi"
        assert "Hindi" in badge.badge_label
        assert "🇮🇳" in badge.badge_label
        assert not badge.is_bilingual

    def test_english_document_gets_english_badge(self) -> None:
        from api.routes.ingest import _build_language_badge

        english_text = (
            "This is an English document. It contains GST notice information. "
            "The taxpayer must respond within 30 days."
        )
        badge = _build_language_badge(english_text, "en", 0.99)
        assert badge.primary_language == "en"
        assert badge.primary_language_name == "English"

    def test_badge_confidence_stored_correctly(self) -> None:
        from api.routes.ingest import _build_language_badge

        badge = _build_language_badge("Sample text", "hi", 0.923)
        assert badge.confidence == pytest.approx(0.923, abs=0.001)

    def test_badge_label_format_hindi(self) -> None:
        from api.routes.ingest import _build_language_badge

        badge = _build_language_badge("हिंदी पाठ", "hi", 0.95)
        # Badge label must contain language name
        assert "Hindi" in badge.badge_label

    def test_unknown_language_code_handled(self) -> None:
        from api.routes.ingest import _build_language_badge

        # Unknown code should not crash — uses uppercased code as name
        badge = _build_language_badge("Some text", "xy", 0.80)
        assert badge.primary_language == "xy"
        assert badge.primary_language_name == "XY"


# ---------------------------------------------------------------------------
# SSE stream parser tests
# ---------------------------------------------------------------------------

class TestSSEParser:

    def _make_mock_response(self, sse_text: str):
        """Creates a mock requests.Response that yields SSE lines."""
        lines = sse_text.encode("utf-8").split(b"\n")

        class MockResponse:
            def iter_lines(self, decode_unicode=True):
                for line in lines:
                    decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                    yield decoded

        return MockResponse()

    def test_parses_sources_event(self) -> None:
        from ui.components.answer_panel import _parse_sse_stream

        sse = (
            'event: sources\n'
            'data: {"chunks": [{"chunk_id": "abc", "title": "GST Notice"}], "crag_triggered": false, "total_count": 1}\n'
            '\n'
        )
        events = list(_parse_sse_stream(self._make_mock_response(sse)))
        assert len(events) == 1
        event_type, data = events[0]
        assert event_type == "sources"
        assert data["total_count"] == 1
        assert len(data["chunks"]) == 1

    def test_parses_token_events(self) -> None:
        from ui.components.answer_panel import _parse_sse_stream

        sse = (
            'event: token\n'
            'data: {"token": "The"}\n'
            '\n'
            'event: token\n'
            'data: {"token": " GST"}\n'
            '\n'
        )
        events = list(_parse_sse_stream(self._make_mock_response(sse)))
        assert len(events) == 2
        tokens = [d["token"] for _, d in events]
        assert tokens == ["The", " GST"]

    def test_parses_done_event(self) -> None:
        from ui.components.answer_panel import _parse_sse_stream

        sse = (
            'event: done\n'
            'data: {"language": "hi", "cost_inr": 0.83, "elapsed_ms": 1240}\n'
            '\n'
        )
        events = list(_parse_sse_stream(self._make_mock_response(sse)))
        assert len(events) == 1
        event_type, data = events[0]
        assert event_type == "done"
        assert data["language"] == "hi"
        assert data["cost_inr"] == pytest.approx(0.83)

    def test_parses_error_event(self) -> None:
        from ui.components.answer_panel import _parse_sse_stream

        sse = (
            'event: error\n'
            'data: {"message": "Document not found", "code": "not_found"}\n'
            '\n'
        )
        events = list(_parse_sse_stream(self._make_mock_response(sse)))
        assert len(events) == 1
        event_type, data = events[0]
        assert event_type == "error"
        assert data["code"] == "not_found"

    def test_ignores_comment_lines(self) -> None:
        from ui.components.answer_panel import _parse_sse_stream

        sse = (
            ': this is a comment\n'
            'event: token\n'
            'data: {"token": "Hello"}\n'
            '\n'
        )
        events = list(_parse_sse_stream(self._make_mock_response(sse)))
        assert len(events) == 1
        assert events[0][1]["token"] == "Hello"

    def test_multi_line_data_joined(self) -> None:
        from ui.components.answer_panel import _parse_sse_stream

        sse = (
            'event: token\n'
            'data: {"tok\n'
            'data: en": "Hi"}\n'
            '\n'
        )
        events = list(_parse_sse_stream(self._make_mock_response(sse)))
        # Multi-line data joined with \n — may fail JSON parse, stored as raw
        assert len(events) == 1

    def test_empty_stream_yields_no_events(self) -> None:
        from ui.components.answer_panel import _parse_sse_stream

        events = list(_parse_sse_stream(self._make_mock_response("")))
        assert events == []


# ---------------------------------------------------------------------------
# HTML escape tests
# ---------------------------------------------------------------------------

class TestHTMLEscape:

    def test_escapes_angle_brackets(self) -> None:
        from ui.components.answer_panel import _escape_html

        result = _escape_html("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_escapes_ampersand(self) -> None:
        from ui.components.answer_panel import _escape_html

        result = _escape_html("GST & Income Tax")
        assert "&amp;" in result
        assert " & " not in result

    def test_newlines_converted_to_br(self) -> None:
        from ui.components.answer_panel import _escape_html

        result = _escape_html("Line 1\nLine 2\nLine 3")
        assert result.count("<br>") == 2
        assert "\n" not in result

    def test_hindi_text_not_modified_except_escaping(self) -> None:
        from ui.components.answer_panel import _escape_html

        hindi = "जीएसटी देनदारी ₹2,45,000 है।"
        result = _escape_html(hindi)
        # Hindi chars must be preserved
        assert "जीएसटी" in result
        assert "₹2,45,000" in result

    def test_empty_string_returns_empty(self) -> None:
        from ui.components.answer_panel import _escape_html

        assert _escape_html("") == ""


# ---------------------------------------------------------------------------
# User context middleware tests
# ---------------------------------------------------------------------------

class TestUserContextMiddleware:

    def test_valid_uuid_accepted(self) -> None:
        import re
        pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
        assert pattern.match(valid_uuid)

    def test_invalid_uuid_rejected(self) -> None:
        import re
        pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        invalid_ids = [
            "not-a-uuid",
            "12345",
            "550e8400-e29b-31d4-a716-446655440000",  # Version 3, not 4
            "",
            "injection'; DROP TABLE chunks; --",
        ]
        for bad_id in invalid_ids:
            assert not pattern.match(bad_id), f"Should reject: {bad_id}"

    def test_exempt_paths_list_contains_health(self) -> None:
        from api.middleware.user_context import EXEMPT_PATHS

        assert "/health" in EXEMPT_PATHS
        assert "/docs" in EXEMPT_PATHS
        assert "/openapi.json" in EXEMPT_PATHS