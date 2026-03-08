# tests/test_retrieval.py
# Part 3 tests — language detection, classification, reranker logic.
# Does NOT require live Supabase. Tests components in isolation.

import asyncio
from retrieval.language_detector import (
    detect_language,
    get_language_badge,
    build_language_system_prompt_instruction,
)
from retrieval.query_classifier import classify_query, QueryType
from reranking.reranker import rerank_chunks, RerankerDecision, RetrievedChunk
from retrieval.crag_fallback import build_crag_context_note


def test_hindi_detection():
    result = detect_language("जीएसटी नोटिस में मुआवजे की राशि क्या है?")
    assert result.language_code == "hi", f"Got: {result.language_code}"
    assert result.confidence >= 0.85
    assert not result.defaulted_to_english
    print(f"✅ Hindi: {result.language_code} ({result.confidence:.2f})")


def test_english_detection():
    result = detect_language("What is the penalty amount in the GST notice?")
    assert result.language_code == "en"
    print(f"✅ English: {result.language_code} ({result.confidence:.2f})")


def test_empty_defaults_english():
    result = detect_language("")
    assert result.language_code == "en"
    assert result.defaulted_to_english
    print("✅ Empty text → English default")


def test_language_badge():
    assert "Hindi" in get_language_badge(["hi"])
    bilingual = get_language_badge(["hi", "en"])
    assert "Hindi" in bilingual and "English" in bilingual
    print(f"✅ Badge: '{bilingual}'")


def test_system_prompt_instruction():
    result = detect_language("जीएसटी नोटिस में राशि क्या है?")
    instruction = build_language_system_prompt_instruction(result)
    assert "Hindi" in instruction
    assert "hard failure" in instruction
    print(f"✅ System prompt: '{instruction[:60]}...'")


def test_factual_classification():
    result = classify_query("What is the penalty amount?")
    assert result.query_type == QueryType.FACTUAL
    assert result.suggested_top_k == 5
    print(f"✅ Factual: top_k={result.suggested_top_k}")


def test_analytical_classification():
    result = classify_query("Why was this GST notice issued?")
    assert result.query_type == QueryType.ANALYTICAL
    assert result.suggested_top_k == 8
    print(f"✅ Analytical: top_k={result.suggested_top_k}")


def test_hindi_analytical():
    result = classify_query("यह नोटिस क्यों जारी किया गया?")
    assert result.query_type in [QueryType.ANALYTICAL, QueryType.FACTUAL]
    print(f"✅ Hindi query type: {result.query_type.value}")
    
def test_hinglish_short():
    r = detect_language("transformer kya hai")
    assert r.language_code == "hi", f"Expected hi, got {r.language_code}"
    assert r.detection_method == "hinglish"
    print(f"✅ Hinglish short: {r.language_code} via {r.detection_method}")


def test_romanized_hindi():
    r = detect_language("mujhe transformer architecture samjhao")
    assert r.language_code == "hi", f"Expected hi, got {r.language_code}"
    assert r.detection_method == "hinglish"
    print(f"✅ Romanized Hindi: {r.language_code} via {r.detection_method}")


def test_mixed_devanagari_english():
    r = detect_language("मुझे transformer architecture explain करो")
    assert r.language_code == "hi", f"Expected hi, got {r.language_code}"
    assert r.detection_method == "script"
    assert r.confidence == 1.0
    print(f"✅ Mixed Devanagari+English: {r.language_code} via {r.detection_method}")


def test_marathi_detection():
    r = detect_language("मला transformer समजावून सांगा")
    assert r.language_code in ["hi", "mr"], f"Expected hi/mr, got {r.language_code}"
    assert r.detection_method == "script"
    print(f"✅ Marathi/Devanagari: {r.language_code} via {r.detection_method}")


def test_pure_english_technical():
    r = detect_language("RAG multiagent orchestration design")
    assert r.language_code == "en", f"Expected en, got {r.language_code}"
    assert not r.defaulted_to_english or r.language_code == "en"
    print(f"✅ Pure English technical: {r.language_code} via {r.detection_method}")


def test_reranker_with_mock_chunks():
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            chunk_text=f"GST notice penalty section {i} content about tax compliance",
            page_number=i,
            title="GST Notice 2024",
            doc_id="doc_1",
            doc_primary_language="en",
            chunk_language="en",
            script_type="latin",
            doc_type="gst_notice",
            section_heading="",
            parent_chunk_id="",
            pii_detected=False,
            rrf_score=0.9 - (i * 0.05),
        )
        for i in range(10)
    ]
    result = rerank_chunks("What is the GST penalty amount?", chunks, top_k=5)
    assert len(result.chunks) <= 5
    assert result.decision in list(RerankerDecision)
    assert result.latency_ms > 0
    print(f"✅ Reranker: score={result.top_score:.3f}, decision={result.decision}")


def test_crag_note_present():
    note = build_crag_context_note()
    assert "web search" in note.lower()
    assert "from web search" in note
    print("✅ CRAG note contains disclosure")


def test_singleton_ranker():
    from reranking.reranker import get_ranker
    r1 = get_ranker()
    r2 = get_ranker()
    assert r1 is r2
    print("✅ Ranker is singleton — same object both calls")


if __name__ == "__main__":
    print("Running Part 3 tests...\n")
    test_hindi_detection()
    test_english_detection()
    test_empty_defaults_english()
    test_language_badge()
    test_system_prompt_instruction()
    test_factual_classification()
    test_analytical_classification()
    test_hindi_analytical()
    test_reranker_with_mock_chunks()
    test_crag_note_present()
    test_singleton_ranker()
    print("\n── Edge Case Tests ──\n")
    test_hinglish_short()
    test_romanized_hindi()
    test_mixed_devanagari_english()
    test_marathi_detection()
    test_pure_english_technical()
    print("\n✅ ALL PART 3 TESTS PASSED")
    