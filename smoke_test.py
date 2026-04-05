"""
smoke_test.py — Post-deploy validation for HF Spaces deployment.
5 queries (3 Hindi + 2 English). All must pass.
language_accuracy must be 1.0 on smoke test queries.

Usage:
    python smoke_test.py --api https://YOUR-USERNAME-smartdocs-api.hf.space --user user_123
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import httpx

SMOKE_QUERIES = [
    {"id": "hi_01", "query": "जीएसटी क्या है?",
     "expected_language": "Hindi", "description": "Hindi: What is GST?"},
    {"id": "hi_02", "query": "जीएसटी की दरें क्या हैं?",
     "expected_language": "Hindi", "description": "Hindi: GST rates"},
    {"id": "hi_03", "query": "GSTIN क्या होता है?",
     "expected_language": "Hindi", "description": "Hindi: GSTIN definition"},
    {"id": "en_01", "query": "What is GST and when was it implemented?",
     "expected_language": "English", "description": "English: GST overview"},
    {"id": "en_02", "query": "What are the GST tax rates?",
     "expected_language": "English", "description": "English: GST rates"},
]


@dataclass
class QueryResult:
    query_id: str
    query: str
    expected_language: str
    actual_language: str
    answer: str
    latency_ms: float
    cost_inr: float
    language_match: bool
    has_answer: bool
    passed: bool
    error: Optional[str] = None


def check_health(api_base: str, client: httpx.Client) -> bool:
    print(f"\n[health] Checking: {api_base}/health/ping")
    try:
        r = client.get(f"{api_base}/health/ping", timeout=30)
        if r.status_code != 200:
            print(f"[health] PING FAILED: HTTP {r.status_code}")
            return False
        print(f"[health] PING OK: {r.json()}")

        r2 = client.get(f"{api_base}/health", timeout=30)
        data = r2.json()
        print(f"[health] Status: {data.get('status')}")
        for check, details in data.get("checks", {}).items():
            s = details.get("status", "?")
            icon = "OK" if s in ("ok", "degraded") else "FAIL"
            print(f"  [{icon}] {check}: {s} -- {details.get('detail', '')}")
        return data.get("status") in ("ok", "degraded")
    except Exception as e:
        print(f"[health] ERROR: {e}")
        return False


def run_query(api_base, query, doc_id, user_id, client):
    start = time.perf_counter()
    try:
        r = client.post(
            f"{api_base}/query",
            json={"query": query, "doc_id": doc_id, "doc_title": "Smoke Test"},
            headers={"X-User-ID": user_id, "Content-Type": "application/json"},
            timeout=180,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        if r.status_code != 200:
            return "", "Unknown", latency_ms, 0.0, f"HTTP {r.status_code}: {r.text[:200]}"
        data = r.json()
        return (
            data.get("answer", ""),
            data.get("language", "Unknown"),
            latency_ms,
            data.get("cost_inr", 0.0),
            None,
        )
    except Exception as e:
        return "", "Unknown", round((time.perf_counter() - start) * 1000, 2), 0.0, str(e)


def run_smoke_test(api_base: str, user_id: str, doc_id: Optional[str]) -> bool:
    api_base = api_base.rstrip("/")
    sep = "=" * 64
    print(f"\n{sep}")
    print("SMARTDOCS POST-DEPLOY SMOKE TEST")
    print(f"API: {api_base}")
    print(f"User: {user_id}")
    print(sep)

    with httpx.Client() as client:
        if not check_health(api_base, client):
            print("\nFAILED -- Health check failed. Is the Space running?")
            return False

        if not doc_id:
            print(f"\n[docs] Auto-detecting document for user={user_id}...")
            try:
                r = client.get(
                    f"{api_base}/ingest/documents",
                    headers={"X-User-ID": user_id},
                    timeout=30,
                )
                docs = r.json().get("documents", [])
                completed = [d for d in docs if d.get("ingestion_status") == "completed"]
                if not completed:
                    print("[docs] No completed documents. Upload a PDF via the UI first.")
                    return False
                doc_id = completed[0]["doc_id"]
                print(f"[docs] Using: {completed[0]['title']} (doc_id={doc_id[:8]}...)")
            except Exception as e:
                print(f"[docs] ERROR: {e}")
                return False

        print(f"\n{sep}")
        print("RUNNING 5 SMOKE QUERIES")
        print(sep)

        results = []
        total_cost = 0.0

        for q in SMOKE_QUERIES:
            print(f"\n[{q['id']}] {q['description']}")
            print(f"  Query: {q['query']}")

            answer, language, latency_ms, cost, error = run_query(
                api_base, q["query"], doc_id, user_id, client
            )
            total_cost += cost

            has_answer = bool(answer and len(answer.strip()) > 10)
            language_match = language == q["expected_language"]
            passed = has_answer and language_match and error is None

            result = QueryResult(
                query_id=q["id"],
                query=q["query"],
                expected_language=q["expected_language"],
                actual_language=language,
                answer=answer[:120] + "..." if len(answer) > 120 else answer,
                latency_ms=latency_ms,
                cost_inr=cost,
                language_match=language_match,
                has_answer=has_answer,
                passed=passed,
                error=error,
            )
            results.append(result)

            print(f"  Status:   {'PASS' if passed else 'FAIL'}")
            print(f"  Language: {'OK' if language_match else f'WRONG (got {language})'}")
            print(f"  Answer:   {'OK' if has_answer else 'EMPTY'}")
            print(f"  Latency:  {latency_ms}ms")
            print(f"  Cost:     Rs.{cost:.4f}")
            if error:
                print(f"  ERROR:    {error}")
            if answer:
                print(f"  Preview:  {result.answer}")

        print(f"\n{sep}")
        print("SMOKE TEST SUMMARY")
        print(sep)

        passed_count = sum(1 for r in results if r.passed)
        hindi_results = [r for r in results if r.query_id.startswith("hi")]
        en_results = [r for r in results if r.query_id.startswith("en")]
        hindi_acc = sum(1 for r in hindi_results if r.language_match) / len(hindi_results)
        en_acc = sum(1 for r in en_results if r.language_match) / len(en_results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)

        print(f"\n  Passed:           {passed_count}/5")
        print(f"  Hindi accuracy:   {hindi_acc:.0%}")
        print(f"  English accuracy: {en_acc:.0%}")
        print(f"  Avg latency:      {avg_latency:.0f}ms")
        print(f"  Total cost:       Rs.{total_cost:.4f}")

        all_passed = passed_count == 5 and hindi_acc == 1.0 and en_acc == 1.0

        if all_passed:
            print(f"\n  SMOKE TEST PASSED -- language_accuracy = 1.0")
        else:
            print(f"\n  SMOKE TEST FAILED")
            for r in results:
                if not r.passed:
                    print(f"  FAIL [{r.query_id}]: {r.error or 'no answer/wrong language'}")

        print(sep)

        with open("smoke_test_results.json", "w", encoding="utf-8") as f:
            json.dump({
                "passed": all_passed,
                "passed_count": passed_count,
                "total": 5,
                "hindi_accuracy": hindi_acc,
                "english_accuracy": en_acc,
                "avg_latency_ms": avg_latency,
                "total_cost_inr": total_cost,
                "api_base": api_base,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nResults written: smoke_test_results.json")
        return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=os.getenv("SMARTDOCS_API_URL", "http://localhost:7860"))
    parser.add_argument("--user", default="user_123")
    parser.add_argument("--doc", default=None)
    args = parser.parse_args()
    sys.exit(0 if run_smoke_test(args.api, args.user, args.doc) else 1)