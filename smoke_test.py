"""
smoke_test.py
WHY THIS EXISTS: Post-deploy validation gate.
5 queries (3 Hindi + 2 English) against the live Railway URL.
All must return answers. language_accuracy must be 1.0.
Run immediately after Railway deploy — before sharing the URL.

Usage:
    python smoke_test.py --api https://your-api.railway.app --user user_123 --doc YOUR_DOC_ID

    # Or with environment variables:
    SMARTDOCS_API_URL=https://your-api.railway.app python smoke_test.py
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


# ── Smoke test queries ────────────────────────────────────────────────────────

SMOKE_QUERIES = [
    {
        "id": "hi_01",
        "language": "hi",
        "query": "जीएसटी क्या है?",
        "expected_language": "Hindi",
        "description": "Basic Hindi GST question",
    },
    {
        "id": "hi_02",
        "language": "hi",
        "query": "जीएसटी की दरें क्या हैं?",
        "expected_language": "Hindi",
        "description": "Hindi GST rates question",
    },
    {
        "id": "hi_03",
        "language": "hi",
        "query": "GSTIN क्या होता है?",
        "expected_language": "Hindi",
        "description": "Hindi GSTIN definition",
    },
    {
        "id": "en_01",
        "language": "en",
        "query": "What is GST and when was it implemented?",
        "expected_language": "English",
        "description": "English GST overview",
    },
    {
        "id": "en_02",
        "language": "en",
        "query": "What are the GST tax rates?",
        "expected_language": "English",
        "description": "English GST rates question",
    },
]


# ── Result dataclass ──────────────────────────────────────────────────────────

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


# ── Health check ──────────────────────────────────────────────────────────────

def check_health(api_base: str, client: httpx.Client) -> bool:
    print(f"\n[health] Checking: {api_base}/health/ping")
    try:
        r = client.get(f"{api_base}/health/ping", timeout=15)
        if r.status_code == 200:
            print(f"[health] PING OK: {r.json()}")
        else:
            print(f"[health] PING FAILED: {r.status_code}")
            return False

        r2 = client.get(f"{api_base}/health", timeout=30)
        data = r2.json()
        print(f"[health] Full check: status={data.get('status')}")
        for check, details in data.get("checks", {}).items():
            status = details.get("status", "?")
            icon = "OK" if status in ("ok", "degraded") else "FAIL"
            print(f"  [{icon}] {check}: {status} — {details.get('detail', '')}")

        return data.get("status") in ("ok", "degraded")
    except Exception as e:
        print(f"[health] ERROR: {e}")
        return False


# ── Single query test ─────────────────────────────────────────────────────────

def run_query(
    api_base: str,
    query: str,
    doc_id: str,
    user_id: str,
    client: httpx.Client,
) -> tuple[str, str, float, float, Optional[str]]:
    start = time.perf_counter()
    try:
        r = client.post(
            f"{api_base}/query",
            json={"query": query, "doc_id": doc_id, "doc_title": "Smoke Test Document"},
            headers={"X-User-ID": user_id, "Content-Type": "application/json"},
            timeout=120,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        if r.status_code != 200:
            return "", "Unknown", latency_ms, 0.0, f"HTTP {r.status_code}: {r.text[:200]}"

        data = r.json()
        answer = data.get("answer", "")
        language = data.get("language", "Unknown")
        cost = data.get("cost_inr", 0.0)
        return answer, language, latency_ms, cost, None

    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return "", "Unknown", latency_ms, 0.0, str(e)


# ── Main smoke test ───────────────────────────────────────────────────────────

def run_smoke_test(
    api_base: str,
    user_id: str,
    doc_id: Optional[str],
) -> bool:
    api_base = api_base.rstrip("/")
    sep = "=" * 64
    print(f"\n{sep}")
    print("SMARTDOCS POST-DEPLOY SMOKE TEST")
    print(f"API:     {api_base}")
    print(f"User:    {user_id}")
    print(f"Doc ID:  {doc_id or 'AUTO-DETECT'}")
    print(sep)

    with httpx.Client() as client:
        # 1. Health check
        if not check_health(api_base, client):
            print("\nSMOKE TEST FAILED — Health check failed")
            return False

        # 2. Detect doc_id if not provided
        if not doc_id:
            print(f"\n[docs] Fetching documents for user={user_id}...")
            try:
                r = client.get(
                    f"{api_base}/ingest/documents",
                    headers={"X-User-ID": user_id},
                    timeout=15,
                )
                docs = r.json().get("documents", [])
                completed = [d for d in docs if d.get("ingestion_status") == "completed"]
                if not completed:
                    print("[docs] No completed documents found.")
                    print("[docs] Upload a PDF via the UI first, then re-run smoke test.")
                    return False
                doc_id = completed[0]["doc_id"]
                print(f"[docs] Using: {completed[0]['title']} (doc_id={doc_id[:8]}...)")
            except Exception as e:
                print(f"[docs] ERROR: {e}")
                return False

        # 3. Run all 5 queries
        print(f"\n{sep}")
        print("RUNNING 5 SMOKE QUERIES")
        print(sep)

        results: list[QueryResult] = []
        total_cost = 0.0

        for q in SMOKE_QUERIES:
            print(f"\n[{q['id']}] {q['description']}")
            print(f"  Query: {q['query']}")

            answer, language, latency_ms, cost, error = run_query(
                api_base=api_base,
                query=q["query"],
                doc_id=doc_id,
                user_id=user_id,
                client=client,
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

            status = "PASS" if passed else "FAIL"
            lang_ok = "OK" if language_match else f"WRONG (got {language}, expected {q['expected_language']})"
            ans_ok = "OK" if has_answer else "EMPTY"

            print(f"  Status:   {status}")
            print(f"  Language: {lang_ok}")
            print(f"  Answer:   {ans_ok}")
            print(f"  Latency:  {latency_ms}ms")
            print(f"  Cost:     Rs.{cost:.4f}")
            if error:
                print(f"  ERROR:    {error}")
            if answer:
                print(f"  Preview:  {result.answer}")

        # 4. Summary
        print(f"\n{sep}")
        print("SMOKE TEST SUMMARY")
        print(sep)

        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        hindi_results = [r for r in results if r.query_id.startswith("hi")]
        english_results = [r for r in results if r.query_id.startswith("en")]
        hindi_accuracy = sum(1 for r in hindi_results if r.language_match) / len(hindi_results)
        english_accuracy = sum(1 for r in english_results if r.language_match) / len(english_results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)

        print(f"\n  Passed:           {passed_count}/5")
        print(f"  Failed:           {failed_count}/5")
        print(f"  Hindi accuracy:   {hindi_accuracy:.0%} (target: 100%)")
        print(f"  English accuracy: {english_accuracy:.0%} (target: 100%)")
        print(f"  Avg latency:      {avg_latency:.0f}ms")
        print(f"  Total cost:       Rs.{total_cost:.4f}")

        all_passed = passed_count == 5
        lang_accuracy_pass = hindi_accuracy == 1.0 and english_accuracy == 1.0

        if all_passed and lang_accuracy_pass:
            print(f"\n  SMOKE TEST PASSED")
            print(f"  Deploy approved — language_accuracy = 1.0")
        else:
            print(f"\n  SMOKE TEST FAILED")
            for r in results:
                if not r.passed:
                    print(f"  FAIL [{r.query_id}]: {r.error or 'no answer / wrong language'}")

        print(sep)

        # Write results to disk
        results_path = "smoke_test_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "passed": all_passed and lang_accuracy_pass,
                "passed_count": passed_count,
                "total": 5,
                "hindi_accuracy": hindi_accuracy,
                "english_accuracy": english_accuracy,
                "avg_latency_ms": avg_latency,
                "total_cost_inr": total_cost,
                "api_base": api_base,
                "results": [
                    {
                        "id": r.query_id,
                        "passed": r.passed,
                        "language_match": r.language_match,
                        "has_answer": r.has_answer,
                        "latency_ms": r.latency_ms,
                        "cost_inr": r.cost_inr,
                        "error": r.error,
                    }
                    for r in results
                ],
            }, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to: {results_path}")

        return all_passed and lang_accuracy_pass


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartDocs post-deploy smoke test")
    parser.add_argument(
        "--api",
        default=os.getenv("SMARTDOCS_API_URL", "http://localhost:8000"),
        help="API base URL (default: SMARTDOCS_API_URL env or localhost:8000)",
    )
    parser.add_argument(
        "--user",
        default="user_123",
        help="User ID for test queries (default: user_123)",
    )
    parser.add_argument(
        "--doc",
        default=None,
        help="Document UUID to query (auto-detected if not provided)",
    )
    args = parser.parse_args()

    success = run_smoke_test(
        api_base=args.api,
        user_id=args.user,
        doc_id=args.doc,
    )
    sys.exit(0 if success else 1)