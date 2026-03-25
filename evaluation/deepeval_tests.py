"""
evaluation/deepeval_tests.py

WHY THIS EXISTS: DeepEval hallucination gate runs in CI on every pull request.
A generated answer must not contradict its retrieved context. If it does,
the answer is hallucinated — SmartDocs told a CA something false about a GST notice.
That destroys trust and exposes professionals to legal and financial risk.

Run with:
    pytest evaluation/deepeval_tests.py -v

Requires SARVAM_API_KEY in environment.
DeepEval docs: https://docs.confident-ai.com
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Custom Sarvam-30B model for DeepEval
# ---------------------------------------------------------------------------

class SarvamDeepEvalLLM:
    """
    DeepEval custom LLM using Sarvam-30B via OpenAI-compatible API.
    Implements both sync (generate) and async (a_generate) interfaces
    required by DeepEval's DeepEvalBaseLLM contract.
    """

    def __init__(self) -> None:
        # Import settings inside __init__ to avoid circular imports at module load
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config.settings import get_settings
        self._settings = get_settings()

    def load_model(self) -> Any:
        """Required by DeepEvalBaseLLM — returns the underlying client."""
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            api_key=self._settings.sarvam_api_key,
            base_url=self._settings.sarvam_base_url,
        )

    async def a_generate(self, prompt: str) -> str:
        """Async generation — used by DeepEval's async evaluation pipeline."""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=self._settings.sarvam_api_key,
            base_url=self._settings.sarvam_base_url,
        )
        response = await client.chat.completions.create(
            model=self._settings.sarvam_model_30b,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def generate(self, prompt: str) -> str:
        """Sync generation — wraps async in a new event loop."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context (e.g., pytest-asyncio)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.a_generate(prompt))
                    return future.result()
            else:
                return loop.run_until_complete(self.a_generate(prompt))
        except RuntimeError:
            return asyncio.run(self.a_generate(prompt))

    def get_model_name(self) -> str:
        return f"sarvam-30b (via {self._settings.sarvam_base_url})"


# ---------------------------------------------------------------------------
# Test data preparation
# ---------------------------------------------------------------------------

def _load_test_cases() -> list[dict]:
    """Loads golden test set and builds DeepEval test cases."""
    path = Path(__file__).parent / "golden_test_set.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_test_case(sample: dict) -> Any:
    """Builds a DeepEval LLMTestCase from a golden test set sample."""
    from deepeval.test_case import LLMTestCase

    # Use ground_truth as actual_output for baseline test.
    # In integration tests, replace with SmartDocs generated answer.
    return LLMTestCase(
        input=sample["query"],
        actual_output=sample.get("generated_answer", sample["ground_truth"]),
        context=sample["contexts"],
        expected_output=sample["ground_truth"],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sarvam_llm() -> SarvamDeepEvalLLM:
    """Session-scoped Sarvam LLM fixture — one client for all tests."""
    return SarvamDeepEvalLLM()


@pytest.fixture(scope="session")
def all_samples() -> list[dict]:
    return _load_test_cases()


@pytest.fixture(scope="session")
def english_samples(all_samples: list[dict]) -> list[dict]:
    return [s for s in all_samples if s["language"] == "en"]


@pytest.fixture(scope="session")
def hindi_samples(all_samples: list[dict]) -> list[dict]:
    return [s for s in all_samples if s["language"] == "hi"]


# ---------------------------------------------------------------------------
# Hallucination tests
# ---------------------------------------------------------------------------

class TestHallucinationGate:
    """
    Hallucination CI gate.
    Threshold: 0.5 — a score above 0.5 means the answer contradicts context.
    Every test must pass before deployment.
    """

    def test_english_samples_no_hallucination(
        self,
        sarvam_llm: SarvamDeepEvalLLM,
        english_samples: list[dict],
    ) -> None:
        """
        Validates that English answers do not hallucinate facts not in context.
        Samples 0-9 (first 10 English pairs) — deterministic subset for CI speed.
        """
        from deepeval import assert_test
        from deepeval.metrics import HallucinationMetric

        metric = HallucinationMetric(
            threshold=0.5,
            model=sarvam_llm,
            async_mode=False,
        )

        test_samples = english_samples[:10]
        test_cases = [_build_test_case(s) for s in test_samples]

        for test_case in test_cases:
            assert_test(test_case, [metric])

    def test_hindi_samples_no_hallucination(
        self,
        sarvam_llm: SarvamDeepEvalLLM,
        hindi_samples: list[dict],
    ) -> None:
        """
        Validates that Hindi answers do not hallucinate facts not in Hindi context.
        Hindi hallucination is the harder case — tests it separately.
        Samples 0-9 (first 10 Hindi pairs).
        """
        from deepeval import assert_test
        from deepeval.metrics import HallucinationMetric

        metric = HallucinationMetric(
            threshold=0.5,
            model=sarvam_llm,
            async_mode=False,
        )

        test_samples = hindi_samples[:10]
        test_cases = [_build_test_case(s) for s in test_samples]

        for test_case in test_cases:
            assert_test(test_case, [metric])

    def test_gst_notice_samples_no_hallucination(
        self,
        sarvam_llm: SarvamDeepEvalLLM,
        all_samples: list[dict],
    ) -> None:
        """
        GST notice samples specifically — a CA acting on hallucinated tax advice
        is a critical failure mode. These run at the strictest threshold (0.3).
        """
        from deepeval import assert_test
        from deepeval.metrics import HallucinationMetric

        gst_samples = [s for s in all_samples if s["doc_type"] == "gst_notice"]
        metric = HallucinationMetric(
            threshold=0.3,         # Stricter for financial/legal documents
            model=sarvam_llm,
            async_mode=False,
        )

        test_cases = [_build_test_case(s) for s in gst_samples]
        for test_case in test_cases:
            assert_test(test_case, [metric])


# ---------------------------------------------------------------------------
# Answer relevancy tests
# ---------------------------------------------------------------------------

class TestAnswerRelevancy:
    """
    Validates that answers are relevant to the question asked.
    Applies to both English and Hindi samples.
    """

    def test_english_answer_relevancy(
        self,
        sarvam_llm: SarvamDeepEvalLLM,
        english_samples: list[dict],
    ) -> None:
        from deepeval import assert_test
        from deepeval.metrics import AnswerRelevancyMetric

        metric = AnswerRelevancyMetric(
            threshold=0.7,
            model=sarvam_llm,
            async_mode=False,
        )

        test_samples = english_samples[:8]
        for sample in test_samples:
            test_case = _build_test_case(sample)
            assert_test(test_case, [metric])

    def test_hindi_answer_relevancy(
        self,
        sarvam_llm: SarvamDeepEvalLLM,
        hindi_samples: list[dict],
    ) -> None:
        from deepeval import assert_test
        from deepeval.metrics import AnswerRelevancyMetric

        metric = AnswerRelevancyMetric(
            threshold=0.7,
            model=sarvam_llm,
            async_mode=False,
        )

        test_samples = hindi_samples[:8]
        for sample in test_samples:
            test_case = _build_test_case(sample)
            assert_test(test_case, [metric])


# ---------------------------------------------------------------------------
# Faithfulness tests
# ---------------------------------------------------------------------------

class TestFaithfulness:
    """
    Validates that every claim in the answer is grounded in retrieved context.
    This is the core anti-hallucination check for RAG systems.
    """

    def test_factual_samples_faithfulness(
        self,
        sarvam_llm: SarvamDeepEvalLLM,
        all_samples: list[dict],
    ) -> None:
        from deepeval import assert_test
        from deepeval.metrics import FaithfulnessMetric

        metric = FaithfulnessMetric(
            threshold=0.75,
            model=sarvam_llm,
            async_mode=False,
        )

        factual_samples = [s for s in all_samples if s["question_type"] == "factual"][:12]
        for sample in factual_samples:
            test_case = _build_test_case(sample)
            assert_test(test_case, [metric])

    def test_hindi_factual_faithfulness_vs_english(
        self,
        sarvam_llm: SarvamDeepEvalLLM,
        english_samples: list[dict],
        hindi_samples: list[dict],
    ) -> None:
        """
        Computes faithfulness for English and Hindi factual samples separately.
        Asserts Hindi faithfulness / English faithfulness > 0.97.
        This test blocks deployment if SmartDocs degrades on Hindi.
        """
        from deepeval.metrics import FaithfulnessMetric

        metric = FaithfulnessMetric(
            threshold=0.0,     # Collect scores only — not pass/fail per sample
            model=sarvam_llm,
            async_mode=False,
        )

        en_factual = [s for s in english_samples if s["question_type"] == "factual"][:8]
        hi_factual = [s for s in hindi_samples if s["question_type"] == "factual"][:8]

        en_scores: list[float] = []
        hi_scores: list[float] = []

        for sample in en_factual:
            from deepeval.test_case import LLMTestCase
            tc = LLMTestCase(
                input=sample["query"],
                actual_output=sample.get("generated_answer", sample["ground_truth"]),
                context=sample["contexts"],
            )
            metric.measure(tc)
            en_scores.append(metric.score)

        for sample in hi_factual:
            from deepeval.test_case import LLMTestCase
            tc = LLMTestCase(
                input=sample["query"],
                actual_output=sample.get("generated_answer", sample["ground_truth"]),
                context=sample["contexts"],
            )
            metric.measure(tc)
            hi_scores.append(metric.score)

        avg_en = sum(en_scores) / len(en_scores) if en_scores else 0.0
        avg_hi = sum(hi_scores) / len(hi_scores) if hi_scores else 0.0
        ratio = avg_hi / avg_en if avg_en > 0 else 0.0

        print(f"\nFaithfulness ratio (hi/en): {ratio:.4f} — English: {avg_en:.4f}, Hindi: {avg_hi:.4f}")
        assert ratio >= 0.97, (
            f"Hindi faithfulness ({avg_hi:.4f}) / English ({avg_en:.4f}) = {ratio:.4f} < 0.97. "
            "SmartDocs is not India-first. Fix Hindi retrieval before deploying."
        )


# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------

def pytest_configure(config: Any) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "ci: marks tests to run in CI (fast subset)")
    config.addinivalue_line("markers", "full: marks full evaluation suite (slow)")
