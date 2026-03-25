"""
evaluation/ragas_evaluator.py

WHY THIS EXISTS: Runs RAGAS evaluation separately on English and Hindi test sets.
Hindi faithfulness / English faithfulness ratio > 0.97 is the product's primary
success metric. Aggregate-only evaluation hides Hindi failures behind English scores.

LAW 16: Evaluation gate before deployment.
Minimum targets:
    Faithfulness (English):     > 0.90
    Faithfulness (Hindi):       > 0.88
    Answer Relevancy (English): > 0.85
    Answer Relevancy (Hindi):   > 0.83
    Context Precision:          > 0.80
    Context Recall:             > 0.75
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAGAS version detection — handles both 0.1.x and 0.2.x APIs
# ---------------------------------------------------------------------------

def _detect_ragas_api() -> str:
    """Returns 'v1' for RAGAS 0.1.x or 'v2' for RAGAS 0.2.x."""
    try:
        import importlib.metadata
        version = importlib.metadata.version("ragas")
        major, minor = map(int, version.split(".")[:2])
        return "v2" if (major == 0 and minor >= 2) or major >= 1 else "v1"
    except Exception:
        return "v1"


RAGAS_API = _detect_ragas_api()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LanguageMetrics:
    language: str
    sample_count: int
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    evaluation_time_seconds: float
    errors: list[str] = field(default_factory=list)

    @property
    def passed_thresholds(self) -> bool:
        if self.language == "en":
            return (
                self.faithfulness >= 0.90
                and self.answer_relevancy >= 0.85
                and self.context_precision >= 0.80
                and self.context_recall >= 0.75
            )
        elif self.language == "hi":
            return (
                self.faithfulness >= 0.88
                and self.answer_relevancy >= 0.83
                and self.context_precision >= 0.80
                and self.context_recall >= 0.75
            )
        return True


@dataclass
class EvaluationReport:
    english: LanguageMetrics
    hindi: LanguageMetrics
    faithfulness_ratio: float          # hindi / english — must be > 0.97
    hallucination_rate: float          # 1 - faithfulness (approximation)
    overall_passed: bool
    blocking_reasons: list[str]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "english": asdict(self.english),
            "hindi": asdict(self.hindi),
            "faithfulness_ratio": self.faithfulness_ratio,
            "hallucination_rate": self.hallucination_rate,
            "overall_passed": self.overall_passed,
            "blocking_reasons": self.blocking_reasons,
            "timestamp": self.timestamp,
        }

    def print_summary(self) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print("SMARTDOCS EVALUATION REPORT")
        print(sep)
        _print_lang(self.english, "ENGLISH")
        _print_lang(self.hindi, "HINDI")
        print(f"\n  Faithfulness Ratio (hi/en): {self.faithfulness_ratio:.4f}  (target > 0.97)")
        print(f"  Hallucination Rate:         {self.hallucination_rate:.4f}  (target < 0.05)")
        status = "✅ PASSED" if self.overall_passed else "🚫 BLOCKED"
        print(f"\n  Deployment Gate: {status}")
        if self.blocking_reasons:
            for r in self.blocking_reasons:
                print(f"    ✗ {r}")
        print(sep)


def _print_lang(m: LanguageMetrics, label: str) -> None:
    status = "✅" if m.passed_thresholds else "🚫"
    print(f"\n  {status} {label} ({m.sample_count} samples, {m.evaluation_time_seconds:.1f}s)")
    print(f"    Faithfulness:     {m.faithfulness:.4f}")
    print(f"    Answer Relevancy: {m.answer_relevancy:.4f}")
    print(f"    Context Precision:{m.context_precision:.4f}")
    print(f"    Context Recall:   {m.context_recall:.4f}")
    if m.errors:
        print(f"    Errors: {m.errors[:3]}")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_test_set(
    path: Path | None = None,
    language: str | None = None,
) -> list[dict]:
    """
    Loads golden_test_set.json.
    Optionally filters by language ('en' or 'hi').
    """
    if path is None:
        path = Path(__file__).parent / "golden_test_set.json"

    if not path.exists():
        raise FileNotFoundError(f"Golden test set not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if language:
        data = [item for item in data if item["language"] == language]

    if not data:
        raise ValueError(f"No test samples found for language='{language}'")

    return data


# ---------------------------------------------------------------------------
# RAGAS evaluation — v1 API (datasets.Dataset)
# ---------------------------------------------------------------------------

def _evaluate_v1(
    samples: list[dict],
    ragas_llm: Any,
    ragas_embeddings: Any,
) -> dict[str, float]:
    """RAGAS 0.1.x evaluation using datasets.Dataset."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    # Wire custom LLM and embeddings into each metric
    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        if hasattr(metric, "llm"):
            metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    dataset = Dataset.from_dict({
        "question":     [s["query"] for s in samples],
        "answer":       [s.get("generated_answer", s["ground_truth"]) for s in samples],
        "contexts":     [s["contexts"] for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    return {
        "faithfulness":      float(result["faithfulness"]),
        "answer_relevancy":  float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "context_recall":    float(result["context_recall"]),
    }


# ---------------------------------------------------------------------------
# RAGAS evaluation — v2 API (EvaluationDataset)
# ---------------------------------------------------------------------------

def _evaluate_v2(
    samples: list[dict],
    ragas_llm: Any,
    ragas_embeddings: Any,
) -> dict[str, float]:
    """RAGAS 0.2.x evaluation using EvaluationDataset."""
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        ResponseRelevancy,
    )

    ragas_samples = [
        SingleTurnSample(
            user_input=s["query"],
            response=s.get("generated_answer", s["ground_truth"]),
            retrieved_contexts=s["contexts"],
            reference=s["ground_truth"],
        )
        for s in samples
    ]

    dataset = EvaluationDataset(samples=ragas_samples)

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ],
    )

    scores = result.to_pandas()
    return {
        "faithfulness":      float(scores["faithfulness"].mean()),
        "answer_relevancy":  float(scores["response_relevancy"].mean()),
        "context_precision": float(scores["context_precision"].mean()),
        "context_recall":    float(scores["context_recall"].mean()),
    }


def _evaluate_language(
    samples: list[dict],
    language: str,
    ragas_llm: Any,
    ragas_embeddings: Any,
) -> LanguageMetrics:
    """Runs RAGAS evaluation for one language slice and returns LanguageMetrics."""
    t0 = time.perf_counter()
    errors: list[str] = []

    try:
        if RAGAS_API == "v2":
            scores = _evaluate_v2(samples, ragas_llm, ragas_embeddings)
        else:
            scores = _evaluate_v1(samples, ragas_llm, ragas_embeddings)
    except Exception as exc:
        logger.exception("RAGAS evaluation failed for language=%s", language)
        errors.append(str(exc))
        # Return zero scores so deployment gate blocks correctly
        scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }

    elapsed = time.perf_counter() - t0

    return LanguageMetrics(
        language=language,
        sample_count=len(samples),
        faithfulness=scores["faithfulness"],
        answer_relevancy=scores["answer_relevancy"],
        context_precision=scores["context_precision"],
        context_recall=scores["context_recall"],
        evaluation_time_seconds=elapsed,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# LLM / embeddings factory for RAGAS
# ---------------------------------------------------------------------------

def _build_ragas_llm_and_embeddings() -> tuple[Any, Any]:
    """
    Builds RAGAS-compatible LLM and embeddings wrappers using Sarvam-30B
    and intfloat/multilingual-e5-large.

    Returns (ragas_llm, ragas_embeddings).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config.settings import get_settings
    settings = get_settings()

    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    sarvam_chat = ChatOpenAI(
        model=settings.sarvam_model_30b,
        api_key=settings.sarvam_api_key,           # type: ignore[arg-type]
        base_url=settings.sarvam_base_url,
        temperature=0.0,
        max_tokens=1000,
    )
    ragas_llm = LangchainLLMWrapper(sarvam_chat)

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        hf_embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            encode_kwargs={"normalize_embeddings": True},
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    except ImportError:
        # Fallback: RAGAS will use its default embeddings
        ragas_embeddings = None

    return ragas_llm, ragas_embeddings


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_evaluation(
    test_set_path: Path | None = None,
    generated_answers: dict[str, str] | None = None,
    output_path: Path | None = None,
) -> EvaluationReport:
    """
    Runs the full evaluation pipeline.

    Args:
        test_set_path: Path to golden_test_set.json. Defaults to same directory.
        generated_answers: Optional dict mapping sample id → generated answer.
            If provided, uses these answers instead of ground_truth for evaluation.
            This is how you evaluate real SmartDocs outputs.
        output_path: Optional path to write JSON report. Defaults to
            evaluation/evaluation_report.json

    Returns:
        EvaluationReport with per-language metrics and deployment gate decision.
    """
    from datetime import datetime, timezone

    logger.info("Loading RAGAS LLM and embeddings wrappers...")
    ragas_llm, ragas_embeddings = _build_ragas_llm_and_embeddings()

    logger.info("Loading golden test set from %s", test_set_path)
    all_samples = load_test_set(test_set_path)

    # Inject generated answers if provided
    if generated_answers:
        for sample in all_samples:
            if sample["id"] in generated_answers:
                sample["generated_answer"] = generated_answers[sample["id"]]

    en_samples = [s for s in all_samples if s["language"] == "en"]
    hi_samples = [s for s in all_samples if s["language"] == "hi"]

    logger.info("Evaluating English (%d samples)...", len(en_samples))
    en_metrics = _evaluate_language(en_samples, "en", ragas_llm, ragas_embeddings)

    logger.info("Evaluating Hindi (%d samples)...", len(hi_samples))
    hi_metrics = _evaluate_language(hi_samples, "hi", ragas_llm, ragas_embeddings)

    # Compute derived metrics
    faithfulness_ratio = (
        hi_metrics.faithfulness / en_metrics.faithfulness
        if en_metrics.faithfulness > 0
        else 0.0
    )
    # Approximation: hallucination rate = 1 - avg faithfulness
    avg_faithfulness = (en_metrics.faithfulness + hi_metrics.faithfulness) / 2
    hallucination_rate = max(0.0, 1.0 - avg_faithfulness)

    # Deployment gate logic (LAW 16)
    blocking_reasons: list[str] = []
    if hi_metrics.faithfulness < 0.85:
        blocking_reasons.append(
            f"Hindi faithfulness {hi_metrics.faithfulness:.4f} < 0.85 — DEPLOYMENT BLOCKED"
        )
    if faithfulness_ratio < 0.97:
        blocking_reasons.append(
            f"Hindi/English faithfulness ratio {faithfulness_ratio:.4f} < 0.97 — product not India-first"
        )
    if hallucination_rate >= 0.05:
        blocking_reasons.append(
            f"Hallucination rate {hallucination_rate:.4f} >= 0.05 — DEPLOYMENT BLOCKED"
        )
    if not en_metrics.passed_thresholds:
        blocking_reasons.append("English metrics below threshold")
    if not hi_metrics.passed_thresholds:
        blocking_reasons.append("Hindi metrics below threshold")

    report = EvaluationReport(
        english=en_metrics,
        hindi=hi_metrics,
        faithfulness_ratio=faithfulness_ratio,
        hallucination_rate=hallucination_rate,
        overall_passed=len(blocking_reasons) == 0,
        blocking_reasons=blocking_reasons,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Write report to disk
    if output_path is None:
        output_path = Path(__file__).parent / "evaluation_report.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Evaluation report written to %s", output_path)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stdout,
    )
    report = run_evaluation()
    report.print_summary()
    sys.exit(0 if report.overall_passed else 1)
