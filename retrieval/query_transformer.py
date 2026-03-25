# retrieval/query_transformer.py
# WHY THIS EXISTS: Transforms one query into multiple retrieval queries.
# Multi-query + HyDE + step-back run IN PARALLEL via asyncio.gather.
# Singleton client. Retry via tenacity. Structured JSON outputs.
# Single query misses 15-30% of relevant chunks. LAW 5.

import json
import logging
import time
import asyncio
from dataclasses import dataclass, field
from functools import lru_cache

import tenacity
from openai import AsyncOpenAI, APIError, RateLimitError

from config.settings import get_settings
from retrieval.language_detector import LanguageDetectionResult

settings = get_settings()
logger = logging.getLogger(__name__)

# ── Singleton async client ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_sarvam_client() -> AsyncOpenAI:
    """
    Singleton AsyncOpenAI client pointed at Sarvam API.
    Created once. Reused for all calls in process lifetime.
    Timeout=30s. max_retries=3 (tenacity handles above that).
    """
    return AsyncOpenAI(
        api_key=settings.sarvam_api_key,
        base_url=settings.sarvam_base_url,
        timeout=30.0,
        max_retries=0,  # tenacity handles retries — not openai's built-in
    )


# ── Retry policy ─────────────────────────────────────────────────────────────

def _is_retryable(exc: Exception) -> bool:
    """Only retry on rate limits and transient API errors."""
    return isinstance(exc, (RateLimitError, APIError))


RETRY_POLICY = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=8),
    stop=tenacity.stop_after_attempt(3),
    reraise=True,
)


# ── Output models ─────────────────────────────────────────────────────────────

@dataclass
class TransformedQueries:
    original: str
    multi_queries: list[str] = field(default_factory=list)
    hyde_passage: str = ""
    step_back_query: str = ""
    all_queries: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    fallback_used: bool = False

    def build_all_queries(self) -> None:
        """
        Combines all query variants into deduplicated list.
        Includes HyDE passage for embedding.
        Enforces minimum token length guard (> 5 chars).
        """
        candidates = [
            self.original,
            *self.multi_queries,
            self.step_back_query,
            self.hyde_passage,
        ]
        seen: set[str] = set()
        result = []
        for q in candidates:
            q_norm = q.strip()
            if len(q_norm) < 6:
                continue
            if q_norm.lower() not in seen:
                seen.add(q_norm.lower())
                result.append(q_norm)
        self.all_queries = result


# ── Core generation functions ─────────────────────────────────────────────────

@RETRY_POLICY
async def _generate_multi_queries(
    query: str,
    lang_result: LanguageDetectionResult,
    num_queries: int = 3,
) -> list[str]:
    """
    Generates {num_queries} variations of the query via Sarvam-30B.
    Returns structured JSON — not free-text parsing.
    Temperature=0.3 — low temperature for query rewriting.
    """
    client = get_sarvam_client()

    prompt = f"""Rewrite the question below in {num_queries} different ways.

Constraints:
- Identical meaning, different wording
- Language: {lang_result.language_name}
- No explanation, no numbering

Return ONLY valid JSON — no markdown, no backticks:
{{"queries": ["variation1", "variation2", "variation3"]}}

Question: {query}"""

    response = await client.chat.completions.create(
        model=settings.sarvam_model_30b,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
        queries = data.get("queries", [])
        # Guard: each query must be a non-empty string
        validated = [
            q.strip() for q in queries
            if isinstance(q, str) and len(q.strip()) > 5
        ]
        return validated[:num_queries]
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(
            "Multi-query JSON parse failed — returning original",
            extra={"error": str(e), "raw_output": raw[:200]},
        )
        return [query]


@RETRY_POLICY
async def _generate_hyde_passage(
    query: str,
    lang_result: LanguageDetectionResult,
) -> str:
    """
    HyDE: generates a hypothetical answer passage.
    This passage is EMBEDDED — not shown to user.
    Answer-space embedding retrieves better than question-space for factual queries.
    Temperature=0.2 — deterministic, no hallucination-risk creativity.
    """
    client = get_sarvam_client()

    prompt = f"""Generate a short hypothetical document passage that would answer the question.

Rules:
- Maximum 2 sentences
- Factual tone only — no speculation, no opinions
- Write as if you are the source document
- Language: {lang_result.language_name}

Return ONLY valid JSON — no markdown, no backticks:
{{"hyde_passage": "your passage here"}}

Question: {query}"""

    response = await client.chat.completions.create(
        model=settings.sarvam_model_30b,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
        passage = data.get("hyde_passage", "").strip()
        return passage if len(passage) > 10 else query
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(
            "HyDE JSON parse failed — using original query",
            extra={"error": str(e), "raw_output": raw[:200]},
        )
        return query


@RETRY_POLICY
async def _generate_step_back_query(
    query: str,
    lang_result: LanguageDetectionResult,
) -> str:
    """
    Step-back: generates a slightly broader version of the query.
    Retrieves context chunks that the specific query misses.
    Constrained to increase scope slightly — not become abstract.
    Temperature=0.2 — controlled broadening, not open-ended.
    """
    client = get_sarvam_client()

    prompt = f"""Rewrite the question into a slightly broader contextual question.

Constraints:
- Keep the core topic and technical terms
- Increase scope slightly — not too abstract
- Do NOT remove domain-specific terms
- Language: {lang_result.language_name}

Return ONLY valid JSON — no markdown, no backticks:
{{"step_back_query": "broader question here"}}

Original question: {query}"""

    response = await client.chat.completions.create(
        model=settings.sarvam_model_30b,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
        step_back = data.get("step_back_query", "").strip()
        return step_back if len(step_back) > 5 else query
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(
            "Step-back JSON parse failed — using original query",
            extra={"error": str(e), "raw_output": raw[:200]},
        )
        return query


# ── Master transform function ─────────────────────────────────────────────────

async def transform_query(
    query: str,
    lang_result: LanguageDetectionResult,
    use_hyde: bool = True,
    use_multi_query: bool = True,
    use_step_back: bool = True,
    deterministic: bool = False,
) -> TransformedQueries:
    """
    Master query transformation — runs all three in PARALLEL.
    asyncio.gather reduces latency by 2-3x vs sequential execution.

    deterministic=True disables all transformations — used for:
        - reproducible eval runs
        - debugging RAG behavior
        - A/B testing baseline

    Args:
        query: Original user query
        lang_result: Language detection result
        use_hyde: Enable HyDE transformation
        use_multi_query: Enable multi-query generation
        use_step_back: Enable step-back query
        deterministic: If True — skip all transformations

    Returns:
        TransformedQueries with all_queries populated
    """
    start = time.perf_counter()

    result = TransformedQueries(original=query)

    if deterministic:
        logger.info("Deterministic mode — skipping all transformations")
        result.all_queries = [query]
        result.fallback_used = True
        result.latency_ms = 0.0
        return result

    # Build coroutine list for parallel execution
    coros = []
    labels = []

    if use_multi_query:
        coros.append(_generate_multi_queries(query, lang_result))
        labels.append("multi_query")

    if use_hyde:
        coros.append(_generate_hyde_passage(query, lang_result))
        labels.append("hyde")

    if use_step_back:
        coros.append(_generate_step_back_query(query, lang_result))
        labels.append("step_back")

    fallback_used = False

    if coros:
        # Run ALL transformations in parallel — 2-3x faster than sequential
        raw_results = await asyncio.gather(*coros, return_exceptions=True)

        for label, outcome in zip(labels, raw_results):
            if isinstance(outcome, Exception):
                logger.error(
                    "Query transformation failed — using original",
                    extra={"transform": label, "error": str(outcome)},
                )
                fallback_used = True
                continue

            if label == "multi_query":
                result.multi_queries = outcome
            elif label == "hyde":
                result.hyde_passage = outcome
            elif label == "step_back":
                result.step_back_query = outcome

    result.build_all_queries()
    result.fallback_used = fallback_used
    result.latency_ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        "Query transformation complete",
        extra={
            "original": query,
            "total_queries": len(result.all_queries),
            "latency_ms": result.latency_ms,
            "fallback_used": fallback_used,
            "transforms": labels,
        },
    )

    return result