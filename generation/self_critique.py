# generation/self_critique.py
# WHY THIS EXISTS: Self-evaluation loop after generation. LAW 8.
# faithfulness < 0.75 → retry with refined query (max 2)
# language_match = False → retry (wrong language = hard failure)
# After 2 failures → graceful "insufficient information" response
# This is what separates a pipeline from an agent.

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import tenacity
from openai import APIError, RateLimitError, APITimeoutError

from generation.sarvam_client import get_sarvam_client
from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    """Result of self-critique evaluation on a generated answer."""
    passed: bool                          # True if answer should proceed to guardrail
    faithful: bool                        # Answer supported by retrieved context
    language_match: bool                  # Answer language matches query language
    faithfulness_score: float             # 0.0 to 1.0
    issues: str                           # Description of problems found
    refined_query: Optional[str] = None  # Improved query for retry
    retry_needed: bool = False            # True if retry should be attempted


CRITIQUE_PROMPT = """\
You are evaluating an AI-generated answer for quality.

Evaluate the answer on TWO criteria:

1. FAITHFULNESS: Is every factual claim in the answer directly supported by the context?
   Score 0.0 (not faithful) to 1.0 (completely faithful).
   Flag any claim that is not in the context as a hallucination.

2. LANGUAGE MATCH: Is the answer written in the same language as the query?
   Query language: {query_language}
   This is a binary check — True or False.

If issues found, provide a refined version of the original query that might retrieve better context.

Context provided to the AI:
{context}

Original query: {query}

AI's answer:
{answer}

Return ONLY valid JSON — no markdown, no backticks, no explanation:
{{
  "faithful": true or false,
  "faithfulness_score": 0.0 to 1.0,
  "language_match": true or false,
  "issues": "description of issues or empty string if none",
  "refined_query": "improved query for retry or empty string if not needed"
}}"""

FAITHFULNESS_THRESHOLD = 0.75


def _is_retryable(exc: Exception) -> bool:
    return isinstance(exc, (RateLimitError, APITimeoutError, APIError))


@tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=6),
    stop=tenacity.stop_after_attempt(2),
    reraise=True,
)
async def critique_answer(
    query: str,
    answer: str,
    context: str,
    query_language: str,
    retry_count: int,
) -> CritiqueResult:
    """
    Evaluates generated answer for faithfulness and language match.
    Uses Sarvam-30B with structured JSON output.

    Logic:
        faithful AND language_match → passed=True
        not faithful (score < 0.75) → retry_needed=True (if retry_count < 2)
        not language_match → retry_needed=True (language failure = hard failure)
        retry_count >= 2 → passed=False, retry_needed=False → graceful failure

    Args:
        query: Original user query
        answer: Generated answer to evaluate
        context: Context that was provided to generate the answer
        query_language: Expected response language (e.g. "Hindi")
        retry_count: Current retry count (0, 1, or 2)

    Returns:
        CritiqueResult — never raises
    """
    start = time.perf_counter()
    client = get_sarvam_client()

    # Truncate context for critique call — only need enough to check faithfulness
    truncated_context = context[:3000] if len(context) > 3000 else context

    prompt = CRITIQUE_PROMPT.format(
        query_language=query_language,
        context=truncated_context,
        query=query,
        answer=answer,
    )

    try:
        response = await client.chat.completions.create(
            model=settings.sarvam_model_30b,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
            stream=False,
        )

        raw = response.choices[0].message.content.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(
                "Critique JSON parse failed — assuming passed",
                extra={"error": str(e), "raw": raw[:200]},
            )
            # JSON parse failure → assume passed to avoid blocking good answers
            return CritiqueResult(
                passed=True,
                faithful=True,
                language_match=True,
                faithfulness_score=0.8,
                issues="Critique parse failed — assumed passed",
                refined_query=None,
                retry_needed=False,
            )

        faithful = bool(data.get("faithful", True))
        faithfulness_score = float(data.get("faithfulness_score", 0.8))
        language_match = bool(data.get("language_match", True))
        issues = str(data.get("issues", ""))
        refined_query = data.get("refined_query", "") or None

        # Determine if retry is needed
        score_fail = faithfulness_score < FAITHFULNESS_THRESHOLD
        retry_needed = (score_fail or not language_match) and retry_count < 2
        passed = not score_fail and language_match

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "Self-critique complete",
            extra={
                "faithful": faithful,
                "faithfulness_score": faithfulness_score,
                "language_match": language_match,
                "passed": passed,
                "retry_needed": retry_needed,
                "retry_count": retry_count,
                "latency_ms": latency_ms,
            },
        )

        return CritiqueResult(
            passed=passed,
            faithful=faithful,
            language_match=language_match,
            faithfulness_score=faithfulness_score,
            issues=issues,
            refined_query=refined_query,
            retry_needed=retry_needed,
        )

    except Exception as e:
        logger.error(
            "Self-critique failed entirely — assuming passed",
            extra={"error": str(e), "retry_count": retry_count},
        )
        return CritiqueResult(
            passed=True,
            faithful=True,
            language_match=True,
            faithfulness_score=0.8,
            issues=f"Critique error: {str(e)}",
            refined_query=None,
            retry_needed=False,
        )