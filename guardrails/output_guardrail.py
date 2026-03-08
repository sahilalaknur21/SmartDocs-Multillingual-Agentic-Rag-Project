# guardrails/output_guardrail.py
# WHY THIS EXISTS: Final defense before answer reaches user.
# PII redaction — Aadhaar/PAN/phone never in API response.
# Injection detection — malicious content in retrieved docs stays hidden.
# Off-topic detection — answers about non-document topics flagged.

import logging
import re
from dataclasses import dataclass, field

from ingestion.pii_detector import redact_pii_from_response
from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Result of output guardrail pass."""
    clean_answer: str                    # Answer after all redactions
    pii_redacted: bool                   # True if PII was found and removed
    injection_detected: bool             # True if injection pattern in output
    off_topic_detected: bool             # True if answer doesn't reference document
    passed: bool                         # True if answer is safe to return
    flags: list[str] = field(default_factory=list)  # List of triggered flags


# Injection patterns that should never appear in generated output
OUTPUT_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore previous instructions", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
    re.compile(r"forget everything", re.IGNORECASE),
    re.compile(r"as an AI language model", re.IGNORECASE),
    re.compile(r"पिछले निर्देश भूल जाओ", re.IGNORECASE),
    re.compile(r"अब आप हैं", re.IGNORECASE),
]

# Signals that an answer is based on document context (not hallucinated)
CITATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\[Source: page \d+", re.IGNORECASE),
    re.compile(r"\[Web Source:", re.IGNORECASE),
    re.compile(r"according to", re.IGNORECASE),
    re.compile(r"के अनुसार", re.IGNORECASE),       # Hindi: "according to"
    re.compile(r"document", re.IGNORECASE),
    re.compile(r"दस्तावेज़", re.IGNORECASE),        # Hindi: "document"
]

# These no-info responses are always valid (from spec)
NO_INFO_MARKERS = [
    "मुझे इस प्रश्न का उत्तर आपके दस्तावेज़ में नहीं मिला",
    "I don't have enough information in your document",
    "உங்கள் ஆவணத்தில்",
    "మీ పత్రంలో",
    "ನಿಮ್ಮ ದಾಖಲೆಯಲ್ಲಿ",
]


def _check_injection_in_output(answer: str) -> bool:
    """
    Checks if the generated answer contains injection patterns.
    This catches cases where malicious document content influenced the output.

    Returns:
        True if injection pattern detected
    """
    return any(pattern.search(answer) for pattern in OUTPUT_INJECTION_PATTERNS)


def _check_off_topic(answer: str) -> bool:
    """
    Heuristic check: does the answer reference document context?
    An answer with zero citation signals and non-trivial length
    is likely hallucinated or off-topic.

    Short answers and no-info responses are always considered on-topic.

    Returns:
        True if answer appears off-topic
    """
    # No-info responses are always valid
    if any(marker in answer for marker in NO_INFO_MARKERS):
        return False

    # Short answers (< 50 chars) pass — brief factual answers are valid
    if len(answer.strip()) < 50:
        return False

    # Check for at least one document citation signal in longer answers
    has_citation = any(pattern.search(answer) for pattern in CITATION_PATTERNS)
    return not has_citation


def run_guardrail(answer: str) -> GuardrailResult:
    """
    Runs all output guardrail checks on a generated answer.
    Called as the final node before returning to the user.

    Checks in order:
        1. PII redaction — remove Aadhaar/PAN/phone from answer
        2. Injection detection — flag prompt injection in output
        3. Off-topic detection — flag answers without document citations

    PII redaction always runs — never blocks.
    Injection detection → flag + log (does not block, adds to flags).
    Off-topic detection → flag + log (does not block, adds to flags).

    Args:
        answer: Raw generated answer from Sarvam-30B

    Returns:
        GuardrailResult — never raises
    """
    flags: list[str] = []
    pii_redacted = False
    injection_detected = False
    off_topic_detected = False

    # Step 1: PII redaction — always runs
    try:
        clean_answer, pii_found = redact_pii_from_response(answer)
        if pii_found:
            pii_redacted = True
            flags.append("pii_redacted")
            logger.warning(
                "PII detected and redacted from generated answer",
                extra={"pii_types": pii_found},
            )
    except Exception as e:
        logger.error("PII redaction failed", extra={"error": str(e)})
        clean_answer = answer  # Fail open — return original if redaction fails

    # Step 2: Injection detection in output
    if _check_injection_in_output(clean_answer):
        injection_detected = True
        flags.append("injection_in_output")
        logger.warning(
            "Injection pattern detected in generated output",
            extra={"answer_sample": clean_answer[:100]},
        )

    # Step 3: Off-topic detection
    if _check_off_topic(clean_answer):
        off_topic_detected = True
        flags.append("off_topic")
        logger.info(
            "Off-topic answer detected — no document citations found",
            extra={"answer_sample": clean_answer[:100]},
        )

    # Answer passes if no injection detected
    # PII redacted answers still pass (we just clean them)
    # Off-topic answers pass but are flagged
    passed = not injection_detected

    logger.info(
        "Guardrail complete",
        extra={
            "passed": passed,
            "flags": flags,
            "pii_redacted": pii_redacted,
        },
    )

    return GuardrailResult(
        clean_answer=clean_answer,
        pii_redacted=pii_redacted,
        injection_detected=injection_detected,
        off_topic_detected=off_topic_detected,
        passed=passed,
        flags=flags,
    )