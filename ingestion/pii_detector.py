# ingestion/pii_detector.py
# WHY THIS EXISTS: Detects PII (Aadhaar, PAN, phone) in every chunk
# at ingestion time. Tags chunks with pii_detected=True.
# Never returns PAN or Aadhaar in any API response. LAW 11.

import re
from dataclasses import dataclass, field


@dataclass
class PIIDetectionResult:
    """Result of PII detection on a text chunk."""
    text: str
    pii_detected: bool
    pii_types: list[str]
    redacted_text: str


# PII patterns for Indian documents
PII_PATTERNS = {
    "aadhaar": r"\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b",
    "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
    "phone_india": r"\b(?:\+91|91|0)?[6-9][0-9]{9}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "gstin": r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b",
    "passport": r"\b[A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9]\b",
    "voter_id": r"\b[A-Z]{3}[0-9]{7}\b",
    "ifsc": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
}

_COMPILED_PII_PATTERNS = {
    name: re.compile(pattern, re.IGNORECASE | re.UNICODE)
    for name, pattern in PII_PATTERNS.items()
}

# Replacement tokens for redacted text
PII_REPLACEMENTS = {
    "aadhaar": "[AADHAAR REDACTED]",
    "pan": "[PAN REDACTED]",
    "phone_india": "[PHONE REDACTED]",
    "email": "[EMAIL REDACTED]",
    "gstin": "[GSTIN REDACTED]",
    "passport": "[PASSPORT REDACTED]",
    "voter_id": "[VOTER ID REDACTED]",
    "ifsc": "[IFSC REDACTED]",
}


def detect_pii_in_chunk(text: str, redact: bool = False) -> PIIDetectionResult:
    """
    Detects PII in a single text chunk.
    Does NOT block ingestion — only tags chunk metadata.
    Redaction happens at response generation time, not ingestion.

    Args:
        text: Text chunk to scan
        redact: If True, returns redacted version of text

    Returns:
        PIIDetectionResult with pii_detected flag and types found
    """
    if not text:
        return PIIDetectionResult(
            text=text,
            pii_detected=False,
            pii_types=[],
            redacted_text=text,
        )

    pii_types_found = []
    redacted_text = text

    for pii_type, compiled_pattern in _COMPILED_PII_PATTERNS.items():
        if compiled_pattern.search(text):
            pii_types_found.append(pii_type)
            if redact:
                replacement = PII_REPLACEMENTS.get(pii_type, "[REDACTED]")
                redacted_text = compiled_pattern.sub(replacement, redacted_text)

    return PIIDetectionResult(
        text=text,
        pii_detected=len(pii_types_found) > 0,
        pii_types=pii_types_found,
        redacted_text=redacted_text,
    )


def redact_pii_from_response(text: str) -> tuple[str, list[str]]:
    """
    Redacts all PII from a generated response before returning to user.
    Called in output_guardrail.py before every API response.
    Never returns PAN or Aadhaar numbers to user. LAW 11.

    Args:
        text: Generated response text

    Returns:
        Tuple of (redacted_text, list_of_pii_types_found)
    """
    if not text:
        return text, []

    redacted = text
    pii_found: list[str] = []

    for pii_type, compiled_pattern in _COMPILED_PII_PATTERNS.items():
        if compiled_pattern.search(redacted):          # check BEFORE replacing
            pii_found.append(pii_type)
        replacement = PII_REPLACEMENTS.get(pii_type, "[REDACTED]")
        redacted = compiled_pattern.sub(replacement, redacted)

    return redacted, pii_found


def scan_document_for_pii(chunks: list[str]) -> list[PIIDetectionResult]:
    """
    Scans all chunks of a document for PII at ingestion time.
    Sets pii_detected metadata flag per chunk.

    Args:
        chunks: List of text chunks

    Returns:
        List of PIIDetectionResult, one per chunk
    """
    return [detect_pii_in_chunk(chunk, redact=False) for chunk in chunks]