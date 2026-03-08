# ingestion/injection_scanner.py
# WHY THIS EXISTS: Scans every uploaded PDF chunk for prompt injection
# attempts before ingestion. Malicious PDFs can contain text that
# hijacks LLM behavior through retrieved context. LAW 10.

import re
from dataclasses import dataclass


@dataclass
class InjectionScanResult:
    """Result of scanning a text chunk for injection attempts."""
    text: str
    injection_risk: bool
    matched_patterns: list[str]
    risk_score: float  # 0.0 to 1.0


# Injection patterns — English + Hindi equivalents
# Add new patterns here as new attack vectors are discovered
INJECTION_PATTERNS = [
    # English patterns
    r"ignore\s+(?:all\s+)?previous\s+instructions",
    r"disregard\s+(?:all\s+)?previous",
    r"you\s+are\s+now\s+(?:a|an)",
    r"forget\s+everything",
    r"system\s+prompt",
    r"new\s+instructions\s*:",
    r"act\s+as\s+(?:a|an|if)",
    r"pretend\s+(?:you\s+are|to\s+be)",
    r"override\s+(?:your\s+)?instructions",
    r"jailbreak",
    r"prompt\s+injection",
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"developer\s+mode",
    r"</?(system|instruction|prompt)>",

    # Hindi equivalents
    r"पिछले\s+निर्देश\s+भूल\s+जाओ",       # forget previous instructions
    r"अब\s+आप\s+हैं",                        # you are now
    r"सभी\s+निर्देश\s+अनदेखा\s+करें",       # ignore all instructions
    r"नए\s+निर्देश",                          # new instructions
    r"सिस्टम\s+प्रॉम्प्ट",                   # system prompt
    r"पिछला\s+सब\s+भूल\s+जाओ",              # forget everything previous
]

# Compile patterns once at module load for performance
_COMPILED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.UNICODE)
    for pattern in INJECTION_PATTERNS
]


def scan_chunk_for_injection(text: str) -> InjectionScanResult:
    """
    Scans a single text chunk for prompt injection patterns.
    Called on every chunk during ingestion — before embedding.

    Args:
        text: Text chunk to scan

    Returns:
        InjectionScanResult with injection_risk flag
    """
    if not text:
        return InjectionScanResult(
            text=text,
            injection_risk=False,
            matched_patterns=[],
            risk_score=0.0,
        )

    matched_patterns = []

    for pattern, compiled in zip(INJECTION_PATTERNS, _COMPILED_PATTERNS):
        if compiled.search(text):
            matched_patterns.append(pattern)

    injection_risk = len(matched_patterns) > 0
    risk_score = min(len(matched_patterns) / 3.0, 1.0)

    return InjectionScanResult(
        text=text,
        injection_risk=injection_risk,
        matched_patterns=matched_patterns,
        risk_score=risk_score,
    )


def scan_document_for_injection(chunks: list[str]) -> list[InjectionScanResult]:
    """
    Scans all chunks of a document for injection attempts.
    Called during ingestion after chunking, before embedding.

    Args:
        chunks: List of text chunks from PDF

    Returns:
        List of InjectionScanResult, one per chunk
    """
    return [scan_chunk_for_injection(chunk) for chunk in chunks]


def wrap_chunk_in_delimiters(chunk_text: str) -> str:
    """
    Wraps retrieved chunk in XML delimiters before passing to Sarvam-30B.
    This is the second layer of injection defense — even if a malicious
    chunk passes the scanner, the LLM sees it as document content,
    not as instructions. LAW 10.

    Args:
        chunk_text: Retrieved chunk text

    Returns:
        Delimited chunk safe for LLM context
    """
    return f"<retrieved_document>{chunk_text}</retrieved_document>"