# ingestion/document_classifier.py
"""
WHY THIS EXISTS: Classifies a document into a known Indian professional
document type based on keywords. Used for display and chunking hints.
"""

from __future__ import annotations

import re

_PATTERNS: dict[str, list[str]] = {
    "gst_notice":           ["gstin", "gst", "cgst", "sgst", "igst", "tax invoice", "itc"],
    "legal_agreement":      ["whereas", "hereinafter", "lessor", "lessee", "arbitration", "jurisdiction"],
    "insurance_policy":     ["insured", "premium", "sum assured", "policyholder", "claim", "nominee"],
    "government_circular":  ["circular", "office memorandum", "ministry", "department", "government of india"],
    "tax_notice":           ["income tax", "assessment year", "pan", "section 143", "section 148", "tds"],
}


def classify_document(full_text: str, filename: str = "") -> str:
    """
    Returns a document type string based on keyword matching.
    Falls back to 'other' if no pattern matches.
    """
    text_lower = full_text.lower()
    name_lower = filename.lower()

    for doc_type, keywords in _PATTERNS.items():
        if any(kw in text_lower or kw in name_lower for kw in keywords):
            return doc_type

    return "other"