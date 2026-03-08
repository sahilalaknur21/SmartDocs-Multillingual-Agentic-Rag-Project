# ingestion/indic_preprocessing.py
# WHY THIS EXISTS: Full 5-step Indic text normalization pipeline.
# Every Hindi/Indic chunk passes through this before any downstream
# processing. Skipping this = silent BM25 failures on Hindi text.
# LAW 1 — non-negotiable.

import re
import os
import unicodedata
from pathlib import Path
from langdetect import detect, LangDetectException

# Set indic_nlp_resources path before importing
INDIC_NLP_RESOURCES = str(
    Path(__file__).parent.parent / "indic_nlp_resources"
)
os.environ["INDIC_RESOURCES_PATH"] = INDIC_NLP_RESOURCES

from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import sentence_tokenize
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator


# ── Constants ────────────────────────────────────────────────────────────────

DEVANAGARI_LANGUAGES = {
    "hi": "hi",   # Hindi
    "mr": "mr",   # Marathi
    "ne": "ne",   # Nepali
    "sa": "sa",   # Sanskrit
    "kok": "kok", # Konkani
}

DRAVIDIAN_LANGUAGES = {
    "ta": "ta",   # Tamil
    "te": "te",   # Telugu
    "kn": "kn",   # Kannada
    "ml": "ml",   # Malayalam
}

ALL_INDIC_LANGUAGES = {**DEVANAGARI_LANGUAGES, **DRAVIDIAN_LANGUAGES}

# Zero-width characters that corrupt BM25 tokenization
ZERO_WIDTH_CHARS = [
    "\u200b",  # Zero Width Space
    "\u200c",  # Zero Width Non-Joiner
    "\u200d",  # Zero Width Joiner
    "\u2060",  # Word Joiner
    "\ufeff",  # Byte Order Mark
    "\u00ad",  # Soft Hyphen
]

# Quality detection thresholds
MIN_MEANINGFUL_CHARS_RATIO = 0.3
MIN_TEXT_LENGTH = 10


# ── Normalizer factory cache ──────────────────────────────────────────────────

_normalizer_cache: dict = {}


def _get_normalizer(lang_code: str):
    """Returns cached normalizer for language code."""
    if lang_code not in _normalizer_cache:
        factory = IndicNormalizerFactory()
        _normalizer_cache[lang_code] = factory.get_normalizer(lang_code)
    return _normalizer_cache[lang_code]


# ── Step 1: Indic Normalize ───────────────────────────────────────────────────

def indic_normalize(text: str, lang_code: str = "hi") -> str:
    """
    Step 1 of Indic preprocessing pipeline.
    Normalizes Unicode representations of Indic scripts.
    Hindi has multiple Unicode representations for the same character.
    Without normalization, BM25 treats them as different tokens.

    Args:
        text: Raw text from PDF
        lang_code: ISO language code (default: "hi" for Hindi)

    Returns:
        Normalized text
    """
    if not text or not text.strip():
        return text

    # Only normalize Indic languages
    if lang_code not in ALL_INDIC_LANGUAGES:
        return unicodedata.normalize("NFC", text)

    try:
        normalizer = _get_normalizer(lang_code)
        normalized = normalizer.normalize(text)
        return normalized
    except Exception:
        # Fallback to Unicode NFC normalization
        return unicodedata.normalize("NFC", text)


# ── Step 2: Remove Zero Width Joiners ────────────────────────────────────────

def remove_zero_width_joiners(text: str) -> str:
    """
    Step 2 of Indic preprocessing pipeline.
    Removes invisible Unicode characters that corrupt BM25 tokenization.
    PDFs generated from Hindi word processors are full of these.
    They are invisible in any text editor but break exact-match retrieval.

    Args:
        text: Text after indic_normalize

    Returns:
        Text with all zero-width characters removed
    """
    if not text:
        return text

    for char in ZERO_WIDTH_CHARS:
        text = text.replace(char, "")

    return text


# ── Step 3: Normalize Devanagari Spaces ──────────────────────────────────────

def normalize_devanagari_spaces(text: str) -> str:
    """
    Step 3 of Indic preprocessing pipeline.
    Normalizes various space characters in Devanagari text.
    Hindi PDFs often contain non-standard space characters that
    prevent word boundary detection in BM25.

    Args:
        text: Text after remove_zero_width_joiners

    Returns:
        Text with normalized spaces
    """
    if not text:
        return text

    # Normalize various Unicode space characters to standard space
    space_variants = [
        "\u00a0",  # Non-breaking space
        "\u1680",  # Ogham space mark
        "\u2000",  # En quad
        "\u2001",  # Em quad
        "\u2002",  # En space
        "\u2003",  # Em space
        "\u2004",  # Three-per-em space
        "\u2005",  # Four-per-em space
        "\u2006",  # Six-per-em space
        "\u2007",  # Figure space
        "\u2008",  # Punctuation space
        "\u2009",  # Thin space
        "\u200a",  # Hair space
        "\u202f",  # Narrow no-break space
        "\u205f",  # Medium mathematical space
        "\u3000",  # Ideographic space
    ]

    for space in space_variants:
        text = text.replace(space, " ")

    # Collapse multiple spaces into single space
    text = re.sub(r" {2,}", " ", text)

    # Remove spaces before Devanagari punctuation
    text = re.sub(r" ([।॥,])", r"\1", text)

    return text.strip()


# ── Step 4: Detect and Flag Low Quality ──────────────────────────────────────

def detect_and_flag_low_quality(text: str) -> tuple[str, bool]:
    """
    Step 4 of Indic preprocessing pipeline.
    Detects low-quality text from PDF extraction — garbled characters,
    encoding errors, or mostly-whitespace chunks.
    Flags them so they can be excluded from embedding but still stored
    for audit purposes.

    Args:
        text: Text after normalize_devanagari_spaces

    Returns:
        tuple: (text, is_low_quality)
        is_low_quality=True means this chunk should not be embedded
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return text, True

    # Check ratio of meaningful characters
    total_chars = len(text)
    meaningful_chars = len(re.sub(r"[\s\n\t\r]", "", text))

    if total_chars > 0:
        ratio = meaningful_chars / total_chars
        if ratio < MIN_MEANINGFUL_CHARS_RATIO:
            return text, True

    # Check for excessive replacement characters (encoding errors)
    replacement_count = text.count("\ufffd")
    if replacement_count > 5:
        return text, True

    # Check for garbled text (too many non-printable characters)
    non_printable = sum(
        1 for c in text
        if unicodedata.category(c) in ("Cc", "Cf") and c not in "\n\t\r"
    )
    if non_printable > len(text) * 0.1:
        return text, True

    return text, False


# ── Step 5: Sentence Tokenize Indic ──────────────────────────────────────────

def sentence_tokenize_indic(text: str, lang_code: str = "hi") -> list[str]:
    """
    Step 5 of Indic preprocessing pipeline.
    Splits Indic text into clean sentences BEFORE chunking.
    Critical for Devanagari — standard sentence splitters miss
    Hindi sentence boundaries (।) and produce chunks that cut
    mid-sentence, destroying context for retrieval.

    Args:
        text: Text after detect_and_flag_low_quality
        lang_code: ISO language code

    Returns:
        List of clean sentences
    """
    if not text or not text.strip():
        return []

    if lang_code not in ALL_INDIC_LANGUAGES:
        # Fallback for non-Indic text
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    try:
        sentences = sentence_tokenize.sentence_split(text, lang=lang_code)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        # Fallback: split on Devanagari danda and standard punctuation
        sentences = re.split(r"[।॥.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]


# ── Master Pipeline ───────────────────────────────────────────────────────────

def indic_preprocessing_pipeline(
    text: str,
    lang_code: str = "hi",
    return_sentences: bool = False
) -> dict:
    """
    Master pipeline — runs all 5 steps in order.
    Import this function into every ingestion function.
    Never call individual steps directly in production code.

    Pipeline:
        raw_text
        → indic_normalize(text)
        → remove_zero_width_joiners(text)
        → normalize_devanagari_spaces(text)
        → detect_and_flag_low_quality(text)
        → sentence_tokenize_indic(text)
        → cleaned_text

    Args:
        text: Raw text from PDF extraction
        lang_code: ISO language code (auto-detected if not provided)
        return_sentences: If True, returns list of sentences

    Returns:
        dict with keys:
            - cleaned_text: fully preprocessed text
            - sentences: list of sentences (if return_sentences=True)
            - lang_code: language code used
            - is_low_quality: True if chunk should not be embedded
            - steps_applied: list of steps that ran
    """
    if not text:
        return {
            "cleaned_text": "",
            "sentences": [],
            "lang_code": lang_code,
            "is_low_quality": True,
            "steps_applied": [],
        }

    steps_applied = []

    # Auto-detect language if not Indic — still run pipeline for safety
    detected_lang = lang_code
    try:
        detected_lang = detect(text[:500])
    except LangDetectException:
        detected_lang = "en"

    # Use provided lang_code if it's Indic, otherwise use detected
    if lang_code not in ALL_INDIC_LANGUAGES:
        lang_code = detected_lang

    # Step 1
    text = indic_normalize(text, lang_code)
    steps_applied.append("indic_normalize")

    # Step 2
    text = remove_zero_width_joiners(text)
    steps_applied.append("remove_zero_width_joiners")

    # Step 3
    text = normalize_devanagari_spaces(text)
    steps_applied.append("normalize_devanagari_spaces")

    # Step 4
    text, is_low_quality = detect_and_flag_low_quality(text)
    steps_applied.append("detect_and_flag_low_quality")

    # Step 5
    sentences = []
    if return_sentences or lang_code in ALL_INDIC_LANGUAGES:
        sentences = sentence_tokenize_indic(text, lang_code)
        steps_applied.append("sentence_tokenize_indic")

    return {
        "cleaned_text": text,
        "sentences": sentences,
        "lang_code": lang_code,
        "is_low_quality": is_low_quality,
        "steps_applied": steps_applied,
    }


def detect_script_type(text: str) -> str:
    """
    Detects the script type of a text chunk.
    Used to set chunk metadata.script_type field.

    Returns:
        "devanagari" | "latin" | "dravidian" | "mixed"
    """
    if not text:
        return "latin"

    devanagari_count = sum(
        1 for c in text if "\u0900" <= c <= "\u097f"
    )
    latin_count = sum(
        1 for c in text if "\u0041" <= c <= "\u007a"
    )
    dravidian_count = sum(
        1 for c in text
        if ("\u0b80" <= c <= "\u0bff")  # Tamil
        or ("\u0c00" <= c <= "\u0c7f")  # Telugu
        or ("\u0c80" <= c <= "\u0cff")  # Kannada
        or ("\u0d00" <= c <= "\u0d7f")  # Malayalam
    )

    total = max(devanagari_count + latin_count + dravidian_count, 1)

    if devanagari_count / total > 0.6:
        return "devanagari"
    elif dravidian_count / total > 0.6:
        return "dravidian"
    elif latin_count / total > 0.6:
        return "latin"
    else:
        return "mixed"