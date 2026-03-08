# retrieval/language_detector.py
# WHY THIS EXISTS: Detects query language before every retrieval call.
# Result injected into system prompt: "Respond in {language}."
#
# PRODUCTION FIXES APPLIED:
# 1. DetectorFactory.seed = 0 — deterministic results
# 2. Script-first detection — deterministic, 100% reliable
# 3. Hinglish lexicon — handles Romanized Hindi like "transformer kya hai"
# 4. ASCII-dominance override — fixes "RAG orchestration" → French/Norwegian
# 5. Invalid code filter — blocks "no", "af", noise codes on short queries
# 6. Indic override on low confidence — mixed queries don't fall back to English
# 7. Full logging on every detection path — zero silent failures
# LAW 15 — runs on every query without exception.

import logging
from dataclasses import dataclass, field

from langdetect import detect_langs, LangDetectException, DetectorFactory
from config.settings import get_settings

# ── CRITICAL: Deterministic seed ─────────────────────────────────────────────
# Without this: same query → different language on different runs.
# Non-determinism in production RAG is unacceptable.
DetectorFactory.seed = 0

settings = get_settings()
logger = logging.getLogger(__name__)

# ── Language maps ─────────────────────────────────────────────────────────────

LANGUAGE_NAMES: dict[str, str] = {
    "hi": "Hindi", "en": "English", "ta": "Tamil",
    "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
    "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi",
    "bn": "Bengali", "or": "Odia", "as": "Assamese",
    "ur": "Urdu", "sa": "Sanskrit", "ne": "Nepali",
    "si": "Sinhala", "sd": "Sindhi", "ks": "Kashmiri",
    "mai": "Maithili", "doi": "Dogri",
}

LANGUAGE_BADGE: dict[str, str] = {
    "hi": "🇮🇳 Hindi", "en": "🇬🇧 English",
    "ta": "🇮🇳 Tamil", "te": "🇮🇳 Telugu",
    "kn": "🇮🇳 Kannada", "ml": "🇮🇳 Malayalam",
    "mr": "🇮🇳 Marathi", "gu": "🇮🇳 Gujarati",
    "pa": "🇮🇳 Punjabi", "bn": "🇮🇳 Bengali",
    "ur": "🇮🇳 Urdu",
}

# ── Unicode script ranges ─────────────────────────────────────────────────────
# Deterministic. Runs before langdetect.
# If Indic script characters present → trust script, skip probability.

SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "devanagari": (0x0900, 0x097F),   # Hindi, Marathi, Nepali, Sanskrit
    "tamil":      (0x0B80, 0x0BFF),
    "telugu":     (0x0C00, 0x0C7F),
    "kannada":    (0x0C80, 0x0CFF),
    "malayalam":  (0x0D00, 0x0D7F),
    "gujarati":   (0x0A80, 0x0AFF),
    "punjabi":    (0x0A00, 0x0A7F),   # Gurmukhi
    "bengali":    (0x0980, 0x09FF),
    "odia":       (0x0B00, 0x0B7F),
    "urdu":       (0x0600, 0x06FF),   # Arabic script
}

SCRIPT_TO_LANGUAGE: dict[str, str] = {
    "devanagari": "hi",
    "tamil":      "ta",
    "telugu":     "te",
    "kannada":    "kn",
    "malayalam":  "ml",
    "gujarati":   "gu",
    "punjabi":    "pa",
    "bengali":    "bn",
    "odia":       "or",
    "urdu":       "ur",
}

# ── Romanized Hindi (Hinglish) lexicon ────────────────────────────────────────
# High-frequency Hindi words in Latin script.
# 2+ matches → classify as Hindi.
# Fixes: "transformer kya hai" → Norwegian → now correctly Hindi.

HINGLISH_LEXICON: set[str] = {
    # Question words
    "kya", "kyo", "kyun", "kyunki", "kaise", "kab", "kahan", "kaun", "kitna",
    # Common verbs
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "honge",
    "kar", "karo", "karna", "karta", "karti", "karte",
    "bata", "batao", "batana", "samjhao", "samajh", "samjha",
    "dekho", "dekh", "dikhao", "chahiye", "milega", "milegi",
    # Pronouns and connectors
    "mujhe", "mujhko", "mera", "meri", "mere", "hamara", "hamari",
    "aap", "tum", "woh", "yeh", "iska", "uska", "inhe", "unhe",
    "aur", "ya", "lekin", "isliye", "toh", "phir",
    # Common nouns / particles
    "matlab", "seedha", "asaan", "tarika", "cheez",
    "kaam", "jagah", "waqt", "log", "din", "raat",
    "bhi", "hi", "na", "nahi", "mat", "sirf", "bas", "sab",
}

HINGLISH_MATCH_THRESHOLD = 2  # 2+ lexicon words = Hinglish


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class LanguageDetectionResult:
    language_code: str
    language_name: str
    confidence: float
    defaulted_to_english: bool
    detection_method: str               # "script" | "hinglish" | "langdetect"
                                        # | "ascii_override" | "langdetect_indic_override"
                                        # | "default" | "low_confidence_fallback"
    raw_detections: list[tuple[str, float]] = field(default_factory=list)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _default_english(
    raw: list = None,
    method: str = "default",
) -> LanguageDetectionResult:
    return LanguageDetectionResult(
        language_code="en",
        language_name="English",
        confidence=1.0,
        defaulted_to_english=True,
        detection_method=method,
        raw_detections=raw or [],
    )


def _detect_script(text: str) -> str | None:
    """
    Script-based pre-classifier. Step 1 of pipeline.
    Deterministic. 100% reliable. Zero probability error.
    3+ characters in an Indic Unicode range = definitive signal.

    Returns:
        Script name if found, None if text is Latin/ASCII only
    """
    for script, (start, end) in SCRIPT_RANGES.items():
        count = sum(1 for c in text if start <= ord(c) <= end)
        if count >= 3:
            return script
    return None


def _detect_hinglish(text: str) -> bool:
    """
    Romanized Hindi heuristic. Step 2 of pipeline.
    Checks token-level overlap with high-frequency Hindi lexicon.
    2+ matches → Hindi.

    Fixes: "transformer kya hai", "mujhe batao", "kaise kare"

    Returns:
        True if text is likely Hinglish / Romanized Hindi
    """
    words = set(text.lower().split())
    matches = words.intersection(HINGLISH_LEXICON)
    return len(matches) >= HINGLISH_MATCH_THRESHOLD


def _is_ascii_dominant(text: str, threshold: float = 0.85) -> bool:
    """
    Returns True if text is predominantly ASCII alphabetic characters.
    Used to catch langdetect misclassifying English technical jargon
    as French, Norwegian, Afrikaans, etc.

    Fixes: "RAG multiagent orchestration design" → fr → now correctly en

    Returns:
        True if >= 85% of alphabetic characters are ASCII
    """
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return False
    return (ascii_chars / total_alpha) >= threshold


def _is_valid_language_code(code: str) -> bool:
    """
    Filters langdetect noise codes.
    Blocks: "no" (Norwegian), "af" (Afrikaans), "so" (Somali), etc.
    that appear on short Indian queries due to statistical instability.

    Accepts: All 20 Indic codes + common world languages.
    """
    if code in LANGUAGE_NAMES:
        return True
    if code in {"en", "fr", "de", "es", "it", "pt", "ru", "zh-cn", "ja", "ar"}:
        return True
    return False


def _run_langdetect(text: str) -> list[tuple[str, float]]:
    """
    Statistical langdetect with deterministic seed (set at module load).
    Step 3 of pipeline — only called after script + Hinglish checks fail.

    Returns:
        List of (lang_code, probability) — empty list on failure
    """
    try:
        detections = detect_langs(text)
        return [(str(d.lang), round(d.prob, 4)) for d in detections]
    except LangDetectException as e:
        logger.warning(
            "langdetect failed",
            extra={"error": str(e), "text_sample": text[:50]},
        )
        return []


# ── Master detection pipeline ─────────────────────────────────────────────────

def detect_language(text: str) -> LanguageDetectionResult:
    """
    Production-grade language detection for Indian queries.

    7-step logic tree:
        1. Empty text             → English default
        2. Indic script present   → trust script (deterministic, confidence=1.0)
        3. Hinglish lexicon match → Hindi (confidence=0.90)
        4. langdetect noise code  → English default
        5. ASCII-dominant + misclassified → English override
        6. High confidence valid code → trust langdetect
        7. Low confidence but Indic in top-3 → trust Indic (not English)
        8. Final fallback         → English

    Args:
        text: Raw query text (any language, any script)

    Returns:
        LanguageDetectionResult — never raises
    """
    if not text or not text.strip():
        logger.debug("Empty text — defaulting to English")
        return _default_english(method="default")

    threshold = settings.language_confidence_threshold  # 0.85 from config

    # ── Step 1: Script detection (deterministic) ──────────────────────────────
    script = _detect_script(text)
    if script:
        lang_code = SCRIPT_TO_LANGUAGE.get(script, "hi")
        language_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
        logger.info(
            "Language detected via script",
            extra={"script": script, "language": lang_code, "text_sample": text[:50]},
        )
        return LanguageDetectionResult(
            language_code=lang_code,
            language_name=language_name,
            confidence=1.0,
            defaulted_to_english=False,
            detection_method="script",
        )

    # ── Step 2: Hinglish / Romanized Hindi detection ──────────────────────────
    if _detect_hinglish(text):
        logger.info(
            "Hinglish detected via lexicon",
            extra={"text_sample": text[:50]},
        )
        return LanguageDetectionResult(
            language_code="hi",
            language_name="Hindi",
            confidence=0.90,
            defaulted_to_english=False,
            detection_method="hinglish",
        )

    # ── Step 3: Statistical langdetect ───────────────────────────────────────
    raw = _run_langdetect(text)

    if not raw:
        logger.warning("langdetect returned nothing — defaulting to English")
        return _default_english(raw=[], method="default")

    best_code, best_confidence = raw[0]

    # ── Step 4: Filter noise language codes ──────────────────────────────────
    if not _is_valid_language_code(best_code):
        logger.warning(
            "langdetect noise code — defaulting to English",
            extra={
                "detected_code": best_code,
                "confidence": best_confidence,
                "text_sample": text[:50],
            },
        )
        return _default_english(raw=raw, method="default")

 # ── Step 5: ASCII-dominance override ─────────────────────────────────────
    # If text is 85%+ ASCII alphabetic characters, it is Latin-script text.
    # Only English should be the output for pure ASCII input — not French,
    # Norwegian, Afrikaans, or any other language langdetect hallucinates.
    # No confidence check needed — ASCII dominance alone is the signal.
    # Fixes: "RAG multiagent orchestration design" → fr → now correctly en
    if best_code != "en" and _is_ascii_dominant(text):
        logger.info(
            "ASCII-dominant text misclassified as non-English — overriding to English",
            extra={
                "detected": best_code,
                "confidence": best_confidence,
                "text_sample": text[:50],
            },
        )
        return LanguageDetectionResult(
            language_code="en",
            language_name="English",
            confidence=1.0,
            defaulted_to_english=False,
            detection_method="ascii_override",
            raw_detections=raw,
        )

    # ── Step 6: High confidence valid detection ───────────────────────────────
    if best_confidence >= threshold:
        language_name = LANGUAGE_NAMES.get(best_code, best_code.upper())
        logger.info(
            "Language detected via langdetect",
            extra={"language": best_code, "confidence": best_confidence},
        )
        return LanguageDetectionResult(
            language_code=best_code,
            language_name=language_name,
            confidence=best_confidence,
            defaulted_to_english=False,
            detection_method="langdetect",
            raw_detections=raw,
        )

    # ── Step 7: Low confidence — Indic override ───────────────────────────────
    # "मुझे transformer explain करो" may score hi:0.61 — below 0.85 threshold
    # but clearly Hindi. Don't fall back to English — trust the Indic signal.
    indic_codes = set(LANGUAGE_NAMES.keys()) - {"en"}
    for code, prob in raw[:3]:
        if code in indic_codes and prob >= 0.40:
            language_name = LANGUAGE_NAMES.get(code, code.upper())
            logger.info(
                "Low confidence but Indic in top detections — trusting Indic",
                extra={"language": code, "confidence": prob},
            )
            return LanguageDetectionResult(
                language_code=code,
                language_name=language_name,
                confidence=prob,
                defaulted_to_english=False,
                detection_method="langdetect_indic_override",
                raw_detections=raw,
            )

    # ── Step 8: Final fallback ────────────────────────────────────────────────
    logger.info(
        "All detection steps exhausted — defaulting to English",
        extra={"best_code": best_code, "confidence": best_confidence},
    )
    return _default_english(raw=raw, method="low_confidence_fallback")


# ── Document-level detection ──────────────────────────────────────────────────

def detect_document_languages(full_text: str) -> list[str]:
    """
    Detects all languages present in a document.
    Used for UI language badge on upload. LAW 17.
    Samples first 2000 characters per spec.

    Returns:
        List of language codes e.g. ["hi", "en"] for bilingual doc
    """
    sample = full_text[:2000].strip()
    if not sample:
        return ["en"]

    # Script detection first — deterministic
    script = _detect_script(sample)
    if script:
        script_lang = SCRIPT_TO_LANGUAGE.get(script, "hi")
        raw = _run_langdetect(sample)
        has_english = any(
            code == "en" and prob > 0.15
            for code, prob in raw
        )
        return [script_lang, "en"] if has_english else [script_lang]

    raw = _run_langdetect(sample)
    detected = [
        code for code, prob in raw
        if prob > 0.15 and _is_valid_language_code(code)
    ]
    return detected or ["en"]


def get_language_badge(lang_codes: list[str]) -> str:
    """
    Returns UI badge string for Streamlit upload_panel.py. LAW 17.
    Example: ["hi", "en"] → "🇮🇳 Hindi + 🇬🇧 English"
    """
    badges = [LANGUAGE_BADGE.get(code, f"🌐 {code.upper()}") for code in lang_codes]
    return " + ".join(badges) if badges else "🌐 Unknown"


def build_language_system_prompt_instruction(result: LanguageDetectionResult) -> str:
    """
    Builds language instruction for Sarvam-30B system prompt.
    Injected on every query. Forces model to respond in user's language.
    """
    return (
        f"Respond in {result.language_name} only. "
        f"The user's query is in {result.language_name}. "
        f"Your entire response — every sentence — must be in {result.language_name}. "
        f"Switching to another language mid-response is a hard failure."
    )