# retrieval/query_classifier.py
# WHY THIS EXISTS: Classifies query type before retrieval.
# Each type has a different suggested_top_k for retrieval.
# Fast keyword matching — no API call, no latency.

import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    COMPARATIVE = "comparative"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class QueryClassificationResult:
    query_type: QueryType
    confidence: float
    reasoning: str
    suggested_top_k: int


# Keyword signals per query type — English + Hindi + Tamil + Telugu
QUERY_SIGNALS: dict[QueryType, list[str]] = {
    QueryType.FACTUAL: [
        "what is", "how much", "when", "who", "which", "where",
        "क्या है", "कितना", "कब", "कौन", "कहाँ",
        "என்ன", "எவ்வளவு", "ఏమిటి", "ఎంత",
    ],
    QueryType.ANALYTICAL: [
        "why", "how does", "explain", "reason", "cause", "impact",
        "क्यों", "कैसे", "समझाइए", "कारण", "प्रभाव",
        "ஏன்", "எப்படி", "ఎందుకు", "ఎలా",
    ],
    QueryType.CONVERSATIONAL: [
        "summarize", "overview", "tell me about", "describe", "brief",
        "सारांश", "बताइए", "संक्षेप", "विवरण",
        "சுருக்கம்", "సారాంశం",
    ],
    QueryType.COMPARATIVE: [
        "difference", "compare", "versus", "vs", "between", "unlike",
        "अंतर", "तुलना", "बनाम", "वित्तीय अंतर",
        "வித்தியாசம்", "తేడా",
    ],
}

QUERY_TYPE_TOP_K: dict[QueryType, int] = {
    QueryType.FACTUAL: 5,
    QueryType.ANALYTICAL: 8,
    QueryType.CONVERSATIONAL: 10,
    QueryType.COMPARATIVE: 8,
    QueryType.OUT_OF_SCOPE: 0,
}


def classify_query(query: str) -> QueryClassificationResult:
    """
    Classifies query into one of 5 types via keyword matching.
    Zero API calls — pure local logic.
    Determines suggested_top_k passed to hybrid_retriever.

    Args:
        query: Raw user query

    Returns:
        QueryClassificationResult — never raises
    """
    if not query or not query.strip():
        return QueryClassificationResult(
            query_type=QueryType.FACTUAL,
            confidence=0.5,
            reasoning="Empty query — defaulting to factual",
            suggested_top_k=5,
        )

    query_lower = query.lower().strip()
    scores: dict[QueryType, int] = {qt: 0 for qt in QueryType}

    for query_type, signals in QUERY_SIGNALS.items():
        for signal in signals:
            if signal in query_lower:
                scores[query_type] += 1

    scoreable = [qt for qt in QueryType if qt != QueryType.OUT_OF_SCOPE]
    best_type = max(scoreable, key=lambda qt: scores[qt])
    best_score = scores[best_type]

    if best_score == 0:
        best_type = QueryType.FACTUAL
        confidence = 0.6
        reasoning = "No signals matched — defaulting to factual"
    else:
        total = max(sum(scores.values()), 1)
        confidence = round(best_score / total, 2)
        reasoning = f"Matched {best_score} {best_type.value} signals"

    logger.debug(
        "Query classified",
        extra={
            "query_type": best_type.value,
            "confidence": confidence,
            "suggested_top_k": QUERY_TYPE_TOP_K[best_type],
        },
    )

    return QueryClassificationResult(
        query_type=best_type,
        confidence=confidence,
        reasoning=reasoning,
        suggested_top_k=QUERY_TYPE_TOP_K[best_type],
    )