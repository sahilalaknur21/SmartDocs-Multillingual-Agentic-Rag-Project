# reranking/reranker.py
# WHY THIS EXISTS: FlashRank reranks top-20 hybrid candidates → top-5.
# Singleton ranker — loads once on CPU, zero GPU memory.
# Score thresholds: >0.7 proceed, 0.3-0.7 CRAG, <0.3 insufficient.
# LAW 5 + LAW 13.

import logging
import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Optional

from flashrank import Ranker, RerankRequest

from retrieval.hybrid_retriever import RetrievedChunk
from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class RerankerDecision(str, Enum):
    PROCEED = "proceed"           # score > 0.7 → Sarvam-30B
    CRAG_FALLBACK = "crag"        # score 0.3-0.7 → Tavily search
    INSUFFICIENT = "insufficient" # score < 0.3 → return no-info message


@dataclass
class RerankedResult:
    chunks: list[RetrievedChunk]
    top_score: float
    decision: RerankerDecision
    proceed_threshold: float
    crag_threshold: float
    latency_ms: float = 0.0


@lru_cache(maxsize=1)
def get_ranker() -> Ranker:
    """
    Singleton FlashRank ranker.
    Loads ms-marco-MiniLM-L-12-v2 once — ~80MB download on first run.
    Runs on CPU — zero GPU memory. Zero API cost.
    """
    logger.info("Loading FlashRank ranker...")
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    logger.info("FlashRank ranker loaded")
    return ranker


def _get_decision(
    score: float,
    proceed_threshold: float,
    crag_threshold: float,
) -> RerankerDecision:
    if score >= proceed_threshold:
        return RerankerDecision.PROCEED
    elif score >= crag_threshold:
        return RerankerDecision.CRAG_FALLBACK
    return RerankerDecision.INSUFFICIENT


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
) -> RerankedResult:
    """
    Reranks top-20 hybrid candidates using FlashRank.
    Returns top-k with decision.

    Fallback: if FlashRank fails → use RRF score ordering.
    Never raises — always returns a valid RerankedResult.

    Args:
        query: Original user query
        chunks: Top-20 candidates from hybrid_retriever
        top_k: Final number to return (default: 5)

    Returns:
        RerankedResult with decision and top-5 chunks
    """
    start = time.perf_counter()
    proceed_threshold = settings.reranker_proceed_threshold  # 0.7
    crag_threshold = settings.reranker_crag_threshold        # 0.3

    if not chunks:
        return RerankedResult(
            chunks=[],
            top_score=0.0,
            decision=RerankerDecision.INSUFFICIENT,
            proceed_threshold=proceed_threshold,
            crag_threshold=crag_threshold,
        )

    ranker = get_ranker()

    passages = [
        {"id": i, "text": chunk.chunk_text}
        for i, chunk in enumerate(chunks)
    ]

    try:
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked = ranker.rerank(rerank_request)

        chunk_map = {i: chunk for i, chunk in enumerate(chunks)}
        reranked_chunks: list[RetrievedChunk] = []

        for item in reranked[:top_k]:
            idx = item.get("id", 0)
            score = float(item.get("score", 0.0))
            chunk = chunk_map.get(idx)
            if chunk:
                chunk.reranker_score = score
                reranked_chunks.append(chunk)

        if not reranked_chunks:
            raise ValueError("Reranker returned empty results")

        top_score = reranked_chunks[0].reranker_score
        decision = _get_decision(top_score, proceed_threshold, crag_threshold)

        latency = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "Reranking complete",
            extra={
                "top_score": top_score,
                "decision": decision.value,
                "chunks_in": len(chunks),
                "chunks_out": len(reranked_chunks),
                "latency_ms": latency,
            },
        )

        return RerankedResult(
            chunks=reranked_chunks,
            top_score=top_score,
            decision=decision,
            proceed_threshold=proceed_threshold,
            crag_threshold=crag_threshold,
            latency_ms=latency,
        )

    except Exception as e:
        # Fallback to RRF ordering — log clearly, never silently fail
        logger.error(
            "FlashRank failed — falling back to RRF ordering",
            extra={"error": str(e), "query": query[:80]},
        )
        fallback_chunks = sorted(
            chunks, key=lambda c: c.rrf_score, reverse=True
        )[:top_k]
        for c in fallback_chunks:
            c.reranker_score = c.rrf_score

        top_score = fallback_chunks[0].reranker_score if fallback_chunks else 0.0

        return RerankedResult(
            chunks=fallback_chunks,
            top_score=top_score,
            decision=_get_decision(top_score, proceed_threshold, crag_threshold),
            proceed_threshold=proceed_threshold,
            crag_threshold=crag_threshold,
        )
