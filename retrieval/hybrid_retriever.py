# retrieval/hybrid_retriever.py
# WHY THIS EXISTS: Dense + BM25 + RRF fusion retrieval. LAW 5.
# Dense-only loses 15-25% of retrievable Hindi answers.
# BM25 catches exact keyword matches dense misses.
# top-20 candidates → reranker → top-5.

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from embeddings.dense_embedder import get_embedder
from embeddings.sparse_embedder import SparseEmbedder
from vectorstore.pgvector_client import get_pgvector_client
from retrieval.language_detector import LanguageDetectionResult
from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    chunk_text: str
    page_number: int
    title: str
    doc_id: str
    doc_primary_language: str
    chunk_language: str
    script_type: str
    doc_type: str
    section_heading: str
    parent_chunk_id: str
    pii_detected: bool
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    reranker_score: float = 0.0


def _reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[tuple],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    RRF score = dense_weight * 1/(k + dense_rank)
              + sparse_weight * 1/(k + sparse_rank)

    k=60 is standard RRF constant.
    Reduces impact of top-ranked outliers.

    Args:
        dense_results: pgvector similarity search results
        sparse_results: BM25 (chunk_index, score, text) tuples
        dense_weight: 0.7 per spec
        sparse_weight: 0.3 per spec
        k: RRF constant

    Returns:
        Sorted list of (chunk_id, rrf_score) descending
    """
    rrf_scores: dict[str, float] = {}

    # Dense ranking
    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0)
        rrf_scores[chunk_id] += dense_weight * (1.0 / (k + rank))

    # Sparse ranking — match text prefix to dense result chunk_ids
    text_to_chunk_id = {
        r.get("chunk_text", "")[:100]: r["chunk_id"]
        for r in dense_results
    }
    for rank, (_, _, chunk_text) in enumerate(sparse_results, start=1):
        prefix = chunk_text[:100]
        chunk_id = text_to_chunk_id.get(prefix)
        if chunk_id:
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0)
            rrf_scores[chunk_id] += sparse_weight * (1.0 / (k + rank))

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


async def retrieve(
    query: str,
    user_id: str,
    lang_result: LanguageDetectionResult,
    doc_id: Optional[str] = None,
    top_k_candidates: int = 20,
) -> list[RetrievedChunk]:
    """
    Single-query hybrid retrieval: dense + BM25 + RRF.
    Mandatory "query: " prefix on embedding. LAW 4.
    Per-user isolation enforced via pgvector_client. LAW 6.

    Args:
        query: User query text
        user_id: Current user for RLS
        lang_result: Language detection result
        doc_id: Optional — restrict to specific document
        top_k_candidates: Candidates before reranking (default: 20)

    Returns:
        List of RetrievedChunk sorted by RRF score
    """
    start = time.perf_counter()

    embedder = get_embedder()
    client = await get_pgvector_client()

    # Step 1: Embed query — MANDATORY "query: " prefix (LAW 4)
    query_embedding = embedder.embed_query(query)

    # Step 2: Dense retrieval from pgvector
    dense_raw = await client.similarity_search(
        query_embedding=query_embedding,
        user_id=user_id,
        doc_id=doc_id,
        top_k=top_k_candidates,
        exclude_injection_risk=True,
    )

    if not dense_raw:
        logger.warning(
            "Dense retrieval returned zero results",
            extra={"query": query, "user_id": user_id, "doc_id": doc_id},
        )
        return []

    # Step 3: BM25 sparse retrieval on same corpus
    bm25_corpus = [r.get("chunk_text", "") for r in dense_raw]
    sparse_embedder = SparseEmbedder(lang_code=lang_result.language_code)
    sparse_embedder.build_index(bm25_corpus)
    sparse_raw = sparse_embedder.get_top_n(query, n=top_k_candidates)

    # Step 4: RRF fusion
    rrf_ranked = _reciprocal_rank_fusion(
        dense_results=dense_raw,
        sparse_results=sparse_raw,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
    )

    # Step 5: Assemble RetrievedChunk objects
    chunk_map = {r["chunk_id"]: r for r in dense_raw}
    sparse_text_score_map = {t[:100]: s for _, s, t in sparse_raw}

    results: list[RetrievedChunk] = []
    for chunk_id, rrf_score in rrf_ranked[:top_k_candidates]:
        raw = chunk_map.get(chunk_id)
        if not raw:
            continue
        sparse_score = sparse_text_score_map.get(
            raw.get("chunk_text", "")[:100], 0.0
        )
        results.append(RetrievedChunk(
            chunk_id=chunk_id,
            chunk_text=raw.get("chunk_text", ""),
            page_number=raw.get("page_number", 0),
            title=raw.get("title", ""),
            doc_id=str(raw.get("doc_id", "")),
            doc_primary_language=raw.get("doc_primary_language", "en"),
            chunk_language=raw.get("chunk_language", "en"),
            script_type=raw.get("script_type", "latin"),
            doc_type=raw.get("doc_type", "other"),
            section_heading=raw.get("section_heading", ""),
            parent_chunk_id=raw.get("parent_chunk_id", ""),
            pii_detected=raw.get("pii_detected", False),
            dense_score=float(raw.get("similarity", 0.0)),
            sparse_score=float(sparse_score),
            rrf_score=rrf_score,
        ))

    latency = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "Hybrid retrieval complete",
        extra={
            "query": query[:80],
            "dense_results": len(dense_raw),
            "rrf_results": len(results),
            "latency_ms": latency,
        },
    )

    return results


async def retrieve_multi_query(
    queries: list[str],
    user_id: str,
    lang_result: LanguageDetectionResult,
    doc_id: Optional[str] = None,
    top_k_candidates: int = 20,
) -> list[RetrievedChunk]:
    """
    Runs hybrid retrieval for ALL query variants in PARALLEL.
    Deduplicates by chunk_id — keeps highest RRF score.
    Called with TransformedQueries.all_queries from query_transformer.

    Args:
        queries: All variants from transform_query().all_queries
        user_id: Current user
        lang_result: Language detection result
        doc_id: Optional document filter
        top_k_candidates: Candidates per query

    Returns:
        Deduplicated top-20 by best RRF score
    """
    if not queries:
        return []

    # Run all queries in parallel
    all_results = await asyncio.gather(
        *[
            retrieve(
                query=q,
                user_id=user_id,
                lang_result=lang_result,
                doc_id=doc_id,
                top_k_candidates=top_k_candidates,
            )
            for q in queries
        ],
        return_exceptions=True,
    )

    # Deduplicate — keep highest RRF score per chunk_id
    best: dict[str, RetrievedChunk] = {}
    for outcome in all_results:
        if isinstance(outcome, Exception):
            logger.error("Multi-query retrieval error", extra={"error": str(outcome)})
            continue
        for chunk in outcome:
            if chunk.chunk_id not in best or chunk.rrf_score > best[chunk.chunk_id].rrf_score:
                best[chunk.chunk_id] = chunk

    return sorted(best.values(), key=lambda c: c.rrf_score, reverse=True)[:top_k_candidates]


