# generation/context_assembler.py
# WHY THIS EXISTS: Assembles retrieved chunks into LLM-ready context. LAW 14.
# Rules:
#   1. Sort by reranker score descending
#   2. Source diversity — max 2 chunks per doc_id
#   3. Citation injection: [Source: page X, title]
#   4. Delimiter wrap: <retrieved_document>...</retrieved_document>
#   5. Context budget: 70% chunks / 30% system prompt + query + output

import logging
from dataclasses import dataclass, field

from retrieval.hybrid_retriever import RetrievedChunk
from retrieval.crag_fallback import CRAGResult

logger = logging.getLogger(__name__)

# 70% of context window reserved for retrieved chunks
# Remaining 30% for system prompt + query + output generation
CONTEXT_BUDGET_CHARS = 12_000      # ~3000 tokens at 4 chars/token
MAX_CHUNKS_PER_SOURCE = 2          # Source diversity — LAW 14
DELIMITER_OPEN = "<retrieved_document>"
DELIMITER_CLOSE = "</retrieved_document>"


@dataclass
class AssembledContext:
    """Complete assembled context ready for system prompt injection."""
    context_text: str                    # Final string for {context_with_citations}
    cited_sources: list[dict]            # Source metadata for UI panel
    chunk_count: int                     # Total chunks in context
    crag_chunks_included: int            # Web search chunks included
    budget_used_chars: int               # Characters used of budget
    budget_total_chars: int = CONTEXT_BUDGET_CHARS


@dataclass
class SourceCitation:
    """Citation metadata for a single source chunk."""
    chunk_id: str
    title: str
    page_number: int
    doc_id: str
    reranker_score: float
    is_crag: bool = False
    url: str = ""


def _build_citation_tag(title: str, page_number: int, is_crag: bool = False) -> str:
    """
    Builds citation string injected into every chunk.
    Format: [Source: page X, Title] or [Web Source: Title]
    """
    if is_crag:
        return f"[Web Source: {title}]"
    return f"[Source: page {page_number}, {title}]"


def _wrap_in_delimiter(chunk_text: str, citation: str) -> str:
    """
    Wraps chunk in XML delimiters with citation.
    Mandatory — prevents indirect prompt injection. LAW 10 + LAW 14.
    """
    return (
        f"{DELIMITER_OPEN}\n"
        f"{citation}\n"
        f"{chunk_text}\n"
        f"{DELIMITER_CLOSE}"
    )


def assemble_context(
    chunks: list[RetrievedChunk],
    crag_result: CRAGResult | None = None,
    budget_chars: int = CONTEXT_BUDGET_CHARS,
) -> AssembledContext:
    """
    Assembles retrieved chunks into context string for Sarvam-30B.

    Pipeline:
        1. Sort all chunks by reranker_score descending
        2. Apply source diversity filter (max 2 per doc_id)
        3. Add CRAG web chunks if present (appended after document chunks)
        4. Inject citation tag into each chunk
        5. Wrap each chunk in <retrieved_document> delimiters
        6. Enforce budget: stop adding chunks when chars exceed budget
        7. Return assembled context + source metadata

    Args:
        chunks: Reranked chunks from FlashRank (top-5)
        crag_result: Optional web search results from CRAG fallback
        budget_chars: Character budget for context (default: 12,000)

    Returns:
        AssembledContext with context_text ready for system prompt
    """
    # Step 1: Sort by reranker score descending
    sorted_chunks = sorted(chunks, key=lambda c: c.reranker_score, reverse=True)

    # Step 2: Source diversity filter — max 2 chunks per doc_id
    source_counts: dict[str, int] = {}
    filtered_chunks: list[RetrievedChunk] = []

    for chunk in sorted_chunks:
        count = source_counts.get(chunk.doc_id, 0)
        if count < MAX_CHUNKS_PER_SOURCE:
            filtered_chunks.append(chunk)
            source_counts[chunk.doc_id] = count + 1

    assembled_parts: list[str] = []
    cited_sources: list[dict] = []
    chars_used = 0
    crag_chunks_included = 0

    # Step 3 + 4 + 5 + 6: Process document chunks
    for chunk in filtered_chunks:
        citation = _build_citation_tag(chunk.title, chunk.page_number, is_crag=False)
        wrapped = _wrap_in_delimiter(chunk.chunk_text, citation)

        if chars_used + len(wrapped) > budget_chars:
            logger.debug(
                "Context budget reached — stopping chunk addition",
                extra={"chars_used": chars_used, "budget": budget_chars},
            )
            break

        assembled_parts.append(wrapped)
        chars_used += len(wrapped)

        cited_sources.append({
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "page_number": chunk.page_number,
            "doc_id": chunk.doc_id,
            "reranker_score": round(chunk.reranker_score, 4),
            "is_crag": False,
            "url": "",
        })

    # Step 3 cont: Add CRAG web chunks after document chunks
    if crag_result and crag_result.triggered and crag_result.web_chunks:
        for web_chunk in crag_result.web_chunks:
            citation = _build_citation_tag(
                web_chunk.get("title", "Web Source"),
                page_number=0,
                is_crag=True,
            )
            wrapped = _wrap_in_delimiter(
                web_chunk.get("chunk_text", ""),
                citation,
            )

            if chars_used + len(wrapped) > budget_chars:
                break

            assembled_parts.append(wrapped)
            chars_used += len(wrapped)
            crag_chunks_included += 1

            cited_sources.append({
                "chunk_id": f"crag_{crag_chunks_included}",
                "title": web_chunk.get("title", "Web Source"),
                "page_number": 0,
                "doc_id": "web",
                "reranker_score": web_chunk.get("reranker_score", 0.5),
                "is_crag": True,
                "url": web_chunk.get("url", ""),
            })

    context_text = "\n\n".join(assembled_parts)

    logger.info(
        "Context assembled",
        extra={
            "chunks_in": len(chunks),
            "chunks_out": len(filtered_chunks),
            "crag_chunks": crag_chunks_included,
            "chars_used": chars_used,
            "budget": budget_chars,
            "sources": len(cited_sources),
        },
    )

    return AssembledContext(
        context_text=context_text,
        cited_sources=cited_sources,
        chunk_count=len(assembled_parts) - crag_chunks_included,
        crag_chunks_included=crag_chunks_included,
        budget_used_chars=chars_used,
    )