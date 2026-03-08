# retrieval/crag_fallback.py
# WHY THIS EXISTS: Tavily web search when reranker score is 0.3-0.7.
# Results NEVER stored in pgvector — ephemeral only.
# Always marked "from web search" in user response. LAW 13.

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import tenacity
from tavily import TavilyClient

from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class CRAGResult:
    web_chunks: list[dict]
    query_used: str
    triggered: bool
    sources: list[str] = field(default_factory=list)
    error: Optional[str] = None


@lru_cache(maxsize=1)
def get_tavily_client() -> TavilyClient:
    """Singleton Tavily client."""
    return TavilyClient(api_key=settings.tavily_api_key)


@tenacity.retry(
    wait=tenacity.wait_exponential(min=1, max=6),
    stop=tenacity.stop_after_attempt(2),
    reraise=False,
)
async def crag_web_search(
    query: str,
    max_results: int = 5,
) -> CRAGResult:
    """
    Executes CRAG fallback web search via Tavily.
    Called ONLY when reranker score is between 0.3 and 0.7.

    All results tagged crag_source=True.
    Never persisted to pgvector.
    Response marked "from web search" via build_crag_context_note().

    Args:
        query: User query
        max_results: Web results count (default: 5)

    Returns:
        CRAGResult — never raises
    """
    try:
        client = get_tavily_client()

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True,
        )

        web_chunks: list[dict] = []
        sources: list[str] = []

        # Tavily synthesized answer — highest priority
        if response.get("answer"):
            web_chunks.append({
                "chunk_text": f"[WEB SEARCH SUMMARY]\n{response['answer']}",
                "title": "Web Search Summary",
                "url": "tavily_synthesis",
                "page_number": 0,
                "crag_source": True,
                "pii_detected": False,
                "injection_risk": False,
                "reranker_score": 0.8,
            })

        for result in response.get("results", []):
            web_chunks.append({
                "chunk_text": (
                    f"[WEB SOURCE: {result.get('title', 'Unknown')}]\n"
                    f"{result.get('content', '')}"
                ),
                "title": result.get("title", "Web Result"),
                "url": result.get("url", ""),
                "page_number": 0,
                "crag_source": True,
                "pii_detected": False,
                "injection_risk": False,
                "reranker_score": float(result.get("score", 0.5)),
            })
            sources.append(result.get("url", ""))

        logger.info(
            "CRAG web search complete",
            extra={"query": query[:80], "results": len(web_chunks)},
        )

        return CRAGResult(
            web_chunks=web_chunks,
            query_used=query,
            triggered=True,
            sources=sources,
        )

    except Exception as e:
        logger.error(
            "CRAG web search failed",
            extra={"query": query[:80], "error": str(e)},
        )
        return CRAGResult(
            web_chunks=[],
            query_used=query,
            triggered=False,
            error=str(e),
        )


def build_crag_context_note() -> str:
    """
    Mandatory CRAG disclosure note injected into system prompt.
    User must always know when answer comes from web, not their document.
    """
    return (
        "\n\nIMPORTANT: Some context below is from web search, not the user's document. "
        "Mark those parts explicitly with '(from web search)' in your response.\n"
    )