# observability/langsmith_tracer.py
# WHY THIS EXISTS: Traces every Sarvam call with mandatory tags. LAW 7.
# Language routing failures = hardest bugs in multilingual RAG.
# Without traces from Day 1, debugging requires re-running experiments.
# Mandatory tags: language_code, query_type, user_id,
#   top_reranker_score, crag_triggered, language_accuracy, sarvam_model.

import logging
import os
from dataclasses import dataclass
from typing import Optional

from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class TraceMetadata:
    """Mandatory metadata tags for every SmartDocs LangSmith trace. LAW 7."""
    user_id: str
    language_code: str                   # Detected query language
    query_type: str                      # factual/analytical/conversational/etc
    top_reranker_score: float            # Highest reranker score in this query
    crag_triggered: bool                 # Whether Tavily fallback ran
    sarvam_model: str                    # Always sarvam-m (from settings)
    language_accuracy: Optional[float] = None   # Post-eval async metric
    retry_count: int = 0                 # Self-critique retry count
    cache_hit: bool = False              # Whether response was from cache
    total_cost_inr: float = 0.0          # Total query cost


def setup_langsmith() -> bool:
    """
    Configures LangSmith tracing via environment variables.
    LangGraph automatically traces through LangChain's callback system
    when these environment variables are set.

    Returns:
        True if LangSmith configured successfully, False if key missing.
    """
    if not settings.langsmith_api_key:
        logger.warning("LANGSMITH_API_KEY not set — tracing disabled")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    logger.info(
        "LangSmith tracing enabled",
        extra={"project": settings.langsmith_project},
    )
    return True


def build_run_metadata(trace: TraceMetadata) -> dict:
    """
    Builds the metadata dict passed to LangGraph's config.
    Appears as tags in LangSmith trace UI.

    Usage:
        graph.ainvoke(state, config={"metadata": build_run_metadata(trace)})

    Args:
        trace: TraceMetadata with all mandatory fields

    Returns:
        Dict suitable for LangGraph config["metadata"]
    """
    return {
        "user_id": trace.user_id,
        "language_code": trace.language_code,
        "query_type": trace.query_type,
        "top_reranker_score": trace.top_reranker_score,
        "crag_triggered": trace.crag_triggered,
        "sarvam_model": trace.sarvam_model,
        "language_accuracy": trace.language_accuracy,
        "retry_count": trace.retry_count,
        "cache_hit": trace.cache_hit,
        "total_cost_inr": trace.total_cost_inr,
        "smartdocs_version": "1.0.0",
    }


# Configure LangSmith at module import — runs once
_langsmith_enabled = setup_langsmith()