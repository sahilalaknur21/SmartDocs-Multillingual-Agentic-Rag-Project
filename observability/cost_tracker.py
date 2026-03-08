# observability/cost_tracker.py
# WHY THIS EXISTS: Tracks per-query cost in INR. LAW 18.
# Embedding: ₹0.00 (local multilingual-e5-large on RTX GPU).
# Reranking: ₹0.00 (local FlashRank on CPU).
# Generation: Sarvam-30B API — only paid component.
# Solo operator must know cost per query before scaling.

import logging
from dataclasses import dataclass, field

from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class QueryCost:
    """Complete cost breakdown for a single SmartDocs query."""
    embedding_inr: float = 0.0       # Always ₹0 — local RTX GPU
    reranking_inr: float = 0.0       # Always ₹0 — local FlashRank CPU
    generation_inr: float = 0.0      # Sarvam-30B API cost
    crag_search_inr: float = 0.0     # Tavily web search cost (if triggered)
    total_inr: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    crag_triggered: bool = False

    def __post_init__(self) -> None:
        self.total_inr = round(
            self.embedding_inr
            + self.reranking_inr
            + self.generation_inr
            + self.crag_search_inr,
            4,
        )

    def display(self) -> str:
        """Human-readable cost string for Streamlit UI."""
        return (
            f"₹{self.total_inr:.4f} "
            f"(embed ₹{self.embedding_inr:.2f} + "
            f"rerank ₹{self.reranking_inr:.2f} + "
            f"generate ₹{self.generation_inr:.4f}"
            + (f" + web ₹{self.crag_search_inr:.4f}" if self.crag_triggered else "")
            + ")"
        )


def calculate_query_cost(
    input_tokens: int,
    output_tokens: int,
    crag_triggered: bool = False,
) -> QueryCost:
    """
    Calculates complete INR cost for a single query.

    Cost formula:
        generation_inr = (input_tokens / 1000) * cost_per_1k_input
                       + (output_tokens / 1000) * cost_per_1k_output
        crag_search_inr = ₹0.08 per search (Tavily free tier approximation)

    Args:
        input_tokens: Tokens in context (system prompt + chunks + query)
        output_tokens: Tokens in generated answer
        crag_triggered: Whether Tavily web search was used

    Returns:
        QueryCost with all fields populated
    """
    generation_inr = round(
        (input_tokens / 1000) * settings.sarvam_cost_per_1k_input_tokens_inr
        + (output_tokens / 1000) * settings.sarvam_cost_per_1k_output_tokens_inr,
        4,
    )

    # Tavily free tier: 1000 searches/month free, ~₹0.08 per search after
    crag_cost = 0.08 if crag_triggered else 0.0

    cost = QueryCost(
        embedding_inr=0.0,
        reranking_inr=0.0,
        generation_inr=generation_inr,
        crag_search_inr=crag_cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        crag_triggered=crag_triggered,
    )

    logger.info(
        "Query cost calculated",
        extra={
            "total_inr": cost.total_inr,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "crag_triggered": crag_triggered,
        },
    )

    return cost