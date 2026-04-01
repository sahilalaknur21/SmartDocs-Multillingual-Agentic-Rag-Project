# agents/smartdocs_graph.py
# WHY THIS EXISTS: Complete LangGraph agentic loop. LAW 8 + LAW 9.
# Not a pipeline — a stateful graph with conditional edges and retry cycles.
# Every decision point is a conditional edge. Every failure has a fallback.
# Streaming via astream_events v2 — sources before answer tokens.

import logging
import time
from typing import Optional, TypedDict


from langgraph.graph import StateGraph, START, END

from config.settings import get_settings
from retrieval.language_detector import (
    detect_language,
    build_language_system_prompt_instruction,
    LanguageDetectionResult,
)
from retrieval.query_classifier import classify_query, QueryClassificationResult
from retrieval.query_transformer import transform_query, TransformedQueries
from retrieval.hybrid_retriever import retrieve_multi_query, RetrievedChunk
from retrieval.cache import get_cached_response, set_cached_response
from retrieval.crag_fallback import crag_web_search, build_crag_context_note, CRAGResult
from reranking.reranker import rerank_chunks, RerankedResult, RerankerDecision
from generation.context_assembler import assemble_context, AssembledContext
from generation.sarvam_client import (
    stream_answer,
    build_system_prompt,
    get_no_info_response,
    GenerationResult,
)
from generation.self_critique import critique_answer, CritiqueResult
from guardrails.output_guardrail import run_guardrail, GuardrailResult
from observability.cost_tracker import calculate_query_cost, QueryCost
from observability.langsmith_tracer import TraceMetadata, build_run_metadata

settings = get_settings()
logger = logging.getLogger(__name__)

MAX_RETRIES = 2


# ── Graph State ───────────────────────────────────────────────────────────────

class SmartDocsState(TypedDict, total=False):
    """
    Complete state passed between all LangGraph nodes.
    total=False — all fields are optional (not required at init).
    Nodes update only the fields they own.
    """
    # ── Input fields (required at graph entry) ────────────────────────────────
    query: str
    user_id: str
    doc_id: Optional[str]
    doc_title: str

    # ── Language and classification ───────────────────────────────────────────
    lang_result: Optional[LanguageDetectionResult]
    query_classification: Optional[QueryClassificationResult]

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_hit: bool
    cached_response: Optional[dict]

    # ── Query transformation ──────────────────────────────────────────────────
    transformed_queries: Optional[TransformedQueries]

    # ── Retrieval and reranking ───────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]
    rerank_result: Optional[RerankedResult]

    # ── CRAG ──────────────────────────────────────────────────────────────────
    crag_result: Optional[CRAGResult]
    crag_triggered: bool

    # ── Context and generation ────────────────────────────────────────────────
    assembled_context: Optional[AssembledContext]
    system_prompt: str
    raw_answer: str
    stream_tokens: list[str]
    input_tokens: int
    output_tokens: int

    # ── Self-critique ─────────────────────────────────────────────────────────
    critique_result: Optional[CritiqueResult]
    retry_count: int
    active_query: str              # May be refined query on retry

    # ── Guardrail ─────────────────────────────────────────────────────────────
    guardrail_result: Optional[GuardrailResult]

    # ── Final output ──────────────────────────────────────────────────────────
    final_answer: str
    cited_sources: list[dict]
    cost: Optional[QueryCost]
    total_latency_ms: float
    graph_start_time: float


# ── Node functions ────────────────────────────────────────────────────────────

async def preprocess_node(state: SmartDocsState) -> dict:
    """
    Node 1: Language detection + query classification + cache check.
    All three run before any retrieval or API call.
    Cache hit → short-circuits entire graph.
    """
    query = state["query"]
    user_id = state["user_id"]

    # Language detection
    lang_result = detect_language(query)

    # Query classification
    query_classification = classify_query(query)

    # Cache check
    cached = await get_cached_response(query=query, user_id=user_id)

    if cached:
        logger.info(
            "Cache hit — short-circuiting graph",
            extra={"user_id": user_id, "similarity": cached.get("cache_similarity")},
        )
        return {
            "lang_result": lang_result,
            "query_classification": query_classification,
            "cache_hit": True,
            "cached_response": cached,
            "active_query": query,
        }

    return {
        "lang_result": lang_result,
        "query_classification": query_classification,
        "cache_hit": False,
        "cached_response": None,
        "active_query": query,
    }


async def transform_node(state: SmartDocsState) -> dict:
    """
    Node 2: Multi-query + HyDE + step-back transformation.
    Runs in parallel via asyncio.gather inside transform_query().
    """
    transformed = await transform_query(
        query=state["active_query"],
        lang_result=state["lang_result"],
    )

    return {"transformed_queries": transformed}


async def retrieve_node(state: SmartDocsState) -> dict:
    """
    Node 3: Hybrid retrieval across all transformed query variants.
    Uses active_query (may be refined query on retry cycles).
    Dense 0.7 + BM25 0.3 + RRF. top-20 candidates.
    """
    # ── DEPENDENCY INJECTION FIX FOR LANGGRAPH STUDIO ──
    # Intercepts manual JSON payloads from the UI to test documents without a database.
    if state.get("retrieved_chunks"):
        raw_chunks = state["retrieved_chunks"]
        if len(raw_chunks) > 0 and isinstance(raw_chunks[0], dict):
            from retrieval.hybrid_retriever import RetrievedChunk
            try:
                # Convert Studio's raw JSON dicts into your strict Python objects
                object_chunks = [RetrievedChunk(**c) for c in raw_chunks]
                return {"retrieved_chunks": object_chunks}
            except Exception as e:
                logger.warning(f"Mock injection failed kwargs mapping: {e}")
        return {"retrieved_chunks": raw_chunks}
    # ───────────────────────────────────────────────────

    transformed = state.get("transformed_queries")
    queries = (
        transformed.all_queries
        if transformed and transformed.all_queries
        else [state["active_query"]]
    )

    chunks = await retrieve_multi_query(
        queries=queries,
        user_id=state["user_id"],
        lang_result=state["lang_result"],
        doc_id=state.get("doc_id"),
        top_k_candidates=20,
    )

    return {"retrieved_chunks": chunks}


async def rerank_node(state: SmartDocsState) -> dict:
    """
    Node 4: FlashRank reranking of top-20 candidates → top-5.
    Score thresholds determine next node:
        > 0.7  → assemble_context_node
        0.3–0.7 → crag_node
        < 0.3  → END (insufficient)
    """
    chunks = state.get("retrieved_chunks", [])
    query = state["active_query"]

    rerank_result = rerank_chunks(
        query=query,
        chunks=chunks,
        top_k=5,
    )

    return {"rerank_result": rerank_result}


async def crag_node(state: SmartDocsState) -> dict:
    """
    Node 5: Tavily web search fallback. LAW 13.
    Only runs when reranker score is 0.3–0.7.
    Results are ephemeral — never stored in pgvector.
    """
    crag_result = await crag_web_search(query=state["active_query"])

    return {
        "crag_result": crag_result,
        "crag_triggered": crag_result.triggered,
    }


async def assemble_context_node(state: SmartDocsState) -> dict:
    """
    Node 6: Context assembly with citation injection and delimiter wrapping.
    Combines document chunks + CRAG web chunks if present.
    Builds full system prompt for Sarvam-30B.
    """
    rerank_result = state["rerank_result"]
    lang_result = state["lang_result"]
    query_classification = state["query_classification"]
    crag_result = state.get("crag_result")

    # Assemble context
    assembled = assemble_context(
        chunks=rerank_result.chunks,
        crag_result=crag_result,
    )

    # Build context text — prepend CRAG note if web results included
    context_text = assembled.context_text
    if state.get("crag_triggered"):
        context_text = build_crag_context_note() + context_text

    # Build system prompt
    language_instruction = build_language_system_prompt_instruction(lang_result)

    system_prompt = build_system_prompt(
        language_instruction=language_instruction,
        context_with_citations=context_text,
        detected_language=lang_result.language_name,
        query_type=query_classification.query_type.value,
        doc_title=state.get("doc_title", "your document"),
    )

    return {
        "assembled_context": assembled,
        "system_prompt": system_prompt,
        "cited_sources": assembled.cited_sources,
    }


async def generate_node(state: SmartDocsState) -> dict:
    """
    Node 7: Sarvam-30B generation with streaming.
    Streams tokens into state. FastAPI SSE reads from state in Part 5.
    """
    full_answer, input_tokens, output_tokens = await stream_answer(
        query=state["active_query"],
        system_prompt=state["system_prompt"],
    )

    return {
        "raw_answer": full_answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


async def critique_node(state: SmartDocsState) -> dict:
    """
    Node 8: Self-critique — faithfulness + language match. LAW 8.
    faithfulness < 0.75 → retry_needed=True
    language_match = False → retry_needed=True
    retry_count >= 2 → passed=False → graceful failure
    """
    assembled = state.get("assembled_context")
    context = assembled.context_text if assembled else ""
    retry_count = state.get("retry_count", 0)

    critique_result = await critique_answer(
        query=state["active_query"],
        answer=state["raw_answer"],
        context=context,
        query_language=state["lang_result"].language_name,
        retry_count=retry_count,
    )

    # If retry needed — use refined query if critique provided one
    active_query = state["active_query"]
    if critique_result.retry_needed and critique_result.refined_query:
        active_query = critique_result.refined_query
        logger.info(
            "Critique refined query for retry",
            extra={"refined": active_query[:80]},
        )

    return {
        "critique_result": critique_result,
        "retry_count": retry_count + (1 if critique_result.retry_needed else 0),
        "active_query": active_query,
    }


async def guardrail_node(state: SmartDocsState) -> dict:
    """
    Node 9: Final output guardrail — PII, injection, off-topic.
    Calculates final cost. Stores in cache. Returns clean answer.
    """
    guardrail_result = run_guardrail(state["raw_answer"])

    # Calculate total cost
    cost = calculate_query_cost(
        input_tokens=state.get("input_tokens", 0),
        output_tokens=state.get("output_tokens", 0),
        crag_triggered=state.get("crag_triggered", False),
    )

    # Total latency
    start_time = state.get("graph_start_time", time.perf_counter())
    total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    final_answer = guardrail_result.clean_answer

    # Store in cache (non-blocking)
    try:
        await set_cached_response(
            query=state["query"],
            user_id=state["user_id"],
            response={
                "answer": final_answer,
                "cited_sources": state.get("cited_sources", []),
                "cost_inr": cost.total_inr,
                "language": state["lang_result"].language_code,
            },
        )
    except Exception as e:
        logger.warning("Cache store failed — non-fatal", extra={"error": str(e)})

    return {
        "guardrail_result": guardrail_result,
        "final_answer": final_answer,
        "cost": cost,
        "total_latency_ms": total_latency_ms,
    }


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_preprocess(state: SmartDocsState) -> str:
    """Cache hit → return cached answer. No cache → transform query."""
    if state.get("cache_hit"):
        return "serve_cache_node"
    return "transform_node"


def route_after_rerank(state: SmartDocsState) -> str:
    """
    Routes based on reranker decision:
        PROCEED     → assemble context
        CRAG        → web search fallback
        INSUFFICIENT → return no-info response
    """
    rerank_result = state.get("rerank_result")
    if not rerank_result:
        return "insufficient_node"

    decision = rerank_result.decision
    if decision == RerankerDecision.PROCEED:
        return "assemble_context_node"
    elif decision == RerankerDecision.CRAG_FALLBACK:
        return "crag_node"
    else:
        return "insufficient_node"


def route_after_critique(state: SmartDocsState) -> str:
    """
    Routes based on critique result:
        passed=True  → guardrail
        retry_needed=True AND retry_count < MAX_RETRIES → retrieve again
        all else → graceful failure
    """
    critique_result = state.get("critique_result")
    if not critique_result:
        return "guardrail_node"

    if critique_result.passed:
        return "guardrail_node"

    if critique_result.retry_needed and state.get("retry_count", 0) <= MAX_RETRIES:
        logger.info(
            "Self-critique retry",
            extra={
                "retry_count": state.get("retry_count"),
                "issues": critique_result.issues,
            },
        )
        return "retrieve_node"

    return "insufficient_node"


# ── Terminal nodes ────────────────────────────────────────────────────────────

async def serve_cache_node(state: SmartDocsState) -> dict:
    """Returns cached response directly — zero retrieval or generation."""
    cached = state.get("cached_response", {})
    return {
        "final_answer": cached.get("answer", ""),
        "cited_sources": cached.get("cited_sources", []),
        "cost": calculate_query_cost(0, 0),
        "total_latency_ms": 0.0,
    }


async def insufficient_node(state: SmartDocsState) -> dict:
    """
    Returns graceful no-information response.
    Triggered when: reranker score < 0.3, or critique fails after max retries.
    Response is in the user's detected language.
    """
    lang_code = state.get("lang_result").language_code if state.get("lang_result") else "en"
    no_info = get_no_info_response(lang_code)

    logger.info(
        "Insufficient information — returning graceful response",
        extra={"language": lang_code},
    )

    return {
        "final_answer": no_info,
        "cited_sources": [],
        "cost": calculate_query_cost(0, 0),
        "total_latency_ms": 0.0,
    }


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Builds and compiles the complete SmartDocs LangGraph.
    Called once at application startup.

    Graph topology:
        START
          → preprocess_node
          → [cache_hit] serve_cache_node → END
          → [no_cache] transform_node
          → retrieve_node
          → rerank_node
          → [proceed] assemble_context_node
          → [crag] crag_node → assemble_context_node
          → [insufficient] insufficient_node → END
          → generate_node
          → critique_node
          → [pass] guardrail_node → END
          → [retry] retrieve_node (cycle — max 2x)
          → [fail] insufficient_node → END
    """
    builder = StateGraph(SmartDocsState)

    # Register all nodes
    builder.add_node("preprocess_node", preprocess_node)
    builder.add_node("serve_cache_node", serve_cache_node)
    builder.add_node("transform_node", transform_node)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("rerank_node", rerank_node)
    builder.add_node("crag_node", crag_node)
    builder.add_node("assemble_context_node", assemble_context_node)
    builder.add_node("generate_node", generate_node)
    builder.add_node("critique_node", critique_node)
    builder.add_node("guardrail_node", guardrail_node)
    builder.add_node("insufficient_node", insufficient_node)

    # Entry
    builder.add_edge(START, "preprocess_node")

    # After preprocess — cache hit or continue
    builder.add_conditional_edges(
        "preprocess_node",
        route_after_preprocess,
        {
            "serve_cache_node": "serve_cache_node",
            "transform_node": "transform_node",
        },
    )

    # Cache hit → END
    builder.add_edge("serve_cache_node", END)

    # Linear path: transform → retrieve → rerank
    builder.add_edge("transform_node", "retrieve_node")
    builder.add_edge("retrieve_node", "rerank_node")

    # After rerank — proceed, CRAG, or insufficient
    builder.add_conditional_edges(
        "rerank_node",
        route_after_rerank,
        {
            "assemble_context_node": "assemble_context_node",
            "crag_node": "crag_node",
            "insufficient_node": "insufficient_node",
        },
    )

    # CRAG → assemble context
    builder.add_edge("crag_node", "assemble_context_node")

    # Linear path: assemble → generate → critique
    builder.add_edge("assemble_context_node", "generate_node")
    builder.add_edge("generate_node", "critique_node")

    # After critique — pass, retry, or fail
    builder.add_conditional_edges(
        "critique_node",
        route_after_critique,
        {
            "guardrail_node": "guardrail_node",
            "retrieve_node": "retrieve_node",   # Retry cycle
            "insufficient_node": "insufficient_node",
        },
    )

    # Terminal nodes → END
    builder.add_edge("guardrail_node", END)
    builder.add_edge("insufficient_node", END)

    return builder.compile()


# ── Singleton graph ───────────────────────────────────────────────────────────

from functools import lru_cache

@lru_cache(maxsize=1)
def get_smartdocs_graph():
    """
    Singleton compiled graph — built once at startup.
    LangGraph compilation validates all nodes and edges.
    """
    graph = build_graph()
    logger.info("SmartDocs graph compiled and ready")
    return graph


# ── Public API ────────────────────────────────────────────────────────────────

async def run_query(
    query: str,
    user_id: str,
    doc_id: Optional[str] = None,
    doc_title: str = "your document",
) -> dict:
    """
    Main entry point for SmartDocs queries.
    Runs the complete graph — returns final state.
    """
    graph = get_smartdocs_graph()

    initial_state: SmartDocsState = {
        "query": query,
        "user_id": user_id,
        "doc_id": doc_id,
        "doc_title": doc_title,
        "retry_count": 0,
        "crag_triggered": False,
        "cache_hit": False,
        "stream_tokens": [],
        "retrieved_chunks": [],
        "cited_sources": [],
        "active_query": query,
        "graph_start_time": time.perf_counter(),
    }

    # ── LANGSMITH TRACING INJECTION ───────────────────────────────────────────
    trace_data = TraceMetadata(
        user_id=user_id,
        language_code="pending",
        query_type="pending",
        top_reranker_score=0.0,
        crag_triggered=False,
        sarvam_model="sarvam-m",
        retry_count=0,
        cache_hit=False,
        total_cost_inr=0.0
    )
    
    run_config = {
        "metadata": build_run_metadata(trace_data),
        "tags": ["smartdocs_v1", f"user:{user_id}"]
    }
    # ──────────────────────────────────────────────────────────────────────────

    final_state = await graph.ainvoke(initial_state, config=run_config)

    logger.info(
        "SmartDocs query complete",
        extra={
            "user_id": user_id,
            "language": final_state.get("lang_result", {}).language_code
            if final_state.get("lang_result") else "unknown",
            "cache_hit": final_state.get("cache_hit", False),
            "crag_triggered": final_state.get("crag_triggered", False),
            "cost_inr": final_state.get("cost", {}).total_inr
            if final_state.get("cost") else 0.0,
            "latency_ms": final_state.get("total_latency_ms", 0),
        },
    )

    return final_state


async def stream_query(
    query: str,
    user_id: str,
    doc_id: Optional[str] = None,
    doc_title: str = "your document",
):
    """
    Streaming entry point — yields events for FastAPI SSE.
    """
    graph = get_smartdocs_graph()

    initial_state: SmartDocsState = {
        "query": query,
        "user_id": user_id,
        "doc_id": doc_id,
        "doc_title": doc_title,
        "retry_count": 0,
        "crag_triggered": False,
        "cache_hit": False,
        "stream_tokens": [],
        "retrieved_chunks": [],
        "cited_sources": [],
        "active_query": query,
        "graph_start_time": time.perf_counter(),
    }

    # ── LANGSMITH TRACING INJECTION ───────────────────────────────────────────
    trace_data = TraceMetadata(
        user_id=user_id,
        language_code="pending",
        query_type="pending",
        top_reranker_score=0.0,
        crag_triggered=False,
        sarvam_model="sarvam-m",
        retry_count=0,
        cache_hit=False,
        total_cost_inr=0.0
    )
    
    run_config = {
        "metadata": build_run_metadata(trace_data),
        "tags": ["smartdocs_v1", f"user:{user_id}"]
    }
    # ──────────────────────────────────────────────────────────────────────────

    async for event in graph.astream_events(initial_state, version="v2", config=run_config):
        yield event
        
# ── LANGGRAPH STUDIO EXPORT ───────────────────────────────────────────────────
# Expose the compiled graph as a module-level variable for the langgraph-cli.
graph = get_smartdocs_graph()