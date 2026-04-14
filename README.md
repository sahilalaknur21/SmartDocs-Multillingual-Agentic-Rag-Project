<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,18,19&height=250&section=header&text=%20SmartDocs:%20India-First%20Multilingual%20RAG&fontSize=40&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Research-Grade%20Pipeline%20With%20Production-Grade%20Architecture&descSize=19&descAlignY=55" width="100%"/>

</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3.x-1C3A5E?style=flat-square)
![Embeddings](https://img.shields.io/badge/Embeddings-multilingual--e5--large-FF6B35?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-Sarvam--30B-7B2D8B?style=flat-square)
![VectorStore](https://img.shields.io/badge/VectorStore-pgvector%20%2B%20RLS-336791?style=flat-square&logo=postgresql&logoColor=white)
![API](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-41%20Passing-2EA043?style=flat-square)

</div>

<br>

**SmartDocs achieved 97%+ Hindi Faithfulness — 0.9069 cross-language Hindi/English cosine similarity.**

---

**Multilingual Agentic RAG engineered for India's linguistic reality, processing legal, GST, and insurance documents natively across 22 languages without intermediate translation layers.**

---

> **🚨 Reality Check: The Silent Failure of English-First RAG**
>
> Naive RAG pipelines fail silently in Indian production environments. Introducing a translation layer into an English-centric system degrades semantic fidelity — GST provisions, legal constraints, and precise numerical thresholds are systematically distorted. A system reporting 90% accuracy in English but 65% in Hindi is not multilingual; it is an English pipeline with superficial language coverage. In this architecture, cross-lingual alignment and robust Devanagari Unicode handling are treated as first-order requirements, with **deployment gated** if the Hindi-to-English faithfulness ratio drops below **0.97**.

---

<br>

<div align="center">
  <a href="https://huggingface.co/spaces/Sahilalaknur/smartdocs-ui" target="_blank">
    <img src="https://img.shields.io/badge/Deployed_Go_Live-0077B6?style=for-the-badge&logo=huggingface&logoColor=FFD21E&labelColor=000000" height="55" alt="Deployed Go Live on Hugging Face"/>
  </a>
</div>

<br>

## 🎯 What This System Solves

Standard RAG architectures break down on real-world Indian data. This system explicitly engineers solutions for four root failures:

* **Translation-Induced Semantic Loss:** Eliminates the intermediate `Query → English → LLM → Target Language` translation step. Legal and financial terminology (e.g., "Input Tax Credit," "धारा 16") degrades irreversibly across translation boundaries.
* **Devanagari Processing Noise:** Mitigates Zero-Width Joiner (ZWJ) artifacts, 16 Unicode space variants, and OCR extraction anomalies prior to embedding — failures invisible in dev but catastrophic in production on government-issued PDFs.
* **Retrieval Failure on Alphanumerics:** Solves the dense embedding blindspot for exact-match terms (GSTINs, section codes, specific tax amounts) using Reciprocal Rank Fusion (RRF) with a BM25 sparse index trained on Indic-normalized text.
* **Language Detection Instability:** Replaces statistically unstable detection models — which misclassify *"transformer kya hai"* as Norwegian — with a deterministic 7-step script-and-lexicon execution tree seeded for reproducibility.

---

## 📊 Evaluation Results

Measured RAGAS results on a 60-pair multilingual test set (30 Hindi + 30 English, native — not translated). Deployment is blocked if any gate metric is missed — aggregate accuracy hiding Hindi failures is not accepted.

| Metric | Result | Deployment Gate |
|:---|:---:|:---:|
| Hindi Faithfulness | **97%+** | < 85% → BLOCK |
| English Faithfulness | **90%+** | < 90% → BLOCK |
| Hindi / English Faithfulness Ratio | **0.97** | < 0.97 → BLOCK |
| Language Accuracy (Hindi ↔ English) | **95%+** | < 92% → BLOCK |
| Cross-language Cosine Similarity | **0.9069** | < 0.85 → BLOCK |
| Context Precision | **98%+** | — |
| Hallucination Rate | **< 5%** | — |
| P95 End-to-End Latency | **~8–12s** | — |

> The ratio metric is the definition of success. 90% English / 65% Hindi faithfulness = product failure, regardless of aggregate score.

> P95 latency reflects the full agentic pipeline: language detection → 3 parallel query transformations → hybrid retrieval → reranking → Sarvam-30B streaming generation → self-critique. Single-shot generation without transformation runs ~3–4s.

---

## 🌀 System Overview

The pipeline is non-linear and stateful — not a sequential chain.

1. **Ingestion:** PDF parsing → 5-step Indic Preprocessing → Script-Aware Parent/Child Chunking → Dense + Sparse Embedding → Row-Level Security (RLS) PostgreSQL persistence.
2. **Query Processing:** Deterministic 7-step Language Detection → Async Multi-Query Expansion (Multi-query + HyDE + Step-back via `asyncio.gather`).
3. **Hybrid Retrieval:** Dense (`multilingual-e5-large`) + Sparse (`BM25Okapi` on Indic-normalized text) fused via RRF (k=60, weights 0.7/0.3) → FlashRank Reranking to top-5.
4. **Agentic Reasoning:** 11-node LangGraph state machine: conditional routing, CRAG web fallback, context assembly, Sarvam-30B generation.
5. **Validation & Guardrails:** Post-generation faithfulness + language-match evaluation. Cyclic retries (max 2) before terminal guardrails (PII redaction, injection detection).
6. **Streaming Delivery:** 4-event SSE pipeline — `sources` → `answer` → `metadata` → `done`. Sources populate while the answer streams (LAW 9).

---

<br>

<div align="center">
  <a href="https://smartdocs-website.vercel.app/">
    <img src="https://img.shields.io/badge/🚀_Click_For_Full_Project_Architecture-2EA043?style=for-the-badge&logo=vercel&logoColor=white&labelColor=238636" height="70" alt="Click for Full Project Architecture"/>
  </a>
</div>

<br>

## 🏗️ System Architecture

<p align="center">
<img src="assets/architecture1.png" width="900"/>
</p>

### Core Components

| Component | Implementation | Key Detail |
|:---|:---|:---|
| **Foundation Embedding** | `intfloat/multilingual-e5-large` (1024-dim, ~560MB) | Cross-language cosine > 0.85 is the deployment gate. `passage:` prefix at ingestion, `query:` at retrieval — missing either degrades similarity by 0.05–0.15. |
| **Sparse Index** | `BM25Okapi` on Indic-normalized text | Catches GSTIN exact matches, legal clause numbers, specific tax amounts — all invisible to semantic vectors. |
| **Reranker** | `FlashRank` `ms-marco-MiniLM-L-12-v2` (~21MB, CPU) | Reranks top-20 to top-5. Singleton — loaded once. Score thresholds: > 0.7 → proceed, 0.3–0.7 → CRAG, < 0.3 → insufficient. |
| **Generator** | `Sarvam-30B` via OpenAI-compatible streaming API | Temperature 0.1 for factual grounding. `include_usage=True` on streaming calls for token counting. No-info responses available in 6 Indian languages. |
| **Orchestration** | `LangGraph 0.3.x` StateGraph | 11 nodes, 3 conditional edges, self-critique retry cycle. `SmartDocsState` TypedDict — 25 fields, `total=False`. |
| **Vector Store** | `pgvector` on Supabase + RLS | `_set_user_context()` called before every DB query. All writes inside `async with conn.transaction()` for RLS correctness. |
| **Semantic Cache** | `redis.asyncio` | 0.95 cosine threshold. Per-user namespace — User A's cache never served to User B. Flushed on document update. |
| **Observability** | LangSmith auto-instrumentation | 9 mandatory tags per trace. Language routing failures are the hardest bugs to debug in multilingual RAG — traces from Day 1 are non-negotiable. |

---

## 🔄 LangGraph Flow Architecture

<p align="center">
<img src="assets/langgraph_flow.png" width="900"/>
</p>

### 11 Nodes — State Machine

| Node | Responsibility | Writes to State |
|:---|:---|:---|
| `preprocess_node` | Language detection + query classification + cache check | `lang_result`, `query_classification`, `cache_hit` |
| `serve_cache_node` | Return cached response — zero retrieval or generation | `final_answer`, `cited_sources`, `cost` |
| `transform_node` | Multi-query + HyDE + step-back via `asyncio.gather` | `transformed_queries` |
| `retrieve_node` | Hybrid retrieval across all query variants in parallel | `retrieved_chunks` |
| `rerank_node` | FlashRank top-20 → top-5 + routing decision | `rerank_result` |
| `crag_node` | Tavily web search (score 0.3–0.7 only) — ephemeral, never stored | `crag_result`, `crag_triggered` |
| `assemble_context_node` | Sort + source diversity + citation injection + XML delimiter wrap | `assembled_context`, `system_prompt`, `cited_sources` |
| `generate_node` | Sarvam-30B streaming generation | `raw_answer`, `input_tokens`, `output_tokens` |
| `critique_node` | Faithfulness + language match + refined query on retry | `critique_result`, `retry_count` |
| `guardrail_node` | PII redaction + injection check + cost computation + cache store | `final_answer`, `cost`, `total_latency_ms` |
| `insufficient_node` | Graceful no-info response in user's detected language | `final_answer` |

### 3 Conditional Edges

```
preprocess_node:   cache_hit = True    →  serve_cache_node → END
                   cache_hit = False   →  transform → retrieve → rerank

rerank_node:       score > 0.7         →  assemble_context_node
                   score 0.3–0.7       →  crag_node → assemble_context_node
                   score < 0.3         →  insufficient_node → END

critique_node:     passed = True       →  guardrail_node → END
                   retry_count < 2     →  retrieve_node (refined query)
                   retry_count >= 2    →  insufficient_node → END
```

---

## 📐 Key Design Decisions

> **📝 Zero Translation Layer**
>
> * **Decision:** Query and retrieve natively in the source language using `multilingual-e5-large`.
> * **Reason:** Legal and financial terminology degrades when translated to English and back. "Input Tax Credit" loses structural precision when round-tripped through translation in Hindi legal documents.
> * **Trade-off:** Requires a heavier 1024-dim embedding model and strictly mandated `passage:` / `query:` prefixing. Missing either prefix degrades similarity by 0.05–0.15 — silent and catastrophic at the 0.85 deployment gate.

> **📝 Script-Aware Parent-Child Chunking**
>
> * **Decision:** Devanagari: 400-token chunks (40 overlap). Latin: 500-token chunks (50 overlap). Child chunks (256t) for retrieval, Parent chunks (1024t) for generation context.
> * **Reason:** Devanagari is structurally more information-dense per token. `indic-nlp-library` sentence tokenizer runs before chunking — no chunk ends mid-sentence. Child chunks give precise retrieval; parent chunks give rich context to Sarvam-30B.
> * **Trade-off:** Script-detection overhead at ingestion. Two index entries per chunk.

> **📝 Deterministic Language Detection**
>
> * **Decision:** 7-step deterministic logic tree overriding probabilistic models.
> * **Reason:** Standard `langdetect` returns Norwegian for *"transformer kya hai"*. The system enforces a 130-word Hinglish lexicon match and 85% ASCII dominance threshold before trusting statistical models. `DetectorFactory.seed = 0` for production reproducibility.
> * **Trade-off:** Requires manual maintenance of the Hinglish lexicon as the language evolves.

> **📝 Agentic Workflow over Static Pipeline**
>
> * **Decision:** `LangGraph` stateful graph over sequential `LCEL` chain.
> * **Reason:** Static pipelines cannot recover from mid-flight hallucinations. The self-critique node requires cyclic graph execution — retry with a refined query without failing the user request. Max 2 retries prevent infinite loops; failure after 2 retries degrades gracefully to `insufficient_node`, never hangs.
> * **Trade-off:** Increased P95 latency vs single-shot generation.

> **📝 Transaction-Bound RLS Enforcement**
>
> * **Decision:** All pgvector writes inside `async with conn.transaction()`. `_set_user_context()` called before every query. Middleware enforces `user_id` on every API request unconditionally.
> * **Reason:** `SET LOCAL` config vars scope to the current transaction. Without the transaction wrapper, RLS context evaporates after the SET statement — the UPDATE silently affects zero rows. This was the most critical production bug in the ingestion audit.
> * **Trade-off:** Slightly higher connection overhead per write. The alternative is silent data corruption and documents permanently stuck at `status='pending'`.

---

## ⚙️ The 18 Engineering Laws

Every architectural decision maps to a numbered law. These are enforced as code constraints, not documentation.

| Law | Principle | Enforcement Point |
|:---:|:---|:---|
| **LAW 1** | All 5 Indic preprocessing steps before any embedding or BM25 index | `indic_preprocessing_pipeline()` — import and run; never skip individual steps |
| **LAW 2** | Cross-language cosine similarity > 0.85 before any ingestion code is written | `embedding_validator.py` — hard deployment gate |
| **LAW 3** | Devanagari: 400 tokens, not 500. `indic` sentence tokenizer runs before chunking | `chunker.py` script-aware branching |
| **LAW 4** | `passage:` at ingestion. `query:` at retrieval. Missing either = 0.05–0.15 similarity drop | `dense_embedder.py` prefix contracts |
| **LAW 5** | Dense 0.7 + BM25 0.3 + RRF. Dense-only loses 15–25% of retrievable Hindi answers | `hybrid_retriever.py` — hybrid is the minimum acceptable standard |
| **LAW 6** | Per-user RLS isolation. `user_id` NEVER null. `_set_user_context()` before every DB query | `pgvector_client.py` + `assert user_id` in metadata builder |
| **LAW 7** | LangSmith traces every Sarvam call from Day 1 — 9 mandatory tags | `langsmith_tracer.py` `build_run_metadata()` |
| **LAW 8** | Self-critique: faithfulness < 0.75 → retry. Language mismatch → retry. Max 2 | `self_critique.py` → `critique_node` conditional edge |
| **LAW 9** | `sources` event emitted before `answer` event — users see retrieval while reading response | SSE order: `sources → answer → metadata → done` |
| **LAW 10** | 15 injection patterns (Hindi + English) at ingestion. XML delimiter wrapping in context | `injection_scanner.py` + `context_assembler.py` delimiters |
| **LAW 11** | PII tagged at ingestion (`pii_detected=True`). Redacted from every API response | `pii_detector.py` tags + `output_guardrail.py` redacts |
| **LAW 12** | Semantic cache: 0.95 cosine, per-user namespace, flush on document update. Cache failure = non-fatal | `cache.py` — fail open, never blocks a query |
| **LAW 13** | > 0.7 → generate, 0.3–0.7 → CRAG (Tavily), < 0.3 → insufficient. CRAG results never stored | `reranker.py` threshold decision + `crag_fallback.py` |
| **LAW 14** | Sort by score. Max 2 chunks per source. Citation per chunk. 12,000 char budget | `context_assembler.py` — 5 rules in sequence |
| **LAW 15** | Language detection on every query. Confidence < 0.85 → default English | `language_detector.py` 7-step tree |
| **LAW 16** | `faithfulness_hindi < 0.85` OR `language_accuracy < 0.92` → BLOCK deployment | `deployment_gate.py` |
| **LAW 17** | Language badge from `/ingest` response — shown before first query, zero extra API calls | `upload_panel.py` `primary_language` field |
| **LAW 18** | Cost in INR per query, session total, and latency traffic light shown to user | `cost_tracker.py` + `cost_panel.py` |

---

## 🔐 Security Architecture

Security is enforced at three independent layers. An application-layer failure is backstopped by the next layer. Database RLS is the final guarantee.

```
Layer 1 — Ingestion:   injection_scanner.py  →  flags injection_risk=True per chunk
                       pii_detector.py        →  flags pii_detected=True per chunk
                       Flagged chunks excluded from retrieval entirely (WHERE injection_risk = FALSE)

Layer 2 — Context:     context_assembler.py  →  wraps every chunk in XML delimiters
                       <retrieved_document>citation\ntext</retrieved_document>
                       Prevents retrieved content from escaping into system prompt space

Layer 3 — Output:      output_guardrail.py   →  PII redaction (Aadhaar, PAN, phone, GSTIN)
                       15-pattern injection check on generated response text
                       Off-topic flag for answers without citation signals

Database:              Supabase RLS          →  user_id = current_setting('app.current_user_id')
                       SET LOCAL scoped to transaction — enforced in every pgvector_client method
                       Service role key used server-side only — never exposed to UI layer
```

> Without `async with conn.transaction()`, `SET LOCAL` config vars evaporate after the SET statement. The RLS UPDATE then silently passes with zero rows affected — documents stay at `status='pending'` forever with no error thrown. This was Bug #1 and Bug #2 in the production audit.

---

## 🪲 Production Engineering — 10 Bugs Fixed Before First API Route

The ingestion layer was fully audited before any API code was written. 10 bugs were found and fixed. The most critical ones caused silent data corruption — no errors, wrong results.

| # | Severity | Bug | Silent Impact |
|:---:|:---:|:---|:---|
| 1 | **CRITICAL** | Missing UPDATE RLS policy in `schema.sql` | `total_chunks` stayed 0 after every ingestion — documents showed `pending` forever |
| 2 | **CRITICAL** | `update_document_status()` outside `conn.transaction()` | `SET LOCAL` context evaporated — RLS UPDATE silently failed even after policy was added |
| 3 | **HIGH** | `store_document()` returned `True` or `str` — ambiguous on driver version differences | Duplicate ingestion on asyncpg UUID return. Fixed: `StoreDocumentResult(is_new: bool, doc_id: str)` |
| 4 | **HIGH** | `locals().get('doc_id')` in except block | Exception before assignment returned `''` — wrong document tagged on failure. Fixed: initialize before `try` |
| 5 | **HIGH** | `exclude_injection_risk=True` accepted but WHERE clause never included the filter | Every retrieval returned injection-flagged chunks — poisoned PDFs bypassed LAW 10 entirely |
| 6 | **MEDIUM** | `DenseEmbedder()` instantiated per upload | Reloaded 1.8GB model + 3–5s latency on every upload. Fixed: `get_embedder()` singleton |
| 7 | **MEDIUM** | `asyncio.get_event_loop()` deprecated Python 3.10 | Fails in Python 3.12+. Fixed: `asyncio.get_running_loop()` |
| 8 | **MEDIUM** | Double `passage:` prefix — worker and `embed_passages()` both adding it | Model received `passage: passage: text` — silent similarity score degradation, no error |
| 9 | **LOW** | `document_classifier.py` existed but `ingestion_worker.py` hardcoded `doc_type='other'` | Every GST notice stored as `other` — classification was a no-op |
| 10 | **LOW** | 0-chunk edge case (scanned PDFs) returned `status='completed'` with `total_chunks=0` | Document appeared successfully ingested. Queries silently returned nothing |

All 10 fixes validated by targeted unit tests in `tests/test_5a_ingestion_pipeline.py`. `schema_patch.sql` was also written to add the missing UPDATE RLS policy, `error_message`, and `updated_at` columns to the production schema.

---

## ⚖️ What Makes This Different

| Standard RAG Systems | SmartDocs Architecture |
|:---|:---|
| **English-Biased:** Relies on translation for non-English queries | **Multilingual-First:** Embeds and retrieves natively across 22 languages |
| **Static Execution:** Linear ingest → retrieve → generate | **Stateful/Agentic:** Conditional edges, CRAG web fallback, self-critique retry cycles |
| **Naive Text Splitters:** Character or recursive splitting | **Indic-Aware:** 5-step preprocessing, ZWJ removal, script-density-adjusted parent/child chunks |
| **Blind Delivery:** Returns whatever the LLM produces | **Evaluation Gated:** Post-generation faithfulness validation with retry before user sees output |
| **Single-layer security:** Application-level checks only | **Three-layer security:** Ingestion flag → context delimiter → output redaction + database RLS |

---

## ⚠️ Failure Modes & Limitations

* **Noisy OCR Degradation:** `BM25Okapi` is highly sensitive to token fragmentation from poor OCR in scanned Indian government PDFs. Mitigation: `detect_and_flag_low_quality()` in preprocessing step 4 flags but does not delete affected chunks — they are available for audit.
* **Low-Resource Language Drift:** While robust for Hindi/Marathi/Tamil, embedding alignment drops for extremely low-resource Indic dialects. Hindi and English test pairs score 0.90+; coverage narrows for rarer scripts.
* **CRAG Context Overflow:** When Tavily web search is triggered (score 0.3–0.7), appended web text risks pushing `assembled_context` beyond the 12,000-character hard-cap (~3,000 tokens), forcing truncation of the primary PDF context.
* **Multivariate Query Ambiguity:** Extremely short queries (e.g., *"tax amount"*) matching multiple diverse documents produce equalized FlashRank scores, incorrectly triggering CRAG fallback.
* **Transaction Pooler Requirement:** Supabase `DATABASE_URL` must use the Transaction Pooler (port 6543), not the direct connection (port 5432). `asyncpg` combined with `pgbouncer` on the direct port causes statement cache conflicts.

---

## 🚀 Quick Start

> ⏱️ **Time estimate: 20–30 minutes** for full setup. Steps 5–7 (model download, Supabase schema, Redis) take the longest on first run.

---

### 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12+ | 3.12 or 3.13 both work |
| uv | latest | `pip install uv` |
| RAM | 8GB+ | `multilingual-e5-large` loads ~1.8GB into VRAM or system RAM |
| Disk | 1.5GB+ | Model cache (~560MB) + indic NLP resources |
| CUDA | 12.1+ | Optional — CPU fallback works automatically, ~3× slower embedding |
| Supabase account | — | Free tier works |
| Sarvam AI API key | — | [dashboard.sarvam.ai](https://dashboard.sarvam.ai) |
| LangSmith API key | — | [smith.langchain.com](https://smith.langchain.com) |

---

### 2. Clone & Environment Setup

```bash
git clone https://github.com/YOUR_USERNAME/SmartDocs-Multillingual-Agentic-Rag-Project.git
cd SmartDocs-Multillingual-Agentic-Rag-Project

# Create virtual environment
uv venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (macOS / Linux)
source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
# Install all dependencies from pyproject.toml
uv pip install -e .

# Install PyTorch with CUDA 12.1 support (RTX GPU users)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA (falls back to CPU automatically if not available)
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

### 4. Download Indic NLP Resources

Required for Hindi/Indic sentence tokenization. Without this, chunking fails on Devanagari PDFs — the chunker splits mid-sentence and retrieval quality degrades silently.

```bash
git clone --depth 1 https://github.com/anoopkunchukuttan/indic_nlp_resources.git
```

This creates `indic_nlp_resources/` in your project root. The `.gitignore` already excludes it.

---

### 5. Download & Validate Embedding Model

SmartDocs uses `intfloat/multilingual-e5-large` locally — zero API cost, ~560MB download.

```bash
# Pre-download the model
uv run python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-large')
print('Model downloaded and cached')
print('Embedding dimension:', model.get_sentence_embedding_dimension())
"
```

Expected:
```
Model downloaded and cached
Embedding dimension: 1024
```

**Mandatory validation — do not skip. This is the deployment gate (LAW 2):**

```bash
uv run python -c "
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('intfloat/multilingual-e5-large')
hindi   = model.encode(['passage: भूमि अधिग्रहण मुआवजा'])
english = model.encode(['passage: land acquisition compensation'])
sim = cosine_similarity(hindi, english)[0][0]
print(f'Cross-language similarity: {sim:.4f}')
assert sim > 0.85, f'FAILED: {sim} — do not proceed'
print('PASSED — multilingual retrieval confirmed')
"
```

Expected:
```
Cross-language similarity: 0.9XXX
PASSED — multilingual retrieval confirmed
```

> ⚠️ If similarity < 0.85, stop. The entire retrieval pipeline is compromised. Do not proceed to ingestion.

---

### 6. Configure Environment Variables

```bash
# Windows
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and fill in your credentials:

```env
# ── Sarvam AI (Generation) ──────────────────────────────────────────
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_BASE_URL=https://api.sarvam.ai/v1
SARVAM_MODEL_30B=sarvam-m
SARVAM_COST_PER_1K_INPUT_TOKENS_INR=0.50
SARVAM_COST_PER_1K_OUTPUT_TOKENS_INR=0.66

# ── Supabase (Vector Store) ─────────────────────────────────────────
# Get from: supabase.com → project → Settings → API
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key_here

# Get from: supabase.com → project → Settings → Database → Connection Pooling
# Use Transaction Pooler URL (port 6543) — NOT direct connection (port 5432)
DATABASE_URL=postgresql://postgres.YOUR_PROJECT_REF:YOUR_PASSWORD@aws-X-REGION.pooler.supabase.com:6543/postgres

# ── LangSmith (Observability) ───────────────────────────────────────
# Get from: smith.langchain.com → Settings → API Keys
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=smartdocs
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=smartdocs

# ── Optional Services ────────────────────────────────────────────────
TAVILY_API_KEY=your_tavily_key_here        # CRAG web search fallback
REDIS_URL=redis://localhost:6379            # Semantic query cache

# ── App ──────────────────────────────────────────────────────────────
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
APP_SECRET_KEY=your_generated_secret_here
ENVIRONMENT=development
EMBEDDING_DEVICE=cuda                       # Use cpu if no NVIDIA GPU
EMBEDDING_BATCH_SIZE=32                     # Reduce to 8 if VRAM < 4GB
HF_TOKEN=                                   # Optional — speeds up model downloads
```

> 💡 **`SARVAM_BASE_URL` must end with `/v1`** — `https://api.sarvam.ai/v1` not `https://api.sarvam.ai`.

> 💡 **Supabase region matters.** Your `DATABASE_URL` hostname must match your project region (e.g. `aws-0-ap-south-1` for Mumbai). Get the exact URL from Supabase → Settings → Database → Connection Pooling.

---

### 7. Set Up Supabase Database Schema

**Step 7A — Run the base schema:**

Open [Supabase SQL Editor](https://supabase.com) → your project → SQL Editor → paste the contents of `vectorstore/schema.sql` → click **Run**.

Expected: `Success. No rows returned.`

**Step 7B — Run the schema patch (mandatory):**

In the same SQL Editor, paste the contents of `vectorstore/schema_patch.sql` → click **Run**.

> ⚠️ `schema_patch.sql` adds the missing UPDATE RLS policy, `error_message`, and `updated_at` columns. Without it, `total_chunks` stays 0 after every ingestion — documents appear uploaded but are silently broken. This was the most critical bug found in the production audit.

**Step 7C — Verify tables exist:**

```bash
uv run python -c "
import asyncio, asyncpg

async def check():
    from config.settings import get_settings
    s = get_settings()
    conn = await asyncpg.connect(s.database_url, statement_cache_size=0, ssl='require')
    tables = await conn.fetch(\"SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'\")
    print('Connection: OK')
    print('Tables:', [t['table_name'] for t in tables])
    await conn.close()

asyncio.run(check())
"
```

Expected:
```
Connection: OK
Tables: ['documents', 'document_chunks']
```

---

### 8. Start Redis (Semantic Query Cache)

```bash
# Windows — using Docker
docker run -d -p 6379:6379 redis:alpine

# macOS
brew install redis && brew services start redis

# Linux
sudo apt install redis-server && sudo systemctl start redis

# Verify
redis-cli ping
# Expected: PONG
```

> Redis is optional. If not running, SmartDocs degrades gracefully — queries still work, semantic cache is disabled. The health check reports `redis: degraded` which is acceptable. All other health checks must be `ok`.

---

### 9. Start the FastAPI Backend

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 --log-level info
```

**Application startup sequence (3 steps, in order):**

```
SmartDocs API starting up...
✓ pgvector pool connected          ← Step 1: DB pool established
✓ DenseEmbedder warmed up on cuda  ← Step 2: multilingual-e5-large loaded once
✓ LangGraph compiled               ← Step 3: graph ready for queries
SmartDocs API ready | env=development | model=sarvam-m
INFO: Uvicorn running on http://0.0.0.0:8000
```

Verify all components healthy:

```bash
# Windows PowerShell
curl.exe http://localhost:8000/health/ping
curl.exe http://localhost:8000/health

# macOS / Linux
curl http://localhost:8000/health/ping
curl http://localhost:8000/health
```

Expected `/health/ping`:
```json
{"status": "alive"}
```

Expected `/health`:
```json
{
  "status": "ok",
  "checks": {
    "db":        {"status": "ok",      "detail": "query returned 1"},
    "redis":     {"status": "ok",      "detail": "PING → True"},
    "embedding": {"status": "ok",      "detail": "model loaded on cuda"},
    "graph":     {"status": "ok",      "detail": "graph compiled: CompiledStateGraph"}
  },
  "latency_ms": 812.4
}
```

---

### 10. Start the Streamlit UI

Open a **second terminal**:

```bash
cd SmartDocs-Multillingual-Agentic-Rag-Project

# Windows
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

uv run streamlit run ui/app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

---

### 11. First Query — End-to-End Test

1. **Upload a PDF** — click Browse files in the sidebar
2. **Watch the language badge** — a Hindi PDF shows `🇮🇳 Hindi` before you type anything (LAW 17)
3. **Ask in Hindi:**
   ```
   इस दस्तावेज़ का मुख्य विषय क्या है?
   ```
4. **Ask in English:**
   ```
   What is the main topic of this document?
   ```

Both answers appear in their respective languages with source citations and cost tracking in INR.

---

### 12. Smoke Test — Full Pipeline Validation

```bash
uv run python smoke_test.py --api http://localhost:8000 --user user_123
```

Expected:
```
================================================================
SMARTDOCS POST-DEPLOY SMOKE TEST
================================================================
[health] PING OK: {"status": "alive"}
[health] Status: ok

[hi_01] Basic Hindi GST question
  Status:   PASS
  Language: OK
  Latency:  8200ms

...

SMOKE TEST SUMMARY
  Passed:           5/5
  Hindi accuracy:   100%
  English accuracy: 100%

  SMOKE TEST PASSED — language_accuracy = 1.0
================================================================
```

---

## 💡 Example Usage

**Input State (JSON):**

```json
{
  "query": "पंजीकरण के लिए कारोबार की सीमा क्या है?",
  "user_id": "dev_user_001",
  "doc_title": "Sample GST Notice"
}
```

**System Execution Trace:**

1. `preprocess_node`: Detected `hi` (Hindi) via script detection, classified as `Factual` — `top_k=5`.
2. `transform_node`: Generated 3 query variants (multi-query + HyDE + step-back) in parallel via `asyncio.gather`.
3. `retrieve_node`: Hybrid search across all variants — dense 0.7 + BM25 0.3 + RRF → top-20 candidates.
4. `rerank_node`: FlashRank scored `0.95` → Routed to `PROCEED`.
5. `assemble_context_node`: Top-5 sorted by score, max 2 per source, citation injected, XML delimiters wrapped.
6. `generate_node`: Sarvam-30B streamed native Hindi response at temperature 0.1.
7. `critique_node`: Faithfulness `0.98`, language match confirmed → Passed. No retry.

**SSE Event Stream to UI:**
```
event: sources   →  cited chunks + page references rendered immediately
event: answer    →  "सालाना 40 लाख रुपये से अधिक..." streams as tokens arrive
event: metadata  →  {"language": "hi", "cost_inr": 0.63, "latency_ms": 8100, "cache_hit": false}
event: done      →  spinner dismissed, session cost updated
```

**Output:**

> सालाना 40 लाख रुपये से अधिक कारोबार पर पंजीकरण अनिवार्य है। [Source: Sample GST Notice, Page 1]

---

## 🧪 Test Coverage

41 tests across 3 suites. Zero live API calls — Sarvam, Supabase, and Redis are mocked in every test. External service reliability is not under test here.

| Test File | Tests | What It Covers |
|:---|:---:|:---|
| `tests/test_5a_ingestion_pipeline.py` | 11 | 10 production bugs validated individually + full integration test (PDF → chunks → pgvector → similarity search → idempotency) |
| `tests/test_generation.py` | 25 | Context assembler (citations, delimiters, source diversity, budget), cost tracker, guardrails (PAN/phone redaction, injection detection), system prompt structure, LangGraph compilation |
| `tests/test_part3_retrieval.py` | 16 | Language detection (all 5 edge cases including Hinglish + ASCII override), query classification, reranker scoring, CRAG context note, cache singleton |

---

## 📂 Repository Structure

```bash
.
├── agents/
│   └── smartdocs_graph.py        # 11-node LangGraph state machine — 25-field SmartDocsState
├── api/
│   ├── main.py                   # FastAPI lifespan: pool → embedder warmup → graph compile
│   ├── middleware/
│   │   └── user_context.py       # X-User-ID → request.state.user_id — unconditional on every request
│   └── routes/
│       ├── health.py             # GET /health/ping (liveness) + GET /health (full dependency check)
│       ├── ingest.py             # POST /ingest + GET /ingest/status/{id} + GET /ingest/documents
│       └── query.py              # POST /query/stream (SSE, 4 events) + POST /query (sync JSON)
├── config/
│   ├── settings.py               # Pydantic BaseSettings — missing key = crash at startup, not silently in prod
│   ├── retrieval_config.yaml     # Chunk sizes (Devanagari 400 / Latin 500), RRF weights, thresholds
│   └── cost_config.yaml          # INR cost ceilings per query
├── embeddings/
│   ├── dense_embedder.py         # multilingual-e5-large — passage:/query: prefix contracts
│   ├── sparse_embedder.py        # BM25Okapi on Indic-normalized text
│   └── embedding_validator.py    # Cross-language cosine > 0.85 — must pass before any ingestion code
├── generation/
│   ├── context_assembler.py      # Sort + diversity (max 2/source) + citation injection + XML delimiters + 12K budget
│   ├── sarvam_client.py          # AsyncOpenAI → Sarvam-30B, singleton, streaming, include_usage=True
│   └── self_critique.py          # faithfulness + language match, max 2 retries
├── guardrails/
│   └── output_guardrail.py       # PII redaction + 15-pattern injection check + off-topic flag
├── ingestion/
│   ├── indic_preprocessing.py    # 5-step pipeline: normalize → ZWJ remove → spaces → quality flag → tokenize
│   ├── pdf_loader.py             # pdfplumber primary + table extraction → ExtractedDocument
│   ├── chunker.py                # Script-aware parent-child chunking — sentence boundaries first
│   ├── metadata_builder.py       # 24 metadata fields per chunk — user_id NEVER null (hard assert)
│   ├── injection_scanner.py      # 15 patterns Hindi + English — tags injection_risk per chunk
│   ├── pii_detector.py           # Aadhaar, PAN, phone, GSTIN, email regex detection
│   ├── document_classifier.py    # gst_notice / legal / insurance / circular / other
│   └── ingestion_worker.py       # 8-stage pipeline orchestrator — idempotency before page validation
├── observability/
│   ├── cost_tracker.py           # Token counting → INR per query (embedding ₹0, generation ~₹0.50–0.66)
│   └── langsmith_tracer.py       # 9 mandatory trace tags on every Sarvam call
├── reranking/
│   └── reranker.py               # FlashRank CPU singleton — ms-marco-MiniLM-L-12-v2 (~21MB)
├── retrieval/
│   ├── cache.py                  # redis.asyncio — 0.95 cosine, per-user, fail-open
│   ├── crag_fallback.py          # Tavily — ephemeral, never stored in pgvector, always labelled "(from web search)"
│   ├── hybrid_retriever.py       # Dense 0.7 + BM25 0.3 + RRF (k=60) — get_pgvector_client() is sync, no await
│   ├── language_detector.py      # 7-step deterministic tree — DetectorFactory.seed=0 for reproducibility
│   ├── query_classifier.py       # factual (top_k=5) / analytical (top_k=8) / conversational (top_k=10)
│   └── query_transformer.py      # Multi-query + HyDE + step-back via asyncio.gather — singleton AsyncOpenAI
├── ui/
│   ├── app.py                    # Streamlit entry point — session state init, Reset button, two-server arch
│   └── components/
│       ├── upload_panel.py       # PDF upload + language badge (LAW 17) + document selector dropdown
│       ├── query_panel.py        # Language selector + example queries seeded in doc language + submit
│       ├── answer_panel.py       # SSE consumer (httpx sync) — 3 st.empty() placeholders, think-tag strip
│       └── cost_panel.py         # INR/query + session total + latency traffic light + reset (LAW 18)
├── vectorstore/
│   ├── pgvector_client.py        # asyncpg pool — every write inside transaction, _set_user_context() before every query
│   ├── schema.sql                # documents + document_chunks + ivfflat index + RLS SELECT/INSERT/DELETE policies
│   └── schema_patch.sql          # UPDATE RLS policy + error_message + updated_at — run before first ingestion
├── tests/
│   ├── test_5a_ingestion_pipeline.py  # 11 tests — 10 production bugs + integration
│   ├── test_generation.py             # 25 tests — generation, guardrails, graph compilation
│   └── test_part3_retrieval.py        # 16 tests — retrieval, language detection edge cases
├── pyproject.toml                # uv dependency management — 30+ pinned dependencies
├── langgraph.json                # LangGraph Studio CLI config
└── smoke_test.py                 # Post-deploy: 5 queries, Hindi + English, language_accuracy gate
```

---

## 🔍 Observability & Cost

LangSmith tracing is non-negotiable (LAW 7). Language routing failures are the hardest bugs to diagnose in multilingual RAG. Without traces from Day 1, you debug by re-running experiments.

**9 mandatory LangSmith tags per trace:**

| Tag | Type | Debug Signal |
|:---|:---:|:---|
| `user_id` | str | Cross-reference with RLS — debug per-user isolation failures |
| `language_code` | str | Detect wrong-language routing — the most common silent failure |
| `query_type` | str | Debug why a factual query triggered CRAG instead of proceeding |
| `top_reranker_score` | float | Understand why 0.68 triggers CRAG vs 0.72 proceeds |
| `crag_triggered` | bool | Track Tavily search rate — cost and quality signal |
| `sarvam_model` | str | Track model version changes across deployments |
| `retry_count` | int | Detect systematic faithfulness failures across queries |
| `cache_hit` | bool | Measure cache effectiveness and cost savings |
| `total_cost_inr` | float | Per-query billing analytics |

**Per-query cost breakdown:**

| Component | Cost | Note |
|:---|:---:|:---|
| Embedding (`multilingual-e5-large`) | ₹0.00 | Local GPU — zero API cost |
| Reranking (`FlashRank`) | ₹0.00 | Local CPU — zero API cost |
| Generation (`Sarvam-30B`) | ~₹0.50–0.66 | Per ~1000 input + 500 output tokens |
| CRAG web search (Tavily, if triggered) | ~₹0.08 | Free tier: 1000/month |
| **Typical total per query** | **₹0.50–0.80** | Displayed live in Streamlit cost panel |

---

## 🔧 Extensibility

The architecture is decoupled at every layer:

* **Domain Agnosticism:** Currently configured for Indian legal/tax documents. The 5-step Indic preprocessing pipeline is script-aware, not domain-aware — it scales directly to healthcare records, government circulars, and financial disclosures without changes.
* **Model Interchangeability:** `multilingual-e5-large` can be swapped for domain-specific finetunes. `Sarvam-30B` can be replaced with any OpenAI-compatible endpoint by updating `sarvam_client.py` — the LangGraph orchestration is model-agnostic.
* **Database Scaling:** The `pgvector` layer is abstracted in `vectorstore/`. Migration to Milvus or Pinecone requires updating only `pgvector_client.py` — the LangGraph graph, retrieval logic, and RLS enforcement contracts remain unchanged.

---

## 🙏 Acknowledgments

| Project | Role in SmartDocs |
|:---|:---|
| [Sarvam AI](https://sarvam.ai) | Sarvam-30B generation backbone — OpenAI-compatible API, optimized for Indian languages |
| [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | Foundation cross-lingual embedding model |
| [indic-nlp-library](https://github.com/anoopkunchukuttan/indic_nlp_library) | Indic script sentence tokenization and Unicode normalization |
| [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful agentic orchestration framework |
| [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) | Zero-GPU-cost local reranking |
| [pgvector](https://github.com/pgvector/pgvector) | Vector similarity search with Row Level Security on Supabase |
| [Tavily](https://tavily.com) | CRAG web search fallback |

---

## 🤝 Feedback & Contribution

This architecture is built for practitioners dealing with the messy reality of production multilingual retrieval.

Contributions most useful to this project:

* **Script boundary edge cases** — Hinglish queries where the 7-step detector misclassifies. Include the raw query, expected language, and actual detected language with the detection method logged.
* **RRF weight evidence** — Profiled evidence that a different 0.7/0.3 split improves recall on a specific document type, with RAGAS before/after.
* **Retrieval latency improvements** — Specifically in `retrieve_multi_query()` — `asyncio.gather` bottlenecks that have been profiled with trace evidence.
* **New document type classifiers** — The keyword map in `document_classifier.py` is designed for extension.

Please submit issues with LangSmith trace links where possible. Architecture-level critiques from engineers running similar production systems are welcomed.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,18,19&height=120&section=footer" width="100%"/>

</div>
