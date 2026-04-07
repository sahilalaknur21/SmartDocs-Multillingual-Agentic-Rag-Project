<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,18,19&height=250&section=header&text=%20SmartDocs:%20India-First%20Multillingual-Rag&fontSize=40&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Research-Grade%20Pipeline%20With%20Production-Grade%20Architecture&descSize=19&descAlignY=55" width="100%"/>

</div>

**Multilingual, Zero-Translation Retrieval Architecture for High-Compliance Domains**

An agentic RAG engineered for India's linguistic reality, processing legal, GST, and insurance documents natively across 22 languages without intermediate translation layers.

---

<br>

<div align="center">
  <a href="https://huggingface.co/spaces/Sahilalaknur/smartdocs-ui" target="_blank">
    <img src="https://img.shields.io/badge/Deployed_Go_Live-0077B6?style=for-the-badge&logo=huggingface&logoColor=FFD21E&labelColor=000000" height="55" alt="Deployed Go Live on Hugging Face"/>
  </a>
</div>

<br>


> **🚨 Reality Check: The Silent Failure of English-First RAG**

> Naive RAG pipelines fail silently in Indian production environments. Introducing a translation layer into an English-centric system degrades semantic fidelity—GST provisions, legal constraints, and precise numerical thresholds are systematically distorted. A system reporting 90% accuracy in English but 65% in Hindi is not multilingual; it is an English pipeline with superficial language coverage. In this architecture, cross-lingual alignment and robust Devanagari Unicode handling are treated as first-order requirements, with deployment gated if the Hindi-to-English faithfulness ratio drops below 0.97.

## 🎯 What This System Solves

Standard RAG architectures break down on real-world Indian data. This system explicitly engineers solutions for:

* **Translation-Induced Semantic Loss:** Eliminates the intermediate `Query -> English -> LLM -> Target Language` translation step.
* **Devanagari Processing Noise:** Mitigates Zero-Width Joiner (ZWJ) artifacts, 16 Unicode space variants, and OCR extraction anomalies prior to embedding.
* **Retrieval Failure on Alphanumerics:** Solves the dense embedding blindspot for exact-match terms (e.g., GSTINs, section codes, specific tax amounts) using optimized Reciprocal Rank Fusion (RRF).
* **Language Detection Instability:** Replaces statistically unstable detection models (which misclassify "transformer kya hai" as Norwegian) with a deterministic 7-step script-and-lexicon execution tree.

---

## 🌀 System Overview

The pipeline is non-linear and stateful.

1. **Ingestion:** PDF parsing → 5-Step Indic Preprocessing → Script-Aware Parent/Child Chunking → Dense/Sparse Embedding → Row-Level Security (RLS) PostgreSQL persistence.
2. **Query Processing:** Deterministic Language Detection → Async Multi-Query Expansion (Synonym, HyDE, Step-Back).
3. **Hybrid Retrieval:** Dense (`multilingual-e5-large`) + Sparse (`BM25Okapi`) fused via RRF (k=60) → FlashRank Reranking.
4. **Agentic Reasoning:** 11-node state machine executing context assembly, LLM generation, and self-critique.
5. **Validation & Guardrails:** Post-generation evaluation of faithfulness and language match. Triggers cyclic retries (max 2) before cascading to terminal guardrails (PII redaction, injection detection).

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
<img src="assets/architecture.png" width="900"/>
</p>

### Core Components

* **Foundation Embedding:** `multilingual-e5-large` (1024-dim). Maps 100+ languages into a shared latent space. Passing a strict >0.85 cross-language cosine similarity threshold is the deployment gate.
* **Sparse Indexing:** `BM25Okapi` trained on Indic-normalized text to catch critical alphanumeric exact matches.
* **Reranker:** `FlashRank` (`ms-marco-MiniLM-L-12-v2`) executing on CPU to score top-20 hybrid candidates down to top-5.
* **Generator:** `Sarvam-30B` via streaming endpoint.
* **Orchestration:** `LangGraph` state machine to manage conditional logic, cyclic self-reflection, and API fallbacks.
* **Vector Store:** `pgvector` layered with strictly enforced async transaction-bound Row-Level Security.

---

## 🔄 LangGraph Flow Architecture

<p align="center">
<img src="assets/langgraph_flow.png" width="900"/>
</p>

### Execution & State Management

The system operates as an 11-node directed graph with 3 critical conditional routing edges:

* **Edge 1 (Preprocess):** Evaluates `aioredis` semantic cache (0.95 cosine threshold). Short-circuits the entire graph on cache hit.
* **Edge 2 (Rerank):** Evaluates system confidence. Score > 0.7 triggers standard context assembly. Score 0.3–0.7 triggers **CRAG** (Corrective RAG via Tavily web search). Score < 0.3 triggers early exit (insufficient information).
* **Edge 3 (Critique):** Validates generated output. If `faithfulness_score < 0.75` or `detected_lang != user_lang`, the graph cycles back to retrieval using a refined query. Capped at 2 retries to prevent infinite loops.

---

## 📐 Key Design Decisions

> **📝 Zero Translation Layer**
>
> * **Decision:** Query and retrieve natively in the source language using `multilingual-e5-large`.
> * **Reason:** Legal and financial terminology (e.g., "Input Tax Credit") degrades when translated to English and back.
> * **Trade-off:** Requires a much heavier embedding model and strictly mandated `passage:` and `query:` prefixing, increasing vector dimensionality and compute cost.

> **📝 Script-Aware Parent-Child Chunking**
>
> * **Decision:** Devanagari chunks at 400 tokens; Latin chunks at 500 tokens. Child chunks (256t) for retrieval, Parent chunks (1024t) for LLM context.
> * **Reason:** Devanagari is structurally more information-dense. 500 Devanagari tokens overflow standard context windows and dilute vector focus.
> * **Trade-off:** Introduces script-detection overhead during the ingestion pipeline.

> **📝 Deterministic Language Detection**
>
> * **Decision:** 7-step deterministic logic tree overriding probabilistic detection.
> * **Reason:** Standard NLP libraries fail chaotically on Hinglish and short code-mixed queries. The system forces a 130-word Hinglish lexicon match and an 85% ASCII dominance threshold before trusting probabilistic models.
> * **Trade-off:** Requires manual maintenance of the fallback lexicons.

> **📝 Agentic Workflow over Static Pipeline**
>
> * **Decision:** Implementing a stateful graph (LangGraph) instead of a sequential chain (LCEL).
> * **Reason:** Static pipelines cannot recover from mid-flight hallucinations. The self-critique node requires cyclic graph execution to refine the query and re-retrieve without failing the user request.
> * **Trade-off:** Increased P95 latency compared to single-shot generation.

---

## ⚖️ What Makes This Different

| Standard RAG Systems                                               | SmartDocs Architecture                                                                               |
| :----------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| **English-Biased:** Relies on translation for non-English queries. | **Multilingual-First:** Embeds and retrieves natively in 22 languages.                               |
| **Static Execution:** Linear ingest → retrieve → generate.         | **Stateful/Agentic:** Conditional edges, web fallback (CRAG), and self-critique cycles.              |
| **Naive Text Splitters:** Character or recursive splitting.        | **Indic-Aware:** 5-step preprocessing, ZWJ removal, and script-density-adjusted parent/child chunks. |
| **Blind Delivery:** Returns whatever the LLM spits out.            | **Evaluation Gated:** Post-generation faithfulness validation; retries if threshold (<0.75) fails.   |

---

## ⚠️ Failure Modes & Limitations

* **Noisy OCR Degradation:** The `BM25Okapi` sparse index is highly sensitive to token fragmentation from poor OCR in scanned Indian government PDFs. It can cause the RRF to heavily down-weight critical exact matches.
* **Low-Resource Language Drift:** While robust for Hindi/Marathi/Tamil, embedding alignment drops significantly for extremely low-resource dialects, leading to irrelevant dense retrieval.
* **CRAG Context Overflow:** When Tavily web search is triggered (score 0.3–0.7), the appended web text risks pushing the `assembled_context` beyond the 12,000 character hard-cap, forcing aggressive truncation of the primary PDF context.
* **Multivariate Query Ambiguity:** Extremely short queries (e.g., "tax amount") matching multiple diverse documents will confuse the `FlashRank` reranker, often resulting in an equalized score distribution that incorrectly triggers a web search fallback.

---

## 📊 Evaluation / Metrics

Deployment is gated by strict RAGAS validation. Aggregate accuracy hiding Hindi failures is not accepted.

* **Hindi/EN Faithfulness Ratio:** Target `0.97` (The defining success metric)
* **Hindi Faithfulness:** `> 97%`
* **Language Accuracy (Hindi ↔ English):** `> 95%`
* **Context Precision:** `> 98%`
* **P95 Retrieval Latency:** `< 12000ms` (Hybrid retrieval + RTX GPU)
* **Hallucination Rate:** `< 5%`

---

## 🚀 Quick Start

Get SmartDocs running locally in under 10 minutes.

---

### 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12+ | 3.12 or 3.13 both work |
| uv | latest | `pip install uv` |
| CUDA | 12.1+ | Optional — falls back to CPU automatically |
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

Required for Hindi/Indic sentence tokenization. Without this, chunking fails on Devanagari PDFs.

```bash
git clone --depth 1 https://github.com/anoopkunchukuttan/indic_nlp_resources.git
```

This creates `indic_nlp_resources/` in your project root. The `.gitignore` already excludes it.

---

### 5. Download & Validate Embedding Model

SmartDocs uses `intfloat/multilingual-e5-large` locally — zero API cost.

```bash
# Pre-download the model (~560 MB)
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

**Mandatory validation — do not skip:**

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

> ⚠️ If similarity < 0.85, stop. The entire retrieval pipeline is compromised.

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
APP_SECRET_KEY=run_this_to_generate: python -c "import secrets; print(secrets.token_hex(32))"
ENVIRONMENT=development
EMBEDDING_DEVICE=cuda                       # Use cpu if no NVIDIA GPU
EMBEDDING_BATCH_SIZE=32                     # Reduce to 8 if VRAM < 4GB
HF_TOKEN=                                   # Optional — speeds up model downloads
```

> 💡 **`SARVAM_BASE_URL` must end with `/v1`** — `https://api.sarvam.ai/v1` not `https://api.sarvam.ai`.

> 💡 **Supabase region matters.** Your `DATABASE_URL` hostname must match your project region (e.g. `aws-0-ap-south-1` for Mumbai). Get the exact URL from Supabase dashboard → Settings → Database → Connection Pooling.

---

### 7. Set Up Supabase Database Schema

**Step 7A — Run the base schema:**

Open [Supabase SQL Editor](https://supabase.com) → your project → SQL Editor → paste the contents of `vectorstore/schema.sql` → click **Run**.

Expected: `Success. No rows returned.`

**Step 7B — Run the schema patch (mandatory):**

In the same SQL Editor, paste the contents of `vectorstore/schema_patch.sql` → click **Run**.

> ⚠️ Without `schema_patch.sql`, the UPDATE RLS policy is missing. `total_chunks` will stay 0 after every ingestion — documents appear uploaded but are silently broken.

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

> Redis is optional. If not running, SmartDocs degrades gracefully — queries still work, semantic cache is disabled.

---

### 9. Start the FastAPI Backend

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 --log-level info
```

Expected startup output:
```
SmartDocs API starting up...
✓ pgvector pool connected
✓ DenseEmbedder warmed up on cuda
✓ LangGraph compiled
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

> `redis` shows `degraded` if Redis is not running — this is acceptable. All other checks must be `ok`.

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

1. `preprocess_node`: Detected `hi` (Hindi), classified as `Factual`.
2. `transform_node`: Generated 3 synonym expansions via async HyDE.
3. `retrieve_node`: Hybrid search retrieved parent chunk (400t).
4. `rerank_node`: Scored `0.95` -> Routed to `PROCEED`.
5. `generate_node`: Streamed native Hindi response.
6. `critique_node`: Faithfulness score `0.98` -> Passed.

**Output:**

> सालाना 40 लाख रुपये से अधिक कारोबार पर पंजीकरण अनिवार्य है। [Source: Sample GST Notice, Page 1]

---

## 📂 Repository Structure

```bash
.
├── agents/
│   └── smartdocs_graph.py     # Core 11-node LangGraph state machine definition
├── config/
│   └── settings.py            # Pydantic base settings and environment validation
├── generation/
│   ├── context_assembler.py   # Citation injection and delimiter wrapping
│   ├── sarvam_client.py       # Async streaming implementation
│   └── self_critique.py       # Faithfulness and language match evaluation
├── guardrails/
│   └── output_guardrail.py    # Prompt injection and PII redaction rules
├── observability/
│   ├── cost_tracker.py        # Token counting and INR conversion
│   └── langsmith_tracer.py    # Tag injection (user_id, lang_code, crag_triggered)
├── reranking/
│   └── reranker.py            # FlashRank CPU singleton
├── retrieval/
│   ├── cache.py               # aioredis semantic caching
│   ├── crag_fallback.py       # Tavily corrective RAG
│   ├── hybrid_retriever.py    # E5 dense + BM25 sparse logic fused via RRF
│   ├── language_detector.py   # 7-step deterministic language detection
│   ├── query_classifier.py    # Factual vs analytical routing
│   └── query_transformer.py   # Async HyDE and Step-Back expansion
├── ui/
│   └── components/            # Streamlit modular frontend components
├── app.py                     # Streamlit UI entry point
├── main.py                    # FastAPI application entry point
├── langgraph.json             # CLI execution config for LangGraph Studio
└── pyproject.toml             # uv dependency management
```

---

## 🔍 Observability / Instrumentation

Tracing is non-negotiable (LAW 7). The system relies on LangSmith auto-instrumentation injected at the graph entry point.

* **Mandatory Tags Attached:** `user_id`, `language_code`, `query_type`, `top_reranker_score`, `crag_triggered`, `sarvam_model`.
* **Cost Tracking:** Embedded token counting calculated locally into INR (₹) per session to prevent API bankruptcy.
* **Latency Telemetry:** Monitored per node. High-latency spikes in `transform_node` indicate `asyncio.gather` failure regressions.

---

## 🔧 Extensibility

This is not a standalone demo; it is an infrastructure foundation. The architecture is decoupled to allow rapid swapping of core components:

* **Domain Agnosticism:** Currently configured for Indian legal/tax documents, the 5-step Indic preprocessing pipeline seamlessly scales to Healthcare records, Government Circulars, and Financial disclosures.
* **Model Interchangeability:** `multilingual-e5-large` can be swapped for specialized finetunes. `Sarvam-30B` can be updated to GPT-4o or Claude-3 via standard LangChain ChatModel interfaces.
* **Database Scaling:** The `pgvector` layer is abstracted. Migration to Milvus or Pinecone requires updating only the `vectorstore/` implementations, leaving the LangGraph orchestration untouched.

---

## 🤝 Feedback & Contribution

This architecture is built for practitioners dealing with the messy reality of production multilingual retrieval.

If you identify edge cases in the script-aware chunking boundaries, experience correlation drift in the hybrid fusion weights, or have optimizations for the async query transformation bottlenecks, please submit an issue or a PR with detailed tracing evidence. 

Serious architectural critiques from engineers operating similar systems are highly welcomed.
