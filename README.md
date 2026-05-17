# Enterprise Knowledge Support Agent 🚀

> *"I used to think the intelligence in a RAG system came from the LLM. After building this, I realized something uncomfortable: A weak retrieval system can make even the best LLM look incompetent."*

This project is a production-grade **Retrieval-Augmented Generation (RAG)** system designed to solve complex developer support workflows for the Stripe ecosystem. It moves beyond static PDF demos into a continuously evolving knowledge infrastructure.

---

## 🏗️ Current Directory Structure

This represents the exact files and folders currently in the project:

```text
├── core/
│   ├── ingestion/         # chunker.py, embedder.py, scraper.py
│   └── retrieval/         # vector_retriever.py (The search engine)
├── database/              # SQL migrations (pgvector setup)
├── scripts/               # check_db_counts.py, seed_neon.py
├── .env                   # Local environment secrets
├── .gitignore             # GitHub exclusion rules
├── config.py              # Centralized settings & validation
├── main.py                # FastAPI entry point
└── requirements.txt       # Project dependencies
```

---

## 🧠 Philosophy: Retrieval > LLM

In production, the challenge isn't just generating text—it's preventing your knowledge system from becoming outdated, noisy, and semantically unreliable. This architecture focuses on **retrieval quality** and **knowledge freshness** as the primary engineering goals.

### 🛡️ Why OpenAPI Over HTML?
For the ingestion layer, we avoided traditional HTML scraping for Stripe Docs and utilized the **OpenAPI specification** instead. 
- **Machine-readable** & Structured
- **Versionable** and significantly more stable
- **Predictable** for downstream retrieval systems

### 🛰️ The "Production Failure" Knowledge Loop
Official documentation explains *intended* behavior, but real production failures live in the cracks. This system bridges that gap by ingesting:
- **Stripe Docs & Changelog**: The ground truth
- **GitHub Issues**: Real-world bugs and workaround discussions
- **StackOverflow**: Edge-case debugging and community solutions

---

## 🛠️ High-Precision 3-Phase Retrieval Engine

To guarantee absolute relevance and data trust, this repository implements a state-of-the-art, multi-stage hybrid search pipeline:

```text
User Query ──> [1. Query Classifier] ──> Parallel Search (pgvector + FTS)
                                                     │
                                                     ▼
               [3. Cross-Encoder] <── [2. Trust Boost & Age Decay]
                       │
                       ▼
             Top-k Precision Chunks
```

### 🧠 Phase 1: Query Classification & Hybrid Search
*   **Adaptive Intent Classification**: Classifies queries into categories (such as `API`, `Problem`, or `Conceptual`) on-the-fly to calibrate optimal search weights.
*   **Parallel Hybrid Matcher**: Concurrently executes semantic vector search (via `pgvector`) and keyword-focused Full-Text Search (FTS) with advanced English stopword filtering.
*   **Reciprocal Rank Fusion (RRF)**: Merges the candidates dynamically, prioritizing keyword matching for syntactical query intents and semantic vector similarity for conceptual intents.

### ⚖️ Phase 2: Metadata-Aware Ranking (Trust & Freshness)
*   **Source Trust Boosts**: Hard-codes corporate authority tiers by applying score boosts to official Stripe Docs and minor penalties to unverified community forum posts (StackOverflow).
*   **Soft Freshness (Temporal Decay)**: Applies a gentle temporal penalty based on a document's age in days. Highly relevant older core documents remain retrievable, while newer matching pages win close tie-breakers.
*   **Hard Freshness Filters**: Strictly blocks officially deprecated or archived articles (`is_stale = TRUE`) at the SQL layer.

### 🎯 Phase 3: Cross-Encoder Reranking
*   **Joint Attention Ranking**: Fused candidates are evaluated by a deep local Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`). It scores query-document pairs by evaluating full semantic relationships, delivering near-human accuracy in final ranking.

---

## ✂️ Semantic Chunking
Embedding quality cannot compensate for broken chunk semantics. We use `chunker.py` to handle:
- **Markdown Header Splitting**: Respecting logical document hierarchy.
- **Recursive Character Splitting**: Preserving the contextual thread within boundaries.


---

## 🚀 Getting Started

### 1. Installation
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Audit Your Knowledge
Run the audit script to verify your Neon connection and existing data:
```bash
python scripts/check_db_counts.py
```

### 3. Run Sample Ingestion
If you need to seed new data:
```bash
python scripts/seed_neon.py
```