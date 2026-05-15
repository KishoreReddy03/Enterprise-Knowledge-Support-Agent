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

## 🛠️ Technical Implementation

### ✂️ Semantic Chunking
Embedding quality cannot compensate for broken chunk semantics. We use **Chunker.py** to handle:
- **Markdown Header Splitting**: Respecting logical structure.
- **Recursive Character Splitting**: Preserving context thread.

### ⚡ Infrastructure & Retrieval
- **DocumentEmbedder**: Handles batch vectorization to maximize throughput.
- **VectorRetriever**: Manages the `pgvector` interface with Neon. While simple audit scripts can count records using raw SQL, the **AI Agents** rely on this retriever to perform high-speed semantic similarity searches.

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