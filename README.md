# Enterprise Knowledge Support Agent 🚀

> *"I used to think the intelligence in a RAG system came from the LLM. After building this, I realized something uncomfortable: A weak retrieval system can make even the best LLM look incompetent."*

This project is a production-grade **Retrieval-Augmented Generation (RAG)** system designed to solve complex developer support workflows for the Stripe ecosystem. It moves beyond static PDF demos into a continuously evolving knowledge infrastructure.

---

## 🏗️ Current Directory Structure

This represents the exact files and folders currently in the project:

```text
├── api/                   # FastAPI route endpoints & analytics pipelines
│   └── routes/            # tickets.py, analytics.py, evaluation.py, feedback.py, health.py
├── core/
│   ├── agents/            # Multi-agent mesh: Intake, Retrieval, Synthesis, Drafting, Quality Gate, Escalation, state.py, orchestrator.py
│   ├── guardrails/        # Security: circuit_breaker.py, grounding_verifier.py, input_guard.py, output_guard.py
│   ├── ingestion/         # Data: chunker.py, embedder.py, scraper.py, scheduler.py
│   ├── intelligence/      # Learning: few_shot_selector.py (dynamic few-shot drift mitigation)
│   └── retrieval/         # Search: vector_retriever.py, reranker.py (Cross-Encoder)
├── data/                  # Few-shot anchors & benchmark ticket data
├── database/              # SQL pgvector migrations & setups
├── scripts/               # check_db_counts.py, run_ingestion.py, E2E test files
├── .env                   # Local environment secrets
├── .gitignore             # GitHub exclusion rules
├── agent_workflow_diagram.md # Technical workflow diagram and guide
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

### 🎯 Phase 3: Cross-Encoder Reranking
*   **Joint Attention Ranking**: Fused candidates are evaluated by a deep local Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`). It scores query-document pairs by evaluating full semantic relationships, delivering near-human accuracy in final ranking.

---

## 🤖 Multi-Agent Orchestration & Reliability Mesh (LangGraph)

The core ticket processing lifecycle is orchestrated as an asynchronous multi-agent graph with dynamic feedback loops and state-tracking. 

For the complete visual breakdown, export guidelines, and a step-by-step flowchart terminology guide, see the [Agent Workflow Diagram](agent_workflow_diagram.md).

```text
                        ┌─────────── Ticket Received ───────────┐
                        │                                       │
                        ▼                                       ▼
            [Input Security Guardrail]               [Circuit Breaker Check]
                        │                                       │
                        ▼                                       ▼
             [Semantic Cache Check] ─────────────────> (Cache Fast Return)
                        │ (Cache Miss)
                        ▼
             [Intake Agent (Classifier)] ────────────> (Confidence Escalation)
                        │
                        ▼
            [Retrieval Agent (Budget)] ──────────────> [Cross-Encoder Reranker]
                        │                                       │
                        ├───────────────────────────────────────┘
                        ▼
            [Synthesis Agent (Resolve)] ─────────────> [Drafting Agent (Few-Shot)]
                        │                                       │
                        ▼                                       ▼
            [Quality Gate (NLI + Override)] <───────── (Self-Correction Loop)
                        │
                        ▼
            [Citation Attribution Verifier] ─────────> [Output Security Guardrail]
                        │                                       │
                        ▼                                       ▼
             (Downstream Ingestion Log) ─────────────> (Customer Sent)
```

### 1. The Core Support Agents
*   **Intake Agent**: Classifies query topics, complexity tiers (simple, moderate, complex), and ticket urgency to set the initial trajectory. Decoupled via a pluggable routing interface to support standard routing rules or dynamic, tier-sensitive backends like the `ConfidenceAdaptiveRouter`.
*   **Retrieval Agent**: Dynamically scales search depth (retrieval budget) based on ticket complexity, fetching coordinates across docs, changelogs, GitHub, and StackOverflow.
*   **Synthesis Agent**: Fuses search context and resolves direct technical conflicts. It applies a **Strict Source Trust Hierarchy** (Changelogs overrule standard Docs, which overrule developer forums) and temporal freshness checks. If a critical contradiction remains unresolved or search results are blank, it initiates auto-escalation.
*   **Drafting Agent**: Generates highly targeted candidate replies. Employs a **Dynamic Few-Shot Selector** to query historical benchmark examples, shielding the model against prompt drift. It reads active state annotations to self-correct during revisions.
*   **Quality Gate Agent**: Automated compliance inspector. Runs deterministic safety checklists and handles the factual verifications.
*   **Escalation Agent**: Graceful safety valve. Compiles structural markdown brief logs detailing ticket state, classification logs, and failure details for human support engineers in case of persistent errors.

### 2. LLM-as-Judge Grounding & Deterministic Token Override Gate
Standard LLM evaluators are highly susceptible to sycophancy (approving fluent but hallucinated answers). To enforce physical truth, we integrated the **Deterministic Token Verification Gate** directly into the post-generation **Output Guardrail** layer:
*   **Factual NLI Analysis**: Scans drafted segments sentence-by-sentence using Natural Language Inference (NLI) to flag claims not semantically supported by retrieved contexts.
*   **Verbatim Token Check**: Extracts snake_case API parameters, URL paths, endpoints, and camelCase headers using custom regex, verifying their exact physical presence inside the source chunks.
*   **Hard Overrule**: If a technical parameter or endpoint is hallucinated or missing from the source context, the Token Gate **forcefully overrules the LLM judge's verdict**, drops the grounding score to `0.0`, and initiates self-correction (or human escalation).

### 3. Citation-Level Attribution Enforcement
Cross-references every inline citation marker (e.g. `[1]`, `[2]`) against the verified NLI grounding metadata map. It verifies that the exact sentence asserting a claim is physically backed by the cited source chunk ID. If citations are swapped, mismatched, or out-of-bounds, the verifier flags a `CitationMismatch` to trigger revision.

### 4. Threshold Calibration & Routing Intelligence Status
> [!NOTE]
> **Heuristic-Driven Foundations**: There is **no learned evaluation calibration yet** in the system. The current routing thresholds (e.g., classification confidence `< 0.60` or quality scores `< 0.60`) are purely **heuristic-driven**.
>
> **Eventually**, we will implement **intelligent evaluation calibration** using:
> 1. **Offline Evals**: Multi-variant prompt sweeps and regression datasets to identify optimal score boundaries.
> 2. **Human Feedback**: engineer triage evaluations and human-in-the-loop validation labels.
> 3. **Historical Outcomes**: Long-term database performance metrics correlating quality-gated ratings against downstream ticket reopen rates.

---

## ⚡ API Services (`api/`)

Built on FastAPI, the API layer provides high-performance telemetry, ticket orchestration, and analytical feedback routes:
*   `POST /api/tickets/process`: Receives developer tickets, executes the LangGraph multi-agent loop asynchronously, and returns cached fast tracks or formatted agent/human escalation JSON.
*   `POST /api/feedback/submit`: Registers developer validation feedback (thumbs-up/down) and updates the local Redis cache.
*   `POST /api/evaluation/run`: Triggers parallel evaluation scorecards over query benchmarks, logging retrieval precision and citation-level grounding details.
*   `GET /api/analytics/summary`: Exposes core metrics, average latency distributions, grounding quality tracking, cache hit rates, and common knowledge gaps.

---

## ✂️ Semantic Chunking
Embedding quality cannot compensate for broken chunk semantics. We use `chunker.py` to handle:
- **Markdown Header Splitting**: Respecting logical document hierarchy.
- **Recursive Character Splitting**: Preserving the contextual thread within boundaries.

---

## 🚀 Getting Started

### 1. Installation
```bash
# Clone and enter directory
cd Enterprise-Knowledge-Support-Agent

# Set up virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify neon Database
Ensure your Neon PostgreSQL is connected and holds seeded records:
```bash
python scripts/check_db_counts.py
```

### 3. Run Ingestion Pipelines
To index standard API documentation, changelogs, and forum issues:
```bash
python scripts/run_ingestion.py
```

### 4. Execute the Verification Test Suite
We built a highly thorough validation suite to verify the multi-agent orchestration, cache layers, routing systems, and guardrails:
```bash
# 1. Test Intake Agent & Caching layers
python scripts/test_intake_caching.py

# 2. Test Pluggable Adaptive Routers (ConfidenceAdaptiveRouter)
python scripts/test_pluggable_router.py

# 3. Test Retrieval Agent Budget Allocation
python scripts/test_retrieval_agent.py

# 4. Test Contradiction Resolution & Trust Weighting Rules
python scripts/test_contradiction_escalation_mitigation.py

# 5. Test Dynamic Few-Shot Selector & Drift Mitigation
python scripts/test_few_shot_drift_mitigation.py

# 6. Test Quality Gate Centering & Robustness
python scripts/test_quality_gate_robustness.py

# 7. Test E2E Grounding Loops & Deterministic Token Override Gates
python scripts/test_grounding_loop_e2e.py
```