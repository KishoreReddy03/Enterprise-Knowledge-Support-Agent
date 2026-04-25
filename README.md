# Stripe Support Knowledge Agent

## Internal AI assistant for support reps

An enterprise-grade RAG system that drafts support responses by combining Stripe documentation, GitHub issues, and StackOverflow answers. Built with a 6-agent pipeline that catches hallucinations before they reach customers, reducing resolution time by [METRIC]% while maintaining [METRIC]% faithfulness to source material.

---

## Measured Results

| Metric | Result |
|--------|--------|
| Avg resolution time | [METRIC] seconds |
| Manual baseline | 510 seconds (8.5 min) |
| Time reduction | [METRIC]% |
| Retrieval Precision@3 | [METRIC]% |
| Answer Faithfulness | [METRIC]% |
| Out-of-scope rejection | [METRIC]% |
| Escalation accuracy | [METRIC]% |
| Cost per query | $[METRIC] |
| p95 latency | [METRIC] seconds |

*Evaluated against 100 manually-verified test cases across 7 adversarial categories.*

---



1. **Quality gate before human review, not after.** Most RAG systems show the LLM output directly to users. I added a dedicated Haiku-based evaluation agent that checks every response against 4 criteria (answers the question, claims have sources, no hallucinated API behavior, appropriate for customer tier) before it reaches the support rep. Responses scoring below 0.8 loop back for revision. This catches ~[METRIC]% of problematic responses before they waste rep time.

2. **Confidence scores that actually mean something.** The synthesis agent calculates confidence from three signals: retrieval relevance scores, source agreement, and information completeness. A response with confidence 0.72 has different failure modes than one at 0.45. The system uses this to route: high confidence drafts go straight to reps, medium confidence gets extra rep guidance, low confidence escalates with context for human investigation.

3. **Contradiction detection across sources.** Stripe's documentation sometimes conflicts with community answers, especially for recently-changed APIs. The synthesis agent explicitly prompts Claude to identify contradictions between sources before drafting. When detected, it flags the newer source (via timestamp) and includes this in rep guidance rather than silently picking one.

4. **Cost control through model routing.** Sonnet is 10x the cost of Haiku. The pipeline uses Haiku for classification (intake), sufficiency checks (synthesis), and quality evaluation (quality gate) — tasks where the cheaper model performs comparably. Sonnet only runs for the actual response generation (drafting) and contradiction detection. This keeps cost per query under $[METRIC] while maintaining quality.

5. **Learning from rep corrections.** Every time a rep edits a response before sending, the feedback loop captures and classifies the edit (factual correction, tone adjustment, completeness issue). Factual corrections trigger source flagging in the vector store. Weekly pattern analysis identifies documentation gaps by finding topics with consistently low confidence scores.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────── ┐
│                           TICKET PROCESSING PIPELINE                         │
├───────────────────────────────────────────────────────────────────────────── ┤
│                                                                              │
│   ┌──────────┐     ┌─────────────┐     ┌───────────┐     ┌──────────┐        │
│   │  INTAKE  │────▶│  RETRIEVAL  │────▶│ SYNTHESIS │────▶│ DRAFTING │      │
│   │ (Haiku)  │     │  (Parallel) │     │  (Haiku)  │     │ (Sonnet) │        │
│   └──────────┘     └─────────────┘     └───────────┘     └──────────┘        │
│        │                  │                  │                  │            │
│        │                  │                  │                  ▼            │
│        │           ┌──────┴──────┐           │          ┌─────────────┐      │
│        │           │   4 SOURCES │           │          │QUALITY GATE │      │
│        │           ├─────────────┤           │          │   (Haiku)   │      │
│        │           │ • Stripe    │           │          └─────────────┘      │
│        │           │   Docs      │           │                  │            │
│        │           │ • GitHub    │     ┌─────┘                  │            │
│        │           │   Issues    │     │ need_more        ┌─────┴─────┐      │
│        │           │ • Stack     │     │ (max 1x)         │           │      │
│        │           │   Overflow  │     ▼                  ▼           ▼      │
│        │           │ • Changelog │ (retry)           approved     revise     │
│        │           └─────────────┘                       │       (max 2x)    │
│        │                                                 │           │       │
│        ▼                                                 ▼           │       │
│   [escalate]────────────────────────────────────▶ ┌───────────┐ ◀────┘      │
│                                                   │ ESCALATION│              │
│                                                   │  (Haiku)  │              │
│                                                   └───────────┘              │
│                                                         │                    │
│                                                         ▼                    │
│                                                  ┌─────────────┐             │
│                                                  │   OUTPUT    │             │
│                                                  │  FORMATTER  │             │
│                                                  └─────────────┘             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            KNOWLEDGE BASE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                 │
│   │   QDRANT    │      │   QDRANT    │      │   QDRANT    │                 │
│   │ stripe_docs │      │github_issues│      │stackoverflow│                 │
│   │  ~15k docs  │      │  ~8k issues │      │ ~12k answers│                 │
│   └─────────────┘      └─────────────┘      └─────────────┘                 │
│          ▲                    ▲                    ▲                        │
│          │                    │                    │                        │
│   ┌──────┴────────────────────┴────────────────────┴──────┐                 │
│   │              HYBRID RETRIEVER (RRF k=60)              │                 │
│   │         Vector Search + BM25 Keyword Search           │                 │
│   └───────────────────────────────────────────────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Responsibilities

| Agent | Model | Purpose | Key Decision |
|-------|-------|---------|--------------|
| **Intake** | Haiku | Classify ticket, extract entities | Route simple vs complex, detect escalation triggers |
| **Retrieval** | — | Parallel search across 4 sources | Hybrid search with RRF merging |
| **Synthesis** | Haiku | Evaluate context sufficiency | Detect contradictions, request more info if needed |
| **Drafting** | Sonnet | Generate customer-appropriate response | Tier-aware language, proper source citation |
| **Quality Gate** | Haiku | Pre-send evaluation | Score 0-1, approve/revise/escalate routing |
| **Escalation** | Haiku | Prepare human handoff | Generate brief with context for human agent |

---

## Adversarial Test Results

| Category | Test Cases | Pass Rate | Notes |
|----------|------------|-----------|-------|
| Standard webhooks | 20 | [METRIC]% | Baseline capability |
| Error codes | 20 | [METRIC]% | Real Stripe error codes |
| Deprecated APIs | 10 | [METRIC]% | Must redirect to PaymentIntents |
| Out-of-scope | 15 | [METRIC]% | Must decline and redirect |
| Contradiction detection | 10 | [METRIC]% | Seeded conflicting sources |
| Missing information | 10 | [METRIC]% | Must acknowledge gap, not hallucinate |
| Multi-source reasoning | 15 | [METRIC]% | Requires combining docs + GitHub + SO |

**Key insight:** [METRIC] category had lowest pass rate because [OBSERVATION]. This revealed [LEARNING] which I addressed by [CHANGE].

---

## Technical Decisions and Trade-offs

### 1. Retrieval Strategy: Hybrid Search with RRF

**What I chose:** Reciprocal Rank Fusion (RRF) with k=60 combining semantic vector search and BM25 keyword search.

**What I considered:**
- Pure semantic search (simpler, but misses exact terminology)
- Query expansion with LLM (better recall, but adds latency and cost)
- Learned sparse embeddings like SPLADE (better performance, but requires training data)

**Why this choice:** Support tickets often contain exact error codes like `card_declined` or API paths like `/v1/payment_intents`. Pure semantic search ranks these poorly because embeddings don't capture lexical similarity. BM25 catches exact matches while vectors handle conceptual similarity. RRF with k=60 is a well-studied constant that balances the two without tuning. Trade-off: slightly lower precision than a tuned learned ranker, but zero training data required and deterministic behavior.

### 2. Loop Prevention: Hard Limits Over Heuristics

**What I chose:** Fixed retry limits (1 retrieval retry, 2 revision retries) enforced in state.

**What I considered:**
- LLM-based "should we retry?" decisions (more flexible but unpredictable)
- Exponential backoff based on confidence delta (elegant but complex to debug)
- No retries, escalate immediately on failure (simple but wasteful)

**Why this choice:** In production, infinite loops are an existential risk. An LLM deciding whether to retry creates non-deterministic control flow that's nearly impossible to debug at 2 AM when the queue backs up. Fixed limits mean the worst case is predictable: max 4 LLM calls per ticket (intake → drafting → quality → revision → quality). Trade-off: occasionally escalates tickets that one more retry might have solved, but this is preferable to unpredictable latency.

### 3. Confidence Scoring: Composite Signals Over Single Score

**What I chose:** Confidence from three weighted factors: mean retrieval relevance (40%), source agreement (35%), information completeness (25%).

**What I considered:**
- LLM self-reported confidence (unreliable, models are poorly calibrated)
- Embedding distance threshold only (misses agreement/completeness)
- Separate scores per dimension (more information but harder routing logic)

**Why this choice:** Retrieval relevance alone doesn't capture whether sources agree or provide complete coverage. A query might retrieve 5 highly-relevant documents that all say different things — high retrieval confidence but low actual reliability. The composite score correlates better with whether reps actually edit responses. Trade-off: requires tuning weights based on feedback data; initial weights are educated guesses that need validation against [METRIC] hours of production data.

---

## What Failed and What I Learned

### Failure 1: Initial prompts caused over-escalation

**What happened:** First version escalated ~40% of tickets. The intake prompt said "escalate if you're uncertain about anything." Haiku interpreted this conservatively and escalated on any ambiguity.

**What I changed:** Rewrote intake prompt with explicit escalation triggers: account access issues, billing disputes, legal/compliance mentions, explicit rep requests. Changed from "uncertain = escalate" to "specific trigger = escalate, uncertain = proceed with low confidence."

**Result:** Escalation rate dropped to [METRIC]%. Lesson: LLMs follow instructions literally. "Be conservative" means something different to a model than to a human. Explicit enumeration beats implicit heuristics.

### Failure 2: Quality gate approved hallucinated API parameters

**What happened:** Early quality gate prompt asked "does the response contain hallucinations?" The model would say "no" even when the response mentioned non-existent API parameters — because it didn't have the API spec to check against.

**What I changed:** Restructured quality gate to check a more tractable condition: "does every technical claim in the response have a matching source citation?" This doesn't catch all hallucinations, but it catches claims the model made up without source support. Added `must_not_hallucinate` field to test cases for known false claims.

**Result:** Faithfulness score improved from ~0.72 to [METRIC]. Lesson: "detect hallucinations" is too abstract for reliable execution. "Check that claims match citations" is concrete and verifiable.

### Failure 3: Contradiction detection missed timestamp-based conflicts

**What happened:** When a Stack Overflow answer from 2021 conflicted with a Stripe doc from 2024, the system sometimes chose the older source because it had higher retrieval relevance (more detailed answer).

**What I changed:** Added `last_updated` to chunk metadata during ingestion. Synthesis prompt now explicitly says "when sources conflict, note the conflict and prefer information from the more recent source." Contradiction output includes timestamps.

**Result:** Deprecated API detection improved to [METRIC]%. Lesson: relevance ≠ correctness. Temporal signals matter for APIs that evolve. Should have designed for freshness from day one.

---

## Running Locally

### Prerequisites

- Python 3.11+
- Docker (for local Qdrant)
- Accounts: Anthropic, Supabase, Qdrant Cloud (or local), Upstash Redis, Langfuse

### Setup

```bash
# Clone and enter directory
git clone https://github.com/KishoreReddy03/Enterprise-Knowledge-Support-Agent.git
cd Enterprise-Knowledge-Support-Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template and fill in values
cp .env.example .env
# Edit .env with your API keys

# Run database migrations (Supabase)
# Apply database/migrations/001_initial_schema.sql via Supabase dashboard

# Start local Qdrant (optional, can use Qdrant Cloud)
docker run -p 6333:6333 qdrant/qdrant

# Ingest initial data (takes ~30 minutes)
python -m core.ingestion.scheduler

# Start the API server
uvicorn api.main:app --reload
```

### Verify installation

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Should return:
# {"status":"healthy","qdrant":"connected","supabase":"connected",...}

# Process a test ticket
curl -X POST http://localhost:8000/api/v1/tickets/process \
  -H "Content-Type: application/json" \
  -d '{"ticket_content": "Why am I getting card_declined errors?", "customer_tier": "standard"}'
```

---

## Evaluation

### Running the full evaluation suite

```bash
# Full evaluation (100 cases, ~30-45 minutes)
python -m evaluation.run_full_eval

# Quick test (5 cases per category, ~5 minutes)
python -m evaluation.run_full_eval --quick

# Single category
python -m evaluation.run_full_eval --category webhook

# Limited cases
python -m evaluation.run_full_eval --limit 10

# Export results
python -m evaluation.run_full_eval --output results.json
```

### Interpreting results

| Metric | Target | What it measures |
|--------|--------|------------------|
| Pass rate | >85% | Cases passing all criteria |
| Retrieval P@3 | >80% | Correct source types in top 3 |
| Faithfulness | >90% | No hallucinated claims |
| Mention compliance | >90% | Required terms present |
| Escalation accuracy | >95% | Correct escalation decisions |

**Category-specific expectations:**
- `webhook`, `error_codes`: Should have high pass rates (>90%) — bread and butter cases
- `deprecated_api`: Moderate pass rate (>80%) — requires detecting outdated patterns
- `out_of_scope`: Very high pass rate (>95%) — must reliably reject
- `contradiction_detection`: Moderate (>75%) — hardest category, requires nuanced reasoning
- `missing_information`: High (>85%) — must acknowledge gaps, not fabricate

### Adding new test cases

Edit `evaluation/ground_truth.json`:

```json
{
  "id": "new_case_001",
  "ticket": "Your test ticket content here",
  "category": "webhook",
  "difficulty": "standard",
  "expected_topics": ["topic1", "topic2"],
  "must_mention": ["required term"],
  "must_not_hallucinate": ["false claim to reject"],
  "acceptable_confidence_range": [0.75, 0.95],
  "source_should_include": ["stripe_doc"],
  "escalation_expected": false
}
```

---

## Project Structure

```
stripe_support_agent/
├── api/
│   ├── main.py                  # FastAPI application
│   └── routes/
│       ├── tickets.py           # POST /tickets/process
│       ├── feedback.py          # POST /feedback
│       ├── analytics.py         # GET /analytics/roi, /patterns
│       └── health.py            # GET /health
│
├── core/
│   ├── agents/
│   │   ├── state.py             # TicketState TypedDict
│   │   ├── intake.py            # Ticket classification
│   │   ├── retrieval_agent.py   # Parallel source retrieval
│   │   ├── synthesis.py         # Context evaluation
│   │   ├── drafting.py          # Response generation
│   │   ├── quality_gate.py      # Pre-send evaluation
│   │   ├── escalation.py        # Human routing
│   │   └── orchestrator.py      # LangGraph master graph
│   │
│   ├── ingestion/
│   │   ├── scrapers.py          # Data collection
│   │   ├── chunker.py           # Semantic chunking
│   │   ├── embedder.py          # Embedding + storage
│   │   └── scheduler.py         # Ingestion orchestration
│   │
│   ├── retrieval/
│   │   ├── vector_retriever.py  # Qdrant search
│   │   └── hybrid.py            # RRF hybrid search
│   │
│   └── intelligence/
│       ├── pattern_detector.py  # Weekly analysis
│       ├── feedback_loop.py     # Rep edit capture
│       └── roi_calculator.py    # Business metrics
│
├── evaluation/
│   ├── ground_truth.json        # 100 test cases
│   └── run_full_eval.py         # Evaluation runner
│
├── database/
│   └── migrations/              # SQL schemas
│
├── config.py                    # Environment configuration
├── requirements.txt
└── README.md
```



## License

MIT





