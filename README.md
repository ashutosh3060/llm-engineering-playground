# llm-engineering-playground

**Month 1 · Aug 2026** — Experiment with models, prompts, parameters, latency, and cost — and see the tradeoffs as numbers.

> Depends on [`ai-core`](https://github.com/ashutosh3060/ai-core) — the shared provider gateway, cost accounting, and evaluation primitives used across this portfolio.

> **Status:** 🚧 Scaffolded. Implementation begins Aug 2026.
> Sections 7 (Evaluation Results) and 8 (Demo) are filled in as the work lands — they are
> the point of the repository, not an afterthought.

---

## 1. Problem

Teams pick a model by reputation and a prompt by intuition, then discover the cost and latency consequences in production. There is usually no artifact answering: for *this* workload, which model is good enough, how much does it cost per call, and what does the prompt change actually buy?

## 2. Business Value

Model selection is one of the highest-leverage cost decisions in an AI product. Moving a classification workload from a frontier model to a small one can cut inference cost by 80% with no measurable quality loss — but only if you can measure it. This platform makes that decision evidence-based and repeatable instead of a debate.

## 3. Architecture

```
  User
    |
  Streamlit UI  ── model comparison · prompt lab · token & cost analyzer
    |
  FastAPI       ── /complete  /compare  /count-tokens  /models  (SSE streaming)
    |
  ai_core.gateway
    |
    +--------------+--------------+
    |              |              |
Claude tiers   GPT models*   Open models*      (* if a key is configured)
    |              |              |
    +--------------+--------------+
                   |
    MLflow (experiments) + SQLite (per-request usage)
```

## 4. Technology Choices

| Technology | Why this one |
|---|---|
| **FastAPI** | Async, typed, and SSE streaming out of the box. The UI is one client of the API — not the only possible one. |
| **Streamlit** | The audience is engineers evaluating models, not end users. Streamlit gets a usable comparison UI in hours instead of days. |
| **MLflow** | Prompt/model/parameter sweeps are experiments. MLflow gives run comparison and artifact storage without building a results database. |
| **ai-core** | Provider gateway, pricing table, and usage records. This repo owns the experiment layer, not the transport layer. |

## 5. Design Decisions

### 1. Prompts are versioned by content hash, not filename

A prompt is an experimental variable. Hashing the rendered prompt means a run is always attributable to exactly the text that was sent, including whitespace changes that would otherwise silently invalidate a comparison.

### 2. Every request records tokens, latency, and cost — always

Cost is not a reporting feature bolted on later. If it is not recorded on every call from day one, the historical comparison you want in week 4 does not exist.

### 3. Comparison runs are fanned out concurrently, then aligned

Sequential comparison inflates the latency numbers with queueing. Concurrent dispatch with per-request timing gives latency figures that reflect the provider, not the harness.

### 4. Latency is reported as p50/p95 over repeats, not a single sample

A single call's latency is noise. Anything quoted in the README is a distribution over at least 5 runs.

## 6. Trade-offs

What this project deliberately does **not** do, and why:

- Streamlit, not a production frontend. This is an internal engineering tool; the production UI work happens in Month 5.
- MLflow runs locally against a file store. A tracking server is unnecessary for a single-operator experiment log and adds setup friction for anyone cloning the repo.
- Quality scoring here is deliberately shallow (heuristics + spot LLM-judge). Rigorous scoring is Month 4's job; duplicating it here would fork the eval logic.

## 7. Evaluation Results

> _To be populated during Aug 2026._
> Real, measured numbers only — no estimates. See [`docs/evaluation.md`](docs/evaluation.md)
> for methodology and [`docs/cost-analysis.md`](docs/cost-analysis.md) for the cost breakdown.

## 8. Demo

> _2–4 minute walkthrough — to be recorded at the end of Aug 2026._

## 9. Future Improvements

- Import the Month 4 evaluation platform as the quality scorer so comparison tables carry real faithfulness/accuracy columns.
- Prompt-caching-aware cost modelling — show cost with and without cache hits, since that changes the model-choice calculus.
- Batch-API cost mode for workloads that tolerate latency (50% cheaper on supported providers).

---

## Quickstart

```bash
git clone https://github.com/ashutosh3060/llm-engineering-playground.git
cd llm-engineering-playground

python -m venv .venv && source .venv/bin/activate
make install

cp .env.example .env      # add ANTHROPIC_API_KEY (the only required key)
python -m ai_core.probe   # confirm which providers are reachable
```

Everything except Anthropic is optional — the gateway registers a provider only when its key
is present, and each view renders whatever is available.

## Repository Layout

```
src/playground/    application code
tests/             unit + integration tests
docs/              architecture · design-decisions · evaluation · cost-analysis · future-roadmap
```

## Documentation

- [Architecture](docs/architecture.md)
- [Design Decisions](docs/design-decisions.md)
- [Evaluation](docs/evaluation.md)
- [Cost Analysis](docs/cost-analysis.md)
- [Future Roadmap](docs/future-roadmap.md)

---

Part of a [6-month Product AI Engineer portfolio](https://github.com/ashutosh3060) —
`ai-core` · `llm-engineering-playground` · `enterprise-rag-platform` ·
`multi-agent-ai-platform` · `llm-evaluation-platform` · `production-ai-assistant` ·
`ai-model-router`
