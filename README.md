# llm-engineering-playground

**Month 1 · Aug 2026** — Compare LLMs on quality, latency, and cost, and see the tradeoffs as numbers.

> Depends on [`ai-core`](https://github.com/ashutosh3060/ai-core) — the shared provider
> gateway, versioned cost accounting, and shared types used across this portfolio.

**Runs with no API key.** An offline mock provider is built in, so `make install &&
playground bench` works on a clean clone. Add `ANTHROPIC_API_KEY` when you want real numbers.

---

## 1. Problem

Teams pick a model by reputation and a prompt by intuition, then discover the cost and
latency consequences in production. There is usually no artifact answering the only
question that matters: **for this specific workload, which model is good enough, what does
it cost per call, and what did that prompt change actually buy?**

## 2. Business Value

Model selection is one of the highest-leverage cost decisions in an AI product. Moving a
classification workload from a frontier model to a small one can cut inference cost
substantially with no measurable quality loss — but only if you can measure it. Without
evidence the choice defaults to the biggest model, which is the expensive answer to a
question nobody asked.

This makes that decision evidence-based and repeatable rather than a debate.

## 3. Architecture

```
                    ┌──────────────┐   ┌──────────────┐
                    │ Streamlit UI │   │  playground  │
                    │   (4 tabs)   │   │     CLI      │
                    └──────┬───────┘   └──────┬───────┘
                           └────────┬─────────┘
                          ┌─────────▼─────────┐
                          │  FastAPI gateway  │  /complete /compare /count-tokens
                          └─────────┬─────────┘  /models /providers /runs /spend
              ┌─────────────────────▼─────────────────────┐
              │            playground.runtime             │
              │  one gateway + one store for every entry  │
              │  point, so "what's available" answers     │
              │  the same everywhere                      │
              └──────┬─────────────────────────┬──────────┘
          ┌──────────▼──────────┐   ┌──────────▼──────────┐
          │   ai_core.Gateway   │   │  playground.Store   │
          │  provider registry  │   │  SQLite: runs +     │
          │  + concurrent fanout│   │  results            │
          └──────────┬──────────┘   └─────────────────────┘
       ┌─────────────┼─────────────┐
  Anthropic     OpenAI*       mock (offline)
       └─────────────┴─────────────┘
        ai_core: versioned pricing · cost accounting · token counting

  * registered only when a key is present
```

## 4. Technology Choices

| Technology | Why this one |
|---|---|
| **FastAPI** | Async, typed, SSE-capable. The UI is one client of the API — not the only possible one. |
| **Streamlit** | The audience is engineers evaluating models, not end users. A usable comparison UI in hours instead of days. |
| **SQLite** | Every call's tokens, latency, and cost land somewhere queryable from day one. Schema is Postgres-portable. |
| **Typer + Rich** | Benchmarks belong in a terminal and in CI, not only behind a UI. |
| **MLflow** *(optional)* | Run-to-run comparison in a UI. An extra, not a dependency, and every failure inside it degrades to a logged warning — a tracking hiccup must never destroy a benchmark you already paid for. |
| **ai-core** | Provider gateway, pricing table, cost accounting. This repo owns the experiment layer, not the transport layer. |

## 5. Design Decisions

Eleven ADRs in [`docs/design-decisions.md`](docs/design-decisions.md). The five that shape
everything else:

### Prompts are versioned by content hash, not filename
A prompt is an experimental variable. Hashing the *rendered* text — deliberately
whitespace-sensitive — means a run is always attributable to exactly the bytes that were
sent. Filename-based versioning lets two runs claim to share a prompt while sending
different text, and the failure is silent.

### Token counts come from the provider, never `tiktoken`
`tiktoken` is OpenAI's tokenizer. Against Claude it undercounts by roughly 15–20% on prose
and more on code. Cost estimates built on it are wrong in a direction that only surfaces
on the invoice.

### Cost comes from a versioned pricing table with effective dates
Claude Sonnet 5's introductory rate lapses 2026-08-31. A hardcoded float would keep
reporting the promotional price afterwards *and* retroactively corrupt earlier analysis.
`price_at(model, date)` resolves the band that applied when the request ran. A test asserts
the lapse, so the mechanism cannot rot unnoticed.

### Latency is always p50/p95 over repeats, never a single sample
One measurement is noise — cold starts and provider load move it by multiples. Quoting one
would be indistinguishable from making it up. Everything reported is a nearest-rank
percentile over ≥5 runs, from a single shared implementation.

### Sweeps refuse to run oversized grids
Grids grow multiplicatively and every cell is a paid call. Three axes of five values with
five repeats is 625 calls — easy to request by accident. `SweepTooLarge` fires above 200,
counts repeats in the total, and names the real number.

## 6. Prior Art

This problem is solved, well, by mature tools. Anyone evaluating this repo should
know that, and so should I.

| Tool | Overlap with this project |
|---|---|
| **[Promptfoo](https://promptfoo.dev)** | Near-exact superset. YAML test cases, model comparison across providers, assertions including cost thresholds and latency limits, LLM-graded evals, CI integration, red teaming. 350k+ developers, 25%+ of the Fortune 500, and **being acquired by OpenAI as of March 2026**. |
| **[LiteLLM](https://litellm.ai)** | Superset of `ai-core` and this repo's gateway. 140+ providers, per-key/team/org budgets, semantic caching, lowest-cost and auto routing. |
| **[Langfuse](https://langfuse.com)**, **[Braintrust](https://braintrust.dev)**, **[Arize Phoenix](https://phoenix.arize.com)**, **[DeepEval](https://deepeval.com)** | Overlap the eval, tracing, and regression-testing surface, each from a different angle. |

**If you need this in production, use Promptfoo.** It is more capable, better
tested, and maintained by people who work on it full time. Nothing here is a
reason to choose this over it.

So why does this exist? Because the reasoning does not transfer by reading docs.
Working out first-hand *why* latency has to be a percentile over repeats, *why*
`tiktoken` silently corrupts Claude cost estimates, *why* a pricing table needs
effective dates, and *why* a sweep needs a spend guard produces understanding that
using a finished tool does not. The eleven ADRs in
[`docs/design-decisions.md`](docs/design-decisions.md) are the actual output of this
project; the tool is the thing that forced them to be answered.

Two things here I have not seen elsewhere, offered as observations rather than
claims of novelty:

- **Pricing bands with effective dates.** Most cost trackers hold a current price
  per model. This resolves the band that applied *when the request ran*, so a rate
  change does not retroactively corrupt an earlier analysis, and an introductory
  rate expires on schedule rather than silently persisting.
- **The offline mock as a first-class provider.** The full application — API, UI,
  benchmark harness, tests — runs with no API key, which makes the repo genuinely
  clonable and the test suite genuinely hermetic.

## 7. Trade-offs

What this project deliberately does **not** do:

- **Shallow quality scoring.** Deterministic scorers only — exact, label, JSON-structural,
  numeric. LLM-as-judge with calibration, faithfulness, and regression gating is Month 4's
  `llm-evaluation-platform`. Building a second version here would fork the logic and the
  two would drift.
- **Streamlit, not a production frontend.** No auth, no multi-user state. That work is
  Month 5.
- **MLflow against a local file store.** A tracking server adds setup friction for anyone
  cloning the repo and buys nothing for a single operator.
- **No response caching.** Repeating a benchmark re-pays for every call. Caching results by
  default would mask genuine run-to-run variance, which is the thing being measured.
- **Synthetic output can reach the UI.** The mock provider is the price of a keyless clone.
  Mitigated by labelling at every exit — CLI warning, UI banner, `SYNTHETIC` in the price
  note, and an explicit statement in `docs/evaluation.md`.

## 8. Evaluation Results

> **Pending a live API key.** The harness is complete and verified end-to-end; the tables in
> [`docs/evaluation.md`](docs/evaluation.md) are deliberately empty because filling them
> requires real calls. Nothing is estimated or extrapolated — an empty table is more useful
> than a fabricated one.

Reproduce in two commands once a key is set (~450 calls, well under $1):

```bash
playground bench datasets/sentiment-classification.yaml \
  -m claude-haiku-4-5,claude-sonnet-5,claude-opus-5 -r 5 -o docs/results-sentiment.md
playground bench datasets/structured-extraction.yaml \
  -m claude-haiku-4-5,claude-sonnet-5,claude-opus-5 -r 5 -o docs/results-extraction.md
```

The question these answer is not "which model scored highest" — that is known in advance
and uninteresting. It is **where the cheap model stops being good enough**, and whether that
point moves between the easy classification suite and the harder extraction suite. If it
does, that difference is the whole argument for per-workload model selection, and it is what
Month 6's router acts on.

## 9. Demo

> _2–4 minute walkthrough — to be recorded once real results are in._

## 10. Future Improvements

- Import the Month 4 evaluation platform as the quality scorer, so comparison tables carry
  real faithfulness and hallucination columns instead of label matching.
- Prompt-caching-aware cost modelling — show cost with and without cache hits, since that
  materially changes the model-choice calculus on long shared prefixes.
- Batch-API cost mode for latency-tolerant workloads (50% cheaper on supported providers).
- Async dispatch with a rate-limit-aware semaphore, for runs in the thousands of calls.

---

## Quickstart

```bash
git clone https://github.com/ashutosh3060/llm-engineering-playground.git
cd llm-engineering-playground

python -m venv .venv && source .venv/bin/activate
make install

playground probe                                      # works immediately — mock is built in
playground bench datasets/sentiment-classification.yaml -r 3
```

For real numbers:

```bash
cp .env.example .env    # add ANTHROPIC_API_KEY
playground probe        # expect a green `anthropic` row
```

## Usage

```bash
playground probe                       # which providers are reachable
playground models                      # registry with today's prices and availability
playground bench <suite.yaml> -r 5     # run a benchmark suite
playground spend                       # cumulative cost per model
playground serve                       # FastAPI on :8000  (docs at /docs)
playground ui                          # Streamlit on :8501
```

### The four UI tabs

| Tab | Answers |
|---|---|
| **Comparison** | Which model should serve this prompt? |
| **Prompt Lab** | Which prompt and parameter combination is best? |
| **Cost Analyzer** | What does this prompt cost before I run it at scale? |
| **Runs** | What have I already measured? |

### API

```bash
curl -X POST localhost:8000/compare -H 'content-type: application/json' \
  -d '{"prompt":"Explain prompt caching","repeats":3}'
```

`/complete` · `/complete/stream` · `/compare` · `/count-tokens` · `/models` · `/providers` ·
`/runs` · `/runs/{id}` · `/spend`

### As a library

```python
from playground.benchmark import BenchmarkSuite, run_benchmark
from playground.runtime import build_gateway, get_store

suite = BenchmarkSuite.from_yaml("datasets/sentiment-classification.yaml")
run_id, summaries, _ = run_benchmark(
    suite, ["claude-haiku-4-5", "claude-opus-5"],
    gateway=build_gateway(), store=get_store(), repeats=5,
)
for s in summaries:                    # cheapest first
    print(s.model, s.accuracy, s.cost_per_call_usd, s.p50_latency_ms)
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | The only key needed for real results |
| `PLAYGROUND_ALLOW_MOCK` | `1` | Register the offline mock provider |
| `PLAYGROUND_STORE` | `./data/playground.db` | SQLite location |
| `PLAYGROUND_REPEATS` | `5` | Default benchmark repeats |
| `PLAYGROUND_MLFLOW` | `0` | Mirror runs into MLflow (needs the `tracking` extra) |
| `MLFLOW_TRACKING_URI` | `sqlite:///<store_dir>/mlflow.db` | Override the tracking backend. The default is a SQLite database — MLflow 3.x rejects the legacy file store. |
| `AI_CORE_DEFAULT_MODEL` | `claude-opus-5` | Model used when a request omits one |

## Repository Layout

```
src/playground/
  config.py      env-driven settings
  runtime.py     the one gateway + the one store, shared by API, CLI, and UI
  mock.py        deterministic offline provider — why this runs without a key
  prompts.py     templates, $-substitution, content-hash versioning
  sweep.py       parameter grid expansion with a spend guard
  scoring.py     deterministic scorers
  benchmark.py   suite execution, repeats, aggregation
  stats.py       nearest-rank percentile (one implementation, three callers)
  store.py       SQLite: runs + results
  tracking.py    optional MLflow mirror
  api/           FastAPI app + schemas
  ui/app.py      Streamlit, 4 tabs
datasets/        benchmark suites (YAML)
tests/           61 tests, green with zero API keys
docs/            architecture · design-decisions · evaluation · cost-analysis · future-roadmap
```

## Documentation

- [Architecture](docs/architecture.md)
- [Design Decisions](docs/design-decisions.md) — 11 ADRs
- [Evaluation](docs/evaluation.md)
- [Cost Analysis](docs/cost-analysis.md)
- [Future Roadmap](docs/future-roadmap.md)

---

Part of a [6-month Product AI Engineer portfolio](https://github.com/ashutosh3060) —
`ai-core` · `llm-engineering-playground` · `enterprise-rag-platform` ·
`multi-agent-ai-platform` · `llm-evaluation-platform` · `production-ai-assistant` ·
`ai-model-router`
