# Architecture — llm-engineering-playground

## System diagram

```
                    ┌──────────────┐   ┌──────────────┐
                    │ Streamlit UI │   │  playground  │
                    │  (4 tabs)    │   │     CLI      │
                    └──────┬───────┘   └──────┬───────┘
                           │                  │
                           └────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  FastAPI gateway  │  /complete /compare
                          │  (also standalone)│  /count-tokens /models
                          └─────────┬─────────┘  /providers /runs /spend
                                    │
              ┌─────────────────────▼─────────────────────┐
              │            playground.runtime             │
              │   one gateway + one store for every       │
              │   entry point, so "what's available"      │
              │   answers the same everywhere             │
              └──────┬─────────────────────────┬──────────┘
                     │                         │
          ┌──────────▼──────────┐   ┌──────────▼──────────┐
          │  ai_core.Gateway    │   │  playground.Store   │
          │  provider registry  │   │  SQLite: runs +     │
          │  + concurrent fanout│   │  results            │
          └──────────┬──────────┘   └─────────────────────┘
                     │
       ┌─────────────┼─────────────┐
       │             │             │
  Anthropic     OpenAI*       mock (offline)
       │             │             │
       └─────────────┴─────────────┘
                     │
        ai_core: versioned pricing · cost accounting · token counting

  * registered only when a key is present
```

## Components

| Module | Responsibility | Fails how |
|---|---|---|
| `config` | env → `Settings`, cached once | Missing vars fall back to documented defaults |
| `runtime` | Builds the one gateway and the one store | No providers → `ProviderUnavailable` with a pointer to `probe` |
| `mock` | Deterministic offline provider | Never fails; that is its purpose |
| `prompts` | Templates, `$var` rendering, content-hash versioning | Missing variable → `KeyError` naming it |
| `sweep` | Parameter grid expansion | Oversized grid → `SweepTooLarge` before any call is billed |
| `scoring` | Deterministic scorers | Unknown scorer → `ValueError` listing valid names |
| `benchmark` | Suite execution, repeats, aggregation | Per-call errors are recorded, not raised; the run completes |
| `stats` | Nearest-rank percentile — one implementation | — |
| `store` | SQLite persistence | Creates its parent directory on init |
| `api` | HTTP surface | Unavailable model → 409 with remediation text |
| `ui` | Streamlit, 4 tabs | Degrades to a banner when no provider is configured |
| `tracking` | Optional MLflow mirror | No-op when MLflow is absent or disabled |

## Data flow — one benchmark call

```
BenchmarkCase.variables
   → PromptTemplate.render()          → PromptVersion (content hash)
   → PromptVersion.to_request(model)  → CompletionRequest (+ expected/labels metadata)
   → Gateway.complete()               → provider → CompletionResult (usage, latency)
   → estimate_cost(model, usage)      → cost, from the pricing table at today's date
   → score_result(text, expected)     → Score (value + rationale)
   → Store.record()                   → one row, permanently queryable
   → _summarise()                     → ModelSummary (accuracy, $/call, p50, p95)
```

The prompt hash travels with the request and lands in the store, so any row can be
traced back to the exact bytes that produced it.

## Why the mock provider is in the package, not the tests

It would be conventional to put a fake provider under `tests/`. It lives in
`src/playground/mock.py` instead because it serves three callers, not one:

1. **The test suite** — green on a clean clone, no key, no network.
2. **A recruiter cloning the repo** — the app runs and does something real rather than
   erroring on a missing credential.
3. **Development** — iterate on the UI and aggregation logic without spending money.

The cost of that decision is that synthetic output can reach a user-facing surface, so
every such surface labels it: the CLI prints a warning after any run containing a mock
model, and the UI shows a persistent banner.

## Scaling considerations

Where this design breaks, and what changes first:

- **SQLite under concurrent writers.** The benchmark harness writes from a thread pool.
  Fine at this scale; a shared multi-user deployment needs Postgres. The schema is
  portable — that is why it uses no SQLite-specific types.
- **`ThreadPoolExecutor` for fan-out.** Correct for I/O-bound provider calls and simple
  to reason about. A run in the thousands of calls wants async dispatch with a
  rate-limit-aware semaphore.
- **Full result text in the store.** Convenient for inspecting a run; unbounded growth
  over time. A retention policy or content-addressed blob store is the fix.
- **No response caching.** Repeating an identical benchmark re-pays for every call. A
  cache keyed on `(prompt_hash, model, params)` would make iteration nearly free — it
  is listed as a future improvement rather than built, because caching benchmark results
  by default would mask real run-to-run variance.
