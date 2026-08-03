# Design Decisions — llm-engineering-playground

Architecture decision records. Each states the context, the decision, and what the
decision cost — including the ones that made something else harder.

---

## ADR-001: Prompts are versioned by content hash, not filename

**Status:** Accepted

A prompt is an experimental variable. Identifying it by filename or a hand-maintained
version number means two runs can claim to share a prompt while sending different
bytes — an edit, a trailing space, a re-worded few-shot example. Every comparison
built on that is unsound and the failure is silent.

`PromptVersion.version` is a SHA-256 prefix over the rendered system prompt, all
few-shot pairs, and the user message, joined with a NUL separator. Deliberately
whitespace-sensitive.

**Consequences.** Trivial edits produce a new version, so the run history fragments
more than a human would group it. Accepted: a comparison that is wrong is worse than
one that is over-partitioned. The NUL separator prevents a collision where moving text
between the system and user field would otherwise hash identically.

---

## ADR-002: Token counts come from the provider, never `tiktoken`

**Status:** Accepted

`tiktoken` is OpenAI's tokenizer. Against Claude it undercounts by roughly 15–20% on
prose and considerably more on code. Cost estimates built on it are wrong in a
direction that only surfaces on the invoice.

`/count-tokens` and the Cost Analyzer call the provider's own `count_tokens` endpoint.

**Consequences.** Token counting requires a live API call, so it needs a key and adds
latency — the Cost Analyzer cannot work fully offline. Accepted: a fast wrong number
is worse than a slow right one when the whole point is cost accuracy.

---

## ADR-003: Cost comes from a versioned pricing table with effective dates

**Status:** Accepted

Prices change and introductory rates expire. Claude Sonnet 5's introductory rate lapses
2026-08-31. A hardcoded float keeps reporting the promotional price afterwards, and
retroactively corrupts any earlier analysis.

`ai-core`'s pricing table stores bands with `effective_from` / `effective_until`;
`price_at(model, date)` resolves the applicable one, preferring a bounded promotional
band over the standard rate.

**Consequences.** Adding a model means adding a pricing entry — it cannot be inferred.
Prices must be maintained by hand against the provider's pricing page. That is the
cost of correctness; a test asserts the Sonnet 5 intro rate lapses on schedule so the
mechanism cannot rot unnoticed.

---

## ADR-004: Comparison runs are dispatched concurrently

**Status:** Accepted

Sequential dispatch adds every earlier model's response time to every later model's
wall-clock measurement. The latency column would describe the harness, not the
providers — and it would rank models by dispatch order.

`Gateway.compare` and `run_benchmark` fan out over a thread pool and time each request
independently.

**Consequences.** Concurrent load can hit provider rate limits on large runs; errors
are recorded per call rather than aborting the run. Local resource contention adds
noise, which is part of why latency is reported as a percentile over repeats rather
than a mean.

---

## ADR-005: Latency is always p50/p95 over repeats, never a single sample

**Status:** Accepted

A single latency measurement is noise — cold starts, network jitter, and provider load
move it by multiples. Quoting one would be indistinguishable from making it up.

Every latency figure the project reports is a nearest-rank percentile over `repeats`
runs (default 5). One implementation, in `stats.percentile`, shared by the benchmark
harness, the API, and the UI.

**Consequences.** Costs 5x more per benchmark. Accepted — that is what makes the number
quotable. Nearest-rank rather than interpolated because with 5–20 samples, interpolation
invents precision the data does not support. The shared helper exists because three
copies had already started to drift during development.

---

## ADR-006: The mock provider ships in the package, not in tests

**Status:** Accepted

A fake provider under `tests/` would serve only the test suite. This one also lets a
stranger clone the repo and get a working application with no API key, and lets
development iterate on aggregation and UI logic without spending money.

`playground.mock.MockProvider` is deterministic (seeded from the prompt hash), has
three tiers with distinct price/latency/quality profiles, and honours `expected` /
`labels` metadata so the scoring path is exercisable offline.

**Consequences.** Synthetic output can reach a user-facing surface, which is a real
risk — numbers that look like a benchmark but are not. Mitigated by labelling at every
exit: the CLI prints a warning after any run containing a mock model, the UI shows a
persistent banner, the registry entry carries `SYNTHETIC` in its price note, and
`docs/evaluation.md` states explicitly that mock results never populate it.

---

## ADR-007: Parameter sweeps refuse to run oversized grids

**Status:** Accepted

Grids grow multiplicatively and every cell is a paid call. Three axes of five values
with five repeats is 625 calls — easy to request by accident, expensive to discover
afterwards.

`sweep.expand` raises `SweepTooLarge` above 200 total calls, counting repeats in the
total, and names the actual number in the message. The limit is overridable, but only
explicitly.

**Consequences.** A legitimate large sweep needs an extra argument. Accepted: an
explicit override is a fair price for not silently billing someone. Counting repeats
in the total matters — the guard would be nearly useless otherwise, since repeats are
exactly how a small grid becomes a large bill.

---

## ADR-008: Quality scoring here is deliberately shallow

**Status:** Accepted

Rigorous evaluation — LLM-as-judge with calibration, faithfulness, hallucination rate,
regression gating — is Month 4's `llm-evaluation-platform`. Building a second version
here would fork the logic, and the two would drift.

`scoring.py` provides only deterministic scorers: exact, contains, word-boundary label,
JSON structural, numeric-with-tolerance. Enough to give a model comparison a quality
column, and nothing more.

**Consequences.** Accuracy figures measure whether the right label came back, not
whether the answer was well-reasoned. Comparisons on open-ended generation are limited
to cost and latency until Month 4 lands. The label scorer specifically rejects negated
matches, because "not positive" scoring as "positive" is the classic silent false
positive in this kind of harness.

---

## ADR-009: Streamlit, not a production frontend

**Status:** Accepted

The audience is an engineer evaluating models, not an end user. Streamlit reaches a
usable four-tab comparison UI in hours rather than days, and the production frontend
work belongs to Month 5's `production-ai-assistant`.

**Consequences.** No auth, no multi-user state, limited layout control, and a full
re-run of the script on every interaction — which is why the gateway and store are
`@st.cache_resource`. Not a deployment target.

---

## ADR-010: The FastAPI layer is not just the UI's backend

**Status:** Accepted

Everything the UI can do is reachable over HTTP: `/complete`, `/compare`,
`/count-tokens`, `/models`, `/providers`, `/runs`, `/spend`. The UI is one client.

**Consequences.** Some duplication between the API layer and the UI's direct gateway
calls — the UI talks to `runtime` directly rather than to its own HTTP endpoints, to
avoid a needless network hop in a local tool. The shared `runtime` module keeps them
consistent about which models exist and where results are written.

---

## ADR-011: MLflow is optional; SQLite is the source of truth

**Status:** Accepted

The roadmap called for MLflow. It is a heavy dependency, and making it mandatory would
mean a failed `pip install` blocks a benchmark from running at all.

It ships as the `[tracking]` extra, off by default, and `tracking.log_benchmark`
returns `False` rather than raising when it is absent or disabled. The SQLite store
records everything regardless.

**Consequences.** MLflow's run-comparison UI requires an extra install step and an env
var. Accepted: a missing optional dependency must never break a paid benchmark run.
