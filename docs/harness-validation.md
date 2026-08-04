# Harness Validation

> # ⚠️ THESE ARE NOT MODEL EVALUATION RESULTS
>
> Every number on this page was produced by the **offline mock provider**. It makes
> no network calls. Its outputs are synthetic, seeded from a prompt hash, and its
> "accuracy" is a hardcoded per-tier probability — not a measurement of anything.
>
> **Nothing here says anything about Claude, or about any real model.**
>
> Real model evaluation lives in [`evaluation.md`](evaluation.md), which is
> deliberately empty pending an API key.

## What this page is for

A benchmark harness that has never run end-to-end is an untested claim. This page
records a full execution of every stage — suite loading, prompt rendering,
concurrent dispatch, scoring, persistence, aggregation, percentile computation,
cost accounting, and MLflow mirroring — so that "the harness works" is an
observation rather than an assertion.

The mock provider exists precisely so this is possible without spending money or
requiring a credential. Swapping in a real provider is a one-flag change:

```bash
# what produced this page
playground bench datasets/sentiment-classification.yaml -m mock-small,mock-balanced,mock-frontier -r 5

# what will produce evaluation.md
playground bench datasets/sentiment-classification.yaml -m claude-haiku-4-5,claude-sonnet-5,claude-opus-5 -r 5
```

---

## Run 1 — classification suite

20 cases × 3 mock tiers × 5 repeats = **300 calls**.

| Model | n | Errors | "Accuracy" | $/call | $ total | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| `mock-small` | 100 | 0 | 65.0% | 0.000606 | 0.0606 | 105 | 139 |
| `mock-balanced` | 100 | 0 | 80.0% | 0.002039 | 0.2039 | 261 | 352 |
| `mock-frontier` | 100 | 0 | 90.0% | 0.003180 | 0.3180 | 507 | 664 |

The accuracy column reproduces each mock tier's configured probability (0.65 /
0.80 / 0.90) to within a point over 100 samples. That is the point: it confirms
the **scoring and aggregation path is wired correctly**, because the expected
answer is known by construction. It says nothing about model quality.

## Run 2 — extraction suite

10 cases × 3 mock tiers × 5 repeats = **150 calls**.

| Model | n | Errors | "Accuracy" | $/call | $ total | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| `mock-small` | 50 | 0 | 0.0% | 0.000573 | 0.0286 | 106 | 142 |
| `mock-balanced` | 50 | 0 | 0.0% | 0.001732 | 0.0866 | 251 | 339 |
| `mock-frontier` | 50 | 0 | 0.0% | 0.003366 | 0.1683 | 480 | 657 |

**The 0.0% is a passing result, not a failure.** This suite scores with the
structural JSON scorer, and the mock provider does not emit JSON. A scorer that
awarded credit here would be broken. Seeing it correctly return zero across 150
calls validates that the structural scorer rejects malformed output rather than
finding something to like in it — the failure mode that would quietly inflate
every future extraction benchmark.

---

## What this execution confirms

| Stage | Confirmed by |
|---|---|
| YAML suite loading | 30 cases parsed across two suites with different scorers |
| Prompt rendering + content hashing | 450 requests rendered; per-case variables substituted |
| Concurrent dispatch | 450 calls through the thread pool with no lost or duplicated results |
| Scoring | Classification reproduces configured probabilities; JSON scorer correctly returns 0 |
| Persistence | 450 rows in SQLite, each with case ID, repeat index, tokens, latency, cost |
| Percentile aggregation | p95 ≥ p50 on every row; both derived from 5 repeats, never one sample |
| Cost accounting | Per-call and totals consistent with the pricing table |
| Spend rollup | `playground spend` totals $0.8660 across 450 calls, matching the per-run sums |
| MLflow mirror | 6 runs written with params and metrics |
| Error path | 0 errors here; error recording is covered separately in `tests/test_store.py` |

## What it does not confirm

- **Any claim about a real model.** No real model was called.
- **Real latency.** The mock sleeps for a seeded interval; these numbers describe
  a `time.sleep`, not a provider.
- **Real cost.** The mock's prices are synthetic placeholders. Real unit
  economics — computed from the actual pricing table, with no mock involvement —
  are in [`cost-analysis.md`](cost-analysis.md).
- **Whether a cheaper model is good enough for your workload.** That is the
  question the whole project exists to answer, and it requires an API key.

## Replacing this page

Once `ANTHROPIC_API_KEY` is set, the two commands at the top of this page produce
real tables that belong in [`evaluation.md`](evaluation.md). This page stays as a
record that the harness was validated independently of any provider — which is
also what keeps the test suite green on a clean clone with no credentials.
