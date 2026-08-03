# Evaluation — llm-engineering-playground

> **Status: awaiting a live API key.**
>
> The harness is complete and verified end-to-end against the offline mock provider.
> The tables below are **deliberately empty**, because filling them requires real API
> calls and no provider key was configured when this was built. Nothing here is
> estimated or extrapolated — an empty table is more useful than a fabricated one.

## How to produce these numbers

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY
playground probe              # expect a green `anthropic` row

playground bench datasets/sentiment-classification.yaml \
  --models claude-haiku-4-5,claude-sonnet-5,claude-opus-5 \
  --repeats 5 \
  --out docs/results-sentiment.md

playground bench datasets/structured-extraction.yaml \
  --models claude-haiku-4-5,claude-sonnet-5,claude-opus-5 \
  --repeats 5 \
  --out docs/results-extraction.md
```

Then paste the generated tables into Results and write the Analysis.

**Cost of doing this:** 20 cases × 3 models × 5 repeats = 300 calls for classification,
plus 150 for extraction. At these prompt sizes that is well under $1. `playground spend`
reports the exact figure afterwards.

## Methodology

- **Datasets** — `sentiment-classification.yaml` (20 labelled customer-feedback snippets,
  3 classes) and `structured-extraction.yaml` (10 snippets → JSON with a fixed key set).
  The first is deliberately easy; the second is where small models usually fail. Together
  they bracket the question instead of answering half of it.
- **Scoring** — classification uses a word-boundary label matcher that rejects negated
  matches, so "not positive" does not score as "positive". Extraction uses a structural
  JSON scorer: does it parse, and are the required keys present. Both are deterministic
  and cheap. Semantic scoring is Month 4's job, not this project's.
- **Repeats** — 5 per (case, model). Every latency figure is a nearest-rank percentile
  over those repeats. A single sample is noise and is never reported.
- **Dispatch** — concurrent. Sequential runs would add queueing time to every model after
  the first, so the latency column would describe the harness rather than the providers.

## Results

### Classification (20 cases × 5 repeats)

| Model | n | Errors | Accuracy | $/call | $ total | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| _pending a live key_ | | | | | | | |

### Structured extraction (10 cases × 5 repeats)

| Model | n | Errors | Accuracy | $/call | $ total | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| _pending a live key_ | | | | | | | |

## Analysis

_To be written once the tables are populated._

The question to answer is not "which model scored highest" — that answer is known in
advance and uninteresting. It is **at what point the cheaper model stops being good
enough for this workload**, and whether that point differs between the easy
classification suite and the harder extraction suite. If it does, that difference is
the entire argument for per-workload model selection, and it is what Month 6's router
acts on.

## Limitations

- Two narrow workloads. Results do not generalise to long-context, reasoning, or agentic
  tasks — those are Months 2–4.
- Deterministic scorers only. They measure whether the right label or key set came back,
  not whether the reasoning behind it was sound.
- Latency is client-measured and includes network round-trip. It is a comparison between
  models under identical conditions, not an absolute figure.
- Single geography, single time window. Provider latency varies with load.

## Mock-provider results are not evaluation results

The offline mock provider produces stable, deterministic numbers that look like a real
benchmark. They are synthetic. Every surface that can display them labels them as such —
the CLI prints a warning, the UI shows a banner. They exist to prove the harness works,
never to populate this document.
