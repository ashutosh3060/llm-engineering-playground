# Cost Analysis — llm-engineering-playground

**Status: complete for cost. Quality and latency remain pending an API key.**

Cost is arithmetic — token count × price — so the entire cost analysis is
computable with **zero API calls**. That is what this document contains. What it
does *not* contain is whether the cheap model is good enough, which is a quality
question and needs real calls (see [`evaluation.md`](evaluation.md)).

> **Provenance.** Token counts below are **estimated** at ~4 characters/token,
> because no provider key was configured. The arithmetic is exact *given* those
> counts; the counts themselves are not. Set `ANTHROPIC_API_KEY` and re-run
> `playground cost` to replace them with provider-exact figures — the command
> labels which it used. Everything else here — prices, cache minimums, the shape
> of the curves — is exact and unaffected.

Reproduce everything:

```bash
playground cost datasets/sentiment-classification.yaml --calls 10000
playground cost datasets/structured-extraction.yaml   --calls 10000
```

---

## Workload 1 — classification

Prompt shape: **33** token shared prefix, **25** variable, **16** output.

| Model | Tier | $/call | Monthly @ 10k | Cache min | Cacheable |
|---|---|---|---|---|---|
| `claude-haiku-4-5` | small | $0.000138 | **$1.38** | 4,096 | no |
| `claude-sonnet-5` | balanced | $0.000276 | $2.76 | 1,024 | no |
| `claude-opus-5` | frontier | $0.000690 | $6.90 | 512 | no |
| `claude-opus-4-8` | frontier | $0.000690 | $6.90 | 1,024 | no |

## Workload 2 — structured extraction

Prompt shape: **29** token shared prefix, **41** variable, **256** output.

| Model | Tier | $/call | Monthly @ 10k | Cache min | Cacheable |
|---|---|---|---|---|---|
| `claude-haiku-4-5` | small | $0.001350 | **$13.50** | 4,096 | no |
| `claude-sonnet-5` | balanced | $0.002700 | $27.00 | 1,024 | no |
| `claude-opus-5` | frontier | $0.006750 | $67.50 | 512 | no |
| `claude-opus-4-8` | frontier | $0.006750 | $67.50 | 1,024 | no |

---

## Findings

### 1. Output tokens dominate — and the effect is workload-shaped

Output is priced 5× input across the Claude line, so the split follows the
`max_tokens` budget, not the prompt length:

| Workload | Input tokens | Output tokens | Output share of cost |
|---|---|---|---|
| Classification | 58 | 16 | **58%** |
| Extraction | 70 | 256 | **95%** |

The ratio is identical on every tier, because the 5× multiplier is uniform.

**Consequence:** for the extraction workload, shortening the prompt is close to
pointless — 95% of the bill is output. Capping `max_tokens`, or prompting for
terser output, is the lever that matters. This inverts the usual instinct to
optimise the prompt first, and it is invisible unless you decompose the cost.

### 2. Neither workload can use prompt caching at all

Both prefixes (33 and 29 tokens) are far below every model's minimum cacheable
length. The provider does not error — it simply declines to cache, returning
`cache_creation_input_tokens: 0`. A cost plan that assumes a uniform "~90%
caching discount" would be wrong by the entire saving on both of these.

This is the most useful thing in this document, and it is a **negative** result:
the optimisation everyone reaches for first does nothing here.

### 3. Caching is a step function on prefix length, and the step differs 8× by model

Saving at 10,000 calls, by shared-prefix length:

| Prefix tokens | `opus-5` | `sonnet-5` | `opus-4-8` | `haiku-4-5` |
|---|---|---|---|---|
| 0 | — | — | — | — |
| 256 | — | — | — | — |
| **512** | **75%** | — | — | — |
| **1,024** | 82% | **82%** | **82%** | — |
| 2,048 | 86% | 86% | 86% | — |
| **4,096** | 88% | 88% | 88% | **88%** |
| 8,192 | 89% | 89% | 89% | 89% |

Nothing happens until the prefix crosses the model's minimum; then the saving
appears abruptly and climbs toward 90% as the write premium amortises.

**The minimums are not ordered by price.** `claude-opus-5` — the most expensive
model here — has the *lowest* threshold at 512 tokens, while `claude-haiku-4-5`
— the cheapest — has the highest at 4,096. So a system prompt in the 512–4,096
range caches on the frontier model and not on the small one, which can narrow or
invert a tier gap that looks decisive on sticker price.

**Consequence:** "which model is cheapest?" is not answerable from the price
table alone once caching is in play. It depends on your prefix length.

### 4. The tier spread is exactly 5.0× and is stable across workloads

`claude-haiku-4-5` → `claude-opus-5` is 5× on both input and output, so the
ratio holds regardless of prompt shape. That makes the quality question sharp
and quantifiable: **for the classification workload, Opus 5 must be worth
$5.52/month more per 10k calls than Haiku.** Whether it is cannot be answered
here — it needs a benchmark run.

### 5. Break-even on caching is the second call

A single call costs *more* cached than uncached: it pays the 1.25× write premium
and never reads the cache back. Two calls break even; the saving approaches 90%
asymptotically. Relevant for low-volume or bursty workloads where a cache entry
may expire (5-minute default TTL) before it is read.

---

## Optimizations, evaluated

| Optimization | Applies here? | Effect |
|---|---|---|
| Tier downgrade frontier → small | **Yes** | 5.0× cheaper. Quality cost unknown pending benchmark. |
| Prompt caching | **No** | Both prefixes below every model's minimum. |
| Reduce `max_tokens` | **Yes — biggest lever on extraction** | 95% of that workload's cost is output. |
| Shorten the prompt | **Marginal** | 5% of extraction cost, 42% of classification. |
| Batch API | Likely | 50% on supported providers; not modelled here. |
| Lower `effort` | Untested | Needs real runs to measure the token reduction. |

---

## Method

- Prices come from `ai-core`'s versioned pricing table resolved at today's date,
  so a lapsed introductory rate is not applied retroactively. Note that
  `claude-sonnet-5` carries an introductory rate through 2026-08-31; figures
  dated after that reflect the standard $3/$15.
- Cache modelling uses the real billing shape: first call writes the prefix at
  1.25× base input, later calls read at 0.1×, variable input and output billed
  normally throughout.
- A prefix below a model's minimum is modelled as **uncached**, because that is
  what the provider does.
- Cache minimums are per-model metadata in `ai-core` (`ModelSpec.cache_min_tokens`).

## What is still missing

Cost is only half the decision. These need an API key:

- **Accuracy per tier** — is Haiku good enough for these workloads?
- **Latency p50/p95** — real provider timings.
- **Exact token counts** — replacing the ~4 chars/token estimate.

Until then this document answers "what would each option cost" but not "which
should you choose." See [`evaluation.md`](evaluation.md).
