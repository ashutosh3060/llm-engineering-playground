# Cost Analysis — llm-engineering-playground

> **Status: unit economics pending a live key.** The pricing mechanics below are
> implemented and tested; the measured per-workload figures need real calls.

## How costs are computed

Every cost comes from `ai-core`'s versioned pricing table, evaluated at the date the
request ran — never from a hardcoded constant.

```python
from datetime import date
from ai_core import price_at

price_at("claude-sonnet-5", date(2026, 8, 15))   # $2/$10 — introductory rate
price_at("claude-sonnet-5", date(2026, 10, 1))   # $3/$15 — standard rate
```

That matters more than it sounds. Claude Sonnet 5 carries an introductory rate that
expires 2026-08-31. A hardcoded float would keep reporting the promotional price into
October and quietly understate the bill — and worse, would retroactively corrupt any
cost analysis written before the change. The table records what a request actually
cost at the time it ran. There is a test asserting the rate lapses on schedule.

## Cost drivers, ranked

1. **Model tier.** The spread between the cheapest and most expensive Claude tier is
   5x on input and 5x on output. No other lever comes close. This is why the project
   exists.
2. **Output tokens.** Output is priced 5x input across the Claude line. A prompt that
   induces a verbose answer costs more than a longer prompt that induces a terse one —
   counterintuitive, and worth measuring rather than assuming.
3. **Cache hit rate.** Cache reads bill at ~0.1x base input; 5-minute writes at ~1.25x.
   Break-even is two requests against the same prefix.
4. **Effort level.** On models that support it, `effort` controls thinking depth and
   therefore spend. It is the cost lever — not a token budget.

## Reading the numbers correctly

`usage.input_tokens` is the **uncached remainder**, not the prompt size. Full prompt
size is `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`, exposed
as `usage.total_prompt_tokens`.

Misreading this is the most common way to conclude that caching is broken when it is
working perfectly: `input_tokens` collapses toward zero on a cache hit precisely
*because* the cache is doing its job.

## Unit economics

| Workload | Model | Input tok | Output tok | $/call | $/10k calls |
|---|---|---|---|---|---|
| _pending a live key_ | | | | | |

Produce these with the Cost Analyzer tab, or:

```bash
playground bench datasets/sentiment-classification.yaml --repeats 5
playground spend
```

## Optimizations to evaluate

| Optimization | Expected effect | Measured |
|---|---|---|
| Tier downgrade (frontier → small) for classification | up to 5x cheaper | _pending_ |
| Prompt caching on the shared system prompt | ~90% off the cached prefix after call 2 | _pending_ |
| Lower `effort` on simple workloads | fewer thinking tokens | _pending_ |
| Batch API for latency-tolerant runs | 50% on supported providers | _pending_ |
| Tighter `max_tokens` on classification | caps runaway output | _pending_ |

The point of measuring rather than assuming is that **some of these will not pay off
on these workloads**. A prompt-caching win needs a prefix above the model's minimum
cacheable size — 512 tokens on Claude Opus 5, 1024 on Opus 4.8, 4096 on Opus 4.6 and
Haiku 4.5. A short classification system prompt may never cache at all, and reporting
a saving that did not occur would be worse than reporting none.

## Spend tracking

Cost is recorded on every call from the first one, not added later as reporting.

```bash
playground spend        # per-model totals: calls, tokens, average latency
```

```
GET /spend              # the same data over HTTP
```

Both read the SQLite store that the benchmark harness and the UI also write to, so
there is one number rather than three that disagree.
