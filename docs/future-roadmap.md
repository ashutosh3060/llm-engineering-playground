# Future Roadmap — llm-engineering-playground

## Immediate (unblocks section 7 of the README)

- **Populate the evaluation tables.** Requires `ANTHROPIC_API_KEY`. Two `playground bench`
  commands, ~450 calls, under $1. Everything else is built and verified.
- **Record the demo video.** After the tables are real — a walkthrough of empty tables
  would show the tool, not the finding.

## Planned

- **Import the Month 4 evaluation platform as the quality scorer.** Today's scorers are
  deterministic label and structural matchers, which is enough to give a comparison a
  quality column but cannot judge open-ended generation. Once `llm-evaluation-platform`
  exists, this repo consumes it rather than growing its own second-rate version.
- **Prompt-caching-aware cost modelling.** Show cost with and without cache hits. On a long
  shared prefix this changes the model-choice answer, and the current tables cannot express
  it. Needs the minimum-cacheable-prefix rule per model (512 tokens on Claude Opus 5, 1024
  on Opus 4.8, 4096 on Opus 4.6 and Haiku 4.5) so it does not claim a saving that a short
  prompt never gets.
- **Batch-API cost mode.** 50% cheaper on supported providers for latency-tolerant runs.
  Most benchmark runs are latency-tolerant by definition, so this mostly pays for itself.
- **Async dispatch with a rate-limit-aware semaphore.** The current `ThreadPoolExecutor` is
  correct and simple, but a run in the thousands of calls will hit provider rate limits and
  needs backoff shaped by the response headers rather than blind retries.

## Considered and deferred

- **Response caching keyed on `(prompt_hash, model, params)`.** Would make iteration nearly
  free. Deferred because caching benchmark results *by default* masks genuine run-to-run
  variance — which is exactly what the repeats exist to measure. If it lands it must be
  opt-in and clearly marked in the output.
- **A learned or LLM-based scorer in this repo.** Rejected: it is Month 4's job, and two
  implementations would drift.
- **Postgres instead of SQLite.** Unnecessary for a single-operator tool, and it would add
  a service dependency to `git clone && make install`. The schema is portable when the time
  comes.
- **Real streaming on `/complete/stream`.** The provider interface is synchronous, so the
  endpoint emits one SSE frame. Faking progressive delivery would mislead a consumer about
  where the latency actually is; the client contract is identical either way.

## Open questions

- **Does the cheap-model-is-good-enough threshold move between workload types?** The two
  suites are designed to answer this. If it does not move, per-workload routing is much
  less valuable than the roadmap assumes — and Month 6's premise needs revisiting.
- **How stable are latency percentiles across time of day and provider load?** Five repeats
  in one window may not be enough to quote a p95 with confidence. Worth a repeated run at
  different hours before treating the number as a property of the model.
- **Is `effort` a better cost lever than tier downgrade on these workloads?** Both reduce
  spend. Nobody has measured which trades away less quality per dollar saved.
