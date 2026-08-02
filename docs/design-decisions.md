# Design Decisions — llm-engineering-playground

Architecture decision records. Each captures the context, the decision, and — once
implemented — what the decision actually cost.

## ADR-001: Prompts are versioned by content hash, not filename

**Status:** Accepted

**Context & Decision**

A prompt is an experimental variable. Hashing the rendered prompt means a run is always attributable to exactly the text that was sent, including whitespace changes that would otherwise silently invalidate a comparison.

**Consequences**

_To be recorded once implemented — including anything this decision made harder._

## ADR-002: Every request records tokens, latency, and cost — always

**Status:** Accepted

**Context & Decision**

Cost is not a reporting feature bolted on later. If it is not recorded on every call from day one, the historical comparison you want in week 4 does not exist.

**Consequences**

_To be recorded once implemented — including anything this decision made harder._

## ADR-003: Comparison runs are fanned out concurrently, then aligned

**Status:** Accepted

**Context & Decision**

Sequential comparison inflates the latency numbers with queueing. Concurrent dispatch with per-request timing gives latency figures that reflect the provider, not the harness.

**Consequences**

_To be recorded once implemented — including anything this decision made harder._

## ADR-004: Latency is reported as p50/p95 over repeats, not a single sample

**Status:** Accepted

**Context & Decision**

A single call's latency is noise. Anything quoted in the README is a distribution over at least 5 runs.

**Consequences**

_To be recorded once implemented — including anything this decision made harder._
