# Architecture — llm-engineering-playground

## System Diagram

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

## Components

_One subsection per box above: responsibility, inputs, outputs, failure modes,
and what happens when its dependency is unavailable._

## Data Flow

_End-to-end trace of a single representative request, with the data shape at each hop._

## Scaling Considerations

_Where this design breaks under load, and the first thing that would need to change._
