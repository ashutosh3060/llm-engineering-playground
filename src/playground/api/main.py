"""FastAPI gateway.

Every request is logged to SQLite with tokens, latency, and cost before the
response is returned — cost tracking is not an optional reporting layer.

The UI is one client of this API, not the only possible one; anything the
Streamlit app can do is reachable over HTTP.
"""

from __future__ import annotations

import json
import statistics
import uuid
from collections.abc import AsyncIterator
from datetime import date
from typing import Any

from ai_core.cost import estimate_cost
from ai_core.models import REGISTRY
from ai_core.providers.base import ProviderUnavailable
from ai_core.types import CompletionRequest, Message, Usage
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from ..config import get_settings
from ..runtime import get_gateway, get_store, provider_report
from ..stats import percentile
from .schemas import (
    CompareIn,
    CompareOut,
    CompareRowOut,
    CompleteIn,
    CountTokensIn,
    CountTokensOut,
    ModelOut,
    ResultOut,
    RunOut,
    SpendRow,
    UsageOut,
)

app = FastAPI(
    title="LLM Engineering Playground",
    version="0.1.0",
    description=(
        "Compare models on quality, latency, and cost. Every call is recorded with "
        "token counts and cost so the comparison is evidence, not impression."
    ),
)


# --------------------------------------------------------------------- helpers


def _to_request(payload: CompleteIn, model: str | None = None) -> CompletionRequest:
    settings = get_settings()
    try:
        messages = payload.resolved_messages()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return CompletionRequest(
        model=model or payload.model or settings.default_model,
        messages=[Message(role=m.role, content=m.content) for m in messages],
        system=payload.system,
        max_tokens=payload.max_tokens,
        effort=payload.effort,
        thinking=payload.thinking,
    )


def _usage_out(usage: Usage) -> UsageOut:
    return UsageOut(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        total_prompt_tokens=usage.total_prompt_tokens,
    )


# ---------------------------------------------------------------------- routes


@app.get("/health")
def health() -> dict[str, Any]:
    gw = get_gateway()
    return {
        "status": "ok",
        "providers": gw.provider_names,
        "models_available": len(gw.available_models()),
    }


@app.get("/providers")
def providers() -> list[dict[str, Any]]:
    """Live credential check per provider — the API-side equivalent of `ai_core.probe`."""
    return provider_report()


@app.get("/models", response_model=list[ModelOut])
def models() -> list[ModelOut]:
    gw = get_gateway()
    available = set(gw.available_models())
    today = date.today()
    out = []
    for spec in REGISTRY.values():
        price = spec.price_on(today)
        out.append(
            ModelOut(
                id=spec.id,
                provider=spec.provider,
                display_name=spec.display_name,
                tier=spec.tier,
                context_window=spec.context_window,
                max_output_tokens=spec.max_output_tokens,
                input_per_mtok=price.input_per_mtok,
                output_per_mtok=price.output_per_mtok,
                price_note=price.note,
                available=spec.id in available,
                notes=spec.notes,
            )
        )
    out.sort(key=lambda m: (not m.available, m.input_per_mtok, m.id))
    return out


@app.post("/complete", response_model=ResultOut)
def complete(payload: CompleteIn) -> ResultOut:
    request = _to_request(payload)
    try:
        result = get_gateway().complete(request)
    except ProviderUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    run_id = f"api-{uuid.uuid4().hex[:8]}"
    store = get_store()
    store.start_run(run_id, kind="complete", label=request.model)
    store.record(run_id, result)
    store.finish_run(run_id)

    return ResultOut(
        model=result.model,
        provider=result.provider,
        text=result.text,
        usage=_usage_out(result.usage),
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
        stop_reason=result.stop_reason,
        prompt_hash=result.prompt_hash,
        error=result.error,
        ok=result.ok,
    )


@app.post("/complete/stream")
async def complete_stream(payload: CompleteIn) -> StreamingResponse:
    """Server-sent events.

    The gateway's provider interface is synchronous, so this emits the completed
    result as a single SSE frame rather than pretending to stream tokens. An
    honest single frame beats a fake progressive one — the client contract is
    identical and no consumer is misled about where the latency went.
    """
    request = _to_request(payload)

    async def events() -> AsyncIterator[str]:
        try:
            result = get_gateway().complete(request)
        except ProviderUnavailable as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            return

        store = get_store()
        run_id = f"api-{uuid.uuid4().hex[:8]}"
        store.start_run(run_id, kind="complete-stream", label=request.model)
        store.record(run_id, result)
        store.finish_run(run_id)

        yield f"event: message\ndata: {json.dumps({'text': result.text})}\n\n"
        yield (
            "event: usage\ndata: "
            + json.dumps(
                {
                    "model": result.model,
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "latency_ms": round(result.latency_ms, 1),
                    "cost_usd": result.cost_usd,
                }
            )
            + "\n\n"
        )
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/compare", response_model=CompareOut)
def compare(payload: CompareIn) -> CompareOut:
    """Run one prompt across several models and return a comparison table.

    Dispatch is concurrent inside the gateway: sequential runs would add queueing
    time to every model after the first, so the latency column would describe the
    harness rather than the providers.
    """
    gw = get_gateway()
    targets = payload.models or gw.available_models()
    if not targets:
        raise HTTPException(
            status_code=409,
            detail="No models available. Set a provider key, or enable PLAYGROUND_ALLOW_MOCK.",
        )

    base = _to_request(payload)
    run_id = f"cmp-{uuid.uuid4().hex[:8]}"
    store = get_store()
    store.start_run(run_id, kind="compare", label=base.prompt_hash(), models=targets)

    collected: dict[str, list[Any]] = {}
    for rep in range(payload.repeats):
        try:
            results = gw.compare(base, models=targets)
        except ProviderUnavailable as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        for result in results:
            collected.setdefault(result.model, []).append(result)
            store.record(run_id, result, repeat=rep)
    store.finish_run(run_id)

    rows = []
    for model, results in collected.items():
        ok = [r for r in results if r.ok]
        latencies = [r.latency_ms for r in ok]
        rows.append(
            CompareRowOut(
                model=model,
                provider=results[0].provider,
                n=len(results),
                errors=len(results) - len(ok),
                cost_per_call_usd=(statistics.fmean([r.cost_usd for r in ok]) if ok else 0.0),
                p50_latency_ms=percentile(latencies, 50),
                p95_latency_ms=percentile(latencies, 95),
                sample_text=(ok[0].text if ok else (results[0].error or "")),
                ok=bool(ok),
            )
        )
    rows.sort(key=lambda r: (not r.ok, r.cost_per_call_usd))

    return CompareOut(run_id=run_id, prompt_hash=base.prompt_hash(), rows=rows)


@app.post("/count-tokens", response_model=CountTokensOut)
def count_tokens(payload: CountTokensIn) -> CountTokensOut:
    """Provider-native token count — never a `tiktoken` approximation."""
    settings = get_settings()
    model = payload.model or settings.default_model
    messages = payload.messages or (
        [{"role": "user", "content": payload.prompt}] if payload.prompt else []
    )
    if not messages:
        raise HTTPException(status_code=422, detail="Provide either `prompt` or `messages`.")

    request = CompletionRequest(
        model=model,
        messages=[
            Message(role=m["role"], content=m["content"]) if isinstance(m, dict) else m
            for m in messages
        ],  # type: ignore[arg-type]
        system=payload.system,
    )
    try:
        tokens = get_gateway().count_tokens(request)
    except ProviderUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return CountTokensOut(
        model=model,
        input_tokens=tokens,
        estimated_input_cost_usd=estimate_cost(model, Usage(input_tokens=tokens)),
    )


@app.get("/runs", response_model=list[RunOut])
def runs(limit: int = 50) -> list[RunOut]:
    out = []
    for row in get_store().runs(limit=limit):
        config = row.pop("config", "{}")
        out.append(RunOut(**row, config=json.loads(config) if config else {}))
    return out


@app.get("/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    results = get_store().results(run_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"No results for run {run_id!r}")
    return {"run_id": run_id, "results": results}


@app.get("/spend", response_model=list[SpendRow])
def spend() -> list[SpendRow]:
    """Where the money went, per model, across every run."""
    return [SpendRow(**row) for row in get_store().spend_by_model()]
