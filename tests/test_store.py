from __future__ import annotations

import pytest
from ai_core.types import CompletionResult, Usage

from playground.store import Store


def _result(model: str = "mock-small", cost: float = 0.001, latency: float = 100.0):
    return CompletionResult(
        model=model,
        provider="mock",
        text="hello",
        usage=Usage(input_tokens=100, output_tokens=50),
        latency_ms=latency,
        cost_usd=cost,
        prompt_hash="abc123",
        stop_reason="end_turn",
    )


def test_schema_is_created_on_init(store: Store) -> None:
    assert store.runs() == []
    assert store.total_spend() == 0.0


def test_record_and_read_back(store: Store) -> None:
    store.start_run("r1", kind="benchmark", label="suite")
    store.record("r1", _result(), case_id="c01", repeat=0, score=1.0)
    store.finish_run("r1")

    rows = store.results("r1")
    assert len(rows) == 1
    assert rows[0]["case_id"] == "c01"
    assert rows[0]["score"] == 1.0
    assert rows[0]["input_tokens"] == 100


def test_run_summary_aggregates_cost(store: Store) -> None:
    store.start_run("r1", kind="compare")
    store.record("r1", _result(cost=0.002))
    store.record("r1", _result(cost=0.003))
    store.finish_run("r1")

    run = store.runs()[0]
    assert run["n_results"] == 2
    assert run["total_cost"] == 0.005
    assert run["finished_at"] is not None


def test_spend_by_model_groups_and_orders_by_cost(store: Store) -> None:
    store.start_run("r1", kind="compare")
    store.record("r1", _result(model="mock-small", cost=0.001))
    store.record("r1", _result(model="mock-frontier", cost=0.050))
    store.record("r1", _result(model="mock-small", cost=0.001))

    spend = store.spend_by_model()
    assert spend[0]["model"] == "mock-frontier"
    assert spend[1]["calls"] == 2
    assert store.total_spend() == pytest.approx(0.052)


def test_errored_calls_are_recorded_but_excluded_from_spend_rollup(store: Store) -> None:
    """A failed call still costs latency and must be visible — but not counted as usage."""
    failed = CompletionResult(
        model="mock-small",
        provider="mock",
        text="",
        usage=Usage(),
        latency_ms=12.0,
        cost_usd=0.0,
        prompt_hash="x",
        error="RateLimitError: slow down",
    )
    store.start_run("r1", kind="compare")
    store.record("r1", failed)
    store.record("r1", _result(cost=0.004))

    assert len(store.results("r1")) == 2
    spend = store.spend_by_model()
    assert len(spend) == 1
    assert spend[0]["calls"] == 1


def test_start_run_is_idempotent(store: Store) -> None:
    store.start_run("r1", kind="compare")
    store.start_run("r1", kind="compare")
    assert len(store.runs()) == 1
