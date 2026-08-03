"""Tracking tests.

This module shipped having never executed. The first real run raised twice, and
the second failure was the serious one: the exception escaped and killed a
benchmark that had already been paid for. These tests exist so neither returns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from playground.benchmark import BenchmarkSuite, ModelSummary
from playground.prompts import PromptTemplate
from playground.tracking import log_benchmark


@pytest.fixture
def summary() -> ModelSummary:
    return ModelSummary(
        model="mock-small", n=2, errors=0, accuracy=0.75,
        total_cost_usd=0.002, cost_per_call_usd=0.001,
        p50_latency_ms=100.0, p95_latency_ms=140.0,
        total_input_tokens=200, total_output_tokens=80,
    )


@pytest.fixture
def suite() -> BenchmarkSuite:
    return BenchmarkSuite(name="t", template=PromptTemplate(name="t", user="x"), cases=[])


def _reset_settings() -> None:
    from playground import config

    config.get_settings.cache_clear()


def test_disabled_by_default_is_a_noop(
    suite: BenchmarkSuite, summary: ModelSummary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PLAYGROUND_MLFLOW", "0")
    _reset_settings()
    assert log_benchmark(suite, [summary], run_id="r", repeats=1, effort=None) is False


def test_a_broken_mlflow_never_raises(
    suite: BenchmarkSuite, summary: ModelSummary, monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The bug that mattered: a tracking failure must not destroy a paid benchmark.

    Simulated by pointing MLflow at a URI scheme it cannot possibly serve.
    """
    monkeypatch.setenv("PLAYGROUND_MLFLOW", "1")
    monkeypatch.setenv("PLAYGROUND_STORE", str(tmp_path / "s.db"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "not-a-real-scheme://nowhere")
    _reset_settings()

    # Must return False, not raise.
    assert log_benchmark(suite, [summary], run_id="r", repeats=1, effort=None) is False


def test_default_tracking_uri_is_a_database_not_the_file_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MLflow 3.x rejects the legacy `./mlruns` file store outright."""
    from playground.tracking import _default_tracking_uri

    monkeypatch.setenv("PLAYGROUND_STORE", str(tmp_path / "data" / "s.db"))
    _reset_settings()
    uri = _default_tracking_uri()
    assert uri.startswith("sqlite:///")
    assert "mlruns" not in uri


@pytest.mark.skipif(
    pytest.importorskip("mlflow", reason="mlflow not installed") is None, reason="unreachable"
)
def test_real_mirror_writes_params_and_metrics(
    suite: BenchmarkSuite, summary: ModelSummary, monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end against a real MLflow backend, when one is installed."""
    import mlflow

    monkeypatch.setenv("PLAYGROUND_MLFLOW", "1")
    monkeypatch.setenv("PLAYGROUND_STORE", str(tmp_path / "s.db"))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("PLAYGROUND_MLFLOW_EXPERIMENT", "test-exp")
    _reset_settings()

    assert log_benchmark(suite, [summary], run_id="r1", repeats=2, effort="low") is True

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}")
    runs = mlflow.search_runs(experiment_names=["test-exp"])
    assert len(runs) == 1
    assert runs.iloc[0]["params.model"] == "mock-small"
    assert runs.iloc[0]["metrics.accuracy"] == pytest.approx(0.75)
    assert runs.iloc[0]["metrics.cost_per_call_usd"] == pytest.approx(0.001)
