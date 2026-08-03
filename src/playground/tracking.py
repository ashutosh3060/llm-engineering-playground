"""Optional MLflow mirror.

The SQLite store is the source of truth. MLflow is a convenience for run-to-run
comparison in a UI, so it is an optional extra (`pip install -e ".[tracking]"`)
and **every failure here degrades to a no-op**. A benchmark costs real money;
losing it because an optional tracker misbehaved would be indefensible.

That guarantee is not theoretical. The first time this module was actually
executed it raised twice: MLflow 3.x rejects the legacy filesystem backend
outright, and the exception propagated far enough to kill the benchmark that had
already been paid for. Both are covered by tests now.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .config import get_settings

if TYPE_CHECKING:
    from .benchmark import BenchmarkSuite, ModelSummary

__all__ = ["log_benchmark"]

log = logging.getLogger(__name__)


def _default_tracking_uri() -> str:
    """A SQLite backend alongside the playground's own store.

    MLflow 3.x refuses the old `./mlruns` file store unless you opt out with
    `MLFLOW_ALLOW_FILE_STORE`. SQLite is MLflow's recommended replacement, needs
    no server, and keeps tracking data next to the run data it mirrors.
    """
    settings = get_settings()
    settings.store_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(settings.store_dir / 'mlflow.db').resolve()}"


def log_benchmark(
    suite: BenchmarkSuite,
    summaries: list[ModelSummary],
    *,
    run_id: str,
    repeats: int,
    effort: str | None,
) -> bool:
    """Mirror a benchmark into MLflow.

    Returns True only if the mirror actually succeeded. Returns False — never
    raises — when tracking is disabled, MLflow is absent, or anything at all
    goes wrong inside it.
    """
    settings = get_settings()
    if not settings.mlflow_enabled:
        return False

    try:
        import mlflow
    except ImportError:
        log.debug("MLflow not installed; skipping mirror.")
        return False

    try:
        # Respect an explicit MLFLOW_TRACKING_URI if the operator set one;
        # otherwise pick a backend that works on MLflow 3.x without opt-outs.
        import os

        if not os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(_default_tracking_uri())

        mlflow.set_experiment(settings.mlflow_experiment)
        for summary in summaries:
            # One MLflow run per model: that is the unit you actually compare in the UI.
            with mlflow.start_run(run_name=f"{suite.name}:{summary.model}"):
                mlflow.log_params(
                    {
                        "suite": suite.name,
                        "model": summary.model,
                        "repeats": repeats,
                        "effort": effort or "default",
                        "cases": len(suite.cases),
                        "benchmark_run_id": run_id,
                    }
                )
                metrics: dict[str, Any] = {
                    "cost_total_usd": summary.total_cost_usd,
                    "cost_per_call_usd": summary.cost_per_call_usd,
                    "p50_latency_ms": summary.p50_latency_ms,
                    "p95_latency_ms": summary.p95_latency_ms,
                    "input_tokens": summary.total_input_tokens,
                    "output_tokens": summary.total_output_tokens,
                    "errors": summary.errors,
                }
                if summary.accuracy is not None:
                    metrics["accuracy"] = summary.accuracy
                mlflow.log_metrics(metrics)
    except Exception as exc:
        # Deliberately broad. Any MLflow failure — backend rejected, disk full,
        # server unreachable, API changed under us — must cost the operator a log
        # line, not the benchmark they already paid for. The full traceback goes
        # to DEBUG so a tracking hiccup does not bury the results table.
        log.warning(
            "MLflow mirror failed (%s: %s); benchmark results are unaffected.",
            type(exc).__name__, exc,
        )
        log.debug("MLflow mirror traceback", exc_info=True)
        return False

    return True
