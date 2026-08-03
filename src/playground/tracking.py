"""Optional MLflow mirror.

The SQLite store is the source of truth. MLflow is a convenience for run-to-run
comparison in a UI, so it is an optional extra (`pip install -e ".[tracking]"`)
and every call here degrades to a no-op when it is absent or disabled. A missing
optional dependency must never break a benchmark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import get_settings

if TYPE_CHECKING:
    from .benchmark import BenchmarkSuite, ModelSummary

__all__ = ["log_benchmark"]


def log_benchmark(
    suite: BenchmarkSuite,
    summaries: list[ModelSummary],
    *,
    run_id: str,
    repeats: int,
    effort: str | None,
) -> bool:
    """Mirror a benchmark into MLflow. Returns False if it was skipped."""
    settings = get_settings()
    if not settings.mlflow_enabled:
        return False

    try:
        import mlflow
    except ImportError:
        return False

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
    return True
