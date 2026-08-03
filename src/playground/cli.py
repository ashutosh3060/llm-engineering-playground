"""Command line interface.

playground probe                     which providers are reachable
playground models                    registry with prices and availability
playground bench datasets/x.yaml     run a benchmark suite
playground spend                     cumulative cost per model
playground serve                     start the FastAPI gateway
playground ui                        start the Streamlit app
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from ai_core.models import REGISTRY
from rich.console import Console
from rich.table import Table

from .benchmark import BenchmarkSuite, run_benchmark
from .config import get_settings
from .runtime import build_gateway, get_store, provider_report
from .tracking import log_benchmark

app = typer.Typer(add_completion=False, help="LLM Engineering Playground")
console = Console()

_STATE = {"no-key": "dim", "ok": "green", "error": "red"}


@app.command()
def probe() -> None:
    """Show which providers are reachable right now."""
    table = Table(title="Providers")
    table.add_column("provider")
    table.add_column("state")
    table.add_column("detail")
    for row in provider_report():
        state = str(row["state"])
        table.add_row(str(row["name"]), f"[{_STATE.get(state, '')}]{state}[/]", str(row["detail"]))
    console.print(table)


@app.command()
def models() -> None:
    """List every known model with today's price and whether it is callable."""
    from datetime import date

    gw = build_gateway()
    available = set(gw.available_models())
    table = Table(title="Model registry")
    for col in ("model", "tier", "in $/Mtok", "out $/Mtok", "ctx", "available"):
        table.add_column(col)
    today = date.today()
    for spec in sorted(REGISTRY.values(), key=lambda s: (s.provider, s.tier, s.id)):
        price = spec.price_on(today)
        is_avail = spec.id in available
        table.add_row(
            spec.id,
            spec.tier,
            f"{price.input_per_mtok:.2f}",
            f"{price.output_per_mtok:.2f}",
            f"{spec.context_window:,}",
            "[green]yes[/]" if is_avail else "[dim]no[/]",
        )
    console.print(table)


@app.command()
def bench(
    suite_path: Annotated[Path, typer.Argument(help="Path to a benchmark suite YAML.")],
    models_csv: Annotated[
        str | None, typer.Option("--models", "-m", help="Comma-separated model IDs.")
    ] = None,
    repeats: Annotated[int | None, typer.Option("--repeats", "-r")] = None,
    effort: Annotated[str | None, typer.Option("--effort", "-e")] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write the summary table as markdown.")
    ] = None,
) -> None:
    """Run a benchmark suite and print the comparison table."""
    settings = get_settings()
    suite = BenchmarkSuite.from_yaml(suite_path)
    gw = build_gateway()

    targets = (
        [m.strip() for m in models_csv.split(",") if m.strip()]
        if models_csv
        else gw.available_models()
    )
    if not targets:
        console.print("[red]No models available.[/] Set a provider key or enable the mock.")
        raise typer.Exit(1)

    n_repeats = repeats or settings.benchmark_repeats
    total = len(suite.cases) * len(targets) * n_repeats
    console.print(
        f"[bold]{suite.name}[/] — {len(suite.cases)} cases x {len(targets)} models "
        f"x {n_repeats} repeats = {total} calls"
    )

    with console.status("running...") as status:

        def progress(done: int, of: int) -> None:
            status.update(f"running... {done}/{of}")

        run_id, summaries, _ = run_benchmark(
            suite,
            targets,
            gateway=gw,
            store=get_store(),
            repeats=n_repeats,
            effort=effort,
            progress=progress,
        )

    table = Table(title=f"{suite.name}  (run {run_id})")
    for col in ("model", "n", "err", "accuracy", "$/call", "$ total", "p50 ms", "p95 ms"):
        table.add_column(col)
    for s in summaries:
        table.add_row(
            s.model,
            str(s.n),
            str(s.errors),
            "—" if s.accuracy is None else f"{s.accuracy:.1%}",
            f"{s.cost_per_call_usd:.6f}",
            f"{s.total_cost_usd:.4f}",
            f"{s.p50_latency_ms:.0f}",
            f"{s.p95_latency_ms:.0f}",
        )
    console.print(table)

    if log_benchmark(suite, summaries, run_id=run_id, repeats=n_repeats, effort=effort):
        console.print("[dim]mirrored to MLflow[/]")

    if out:
        out.write_text(_markdown(suite, summaries, run_id, n_repeats))
        console.print(f"[green]wrote[/] {out}")

    if any(s.model.startswith("mock-") for s in summaries):
        console.print(
            "\n[yellow]NOTE:[/] results include the offline mock provider. "
            "Synthetic numbers — do not quote them as evaluation results."
        )


def _markdown(suite: BenchmarkSuite, summaries: list, run_id: str, repeats: int) -> str:
    lines = [
        f"# {suite.name}",
        "",
        f"Run `{run_id}` · {len(suite.cases)} cases · {repeats} repeats per case.",
        "",
        "| Model | n | Errors | Accuracy | $/call | $ total | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        acc = "—" if s.accuracy is None else f"{s.accuracy:.1%}"
        lines.append(
            f"| `{s.model}` | {s.n} | {s.errors} | {acc} | "
            f"{s.cost_per_call_usd:.6f} | {s.total_cost_usd:.4f} | "
            f"{s.p50_latency_ms:.0f} | {s.p95_latency_ms:.0f} |"
        )
    lines.append("")
    lines.append("Latency is nearest-rank percentile over all repeats, not a single sample.")
    return "\n".join(lines) + "\n"


@app.command()
def spend() -> None:
    """Cumulative cost per model across every recorded run."""
    rows = get_store().spend_by_model()
    if not rows:
        console.print("[dim]No recorded calls yet.[/]")
        return
    table = Table(title="Spend by model")
    for col in ("model", "calls", "cost $", "in tok", "out tok", "avg ms"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r["model"],
            str(r["calls"]),
            f"{r['cost'] or 0:.4f}",
            f"{r['input_tokens'] or 0:,}",
            f"{r['output_tokens'] or 0:,}",
            f"{r['avg_latency_ms'] or 0:.0f}",
        )
    console.print(table)
    console.print(f"[bold]Total:[/] ${get_store().total_spend():.4f}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Start the FastAPI gateway."""
    import uvicorn

    uvicorn.run("playground.api.main:app", host=host, port=port, reload=reload)


@app.command()
def ui(port: int = 8501) -> None:
    """Start the Streamlit app."""
    target = Path(__file__).parent / "ui" / "app.py"
    sys.exit(subprocess.call(["streamlit", "run", str(target), "--server.port", str(port)]))


if __name__ == "__main__":
    app()
