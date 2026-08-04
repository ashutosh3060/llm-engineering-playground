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
console = Console(width=118)

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


def _cost_markdown(suite, shape, rows, calls: int, cache_table_fn) -> str:
    """Render the cost analysis as markdown, carrying its own provenance."""
    from datetime import date

    provenance = (
        f"Token counts are **exact** — {shape.source}."
        if shape.exact
        else f"Token counts are **estimated** — {shape.source}. "
        f"The arithmetic below is exact given those counts; the counts themselves "
        f"are not. Set `ANTHROPIC_API_KEY` and re-run for provider-exact figures."
    )

    lines = [
        f"# Cost model — {suite.name}",
        "",
        f"Generated {date.today().isoformat()} · {len(suite.cases)} cases · "
        f"pricing as of today from the versioned table.",
        "",
        provenance,
        "",
        "## Prompt shape",
        "",
        "| Component | Tokens | Notes |",
        "|---|---|---|",
        f"| Shared prefix | {shape.shared_prefix_tokens:,} | system prompt + few-shots; "
        f"identical every call, so the only cacheable part |",
        f"| Variable input | {shape.variable_input_tokens:,} | mean per case |",
        f"| Output budget | {shape.output_tokens:,} | `max_tokens` for this suite |",
        "",
        f"## Unit economics at {calls:,} calls/month",
        "",
        "| Model | Tier | $/call | Monthly | Cache min | Cacheable | Monthly cached | Saving |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.model}` | {r.tier} | ${r.cost_per_call:.6f} | ${r.cost_at_volume:,.2f} | "
            f"{r.cache_min_tokens:,} | {'yes' if r.cacheable else '**no**'} | "
            f"{f'${r.cached_at_volume:,.2f}' if r.cacheable else '—'} | "
            f"{f'{r.saving_pct:.0f}%' if r.cacheable else '—'} |"
        )

    cheapest, dearest = rows[0], rows[-1]
    spread = dearest.cost_per_call / cheapest.cost_per_call
    lines += [
        "",
        "## Findings",
        "",
        f"**1. The tier spread is {spread:.1f}x.** `{cheapest.model}` costs "
        f"${cheapest.cost_at_volume:,.2f}/month for this workload against "
        f"${dearest.cost_at_volume:,.2f} for `{dearest.model}`. Whether that gap is "
        f"worth paying is a quality question, which needs a benchmark run — see "
        f"`evaluation.md`.",
        "",
    ]

    cacheable = [r for r in rows if r.cacheable]
    blocked = [r for r in rows if not r.cacheable]
    if blocked:
        lines += [
            f"**2. The shared prefix is {shape.shared_prefix_tokens:,} tokens, which is too "
            f"short to cache on {', '.join(f'`{r.model}`' for r in blocked)}.** Every model "
            f"has a minimum cacheable prefix, it varies eightfold across the Claude line "
            f"(512 on Opus 5 to 4096 on Haiku 4.5), and below it the provider simply "
            f"declines to cache — no error, just no saving. Any cost plan assuming a "
            f"uniform caching discount is wrong for this workload.",
            "",
        ]
    if cacheable:
        best = min(cacheable, key=lambda r: r.cached_at_volume)
        lines += [
            f"**3. Caching changes the ranking.** With its prefix cached, `{best.model}` "
            f"costs ${best.cached_at_volume:,.2f}/month — against "
            f"${cheapest.cost_at_volume:,.2f} for the cheapest uncached option "
            f"(`{cheapest.model}`). The intuition that the small model is always the "
            f"cheap choice does not survive contact with caching economics.",
            "",
            f"### How the saving scales — `{best.model}`",
            "",
            "| Calls | Uncached | Cached | Saving |",
            "|---|---|---|---|",
        ]
        for row in cache_table_fn(shape, model=best.model):
            lines.append(
                f"| {row['calls']:,} | ${row['uncached_usd']:,.4f} | "
                f"${row['cached_usd']:,.4f} | {row['saving_pct']:+.1f}% |"
            )
        lines += [
            "",
            "Note the first row: a **single** call costs *more* cached than uncached, "
            "because it pays the 1.25x write premium and never reads the cache back. "
            "Break-even is the second call.",
            "",
        ]

    lines += [
        "## Method",
        "",
        "- Costs come from `ai-core`'s versioned pricing table, resolved at today's date, "
        "so an introductory rate that has lapsed is not applied retroactively.",
        "- Cache modelling uses the real billing shape: the first call writes the prefix "
        "at 1.25x the base input rate, later calls read it at 0.1x, and the variable "
        "portion of each prompt is billed normally throughout.",
        "- A prefix shorter than a model's minimum is modelled as **uncached**, because "
        "that is what the provider actually does.",
        "",
        "Reproduce with:",
        "",
        "```bash",
        f"playground cost datasets/{suite.name}.yaml --calls {calls}",
        "```",
        "",
    ]
    return "\n".join(lines) + "\n"


@app.command()
def cost(
    suite_path: Annotated[Path, typer.Argument(help="Benchmark suite YAML to model.")],
    calls: Annotated[int, typer.Option("--calls", "-n", help="Monthly call volume.")] = 10_000,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write the analysis as markdown.")
    ] = None,
) -> None:
    """Model a suite's unit economics. Makes no API calls unless a key is set.

    Cost is arithmetic — token count x price — so the whole cost analysis is
    computable offline. Token counts come from the provider when a key is
    available and from a character heuristic otherwise; the output says which.
    """
    from .costmodel import cache_table, shape_of_suite, tier_table

    suite = BenchmarkSuite.from_yaml(suite_path)
    gw = build_gateway()
    shape = shape_of_suite(suite, gateway=gw)

    console.print(f"\n[bold]{suite.name}[/] — prompt shape")
    console.print(f"  shared prefix   {shape.shared_prefix_tokens:>7,} tokens (cacheable part)")
    console.print(f"  variable input  {shape.variable_input_tokens:>7,} tokens (per case)")
    console.print(f"  output budget   {shape.output_tokens:>7,} tokens")
    marker = "[green]exact[/]" if shape.exact else "[yellow]ESTIMATED[/]"
    console.print(f"  token source    {marker} — {shape.source}\n")

    rows = tier_table(shape, calls=calls)
    table = Table(title=f"Unit economics at {calls:,} calls/month")
    for col in ("model", "tier", "$/call", f"${calls:,} calls", "cache min", "cacheable",
                "cached", "saving"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r.model, r.tier,
            f"{r.cost_per_call:.6f}",
            f"{r.cost_at_volume:,.2f}",
            f"{r.cache_min_tokens:,}",
            "[green]yes[/]" if r.cacheable else "[red]no[/]",
            f"{r.cached_at_volume:,.2f}" if r.cacheable else "—",
            f"{r.saving_pct:.0f}%" if r.cacheable else "—",
        )
    console.print(table)

    cheapest, dearest = rows[0], rows[-1]
    console.print(
        f"\n  Tier spread: [bold]{dearest.cost_per_call / cheapest.cost_per_call:.1f}x[/] "
        f"between {cheapest.model} and {dearest.model}."
    )
    cacheable = [r for r in rows if r.cacheable]
    if cacheable:
        best = min(cacheable, key=lambda r: r.cached_at_volume)
        console.print(
            f"  Cheapest *cached* option is [bold]{best.model}[/] at "
            f"${best.cached_at_volume:,.2f}, vs ${cheapest.cost_at_volume:,.2f} for "
            f"the cheapest uncached ({cheapest.model})."
        )
    if any(not r.cacheable for r in rows):
        blocked = ", ".join(r.model for r in rows if not r.cacheable)
        console.print(
            f"  [yellow]Prefix too short to cache on:[/] {blocked} — "
            f"the saving is unavailable there at any volume."
        )

    if out:
        out.write_text(_cost_markdown(suite, shape, rows, calls, cache_table))
        console.print(f"\n[green]wrote[/] {out}")

    if not shape.exact:
        console.print(
            "\n[yellow]NOTE:[/] token counts are estimated. Set ANTHROPIC_API_KEY and "
            "re-run for exact counts from the provider's tokenizer."
        )


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
