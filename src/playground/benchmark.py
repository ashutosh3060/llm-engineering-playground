"""Benchmark harness.

Runs a suite of cases across several models, with repeats, and aggregates into
the table that answers the Month 1 question: for this workload, what does each
model cost, how fast is it, and is the cheap one good enough?

Two things here are load-bearing:

1. **Repeats, always.** A single latency sample is noise. Every latency figure
   this module produces is a p50/p95 over `repeats` runs.
2. **Concurrent dispatch.** Running models sequentially adds queueing time to
   every model after the first, so the numbers would measure the harness rather
   than the provider.
"""

from __future__ import annotations

import statistics
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from ai_core.gateway import Gateway
from ai_core.types import CompletionResult

from .prompts import FewShot, PromptTemplate
from .scoring import score_result
from .stats import percentile
from .store import Store

__all__ = ["BenchmarkCase", "BenchmarkSuite", "ModelSummary", "run_benchmark"]


@dataclass
class BenchmarkCase:
    id: str
    variables: dict[str, Any] = field(default_factory=dict)
    expected: Any = None
    scorer: str = "label"
    labels: list[str] = field(default_factory=list)


@dataclass
class BenchmarkSuite:
    name: str
    template: PromptTemplate
    cases: list[BenchmarkCase]
    scorer: str = "label"
    max_tokens: int = 512
    description: str = ""

    @classmethod
    def from_yaml(cls, path: Path | str) -> BenchmarkSuite:
        data = yaml.safe_load(Path(path).read_text())
        tpl = data["template"]
        template = PromptTemplate(
            name=data["name"],
            system=tpl.get("system"),
            user=tpl["user"],
            few_shots=[FewShot(**s) for s in tpl.get("few_shots", [])],
            description=data.get("description", ""),
        )
        default_scorer = data.get("scorer", "label")
        cases = [
            BenchmarkCase(
                id=str(c["id"]),
                variables=c.get("variables", {}),
                expected=c.get("expected"),
                scorer=c.get("scorer", default_scorer),
                labels=c.get("labels", data.get("labels", [])),
            )
            for c in data["cases"]
        ]
        return cls(
            name=data["name"],
            template=template,
            cases=cases,
            scorer=default_scorer,
            max_tokens=int(data.get("max_tokens", 512)),
            description=data.get("description", ""),
        )


@dataclass
class ModelSummary:
    model: str
    n: int
    errors: int
    accuracy: float | None
    total_cost_usd: float
    cost_per_call_usd: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int

    def as_row(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n": self.n,
            "errors": self.errors,
            "accuracy": None if self.accuracy is None else round(self.accuracy, 4),
            "cost_total_usd": round(self.total_cost_usd, 6),
            "cost_per_call_usd": round(self.cost_per_call_usd, 6),
            "p50_ms": round(self.p50_latency_ms, 1),
            "p95_ms": round(self.p95_latency_ms, 1),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }


def run_benchmark(
    suite: BenchmarkSuite,
    models: list[str],
    *,
    gateway: Gateway | None = None,
    store: Store | None = None,
    repeats: int = 5,
    effort: str | None = None,
    max_workers: int = 8,
    run_id: str | None = None,
    progress: Any = None,
) -> tuple[str, list[ModelSummary], list[CompletionResult]]:
    """Execute the suite and return (run_id, per-model summaries, raw results)."""
    gw = gateway or Gateway()
    run_id = run_id or f"bench-{uuid.uuid4().hex[:8]}"

    if store:
        store.start_run(
            run_id,
            kind="benchmark",
            label=suite.name,
            models=models,
            repeats=repeats,
            effort=effort,
            cases=len(suite.cases),
        )

    jobs: list[tuple[BenchmarkCase, str, int]] = [
        (case, model, rep) for case in suite.cases for model in models for rep in range(repeats)
    ]

    def execute(job: tuple[BenchmarkCase, str, int]) -> tuple[BenchmarkCase, int, CompletionResult]:
        case, model, rep = job
        version = suite.template.render(**case.variables)
        request = version.to_request(
            model,
            max_tokens=suite.max_tokens,
            effort=effort,
            metadata={
                "case_id": case.id,
                "expected": case.expected,
                "labels": case.labels,
            },
        )
        return case, rep, gw.complete(request)

    results: list[CompletionResult] = []
    per_model: dict[str, list[tuple[CompletionResult, float | None]]] = {m: [] for m in models}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for done, (case, rep, result) in enumerate(pool.map(execute, jobs), start=1):
            results.append(result)
            score: float | None = None
            detail: dict[str, Any] | None = None
            if result.ok and case.expected is not None:
                s = score_result(result.text, case.expected, case.scorer)
                score, detail = s.value, s.detail
            per_model.setdefault(result.model, []).append((result, score))

            if store:
                store.record(
                    run_id,
                    result,
                    case_id=case.id,
                    repeat=rep,
                    prompt_label=suite.name,
                    score=score,
                    score_detail=detail,
                )
            if progress:
                progress(done, len(jobs))

    if store:
        store.finish_run(run_id)

    return run_id, _summarise(per_model), results


def _summarise(
    per_model: dict[str, list[tuple[CompletionResult, float | None]]],
) -> list[ModelSummary]:
    summaries: list[ModelSummary] = []
    for model, entries in per_model.items():
        if not entries:
            continue
        ok = [(r, s) for r, s in entries if r.ok]
        errors = len(entries) - len(ok)
        latencies = [r.latency_ms for r, _ in ok]
        scores = [s for _, s in ok if s is not None]
        total_cost = sum(r.cost_usd for r, _ in ok)

        summaries.append(
            ModelSummary(
                model=model,
                n=len(entries),
                errors=errors,
                accuracy=statistics.fmean(scores) if scores else None,
                total_cost_usd=total_cost,
                cost_per_call_usd=total_cost / len(ok) if ok else 0.0,
                p50_latency_ms=percentile(latencies, 50),
                p95_latency_ms=percentile(latencies, 95),
                total_input_tokens=sum(r.usage.input_tokens for r, _ in ok),
                total_output_tokens=sum(r.usage.output_tokens for r, _ in ok),
            )
        )

    # Cheapest first — the operative question is "what is the least I can spend
    # and still get an acceptable answer?"
    summaries.sort(key=lambda s: s.cost_per_call_usd)
    return summaries
