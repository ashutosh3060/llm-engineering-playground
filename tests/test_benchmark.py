from __future__ import annotations

from pathlib import Path

from ai_core.gateway import Gateway

from playground.benchmark import BenchmarkSuite, run_benchmark
from playground.store import Store


def test_suite_loads_from_yaml(suite_path: Path) -> None:
    suite = BenchmarkSuite.from_yaml(suite_path)
    assert suite.name == "sentiment-classification"
    assert len(suite.cases) == 20
    assert suite.cases[0].expected == "positive"
    # Suite-level labels must propagate to cases that do not override them.
    assert "neutral" in suite.cases[0].labels


def test_benchmark_produces_one_summary_per_model(
    suite_path: Path, gateway: Gateway, store: Store
) -> None:
    suite = BenchmarkSuite.from_yaml(suite_path)
    suite.cases = suite.cases[:3]
    models = ["mock-small", "mock-frontier"]

    run_id, summaries, results = run_benchmark(
        suite, models, gateway=gateway, store=store, repeats=2
    )

    assert len(summaries) == 2
    assert {s.model for s in summaries} == set(models)
    assert len(results) == 3 * 2 * 2  # cases x models x repeats
    assert all(s.n == 6 for s in summaries)
    assert store.runs()[0]["id"] == run_id


def test_summaries_are_ordered_cheapest_first(
    suite_path: Path, gateway: Gateway, store: Store
) -> None:
    suite = BenchmarkSuite.from_yaml(suite_path)
    suite.cases = suite.cases[:2]
    _, summaries, _ = run_benchmark(
        suite,
        ["mock-frontier", "mock-small"],
        gateway=gateway,
        store=store,
        repeats=1,
    )
    costs = [s.cost_per_call_usd for s in summaries]
    assert costs == sorted(costs)


def test_every_result_is_persisted_with_its_case(
    suite_path: Path, gateway: Gateway, store: Store
) -> None:
    suite = BenchmarkSuite.from_yaml(suite_path)
    suite.cases = suite.cases[:2]
    run_id, _, _ = run_benchmark(suite, ["mock-small"], gateway=gateway, store=store, repeats=3)
    rows = store.results(run_id)
    assert len(rows) == 6
    assert {r["case_id"] for r in rows} == {"c01", "c02"}
    assert sorted({r["repeat"] for r in rows}) == [0, 1, 2]


def test_accuracy_is_computed_from_scored_cases(
    suite_path: Path, gateway: Gateway, store: Store
) -> None:
    suite = BenchmarkSuite.from_yaml(suite_path)
    _, summaries, _ = run_benchmark(
        suite, ["mock-frontier", "mock-small"], gateway=gateway, store=store, repeats=2
    )
    for s in summaries:
        assert s.accuracy is not None
        assert 0.0 <= s.accuracy <= 1.0

    # The mock's frontier tier is seeded to be more accurate than its small tier.
    # This asserts the scoring path is actually wired up, not that any real model
    # behaves this way.
    by_model = {s.model: s.accuracy for s in summaries}
    assert by_model["mock-frontier"] > by_model["mock-small"]


def test_latency_percentiles_are_populated(
    suite_path: Path, gateway: Gateway, store: Store
) -> None:
    suite = BenchmarkSuite.from_yaml(suite_path)
    suite.cases = suite.cases[:2]
    _, summaries, _ = run_benchmark(suite, ["mock-small"], gateway=gateway, store=store, repeats=5)
    s = summaries[0]
    assert s.p95_latency_ms >= s.p50_latency_ms > 0


def test_summary_row_is_json_serialisable(suite_path: Path, gateway: Gateway, store: Store) -> None:
    import json

    suite = BenchmarkSuite.from_yaml(suite_path)
    suite.cases = suite.cases[:1]
    _, summaries, _ = run_benchmark(suite, ["mock-small"], gateway=gateway, store=store, repeats=1)
    json.dumps(summaries[0].as_row())


def test_extraction_suite_scores_structurally(gateway: Gateway, store: Store) -> None:
    path = Path(__file__).resolve().parents[1] / "datasets" / "structured-extraction.yaml"
    suite = BenchmarkSuite.from_yaml(path)
    assert suite.scorer == "json"
    assert len(suite.cases) == 10
    _, summaries, _ = run_benchmark(suite, ["mock-small"], gateway=gateway, store=store, repeats=1)
    assert summaries[0].n == 10
