"""Cost model tests — all keyless, since that is the point of the module."""

from __future__ import annotations

from pathlib import Path

import pytest
from ai_core.gateway import Gateway

from playground.benchmark import BenchmarkSuite
from playground.costmodel import (
    cache_breakeven_by_prefix,
    cache_table,
    shape_of_suite,
    tier_table,
)
from playground.mock import MockProvider


@pytest.fixture
def suite(suite_path: Path) -> BenchmarkSuite:
    return BenchmarkSuite.from_yaml(suite_path)


def test_shape_splits_shared_prefix_from_variable_input(suite: BenchmarkSuite) -> None:
    """Only the shared prefix can be cached, so the split is load-bearing."""
    shape = shape_of_suite(suite)
    assert shape.shared_prefix_tokens > 0
    assert shape.variable_input_tokens > 0
    assert shape.total_input_tokens == (
        shape.shared_prefix_tokens + shape.variable_input_tokens
    )


def test_shape_reports_estimated_counts_honestly(suite: BenchmarkSuite) -> None:
    """An estimate presented as exact is worse than no number at all."""
    shape = shape_of_suite(suite)
    assert shape.exact is False
    assert "estimated" in shape.source.lower()


def test_shape_does_not_use_the_mock_provider_for_token_counts(
    suite: BenchmarkSuite,
) -> None:
    """Mock token counts are synthetic — using them would silently fake exactness."""
    gw = Gateway(providers=[MockProvider(latency=False)])
    shape = shape_of_suite(suite, gateway=gw)
    assert shape.exact is False


def test_shape_rejects_an_empty_suite(suite: BenchmarkSuite) -> None:
    suite.cases = []
    with pytest.raises(ValueError, match="no cases"):
        shape_of_suite(suite)


def test_tier_table_is_ordered_cheapest_first(suite: BenchmarkSuite) -> None:
    rows = tier_table(shape_of_suite(suite), calls=10_000)
    assert [r.cost_per_call for r in rows] == sorted(r.cost_per_call for r in rows)
    assert rows[0].tier == "small"


def test_tier_table_excludes_mock_models(suite: BenchmarkSuite) -> None:
    """Synthetic prices must never reach a cost analysis."""
    rows = tier_table(shape_of_suite(suite), calls=100)
    assert all(not r.model.startswith("mock-") for r in rows)


def test_short_prefix_is_reported_as_uncacheable(suite: BenchmarkSuite) -> None:
    """The real finding for both shipped suites: prefixes are far below every minimum."""
    rows = tier_table(shape_of_suite(suite), calls=10_000)
    assert all(not r.cacheable for r in rows), "these suites have ~30-token prefixes"
    assert all(r.cached_at_volume == r.cost_at_volume for r in rows)


def test_caching_is_a_step_function_at_each_model_minimum() -> None:
    rows = cache_breakeven_by_prefix(
        variable_input_tokens=25, output_tokens=16, calls=10_000
    )
    by_prefix = {r["prefix_tokens"]: r for r in rows}

    # Opus 5's minimum is 512; Haiku's is 4096. Between those, the frontier model
    # caches and the cheap one does not — which is the counterintuitive result.
    assert by_prefix[256]["claude-opus-5"] is None
    assert by_prefix[512]["claude-opus-5"] is not None
    assert by_prefix[512]["claude-haiku-4-5"] is None
    assert by_prefix[4096]["claude-haiku-4-5"] is not None


def test_saving_climbs_toward_ninety_percent_with_prefix_length() -> None:
    rows = cache_breakeven_by_prefix(
        variable_input_tokens=0, output_tokens=0, calls=10_000
    )
    savings = [r["claude-opus-5"] for r in rows if r["claude-opus-5"] is not None]
    assert savings == sorted(savings), "saving should increase with prefix length"
    assert savings[-1] < 90.0, "0.1x read rate caps the saving below 90%"


def test_cache_table_shows_a_single_call_costing_more(suite: BenchmarkSuite) -> None:
    """One call pays the write premium and never reads it back."""
    shape = shape_of_suite(suite)
    # Force a cacheable prefix; the shipped suites are all below the minimum.
    shape = type(shape)(
        shared_prefix_tokens=8000,
        variable_input_tokens=shape.variable_input_tokens,
        output_tokens=shape.output_tokens,
        exact=shape.exact,
        source=shape.source,
    )
    rows = cache_table(shape, model="claude-opus-5", volumes=(1, 2, 1000))
    assert rows[0]["saving_usd"] < 0
    assert rows[-1]["saving_usd"] > 0
