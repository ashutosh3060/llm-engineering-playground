from __future__ import annotations

import pytest

from playground.sweep import SweepTooLarge, expand, grid_size


def test_grid_is_the_cartesian_product() -> None:
    cells = expand({"model": ["a", "b"], "effort": ["low", "high"]})
    assert len(cells) == 4
    assert {"model": "a", "effort": "low"} in cells


def test_repeats_duplicate_each_cell() -> None:
    cells = expand({"model": ["a"]}, repeats=3)
    assert len(cells) == 3


def test_grid_size_ignores_repeats() -> None:
    assert grid_size({"a": [1, 2], "b": [1, 2, 3]}) == 6


def test_empty_axes_yields_one_empty_cell_per_repeat() -> None:
    assert expand({}, repeats=2) == [{}, {}]


def test_oversized_grid_is_refused() -> None:
    """Each cell is a paid call — a runaway grid must fail loudly, not silently bill."""
    with pytest.raises(SweepTooLarge, match="above the limit"):
        expand({"m": list(range(30)), "e": list(range(30))})


def test_repeats_count_toward_the_limit() -> None:
    """60 cells x 5 repeats is 300 calls, not 60 — the guard must see the real total."""
    axes = {"m": list(range(20)), "e": list(range(3))}
    expand(axes, limit=200)  # 60 calls, fine
    with pytest.raises(SweepTooLarge):
        expand(axes, repeats=5, limit=200)


def test_limit_can_be_raised_explicitly() -> None:
    assert len(expand({"m": list(range(30)), "e": list(range(30))}, limit=1000)) == 900
