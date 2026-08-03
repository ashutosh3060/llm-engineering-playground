"""Parameter sweeps.

Expands a dict of axis -> candidate values into the full grid, so a prompt
laboratory experiment is declared once and executed exhaustively.

Grids grow multiplicatively and each cell is a paid API call, so `expand` refuses
to silently produce something enormous — an explicit `limit` is required to go
past the guard.
"""

from __future__ import annotations

from itertools import product
from typing import Any

__all__ = ["SweepTooLarge", "expand", "grid_size"]

DEFAULT_MAX_CELLS = 200


class SweepTooLarge(ValueError):
    """Raised when a grid exceeds the safety limit."""


def grid_size(axes: dict[str, list[Any]]) -> int:
    size = 1
    for values in axes.values():
        size *= max(1, len(values))
    return size


def expand(
    axes: dict[str, list[Any]],
    *,
    limit: int = DEFAULT_MAX_CELLS,
    repeats: int = 1,
) -> list[dict[str, Any]]:
    """Cartesian product of every axis.

    ``repeats`` multiplies the cost as well, so it counts toward the limit — a
    5x repeat over a 60-cell grid is 300 calls, which is exactly the kind of
    accidental spend the guard exists to catch.
    """
    if not axes:
        return [{}] * repeats

    total = grid_size(axes) * max(1, repeats)
    if total > limit:
        raise SweepTooLarge(
            f"Sweep would issue {total} calls ({grid_size(axes)} cells x {repeats} repeats), "
            f"above the limit of {limit}. Narrow the grid or pass a higher `limit` explicitly."
        )

    names = list(axes)
    cells = [dict(zip(names, combo, strict=True)) for combo in product(*(axes[n] for n in names))]
    return [cell for cell in cells for _ in range(max(1, repeats))]
