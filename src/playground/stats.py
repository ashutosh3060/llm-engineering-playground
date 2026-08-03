"""Latency aggregation.

One implementation, used by the benchmark harness, the API, and the UI — three
copies would inevitably disagree about what "p50" means, and the whole point of
this project is that the numbers are trustworthy.
"""

from __future__ import annotations

__all__ = ["percentile"]


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile.

    Deliberately not interpolated: with the sample counts a benchmark produces
    (often 5-20 per model), linear interpolation invents precision the data does
    not support.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(pct / 100 * len(ordered))))
    return ordered[rank - 1]
