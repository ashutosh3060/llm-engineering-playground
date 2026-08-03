"""Lightweight scoring.

Deliberately shallow. Rigorous evaluation — LLM-as-judge calibration, faithfulness,
regression gating — is Month 4's `llm-evaluation-platform`. Duplicating it here
would fork the eval logic and guarantee the two drift apart.

What lives here is only what a *model comparison* needs: a cheap, deterministic
signal for "did this model get it right", so a comparison table has a quality
column and not just cost and latency.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["SCORERS", "Score", "get_scorer", "score_result"]


@dataclass(frozen=True)
class Score:
    value: float  # 0.0 - 1.0
    detail: dict[str, Any]


Scorer = Callable[[str, Any], Score]


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower().rstrip(".!?")


def exact_match(output: str, expected: Any) -> Score:
    got, want = _normalise(output), _normalise(str(expected))
    return Score(1.0 if got == want else 0.0, {"got": got[:200], "want": want})


def contains(output: str, expected: Any) -> Score:
    """Substring match — the right scorer for a label buried in a sentence."""
    got, want = _normalise(output), _normalise(str(expected))
    return Score(1.0 if want in got else 0.0, {"got": got[:200], "want": want})


def label_match(output: str, expected: Any) -> Score:
    """Classification scorer.

    Looks for the expected label as a whole word. Word-boundary matching avoids
    the classic false positive where expecting "positive" is satisfied by the
    model writing "not positive".
    """
    want = _normalise(str(expected))
    got = _normalise(output)
    hit = re.search(rf"(?<!\bnot )\b{re.escape(want)}\b", got) is not None
    return Score(1.0 if hit else 0.0, {"got": got[:200], "want": want})


def json_valid(output: str, expected: Any = None) -> Score:
    """Structural scorer: does the output parse, and does it have the right keys?"""
    text = output.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return Score(0.0, {"error": f"invalid json: {exc}"})

    if not expected:
        return Score(1.0, {"keys": sorted(parsed) if isinstance(parsed, dict) else "non-object"})

    required = expected if isinstance(expected, list) else list(expected)
    if not isinstance(parsed, dict):
        return Score(0.0, {"error": "expected a JSON object"})
    present = [k for k in required if k in parsed]
    return Score(
        len(present) / len(required) if required else 1.0,
        {"present": present, "missing": [k for k in required if k not in parsed]},
    )


def numeric_close(output: str, expected: Any) -> Score:
    """Extract the first number and compare within 1% tolerance."""
    match = re.search(r"-?\d+(?:\.\d+)?", output.replace(",", ""))
    if not match:
        return Score(0.0, {"error": "no number in output"})
    got = float(match.group())
    want = float(expected)
    tolerance = abs(want) * 0.01 if want else 1e-9
    return Score(1.0 if abs(got - want) <= tolerance else 0.0, {"got": got, "want": want})


SCORERS: dict[str, Scorer] = {
    "exact": exact_match,
    "contains": contains,
    "label": label_match,
    "json": json_valid,
    "numeric": numeric_close,
}


def get_scorer(name: str) -> Scorer:
    try:
        return SCORERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown scorer {name!r}. Available: {', '.join(sorted(SCORERS))}"
        ) from None


def score_result(output: str, expected: Any, scorer: str = "label") -> Score:
    if expected is None:
        return Score(float("nan"), {"note": "no expected value — unscored"})
    return get_scorer(scorer)(output, expected)
