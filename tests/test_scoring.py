from __future__ import annotations

import math

import pytest

from playground.scoring import get_scorer, score_result


@pytest.mark.parametrize(
    ("output", "expected", "want"),
    [
        ("positive", "positive", 1.0),
        ("Positive.", "positive", 1.0),
        ("  POSITIVE  ", "positive", 1.0),
        ("negative", "positive", 0.0),
    ],
)
def test_exact_match_normalises(output: str, expected: str, want: float) -> None:
    assert get_scorer("exact")(output, expected).value == want


def test_label_scorer_handles_a_label_inside_a_sentence() -> None:
    assert get_scorer("label")("I'd say this is positive overall.", "positive").value == 1.0


def test_label_scorer_rejects_negated_label() -> None:
    """'not positive' must not score as 'positive' — the classic false positive."""
    assert get_scorer("label")("This is not positive.", "positive").value == 0.0


def test_label_scorer_requires_word_boundary() -> None:
    assert get_scorer("label")("nonpositive", "positive").value == 0.0


def test_json_scorer_accepts_fenced_output() -> None:
    out = '```json\n{"name": "Ada", "email": null}\n```'
    assert get_scorer("json")(out, ["name", "email"]).value == 1.0


def test_json_scorer_reports_partial_key_coverage() -> None:
    score = get_scorer("json")('{"name": "Ada"}', ["name", "email"])
    assert score.value == 0.5
    assert score.detail["missing"] == ["email"]


def test_json_scorer_fails_on_invalid_json() -> None:
    score = get_scorer("json")("not json at all", ["name"])
    assert score.value == 0.0
    assert "invalid json" in score.detail["error"]


def test_numeric_scorer_tolerates_one_percent() -> None:
    assert get_scorer("numeric")("about 100.5", 100).value == 1.0  # 0.5% off — inside
    assert get_scorer("numeric")("about 105", 100).value == 0.0  # 5% off — outside
    assert get_scorer("numeric")("1,000", 1000).value == 1.0  # thousands separator
    assert get_scorer("numeric")("it costs -42 dollars", -42).value == 1.0  # negatives


def test_unscored_when_no_expected_value() -> None:
    assert math.isnan(score_result("anything", None).value)


def test_unknown_scorer_lists_the_valid_ones() -> None:
    with pytest.raises(ValueError, match="Available:"):
        get_scorer("vibes")
