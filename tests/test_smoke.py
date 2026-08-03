"""Smoke tests — these must pass on a clean clone with no API key."""

from __future__ import annotations

from ai_core.gateway import Gateway
from ai_core.types import CompletionRequest, Message

import playground
from playground.mock import MockProvider, register_mock_models
from playground.runtime import build_gateway


def test_package_imports() -> None:
    assert playground.__version__


def test_mock_models_register_idempotently() -> None:
    first = register_mock_models()
    second = register_mock_models()
    assert first == second
    assert "mock-small" in first


def test_gateway_is_usable_with_no_api_key() -> None:
    """The whole point of the mock provider: a stranger can clone and run this."""
    gw = build_gateway(include_mock=True, mock_latency=False)
    assert "mock" in gw.provider_names
    assert any(m.startswith("mock-") for m in gw.available_models())


def test_mock_output_is_deterministic() -> None:
    gw = Gateway(providers=[MockProvider(latency=False)])
    request = CompletionRequest(
        model="mock-small", messages=[Message(role="user", content="stable?")]
    )
    assert gw.complete(request).text == gw.complete(request).text


def test_mock_respects_expected_label_metadata() -> None:
    """The scoring path needs the mock to be right sometimes and wrong others."""
    gw = Gateway(providers=[MockProvider(latency=False)])
    outputs = set()
    for i in range(40):
        request = CompletionRequest(
            model="mock-small",
            messages=[Message(role="user", content=f"case {i}")],
            metadata={"expected": "positive", "labels": ["positive", "negative"]},
        )
        outputs.add(gw.complete(request).text)
    assert outputs <= {"positive", "negative"}
    assert len(outputs) == 2, "mock should produce both correct and incorrect labels"


def test_mock_cost_scales_with_tier() -> None:
    gw = Gateway(providers=[MockProvider(latency=False)])
    request = CompletionRequest(
        model="mock-small", messages=[Message(role="user", content="x" * 400)]
    )
    small = gw.complete(request).cost_usd
    big = gw.complete(request.model_copy(update={"model": "mock-frontier"})).cost_usd
    assert big > small

