"""API tests.

Run against the mock provider through the real app, so route wiring, schemas,
and store persistence are all exercised without a key or network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("PLAYGROUND_STORE", str(tmp_path / "api.db"))
    monkeypatch.setenv("PLAYGROUND_ALLOW_MOCK", "1")

    from playground import config, runtime

    config.get_settings.cache_clear()
    runtime.get_gateway.cache_clear()
    runtime.get_store.cache_clear()

    from playground.api.main import app

    return TestClient(app)


def test_health_reports_providers(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "mock" in body["providers"]
    assert body["models_available"] > 0


def test_providers_endpoint_reports_state_per_provider(client: TestClient) -> None:
    rows = client.get("/providers").json()
    states = {r["name"]: r["state"] for r in rows}
    assert states["mock"] == "ok"
    # Anthropic has no key in CI; it must report cleanly rather than error.
    assert states.get("anthropic") in {"no-key", "ok", "error"}


def test_models_lists_availability_and_price(client: TestClient) -> None:
    models = client.get("/models").json()
    assert models
    available = [m for m in models if m["available"]]
    assert available, "mock models should be available"
    assert all("input_per_mtok" in m for m in models)
    # Available models sort ahead of unavailable ones.
    assert models[0]["available"] is True


def test_complete_returns_usage_and_cost(client: TestClient) -> None:
    r = client.post("/complete", json={"model": "mock-small", "prompt": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["usage"]["input_tokens"] > 0
    assert body["cost_usd"] > 0
    assert body["latency_ms"] > 0
    assert len(body["prompt_hash"]) == 16


def test_complete_requires_prompt_or_messages(client: TestClient) -> None:
    assert client.post("/complete", json={"model": "mock-small"}).status_code == 422


def test_complete_rejects_unavailable_model(client: TestClient) -> None:
    r = client.post("/complete", json={"model": "claude-opus-5", "prompt": "hi"})
    assert r.status_code == 409
    assert "probe" in r.json()["detail"]


def test_compare_returns_rows_sorted_by_cost(client: TestClient) -> None:
    r = client.post(
        "/compare",
        json={
            "prompt": "summarise this",
            "models": ["mock-frontier", "mock-small"],
            "repeats": 2,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["rows"]) == 2
    costs = [row["cost_per_call_usd"] for row in body["rows"]]
    assert costs == sorted(costs)
    assert all(row["n"] == 2 for row in body["rows"])


def test_compare_persists_a_run(client: TestClient) -> None:
    run_id = client.post(
        "/compare", json={"prompt": "x", "models": ["mock-small"], "repeats": 2}
    ).json()["run_id"]

    runs = client.get("/runs").json()
    assert any(r["id"] == run_id for r in runs)

    detail = client.get(f"/runs/{run_id}").json()
    assert len(detail["results"]) == 2


def test_run_detail_404s_for_unknown_run(client: TestClient) -> None:
    assert client.get("/runs/does-not-exist").status_code == 404


def test_count_tokens_includes_cost_estimate(client: TestClient) -> None:
    body = client.post("/count-tokens", json={"model": "mock-small", "prompt": "a" * 400}).json()
    assert body["input_tokens"] > 0
    assert body["estimated_input_cost_usd"] > 0


def test_spend_aggregates_across_runs(client: TestClient) -> None:
    client.post("/complete", json={"model": "mock-small", "prompt": "one"})
    client.post("/complete", json={"model": "mock-small", "prompt": "two"})
    spend = client.get("/spend").json()
    assert spend[0]["model"] == "mock-small"
    assert spend[0]["calls"] == 2


def test_stream_emits_message_usage_and_done(client: TestClient) -> None:
    with client.stream(
        "POST", "/complete/stream", json={"model": "mock-small", "prompt": "hi"}
    ) as r:
        payload = "".join(chunk for chunk in r.iter_text())
    assert "event: message" in payload
    assert "event: usage" in payload
    assert "event: done" in payload
