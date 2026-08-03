"""Offline mock provider.

This exists so the entire application — API, UI, benchmark harness, tests — runs
with no API key at all. That matters for two reasons: the test suite must be
green on a clean clone, and a recruiter cloning this repo should see a working
tool rather than a stack trace about a missing credential.

Responses are deterministic (seeded from the prompt hash), so a benchmark over
the mock provider produces stable, reproducible numbers. They are synthetic and
labelled as such everywhere — never quoted as results.
"""

from __future__ import annotations

import hashlib
import random
import time

from ai_core.cost import estimate_cost
from ai_core.models import REGISTRY, ModelSpec, Price, get_model
from ai_core.providers.base import ProviderStatus
from ai_core.types import CompletionRequest, CompletionResult, Usage

MOCK_PROVIDER = "mock"

_MOCK_TIERS = {
    "mock-frontier": ("frontier", 5.00, 25.00, 0.90, 420),
    "mock-balanced": ("balanced", 3.00, 15.00, 0.82, 220),
    "mock-small": ("small", 1.00, 5.00, 0.68, 90),
}


def register_mock_models() -> list[str]:
    """Add mock models to the ai-core registry. Idempotent."""
    from datetime import date

    added = []
    for model_id, (tier, pin, pout, _quality, _lat) in _MOCK_TIERS.items():
        if model_id in REGISTRY:
            added.append(model_id)
            continue
        REGISTRY[model_id] = ModelSpec(
            id=model_id,
            provider=MOCK_PROVIDER,
            display_name=model_id.replace("-", " ").title(),
            context_window=200_000,
            max_output_tokens=16_000,
            tier=tier,
            prices=(Price(pin, pout, date(2020, 1, 1), note="SYNTHETIC — mock provider only"),),
            notes="Offline mock. Deterministic output; never quote these as results.",
        )
        added.append(model_id)
    return added


class MockProvider:
    """Deterministic, offline stand-in for a real provider."""

    name = MOCK_PROVIDER

    def __init__(self, latency: bool = True) -> None:
        register_mock_models()
        # Simulated latency makes the UI and p50/p95 aggregation behave realistically.
        # Tests turn it off so the suite stays fast.
        self._latency = latency

    def available(self) -> bool:
        return True

    def check(self) -> ProviderStatus:
        return ProviderStatus(
            self.name,
            key_present=True,
            reachable=True,
            detail="offline mock — synthetic output, no network",
            models=sorted(_MOCK_TIERS),
        )

    def _seed(self, request: CompletionRequest) -> random.Random:
        raw = f"{request.model}:{request.prompt_hash()}:{request.effort}"
        return random.Random(int(hashlib.sha256(raw.encode()).hexdigest()[:12], 16))

    def complete(self, request: CompletionRequest) -> CompletionResult:
        spec = get_model(request.model)
        if spec.provider != self.name:
            raise ValueError(f"{request.model} is not a mock model")

        _, _, _, quality, base_latency_ms = _MOCK_TIERS[request.model]
        rng = self._seed(request)
        started = time.perf_counter()

        # Jitter around the tier's base latency so p50/p95 are distinguishable.
        simulated_ms = base_latency_ms * rng.uniform(0.75, 1.6)
        if self._latency:
            time.sleep(simulated_ms / 1000.0)

        prompt_chars = sum(len(m.content) for m in request.messages)
        prompt_chars += len(request.system or "")
        # ~4 chars/token is a rough English heuristic. Fine for a synthetic
        # provider; never used for real cost estimation.
        input_tokens = max(1, prompt_chars // 4)
        output_tokens = rng.randint(40, 180)

        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        text = self._synthesize(request, rng, quality)

        return CompletionResult(
            model=request.model,
            provider=self.name,
            text=text,
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000 if self._latency else simulated_ms,
            cost_usd=estimate_cost(request.model, usage),
            stop_reason="end_turn",
            prompt_hash=request.prompt_hash(),
        )

    def _synthesize(self, request: CompletionRequest, rng: random.Random, quality: float) -> str:
        """Produce output that a scorer can meaningfully grade.

        If the prompt looks like a labelled classification task, echo one of the
        labels — correctly at a rate matching the tier's quality. That makes the
        end-to-end scoring path exercisable offline.
        """
        last = request.messages[-1].content if request.messages else ""
        expected = request.metadata.get("expected")

        if expected and rng.random() < quality:
            return str(expected)
        if expected:
            distractors = request.metadata.get("labels") or []
            wrong = [x for x in distractors if x != expected]
            if wrong:
                return str(rng.choice(wrong))
            return f"unsure ({expected[:0]}unknown)"

        return (
            f"[mock:{request.model}] Synthetic response to "
            f"{last[:80]!r}. This provider makes no network calls and its output "
            f"is not a real model's."
        )

    def count_tokens(self, request: CompletionRequest) -> int:
        chars = sum(len(m.content) for m in request.messages) + len(request.system or "")
        return max(1, chars // 4)
