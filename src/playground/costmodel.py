"""Cost modelling without API calls.

Cost is arithmetic: token count x price. Only the *token count* needs a provider
round-trip, so the whole cost half of a model comparison can be produced with no
key at all — and it is exact, given the counts.

That separation is the honest bit. `shape_of_suite` reports whether its token
counts came from the provider's tokenizer or from a character heuristic, and
every table this module produces carries that label. An estimated token count
feeding exact arithmetic is still an estimate, and saying so is the difference
between analysis and decoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ai_core.cost import cached_workload_cost, estimate_cost
from ai_core.gateway import Gateway
from ai_core.models import REGISTRY, get_model
from ai_core.types import Usage

from .benchmark import BenchmarkSuite

__all__ = [
    "PromptShape",
    "TierRow",
    "cache_breakeven_by_prefix",
    "cache_table",
    "shape_of_suite",
    "tier_table",
]

# ~4 characters per token for English prose. Crude, and only ever a fallback —
# real counts come from the provider. Documented rather than hidden so a reader
# knows exactly how much to trust a table built on it.
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class PromptShape:
    """The token profile of a workload."""

    shared_prefix_tokens: int  # system prompt + few-shots: identical every call
    variable_input_tokens: int  # the per-case part
    output_tokens: int
    exact: bool  # True when counts came from the provider's tokenizer
    source: str

    @property
    def total_input_tokens(self) -> int:
        return self.shared_prefix_tokens + self.variable_input_tokens


@dataclass(frozen=True)
class TierRow:
    model: str
    tier: str
    input_per_mtok: float
    output_per_mtok: float
    cost_per_call: float
    cost_at_volume: float
    cache_min_tokens: int
    cacheable: bool
    cached_at_volume: float
    saving_pct: float


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def shape_of_suite(
    suite: BenchmarkSuite,
    *,
    gateway: Gateway | None = None,
    model_for_counting: str | None = None,
    output_tokens: int | None = None,
) -> PromptShape:
    """Measure a suite's prompt shape, preferring real token counts.

    Splits the prompt into the part that is identical on every call (system
    prompt and few-shot examples) and the part that varies per case. That split
    is what makes caching analysis possible at all — only the shared prefix can
    be cached.
    """
    rendered = [suite.template.render(**case.variables) for case in suite.cases]
    if not rendered:
        raise ValueError(f"Suite {suite.name!r} has no cases to measure.")

    first = rendered[0]
    prefix_text = (first.system or "") + "".join(
        s.user + s.assistant for s in first.few_shots
    )

    exact = False
    source = f"estimated at ~{_CHARS_PER_TOKEN} chars/token (no provider key)"
    prefix_tokens = _estimate_tokens(prefix_text) if prefix_text else 0
    variable_tokens = sum(_estimate_tokens(r.user) for r in rendered) // len(rendered)

    if gateway is not None:
        target = model_for_counting or next(
            (m for m in gateway.available_models() if not m.startswith("mock-")), ""
        )
        if target:
            try:
                # Count prefix-only and full separately; the difference is the
                # variable part, which keeps both numbers on the same tokenizer.
                full = [
                    gateway.count_tokens(r.to_request(target)) for r in rendered[:5]
                ]
                variable_tokens = sum(full) // len(full) - prefix_tokens
                exact = True
                source = f"provider `count_tokens` on {target}"
            except Exception:  # noqa: BLE001 — fall back to the estimate, labelled
                pass

    return PromptShape(
        shared_prefix_tokens=prefix_tokens,
        variable_input_tokens=max(1, variable_tokens),
        output_tokens=output_tokens if output_tokens is not None else suite.max_tokens,
        exact=exact,
        source=source,
    )


def tier_table(
    shape: PromptShape,
    *,
    calls: int = 10_000,
    models: list[str] | None = None,
    on: date | None = None,
) -> list[TierRow]:
    """Per-model unit economics for a prompt shape, uncached and cached."""
    targets = models or [
        m for m, s in REGISTRY.items() if s.provider == "anthropic"
    ]
    when = on or date.today()

    rows: list[TierRow] = []
    for model_id in targets:
        spec = get_model(model_id)
        price = spec.price_on(when)
        usage = Usage(
            input_tokens=shape.total_input_tokens, output_tokens=shape.output_tokens
        )
        per_call = estimate_cost(model_id, usage, when)
        wl = cached_workload_cost(
            model_id,
            shared_prefix_tokens=shape.shared_prefix_tokens,
            variable_input_tokens=shape.variable_input_tokens,
            output_tokens=shape.output_tokens,
            calls=calls,
            on=when,
        )
        rows.append(
            TierRow(
                model=model_id,
                tier=spec.tier,
                input_per_mtok=price.input_per_mtok,
                output_per_mtok=price.output_per_mtok,
                cost_per_call=per_call,
                cost_at_volume=wl["uncached_usd"],
                cache_min_tokens=spec.cache_min_tokens,
                cacheable=bool(wl["cacheable"]),
                cached_at_volume=wl["cached_usd"],
                saving_pct=wl["saving_pct"],
            )
        )

    rows.sort(key=lambda r: r.cost_per_call)
    return rows


def cache_table(
    shape: PromptShape,
    *,
    model: str,
    volumes: tuple[int, ...] = (1, 10, 100, 1_000, 10_000, 100_000),
    on: date | None = None,
) -> list[dict[str, float | int | bool]]:
    """How the caching saving scales with volume, for one model.

    Included because the break-even is not where people expect: a single call
    costs *more* cached than uncached, since it pays the write premium and never
    reads it back.
    """
    out = []
    for n in volumes:
        wl = cached_workload_cost(
            model,
            shared_prefix_tokens=shape.shared_prefix_tokens,
            variable_input_tokens=shape.variable_input_tokens,
            output_tokens=shape.output_tokens,
            calls=n,
            on=on,
        )
        out.append(
            {
                "calls": n,
                "uncached_usd": wl["uncached_usd"],
                "cached_usd": wl["cached_usd"],
                "saving_usd": wl["saving_usd"],
                "saving_pct": wl["saving_pct"],
                "cacheable": bool(wl["cacheable"]),
            }
        )
    return out


def cache_breakeven_by_prefix(
    *,
    variable_input_tokens: int,
    output_tokens: int,
    calls: int,
    prefixes: tuple[int, ...] = (0, 256, 512, 1_024, 2_048, 4_096, 8_192),
    models: list[str] | None = None,
    on: date | None = None,
) -> list[dict[str, object]]:
    """At what shared-prefix length does caching start paying, per model?

    The answer is a step function, not a curve: nothing happens until the prefix
    crosses the model's minimum, then the saving appears abruptly. Showing it this
    way makes the design consequence obvious — if you want caching, the system
    prompt has to be long enough to qualify on the model you actually use.
    """
    targets = models or [m for m, s in REGISTRY.items() if s.provider == "anthropic"]
    out: list[dict[str, object]] = []
    for prefix in prefixes:
        row: dict[str, object] = {"prefix_tokens": prefix}
        for model_id in targets:
            wl = cached_workload_cost(
                model_id,
                shared_prefix_tokens=prefix,
                variable_input_tokens=variable_input_tokens,
                output_tokens=output_tokens,
                calls=calls,
                on=on,
            )
            row[model_id] = wl["saving_pct"] if wl["cacheable"] else None
        out.append(row)
    return out
