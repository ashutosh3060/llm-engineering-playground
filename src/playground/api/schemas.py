"""API request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class MessageIn(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str


class CompleteIn(BaseModel):
    model: str | None = None
    prompt: str | None = Field(None, description="Shorthand for a single user message.")
    messages: list[MessageIn] | None = None
    system: str | None = None
    max_tokens: int = 1024
    effort: Effort | None = None
    thinking: bool = True

    def resolved_messages(self) -> list[MessageIn]:
        if self.messages:
            return self.messages
        if self.prompt is not None:
            return [MessageIn(role="user", content=self.prompt)]
        raise ValueError("Provide either `prompt` or `messages`.")


class CompareIn(CompleteIn):
    models: list[str] | None = Field(
        None, description="Defaults to every currently available model."
    )
    repeats: int = Field(
        1, ge=1, le=20, description="Repeats per model; latency is p50 over these."
    )


class CountTokensIn(BaseModel):
    model: str | None = None
    prompt: str | None = None
    messages: list[MessageIn] | None = None
    system: str | None = None


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    total_prompt_tokens: int


class ResultOut(BaseModel):
    model: str
    provider: str
    text: str
    usage: UsageOut
    latency_ms: float
    cost_usd: float
    stop_reason: str | None = None
    prompt_hash: str
    error: str | None = None
    ok: bool


class CompareRowOut(BaseModel):
    model: str
    provider: str
    n: int
    errors: int
    cost_per_call_usd: float
    p50_latency_ms: float
    p95_latency_ms: float
    sample_text: str
    ok: bool


class CompareOut(BaseModel):
    run_id: str
    prompt_hash: str
    rows: list[CompareRowOut]


class ModelOut(BaseModel):
    id: str
    provider: str
    display_name: str
    tier: str
    context_window: int
    max_output_tokens: int
    input_per_mtok: float
    output_per_mtok: float
    price_note: str = ""
    available: bool
    notes: str = ""


class CountTokensOut(BaseModel):
    model: str
    input_tokens: int
    estimated_input_cost_usd: float


class SpendRow(BaseModel):
    model: str
    calls: int
    cost: float
    input_tokens: int
    output_tokens: int
    avg_latency_ms: float


class RunOut(BaseModel):
    id: str
    kind: str
    label: str | None = None
    started_at: str
    finished_at: str | None = None
    n_results: int
    total_cost: float
    config: dict[str, Any] = Field(default_factory=dict)
