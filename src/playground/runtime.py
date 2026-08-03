"""Shared runtime: the gateway and store every entry point uses.

Centralised so the API, the CLI, and the Streamlit UI all see the same provider
set and write to the same store — otherwise "which models are available" would
answer differently depending on where you asked.
"""

from __future__ import annotations

from functools import lru_cache

from ai_core.gateway import Gateway
from ai_core.providers import all_providers
from ai_core.providers.base import LLMProvider

from .config import get_settings
from .mock import MockProvider
from .store import Store

__all__ = ["build_gateway", "get_gateway", "get_store", "provider_report"]


def build_gateway(*, include_mock: bool | None = None, mock_latency: bool = True) -> Gateway:
    settings = get_settings()
    providers: list[LLMProvider] = list(all_providers())

    use_mock = settings.allow_mock if include_mock is None else include_mock
    if use_mock:
        providers.append(MockProvider(latency=mock_latency))

    return Gateway(providers=providers)


@lru_cache(maxsize=1)
def get_gateway() -> Gateway:
    return build_gateway()


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store(get_settings().store_path)


def provider_report() -> list[dict[str, object]]:
    """Live status of every provider, for the /providers endpoint and the UI banner."""
    report = []
    for provider in all_providers():
        report.append(provider.check().to_dict())
    if get_settings().allow_mock:
        report.append(MockProvider().check().to_dict())
    return report
