"""Test fixtures.

Every test runs against the offline mock provider with simulated latency disabled,
so the suite is fast and needs no API key or network.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Set before any playground import so cached settings pick these up.
os.environ.setdefault("PLAYGROUND_ALLOW_MOCK", "1")
os.environ.setdefault("PLAYGROUND_MLFLOW", "0")

from ai_core.gateway import Gateway  # noqa: E402

from playground.mock import MockProvider  # noqa: E402
from playground.store import Store  # noqa: E402


@pytest.fixture
def gateway() -> Gateway:
    """Mock-only gateway — deterministic and instant."""
    return Gateway(providers=[MockProvider(latency=False)])


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "test.db")


@pytest.fixture
def suite_path() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets" / "sentiment-classification.yaml"
