"""Environment-driven settings.

Read once at import and cached. Everything is optional except the provider keys,
which `ai-core` owns — this module only covers playground-specific knobs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    store_path: Path
    default_model: str
    default_effort: str
    benchmark_repeats: int
    allow_mock: bool
    mlflow_enabled: bool
    mlflow_experiment: str

    @property
    def store_dir(self) -> Path:
        return self.store_path.parent


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass

    return Settings(
        store_path=Path(os.getenv("PLAYGROUND_STORE", "./data/playground.db")).expanduser(),
        default_model=os.getenv("AI_CORE_DEFAULT_MODEL", "claude-opus-5"),
        default_effort=os.getenv("AI_CORE_DEFAULT_EFFORT", "medium"),
        # A single latency sample is noise. Nothing quoted anywhere is computed
        # from fewer repeats than this.
        benchmark_repeats=int(os.getenv("PLAYGROUND_REPEATS", "5")),
        # Registers the offline mock provider, so the whole app is runnable with
        # no API key at all. On by default precisely so a stranger can clone and run.
        allow_mock=_flag("PLAYGROUND_ALLOW_MOCK", default=True),
        mlflow_enabled=_flag("PLAYGROUND_MLFLOW", default=False),
        mlflow_experiment=os.getenv("PLAYGROUND_MLFLOW_EXPERIMENT", "llm-playground"),
    )
