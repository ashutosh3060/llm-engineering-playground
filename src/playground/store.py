"""SQLite store for every request the playground makes.

Cost and latency are recorded on every call from day one, not bolted on later —
otherwise the historical comparison you want in week four simply does not exist.

Two tables:
  runs     one benchmark / comparison / sweep invocation
  results  one row per (run, model, case, repeat)
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_core.types import CompletionResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    label       TEXT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    config      TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL REFERENCES runs(id),
    case_id      TEXT,
    repeat       INTEGER NOT NULL DEFAULT 0,
    model        TEXT NOT NULL,
    provider     TEXT NOT NULL,
    prompt_hash  TEXT NOT NULL,
    prompt_label TEXT,
    text         TEXT NOT NULL DEFAULT '',
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens   INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens  INTEGER NOT NULL DEFAULT 0,
    latency_ms   REAL NOT NULL DEFAULT 0,
    cost_usd     REAL NOT NULL DEFAULT 0,
    score        REAL,
    score_detail TEXT,
    stop_reason  TEXT,
    error        TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_results_run   ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_model ON results(model);
CREATE INDEX IF NOT EXISTS idx_results_hash  ON results(prompt_hash);
"""


class Store:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- writes ----------------------------------------------------------

    def start_run(self, run_id: str, kind: str, label: str = "", **config: Any) -> str:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (id, kind, label, started_at, config) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, kind, label, datetime.now(UTC).isoformat(), json.dumps(config)),
            )
        return run_id

    def finish_run(self, run_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET finished_at = ? WHERE id = ?",
                (datetime.now(UTC).isoformat(), run_id),
            )

    def record(
        self,
        run_id: str,
        result: CompletionResult,
        *,
        case_id: str | None = None,
        repeat: int = 0,
        prompt_label: str | None = None,
        score: float | None = None,
        score_detail: dict[str, Any] | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO results (
                    run_id, case_id, repeat, model, provider, prompt_hash, prompt_label,
                    text, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                    latency_ms, cost_usd, score, score_detail, stop_reason, error, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    case_id,
                    repeat,
                    result.model,
                    result.provider,
                    result.prompt_hash,
                    prompt_label,
                    result.text,
                    result.usage.input_tokens,
                    result.usage.output_tokens,
                    result.usage.cache_read_input_tokens,
                    result.usage.cache_creation_input_tokens,
                    result.latency_ms,
                    result.cost_usd,
                    score,
                    json.dumps(score_detail) if score_detail else None,
                    result.stop_reason,
                    result.error,
                    result.created_at.isoformat(),
                ),
            )

    # ---- reads -----------------------------------------------------------

    def runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT r.*, COUNT(res.id) AS n_results, "
                "COALESCE(SUM(res.cost_usd), 0) AS total_cost "
                "FROM runs r LEFT JOIN results res ON res.run_id = r.id "
                "GROUP BY r.id ORDER BY r.started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def results(self, run_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY case_id, model, repeat",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def spend_by_model(self) -> list[dict[str, Any]]:
        """Total spend and call count per model — the 'where did the money go' view."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT model, COUNT(*) AS calls, SUM(cost_usd) AS cost, "
                "SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, "
                "AVG(latency_ms) AS avg_latency_ms "
                "FROM results WHERE error IS NULL GROUP BY model ORDER BY cost DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def total_spend(self) -> float:
        with self._conn() as conn:
            row = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) AS t FROM results").fetchone()
        return float(row["t"])
