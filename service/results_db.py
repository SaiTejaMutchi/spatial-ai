"""Local SQLite catalog of scan and AI results.

Filesystem artifacts remain the source of truth for files the UI downloads.
This database records completed reviews so Groq is not called again for the
same geometry, prompt, and model. Nothing is uploaded.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
  scan_id TEXT PRIMARY KEY,
  name TEXT,
  classification TEXT,
  source_type TEXT,
  source_path TEXT,
  status TEXT,
  model_sha256 TEXT,
  created_at TEXT,
  updated_at TEXT,
  summary_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS ai_reviews (
  cache_key TEXT PRIMARY KEY,
  scan_id TEXT,
  status TEXT NOT NULL,
  model_id TEXT,
  prompt_version TEXT,
  assessment_json TEXT NOT NULL,
  rejected_json TEXT NOT NULL DEFAULT '[]',
  diagnostics_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS opening_resolutions (
  cache_key TEXT PRIMARY KEY,
  scan_id TEXT,
  openings_json TEXT NOT NULL,
  report_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def geometry_result_digest(model: dict) -> str:
    """Identity of the reconstructed room, independent of later AI writes."""
    return stable_digest({
        "measurements": model.get("measurements") or [],
        "surfaceIds": [surface.get("id") for surface in model.get("surfaces") or []],
        "evidenceIds": [view.get("id") for view in model.get("evidence") or []],
    })


class ResultsDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def upsert_scan(self, state: dict) -> None:
        summary = state.get("summary") or {}
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO scans (
                       scan_id, name, classification, source_type, source_path,
                       status, model_sha256, created_at, updated_at, summary_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(scan_id) DO UPDATE SET
                       name=excluded.name,
                       classification=excluded.classification,
                       source_type=excluded.source_type,
                       source_path=excluded.source_path,
                       status=excluded.status,
                       model_sha256=excluded.model_sha256,
                       updated_at=excluded.updated_at,
                       summary_json=excluded.summary_json
                """,
                (
                    state["scanId"],
                    state.get("label") or state["scanId"],
                    state.get("classification"),
                    state.get("sourceType"),
                    state.get("sourcePath"),
                    state.get("status"),
                    summary.get("modelSha256"),
                    state.get("createdAt"),
                    state.get("updatedAt") or _now(),
                    json.dumps(summary),
                ),
            )

    def delete_scan(self, scan_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM scans WHERE scan_id = ?", (scan_id,))

    def get_scan(self, scan_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scans WHERE scan_id = ?", (scan_id,)
            ).fetchone()
        return dict(row) if row else None

    def put_ai(
        self,
        cache_key: str,
        scan_id: str | None,
        assessment: dict,
        rejected_findings: list | None = None,
        diagnostics: dict | None = None,
    ) -> None:
        if assessment.get("status") != "completed":
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO ai_reviews (
                       cache_key, scan_id, status, model_id, prompt_version,
                       assessment_json, rejected_json, diagnostics_json, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                       scan_id=excluded.scan_id,
                       status=excluded.status,
                       model_id=excluded.model_id,
                       prompt_version=excluded.prompt_version,
                       assessment_json=excluded.assessment_json,
                       rejected_json=excluded.rejected_json,
                       diagnostics_json=excluded.diagnostics_json
                """,
                (
                    cache_key,
                    scan_id,
                    assessment["status"],
                    assessment.get("model"),
                    assessment.get("promptVersion"),
                    json.dumps(assessment),
                    json.dumps(rejected_findings or []),
                    json.dumps(diagnostics or {}),
                    _now(),
                ),
            )

    def get_completed_ai(self, cache_key: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ai_reviews WHERE cache_key = ? AND status = 'completed'",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "assessment": json.loads(row["assessment_json"]),
            "rejectedFindings": json.loads(row["rejected_json"]),
            "diagnostics": json.loads(row["diagnostics_json"]),
            "scanId": row["scan_id"],
        }

    def put_openings(
        self,
        cache_key: str,
        scan_id: str | None,
        openings: list,
        report: dict,
    ) -> None:
        if not _openings_are_durable(report):
            return
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO opening_resolutions (
                       cache_key, scan_id, openings_json, report_json, created_at
                   ) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                       scan_id=excluded.scan_id,
                       openings_json=excluded.openings_json,
                       report_json=excluded.report_json
                """,
                (
                    cache_key,
                    scan_id,
                    json.dumps(openings),
                    json.dumps(report),
                    _now(),
                ),
            )

    def get_openings(self, cache_key: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM opening_resolutions WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "openings": json.loads(row["openings_json"]),
            "report": json.loads(row["report_json"]),
            "scanId": row["scan_id"],
        }


def _openings_are_durable(report: dict) -> bool:
    blob = json.dumps(report).lower()
    if "429" in blob or "rate limit" in blob or "provider_failure" in blob:
        return False
    return True


def ai_cache_key(model: dict, config_sha256: str, model_id: str | None) -> str:
    return stable_digest({
        "geometry": geometry_result_digest(model),
        "configSha256": config_sha256,
        "modelId": model_id or "",
    })


def opening_cache_key(model: dict, config_sha256: str, model_id: str | None) -> str:
    return stable_digest({
        "geometry": geometry_result_digest(model),
        "kind": "openings",
        "configSha256": config_sha256,
        "modelId": model_id or "",
    })
