"""Small SQLite persistence layer.

No ORM is used so the project stays easy to run. Every operation opens a short
lived connection, which is safe with FastAPI's threaded request handlers.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterable, Iterator

from .config import DB_PATH, IST, ensure_directories


def now_iso() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    ensure_directories()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    """Yield a committed-and-closed connection.

    ``with sqlite3.connect(...)`` only commits/rolls back; it does not close
    the handle. Closing explicitly avoids lingering WAL file locks on Windows.
    """
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS subscribers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                regions TEXT NOT NULL,
                token TEXT NOT NULL UNIQUE,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_alert_at TEXT
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_key TEXT NOT NULL UNIQUE,
                region_id TEXT NOT NULL,
                region_name TEXT NOT NULL,
                level INTEGER NOT NULL,
                level_label TEXT NOT NULL,
                risk_score REAL NOT NULL,
                target_date TEXT NOT NULL,
                headline TEXT NOT NULL,
                details TEXT NOT NULL,
                recipients INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region_id TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region_id TEXT NOT NULL,
                target_date TEXT NOT NULL,
                risk_score REAL NOT NULL,
                level INTEGER NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_alerts_created
                ON alerts(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_alerts_region
                ON alerts(region_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_predictions_region
                ON predictions(region_id, target_date DESC);
            """
        )


# ---------------------------------------------------------------------------
# Subscribers
# ---------------------------------------------------------------------------

def upsert_subscriber(email: str, regions: Iterable[str], token: str) -> dict[str, Any]:
    regions_json = json.dumps(sorted(set(regions)))
    created_at = now_iso()
    with _connection() as conn:
        row = conn.execute("SELECT * FROM subscribers WHERE email = ?", (email,)).fetchone()
        if row:
            conn.execute(
                "UPDATE subscribers SET regions = ?, active = 1 WHERE email = ?",
                (regions_json, email),
            )
            row = conn.execute("SELECT * FROM subscribers WHERE email = ?", (email,)).fetchone()
        else:
            conn.execute(
                "INSERT INTO subscribers (email, regions, token, active, created_at) "
                "VALUES (?, ?, ?, 1, ?)",
                (email, regions_json, token, created_at),
            )
            row = conn.execute("SELECT * FROM subscribers WHERE email = ?", (email,)).fetchone()
    return _subscriber_row(row)


def _subscriber_row(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        return {}
    data = dict(row)
    try:
        data["regions"] = json.loads(data.get("regions") or "[]")
    except json.JSONDecodeError:
        data["regions"] = []
    data["active"] = bool(data.get("active"))
    return data


def get_subscriber_by_token(token: str) -> dict[str, Any] | None:
    with _connection() as conn:
        row = conn.execute("SELECT * FROM subscribers WHERE token = ?", (token,)).fetchone()
    return _subscriber_row(row) if row else None


def get_subscriber_by_email(email: str) -> dict[str, Any] | None:
    with _connection() as conn:
        row = conn.execute("SELECT * FROM subscribers WHERE email = ?", (email,)).fetchone()
    return _subscriber_row(row) if row else None


def deactivate_subscriber(token: str) -> bool:
    with _connection() as conn:
        cur = conn.execute("UPDATE subscribers SET active = 0 WHERE token = ?", (token,))
        return cur.rowcount > 0


def active_subscribers_for_region(region_id: str) -> list[dict[str, Any]]:
    with _connection() as conn:
        rows = conn.execute("SELECT * FROM subscribers WHERE active = 1").fetchall()
    result = []
    for row in rows:
        subscriber = _subscriber_row(row)
        if region_id in subscriber.get("regions", []):
            result.append(subscriber)
    return result


def count_subscribers() -> int:
    with _connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM subscribers WHERE active = 1").fetchone()
    return int(row["n"]) if row else 0


def mark_subscribers_alerted(ids: list[int]) -> None:
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    with _connection() as conn:
        conn.execute(
            f"UPDATE subscribers SET last_alert_at = ? WHERE id IN ({placeholders})",
            [now_iso(), *ids],
        )


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def alert_exists(alert_key: str) -> bool:
    with _connection() as conn:
        row = conn.execute("SELECT 1 FROM alerts WHERE alert_key = ? LIMIT 1", (alert_key,)).fetchone()
    return row is not None


def recent_region_alert(region_id: str, since_iso: str) -> dict[str, Any] | None:
    with _connection() as conn:
        row = conn.execute(
            "SELECT * FROM alerts WHERE region_id = ? AND created_at >= ? "
            "AND status IN ('sent', 'simulated', 'partial') "
            "ORDER BY created_at DESC LIMIT 1",
            (region_id, since_iso),
        ).fetchone()
    return _alert_row(row) if row else None


def insert_alert(
    *,
    alert_key: str,
    region_id: str,
    region_name: str,
    level: int,
    level_label: str,
    risk_score: float,
    target_date: str,
    headline: str,
    details: dict[str, Any],
    recipients: int,
    status: str,
) -> dict[str, Any]:
    created_at = now_iso()
    with _connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO alerts
            (alert_key, region_id, region_name, level, level_label, risk_score,
             target_date, headline, details, recipients, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert_key,
                region_id,
                region_name,
                level,
                level_label,
                risk_score,
                target_date,
                headline,
                json.dumps(details),
                recipients,
                status,
                created_at,
            ),
        )
        row = conn.execute("SELECT * FROM alerts WHERE alert_key = ?", (alert_key,)).fetchone()
    return _alert_row(row)


def _alert_row(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        return {}
    data = dict(row)
    try:
        data["details"] = json.loads(data.get("details") or "{}")
    except json.JSONDecodeError:
        data["details"] = {}
    return data


def recent_alerts(limit: int = 12) -> list[dict[str, Any]]:
    with _connection() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_alert_row(row) for row in rows]


def alert_counts() -> dict[str, int]:
    with _connection() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM alerts GROUP BY status"
        ).fetchall()
    return {row["status"]: int(row["n"]) for row in rows}


# ---------------------------------------------------------------------------
# Snapshots / predictions
# ---------------------------------------------------------------------------

def save_snapshot(region_id: str, status: str, payload: dict[str, Any]) -> None:
    with _connection() as conn:
        conn.execute(
            "INSERT INTO snapshots (region_id, fetched_at, status, payload) VALUES (?, ?, ?, ?)",
            (region_id, now_iso(), status, json.dumps(payload, default=str)),
        )
        conn.execute(
            "DELETE FROM snapshots WHERE id NOT IN "
            "(SELECT id FROM snapshots WHERE region_id = ? ORDER BY id DESC LIMIT 300)",
            (region_id,),
        )


def save_predictions(region_id: str, predictions: list[dict[str, Any]]) -> None:
    if not predictions:
        return
    created_at = now_iso()
    rows = [
        (
            region_id,
            str(item["date"]),
            float(item["risk_score"]),
            int(item["level"]),
            float(item.get("confidence", 0)),
            created_at,
        )
        for item in predictions
    ]
    with _connection() as conn:
        conn.executemany(
            "INSERT INTO predictions (region_id, target_date, risk_score, level, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.execute(
            "DELETE FROM predictions WHERE id NOT IN "
            "(SELECT id FROM predictions ORDER BY id DESC LIMIT 5000)"
        )


def latest_predictions(region_id: str, limit: int = 12) -> list[dict[str, Any]]:
    with _connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE region_id = ? ORDER BY id DESC LIMIT ?",
            (region_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]
