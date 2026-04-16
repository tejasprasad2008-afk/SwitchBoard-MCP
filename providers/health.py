"""Per-provider rate-limit and error tracking with SQLite persistence."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import _DB_FILE


# ── SQLite helpers ─────────────────────────────────────────────────

def _init_db() -> sqlite3.Connection:
    db_exists = _DB_FILE.exists()
    conn = sqlite3.connect(str(_DB_FILE), check_same_thread=False)
    if not db_exists:
        # Ensure database file has restricted permissions (0600)
        os.chmod(_DB_FILE, 0o600)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS provider_health (
            provider TEXT PRIMARY KEY,
            error_count INTEGER DEFAULT 0,
            error_window_start REAL DEFAULT 0,
            rate_limit_remaining INTEGER DEFAULT 999999,
            rate_limit_reset REAL DEFAULT 0,
            last_updated REAL DEFAULT 0
        )
        """
    )
    conn.commit()
    return conn


_conn: sqlite3.Connection | None = None
_db_lock = asyncio.Lock()


async def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = await asyncio.to_thread(_init_db)
    return _conn


async def _read_row(provider: str) -> dict[str, Any] | None:
    conn = await _get_conn()
    row = conn.execute(
        "SELECT provider, error_count, error_window_start, rate_limit_remaining, rate_limit_reset, last_updated "
        "FROM provider_health WHERE provider = ?",
        (provider,),
    ).fetchone()
    if row is None:
        return None
    return {
        "provider": row[0],
        "error_count": row[1],
        "error_window_start": row[2],
        "rate_limit_remaining": row[3],
        "rate_limit_reset": row[4],
        "last_updated": row[5],
    }


async def _upsert(provider: str, **kwargs: Any) -> None:
    conn = await _get_conn()
    existing = await _read_row(provider)
    if existing is None:
        cols = ", ".join(["provider"] + list(kwargs.keys()))
        placeholders = ", ".join(["?"] + ["?" for _ in kwargs])
        conn.execute(
            f"INSERT INTO provider_health ({cols}) VALUES ({placeholders})",
            [provider] + list(kwargs.values()),
        )
    else:
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        conn.execute(
            f"UPDATE provider_health SET {sets} WHERE provider = ?",
            list(kwargs.values()) + [provider],
        )
    conn.commit()


# ── Sliding-window rate limiter ────────────────────────────────────

@dataclass
class RateLimiter:
    """Simple in-memory sliding window counter, synced to SQLite periodically."""

    max_requests: int = 60
    window_seconds: float = 60.0
    _timestamps: list[float] = field(default_factory=list)

    def record(self) -> None:
        now = time.time()
        self._timestamps.append(now)
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    def remaining(self) -> int:
        now = time.time()
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        return max(0, self.max_requests - len(self._timestamps))

    def is_exhausted(self) -> bool:
        return self.remaining() <= 0


# ── ProviderHealthTracker ──────────────────────────────────────────

class ProviderHealthTracker:
    """Track errors and rate limits per provider. Thread/async safe."""

    ERROR_WINDOW = 60.0  # seconds
    ERROR_THRESHOLD = 2  # errors before marking degraded

    def __init__(self) -> None:
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._error_counts: dict[str, list[float]] = {}  # timestamps of recent errors
        self._degraded: dict[str, bool] = {}

    # ── Public API ─────────────────────────────────────────────────

    async def record_error(self, provider: str) -> None:
        """Record an error for *provider* and persist."""
        now = time.time()
        self._error_counts.setdefault(provider, []).append(now)
        # Prune old errors
        cutoff = now - self.ERROR_WINDOW
        self._error_counts[provider] = [
            t for t in self._error_counts[provider] if t > cutoff
        ]
        # Check if degraded
        self._degraded[provider] = (
            len(self._error_counts[provider]) > self.ERROR_THRESHOLD
        )
        await _upsert(
            provider,
            error_count=len(self._error_counts[provider]),
            error_window_start=min(self._error_counts[provider], default=now),
            last_updated=now,
        )

    async def record_rate_limit(self, provider: str) -> None:
        """Mark that *provider* hit a rate limit."""
        rl = self._rate_limiters.setdefault(
            provider, RateLimiter(max_requests=60)
        )
        rl.record()
        await _upsert(
            provider,
            rate_limit_remaining=rl.remaining(),
            rate_limit_reset=time.time() + 60,
            last_updated=time.time(),
        )

    def is_degraded(self, provider: str) -> bool:
        return self._degraded.get(provider, False)

    def is_rate_limited(self, provider: str) -> bool:
        rl = self._rate_limiters.get(provider)
        if rl is None:
            return False
        return rl.is_exhausted()

    def get_remaining(self, provider: str) -> int:
        rl = self._rate_limiters.get(provider)
        if rl is None:
            return 999999
        return rl.remaining()

    async def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Return health for all known providers."""
        result: dict[str, dict[str, Any]] = {}
        all_providers = set(self._error_counts.keys()) | set(
            self._rate_limiters.keys()
        )
        for p in all_providers:
            row = await _read_row(p) or {}
            result[p] = {
                "degraded": self._degraded.get(p, False),
                "rate_limited": self.is_rate_limited(p),
                "remaining": self.get_remaining(p),
                "error_count": len(self._error_counts.get(p, [])),
                "last_updated": row.get("last_updated"),
            }
        return result

    def get_routing_log(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the last *limit* routing decisions from the JSONL log."""
        log_path = Path.home() / ".switchboard" / "routing.log"
        if not log_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries[-limit:]
