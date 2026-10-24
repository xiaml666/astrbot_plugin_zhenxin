"""SQLite persistence.  One file, no external service."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA = """
CREATE TABLE IF NOT EXISTS hot_slots (
    user_key TEXT NOT NULL,
    slot INTEGER NOT NULL,
    content_seed TEXT NOT NULL DEFAULT '',
    brightness REAL NOT NULL DEFAULT 1.0,
    last_touched REAL NOT NULL,
    type TEXT NOT NULL DEFAULT 'fact',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_key, slot)
);

CREATE TABLE IF NOT EXISTS warm_slots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_key TEXT NOT NULL,
    content_seed TEXT NOT NULL,
    brightness REAL NOT NULL,
    last_touched REAL NOT NULL,
    type TEXT NOT NULL DEFAULT 'fact',
    created_at REAL NOT NULL,
    UNIQUE (user_key, content_seed)
);

CREATE TABLE IF NOT EXISTS cold_seeds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_key TEXT NOT NULL,
    seed_tokens TEXT NOT NULL,
    thin_context TEXT NOT NULL,
    topic_key TEXT,
    source TEXT,
    created_at REAL NOT NULL,
    last_access REAL,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_wake_round INTEGER NOT NULL DEFAULT 0,
    wake_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_key TEXT NOT NULL,
    text TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'fact',
    importance REAL NOT NULL DEFAULT 0.5,
    created_at REAL NOT NULL,
    last_used REAL NOT NULL DEFAULT 0,
    used_count INTEGER NOT NULL DEFAULT 0,
    weight REAL NOT NULL DEFAULT 0.5,
    mentions INTEGER NOT NULL DEFAULT 1,
    pool TEXT NOT NULL DEFAULT 'cycle',
    cycle_at REAL NOT NULL DEFAULT 0,
    trashed_at REAL NOT NULL DEFAULT 0,
    trash_reason TEXT NOT NULL DEFAULT '',
    last_mentioned REAL NOT NULL DEFAULT 0,
    manual_boost REAL NOT NULL DEFAULT 0,
    UNIQUE (user_key, text)
);

-- Keyword-only residue of facts that aged out of the retrieval pool.  The
-- full sentence is deliberately not kept: these rows are a paper trail, never
-- injected into prompts.
CREATE TABLE IF NOT EXISTS fact_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_key TEXT NOT NULL,
    keywords TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'fact',
    importance REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    archived_at REAL NOT NULL,
    reason TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS solidified_paths (
    user_key TEXT NOT NULL,
    signature TEXT NOT NULL,
    response_shape TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0,
    last_hit REAL,
    last_seen_round INTEGER NOT NULL DEFAULT 0,
    neg_streak INTEGER NOT NULL DEFAULT 0,
    degraded_until REAL NOT NULL DEFAULT 0,
    hits INTEGER NOT NULL DEFAULT 0,
    misses INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_key, signature)
);

CREATE TABLE IF NOT EXISTS topic_stack (
    user_key TEXT NOT NULL,
    depth INTEGER NOT NULL,
    topic_key TEXT NOT NULL,
    label TEXT NOT NULL,
    pushed_at REAL NOT NULL,
    PRIMARY KEY (user_key, depth)
);

CREATE TABLE IF NOT EXISTS open_loops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_key TEXT NOT NULL,
    summary TEXT NOT NULL,
    keywords TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    rounds_alive INTEGER NOT NULL DEFAULT 0,
    due_hint TEXT,
    status TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS negative_patterns (
    user_key TEXT NOT NULL,
    feature TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    last_seen REAL NOT NULL,
    PRIMARY KEY (user_key, feature)
);

CREATE TABLE IF NOT EXISTS lexicon (
    user_key TEXT NOT NULL,
    user_word TEXT NOT NULL,
    standard_word TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_key, user_word, standard_word)
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_key TEXT NOT NULL,
    turn_no INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_key TEXT NOT NULL,
    round_no INTEGER NOT NULL,
    mode TEXT NOT NULL,
    wake INTEGER NOT NULL DEFAULT 0,
    wake_trigger TEXT,
    prefix_hash TEXT,
    prefix_stable INTEGER,
    l0_chars INTEGER NOT NULL DEFAULT 0,
    l0_balls INTEGER NOT NULL DEFAULT 0,
    recent_chars INTEGER NOT NULL DEFAULT 0,
    ephemeral_chars INTEGER NOT NULL DEFAULT 0,
    injected_tokens INTEGER NOT NULL DEFAULT 0,
    shortcircuit_hit INTEGER NOT NULL DEFAULT 0,
    solid_signature TEXT,
    solid_confidence REAL,
    l2_llm_calls INTEGER NOT NULL DEFAULT 0,
    main_llm_calls INTEGER NOT NULL DEFAULT 0,
    decision_llm_calls INTEGER NOT NULL DEFAULT 0,
    reply_chars INTEGER NOT NULL DEFAULT 0,
    input_tokens_actual INTEGER NOT NULL DEFAULT 0,
    cached_tokens_actual INTEGER NOT NULL DEFAULT 0,
    output_tokens_actual INTEGER NOT NULL DEFAULT 0,
    actual_usage_available INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    trace_json TEXT
);

CREATE TABLE IF NOT EXISTS state (
    user_key TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (user_key, key)
);

-- Distilled "talk like a real person" knowledge.  One row per source so a
-- local refresh never wipes the last successful online refresh.
CREATE TABLE IF NOT EXISTS style_knowledge (
    user_key TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    chars INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_key, source)
);

-- One row per unsolicited message, so daily caps survive a restart.
CREATE TABLE IF NOT EXISTS proactive_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_key TEXT NOT NULL,
    content TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'memory',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hot_user ON hot_slots(user_key);
CREATE INDEX IF NOT EXISTS idx_warm_user ON warm_slots(user_key);
CREATE INDEX IF NOT EXISTS idx_cold_user ON cold_seeds(user_key);
CREATE INDEX IF NOT EXISTS idx_turns_user ON turns(user_key, id);
CREATE INDEX IF NOT EXISTS idx_metrics_user ON metrics(user_key, round_no);
CREATE INDEX IF NOT EXISTS idx_loops_user ON open_loops(user_key, status);
CREATE INDEX IF NOT EXISTS idx_traces_user ON fact_traces(user_key, archived_at);
CREATE INDEX IF NOT EXISTS idx_facts_created ON facts(created_at);
CREATE INDEX IF NOT EXISTS idx_proactive_user ON proactive_log(user_key, created_at);
"""


class Database:
    def __init__(self, path: str | Path = ":memory:", *, initialize: bool = True) -> None:
        self.path = str(path)
        if self.path not in (":memory:", "") and not self.path.startswith("file:"):
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            if initialize:
                self._conn.executescript(SCHEMA)
                self._migrate()
                self._conn.commit()

    def _migrate(self) -> None:
        """Add instrumentation columns to an existing database in place."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(metrics)").fetchall()}
        additions = {
            "input_tokens_actual": "INTEGER NOT NULL DEFAULT 0",
            "cached_tokens_actual": "INTEGER NOT NULL DEFAULT 0",
            "output_tokens_actual": "INTEGER NOT NULL DEFAULT 0",
            "actual_usage_available": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, decl in additions.items():
            if name not in cols:
                self._conn.execute(f"ALTER TABLE metrics ADD COLUMN {name} {decl}")
        fact_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}
        fact_additions = {
            "importance": "REAL NOT NULL DEFAULT 0.5",
            "weight": "REAL NOT NULL DEFAULT 0.5",
            "mentions": "INTEGER NOT NULL DEFAULT 1",
            "pool": "TEXT NOT NULL DEFAULT 'cycle'",
            "cycle_at": "REAL NOT NULL DEFAULT 0",
            "trashed_at": "REAL NOT NULL DEFAULT 0",
            "trash_reason": "TEXT NOT NULL DEFAULT ''",
            "last_mentioned": "REAL NOT NULL DEFAULT 0",
            "manual_boost": "REAL NOT NULL DEFAULT 0",
        }
        for name, decl in fact_additions.items():
            if name not in fact_cols:
                self._conn.execute(f"ALTER TABLE facts ADD COLUMN {name} {decl}")
        # This index depends on a migrated column, so it cannot live in the
        # initial SCHEMA script: on an old database the column does not exist
        # yet when executescript() runs.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_pool ON facts(user_key, pool, created_at)"
        )

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, seq)
            self._conn.commit()

    def fetchone(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def scalar(self, sql: str, params: Sequence[Any] = (), default: Any = None) -> Any:
        row = self.fetchone(sql, params)
        if row is None:
            return default
        return row[0]

    # -- small state helpers -------------------------------------------------
    def get_state(self, user_key: str, key: str, default: Any = None) -> Any:
        row = self.fetchone("SELECT value FROM state WHERE user_key=? AND key=?", (user_key, key))
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (TypeError, json.JSONDecodeError):
            return default

    def set_state(self, user_key: str, key: str, value: Any) -> None:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        self.execute(
            "INSERT INTO state(user_key,key,value) VALUES(?,?,?) "
            "ON CONFLICT(user_key,key) DO UPDATE SET value=excluded.value",
            (user_key, key, raw),
        )

    def incr_state(self, user_key: str, key: str, amount: int = 1) -> int:
        value = int(self.get_state(user_key, key, 0) or 0) + amount
        self.set_state(user_key, key, value)
        return value

    def append_turn(self, user_key: str, turn_no: int, role: str, content: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self.execute(
            "INSERT INTO turns(user_key,turn_no,role,content,created_at) VALUES(?,?,?,?,?)",
            (user_key, turn_no, role, content, now),
        )
        # Hard cap keeps the hot history bounded; cache-aware pruning is the
        # primary mechanism, this is only a safety net.
        self.execute(
            "DELETE FROM turns WHERE user_key=? AND id NOT IN "
            "(SELECT id FROM turns WHERE user_key=? ORDER BY id DESC LIMIT 400)",
            (user_key, user_key),
        )

    def recent_turns(self, user_key: str, limit: int = 20) -> list[sqlite3.Row]:
        rows = self.fetchall(
            "SELECT id,role,content,turn_no FROM turns WHERE user_key=? ORDER BY id DESC LIMIT ?",
            (user_key, limit),
        )
        return list(reversed(rows))

    def turns_from(self, user_key: str, first_id: int, limit: int = 400) -> list[sqlite3.Row]:
        """All turns from ``first_id`` on, oldest first.

        Used by the cache-aware window: the window keeps a fixed first message
        and only appends, so the prompt prefix stays byte-identical between
        evictions.
        """
        return self.fetchall(
            "SELECT id,role,content,turn_no FROM turns WHERE user_key=? AND id>=? ORDER BY id LIMIT ?",
            (user_key, int(first_id), int(limit)),
        )
