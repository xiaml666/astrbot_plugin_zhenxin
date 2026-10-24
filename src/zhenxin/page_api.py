"""AstrBot plugin-page backend.

AstrBot discovers ``pages/<name>/index.html`` and lists it under the left
"扩展功能" rail.  This module backs that page with two small JSON endpoints:
status and config save.  It intentionally exposes only curated fields, so the
page can stay friendly and cannot change the database path at runtime.
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any

PLUGIN_NAME = "astrbot_plugin_zhenxin"
PAGE_PREFIX = f"/{PLUGIN_NAME}/page"

# Curated, runtime-safe fields. Anything requiring a restart (db_path,
# embedding paths, model endpoints) is deliberately absent.
ALLOWED_PATHS = {
    "shadow_mode",
    "live_allowlist",
    "l0.enabled",
    "l0.inject",
    "l0.tail_injection",
    "l0.max_inject_chars",
    "l0.max_seed_chars",
    "l0.decay",
    "l0.hysteresis",
    "l0.min_dwell_rounds",
    "l0.rebalance_interval",
    "shortcircuit.enabled",
    "shortcircuit.hit_threshold",
    "shortcircuit.max_hit_chars",
    "shortcircuit.neg_degrade_threshold",
    "shortcircuit.degrade_hours",
    "l1.enabled",
    "l1.low_l0_threshold",
    "l1.candidate_min_score",
    "l1.wake_cooldown_rounds",
    "l1.recall_max_chars",
    "l1.max_recalls",
    "l2.enabled",
    "l2.warm_limit",
    "l2.open_loop_limit",
    "l2.open_loop_ttl_rounds",
    "l3.enabled",
    "l3.cache_aware_pruning",
    "l3.recent_turns",
    "l3.window_grow_turns",
    "l3.recent_max_chars",
    "l3.recent_char_low_water",
    "l3.ephemeral_max_chars",
    "l3.host_context_mode",
    "l3.cache_ttl_seconds",
    "aux.topic_stack",
    "aux.compression",
    "aux.compression_threshold",
    "aux.rhythm",
    "aux.style",
    "aux.rhythm_postprocess",
    "aux.lexicon",
    "aux.open_loops",
    "aux.negative_patterns",
    "metrics.enabled",
    "metrics.input_price_per_million",
    "metrics.cached_input_price_per_million",
    "metrics.output_price_per_million",
}

_BOOL_PATHS = {p for p in ALLOWED_PATHS if p.endswith((".enabled", "shadow_mode"))} | {
    "l0.tail_injection",
    "aux.style",
}
_INT_PATHS = {
    "l0.inject",
    "l0.max_inject_chars",
    "l0.max_seed_chars",
    "l0.min_dwell_rounds",
    "l0.rebalance_interval",
    "shortcircuit.max_hit_chars",
    "shortcircuit.neg_degrade_threshold",
    "l1.wake_cooldown_rounds",
    "l1.recall_max_chars",
    "l1.max_recalls",
    "l2.warm_limit",
    "l2.open_loop_limit",
    "l2.open_loop_ttl_rounds",
    "l3.recent_turns",
    "l3.window_grow_turns",
    "l3.recent_max_chars",
    "l3.ephemeral_max_chars",
    "l3.cache_ttl_seconds",
    "aux.compression_threshold",
}


# Tables that hold per-conversation learned state.  ``facts`` is long-term
# memory and ``metrics`` is the cost ledger, so they are reset separately.
_CONVERSATION_TABLES = (
    "hot_slots",
    "warm_slots",
    "cold_seeds",
    "solidified_paths",
    "topic_stack",
    "open_loops",
    "negative_patterns",
    "lexicon",
    "turns",
    "state",
)

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "_conf_schema.json"
_DEFAULT_CACHE: dict[str, Any] | None = None


def schema_defaults() -> dict[str, Any]:
    """Return the shipped default for every allow-listed config path.

    The page must not invent defaults: the schema in ``_conf_schema.json`` is
    what a fresh install writes into ``astrbot_plugin_zhenxin_config.json``.
    """
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is not None:
        return dict(_DEFAULT_CACHE)
    try:
        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        schema = {}

    defaults: dict[str, Any] = {}

    def walk(node: Any, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        for key, spec in node.items():
            if not isinstance(spec, dict):
                continue
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(spec.get("items"), dict) and spec.get("items"):
                walk(spec["items"], path)
            elif path in ALLOWED_PATHS and "default" in spec:
                defaults[path] = spec["default"]

    walk(schema, "")
    _DEFAULT_CACHE = defaults
    return dict(defaults)


def _get_path(source: Any, path: str, default: Any = None) -> Any:
    cur = source
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return cur


def _set_path(target: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = target
    for part in parts[:-1]:
        if isinstance(cur, dict):
            cur = cur.setdefault(part, {})
        else:
            cur = getattr(cur, part)
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)


class PluginPageAPI:
    def __init__(self, plugin: Any, engine: Any, config: Any) -> None:
        self.plugin = plugin
        self.engine = engine
        self.config = config

    def register(self) -> None:
        register = getattr(self.plugin.context, "register_web_api", None)
        if not callable(register):
            return
        register(f"{PAGE_PREFIX}/status", self.status, ["GET"], "Zhenxin page status")
        register(f"{PAGE_PREFIX}/config", self.save_config, ["POST"], "Zhenxin page config")
        register(f"{PAGE_PREFIX}/defaults", self.defaults, ["GET"], "Zhenxin page defaults")
        register(f"{PAGE_PREFIX}/reset", self.reset, ["POST"], "Zhenxin page reset")
        register(f"{PAGE_PREFIX}/memory", self.memory_list, ["GET"], "Zhenxin memory list")
        register(f"{PAGE_PREFIX}/memory/delete", self.memory_delete, ["POST"], "Zhenxin memory delete")
        register(f"{PAGE_PREFIX}/memory/clear", self.memory_clear, ["POST"], "Zhenxin memory clear")

    async def defaults(self) -> dict[str, Any]:
        """Config values a fresh install starts with (for the 恢复默认 button)."""
        return {"ok": True, "defaults": schema_defaults()}

    async def memory_list(self) -> dict[str, Any]:
        try:
            # Opening the page is also a good moment to retire zero-value rows;
            # the sweep is rule-based and capped, so it stays cheap.
            self.engine.facts.sweep(None, max_archive=50)
            rows = self.engine.db.fetchall(
                "SELECT text, kind, COUNT(*) sessions, SUM(used_count) used,"
                " MAX(importance) importance, MAX(last_used) last_used,"
                " MIN(created_at) created_at FROM facts GROUP BY text"
                " ORDER BY importance DESC, used DESC, created_at DESC LIMIT 500"
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"读取记忆库失败：{exc}", "facts": []}
        now = time.time()
        facts = []
        for row in rows:
            created_at = float(row["created_at"] or 0)
            age_days = max(0.0, (now - created_at) / 86400.0) if created_at else 0.0
            importance = float(row["importance"] or 0.5)
            used = int(row["used"] or 0)
            if importance >= 0.75 or used >= 3:
                tier = "important"
            elif age_days <= 7:
                tier = "recent"
            else:
                tier = "normal"
            facts.append(
                {
                    "text": row["text"],
                    "kind": row["kind"],
                    "sessions": int(row["sessions"] or 0),
                    "used": used,
                    "importance": round(importance, 2),
                    "created_at": created_at,
                    "last_used": float(row["last_used"] or 0),
                    "age_days": round(age_days, 1),
                    "tier": tier,
                }
            )
        traces = self.engine.facts.traces(None, limit=100)
        return {
            "ok": True,
            "total": len(facts),
            "facts": facts,
            "traces": traces,
            "archived_total": self.engine.facts.trace_count(),
        }

    async def _json_body(self) -> dict[str, Any]:
        override = getattr(self, "_payload_override", None)
        if override is not None:
            return dict(override)
        try:
            from quart import request  # type: ignore
        except Exception:
            return {}
        try:
            return await request.get_json(silent=True) or {}
        except Exception:  # noqa: BLE001
            return {}

    async def memory_delete(self) -> dict[str, Any]:
        payload = await self._json_body()
        texts = payload.get("texts") or []
        ids = payload.get("ids") or []
        if not isinstance(texts, list) or (not texts and not ids):
            return {"ok": False, "message": "没有选中要删除的记忆"}
        try:
            for text in texts:
                self.engine.db.execute("DELETE FROM facts WHERE text=?", (str(text),))
            for raw in ids:
                self.engine.db.execute("DELETE FROM facts WHERE id=?", (int(raw),))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"删除失败：{exc}"}
        return {"ok": True, "deleted": len(texts) or len(ids)}

    async def memory_clear(self) -> dict[str, Any]:
        payload = await self._json_body()
        user = payload.get("user") or ""
        try:
            if user:
                self.engine.db.execute(
                    "DELETE FROM facts WHERE user_key LIKE ?", (f"%{user}",)
                )
                self.engine.db.execute(
                    "DELETE FROM fact_traces WHERE user_key LIKE ?", (f"%{user}",)
                )
            else:
                self.engine.db.execute("DELETE FROM facts")
                self.engine.db.execute("DELETE FROM fact_traces")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"清空失败：{exc}"}
        return {"ok": True}

    async def reset(self) -> dict[str, Any]:
        """Clear learned state without touching AstrBot's own data.

        ``conversation`` is what "换模型 / 新开对话" needs: recent window, hot
        clues, topic stack, open loops and solidified paths.  Long-term facts
        and the cost ledger survive unless explicitly included.
        """
        payload = await self._json_body()
        scope = str(payload.get("scope") or "").strip()
        user = str(payload.get("user") or "").strip()
        if scope not in ("conversation", "memory", "metrics", "all"):
            return {"ok": False, "message": "未知的重置范围"}
        deleted: dict[str, int] = {}
        try:
            if scope in ("conversation", "all"):
                deleted["conversation"] = self._reset_conversation(user)
            if scope in ("memory", "all"):
                deleted["memory"] = self._delete_rows("facts", user)
                deleted["traces"] = self._delete_rows("fact_traces", user)
            if scope in ("metrics", "all"):
                deleted["metrics"] = self._delete_rows("metrics", user)
                if not user:
                    self.engine.db.execute(
                        "DELETE FROM state WHERE user_key='__global__'"
                    )
            if scope == "all":
                deleted["config"] = self._restore_defaults()
                await self._persist_config()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"重置失败：{exc}"}
        response: dict[str, Any] = {"ok": True, "scope": scope, "deleted": deleted}
        if scope == "all":
            response["config"] = self.engine.cfg.to_dict()
        return response

    def _delete_rows(self, table: str, user: str = "") -> int:
        if user:
            cur = self.engine.db.execute(
                f"DELETE FROM {table} WHERE user_key LIKE ?", (f"%{user}%",)
            )
        else:
            cur = self.engine.db.execute(f"DELETE FROM {table}")
        return int(cur.rowcount or 0)

    def _reset_conversation(self, user: str = "") -> int:
        removed = 0
        for table in _CONVERSATION_TABLES:
            if table == "state":
                # Keep the global error counters: those are health data, not
                # per-conversation state.
                if user:
                    cur = self.engine.db.execute(
                        "DELETE FROM state WHERE user_key LIKE ? AND user_key<>'__global__'",
                        (f"%{user}%",),
                    )
                else:
                    cur = self.engine.db.execute(
                        "DELETE FROM state WHERE user_key<>'__global__'"
                    )
                removed += int(cur.rowcount or 0)
            else:
                removed += self._delete_rows(table, user)
        return removed

    def _restore_defaults(self) -> int:
        defaults = schema_defaults()
        for path, value in defaults.items():
            self._apply(path, value)
        return len(defaults)

    async def _persist_config(self) -> None:
        save = getattr(self.config, "save_config", None)
        if callable(save):
            result = save()
            if inspect.isawaitable(result):
                await result

    async def status(self) -> dict[str, Any]:
        from . import __version__

        stats = self.engine.stats(None)
        cfg = self.engine.cfg.to_dict()
        # Keep the payload small and avoid exposing anything secret-like.
        return {
            "ok": True,
            "plugin": PLUGIN_NAME,
            "version": __version__,
            "db_path": self.engine.db.path,
            "scorer": self.engine.scorer.name,
            "config": cfg,
            "health": self._health(),
            "series": self._series(30),
            "metrics": {
                "rounds": stats.get("rounds", 0),
                "wake_rate": stats.get("wake_rate", 0),
                "last10_wake_rate": stats.get("last10_wake_rate", 0),
                "prefix_stable_last20": stats.get("prefix_stable_last20", 0),
                "prefix_stable_ratio": stats.get("prefix_stable_ratio", 0),
                "l0_chars_max": stats.get("l0_chars_max", 0),
                "solid_hits": stats.get("solid_hits", 0),
                "solid_hit_rate": stats.get("solid_hit_rate", 0),
                "actual_usage_rounds": stats.get("actual_usage_rounds", 0),
                "actual_total_input_tokens": stats.get("actual_total_input_tokens", 0),
                "actual_cached_tokens": stats.get("actual_cached_tokens", 0),
                "actual_output_tokens": stats.get("actual_output_tokens", 0),
                "actual_cost": stats.get("actual_cost", 0),
                "actual_currency": stats.get("actual_currency", "CNY"),
                "l2_llm_calls": stats.get("l2_llm_calls", 0),
            },
        }

    def _health(self) -> dict[str, Any]:
        import os
        import time

        db = self.engine.db
        health: dict[str, Any] = {
            "db_ok": False,
            "db_kb": 0,
            "facts": 0,
            "fact_rows": 0,
            "last_round_at": 0,
            "errors": 0,
            "last_error": "",
            "last_error_at": 0,
        }
        try:
            path = str(db.path)
            if path in (":memory:", "") or path.startswith("file:"):
                health["db_ok"] = True
            elif os.path.exists(path):
                health["db_ok"] = True
                health["db_kb"] = round(os.path.getsize(path) / 1024, 1)
            health["facts"] = int(
                db.scalar("SELECT COUNT(DISTINCT text) FROM facts", (), 0)
            )
            health["fact_rows"] = int(db.scalar("SELECT COUNT(*) FROM facts", (), 0))
            health["last_round_at"] = float(
                db.scalar("SELECT MAX(created_at) FROM metrics", (), 0) or 0
            )
            health["errors"] = int(db.get_state("__global__", "error_count", 0) or 0)
            health["last_error"] = str(db.get_state("__global__", "last_error", "") or "")
            health["last_error_at"] = float(db.get_state("__global__", "last_error_at", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            health["last_error"] = f"health probe failed: {exc}"
            health["last_error_at"] = time.time()
        return health

    def _series(self, limit: int) -> list[dict[str, Any]]:
        try:
            rows = self.engine.db.fetchall(
                "SELECT round_no, input_tokens_actual, cached_tokens_actual,"
                " output_tokens_actual, created_at FROM metrics ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            )
        except Exception:  # noqa: BLE001
            return []
        cfg = self.engine.cfg.metrics
        out: list[dict[str, Any]] = []
        for row in reversed(rows):
            fresh = int(row["input_tokens_actual"] or 0)
            cached = int(row["cached_tokens_actual"] or 0)
            output = int(row["output_tokens_actual"] or 0)
            cost = (
                fresh * cfg.input_price_per_million
                + cached * cfg.cached_input_price_per_million
                + output * cfg.output_price_per_million
            ) / 1e6
            out.append(
                {
                    "r": int(row["round_no"]),
                    "fresh": fresh,
                    "cached": cached,
                    "output": output,
                    "cost": round(cost, 6),
                }
            )
        return out

    async def save_config(self) -> dict[str, Any]:
        payload = await self._json_body()
        values = payload.get("values")
        if not isinstance(values, dict) or not values:
            return {"ok": False, "message": "没有需要保存的配置"}
        clean: dict[str, Any] = {}
        for path, raw in values.items():
            if path not in ALLOWED_PATHS:
                continue
            try:
                clean[path] = self._coerce(path, raw)
            except (TypeError, ValueError):
                continue
        if not clean:
            return {"ok": False, "message": "没有可保存的有效字段"}
        try:
            for path, value in clean.items():
                self._apply(path, value)
            await self._persist_config()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"写入配置失败：{exc}"}
        return {"ok": True, "saved": clean, "config": self.engine.cfg.to_dict()}

    def _apply(self, path: str, value: Any) -> None:
        _set_path(self.engine.cfg, path, value)
        if self.config is not None:
            try:
                _set_path(self.config, path, value)
            except Exception:
                pass
        # Components cache their own enabled flag at construction time.
        if path == "aux.topic_stack":
            self.engine.topic.enabled = bool(value)
        elif path == "aux.rhythm":
            self.engine.rhythm.enabled = bool(value)
        elif path == "aux.style":
            self.engine.style.enabled = bool(value)
        elif path == "aux.lexicon":
            self.engine.lexicon.enabled = bool(value)
        elif path == "aux.open_loops":
            self.engine.loops.enabled = bool(value)
        elif path == "aux.negative_patterns":
            self.engine.negative.enabled = bool(value)
        elif path == "l2.enabled":
            self.engine.l2.cfg.enabled = bool(value)
        elif path == "l1.enabled":
            self.engine.l1.cfg.enabled = bool(value)
        elif path == "shortcircuit.enabled":
            self.engine.shortcircuit.cfg.enabled = bool(value)

    def _coerce(self, path: str, raw: Any) -> Any:
        if path == "live_allowlist":
            if isinstance(raw, str):
                return [line.strip() for line in raw.splitlines() if line.strip()]
            if isinstance(raw, list):
                return [str(x).strip() for x in raw if str(x).strip()]
            raise TypeError
        if path in _BOOL_PATHS:
            if isinstance(raw, bool):
                return raw
            return str(raw).lower() in ("1", "true", "yes", "on")
        if path in _INT_PATHS:
            return int(float(raw))
        if path in {
            "l0.decay",
            "l0.hysteresis",
            "l3.recent_char_low_water",
            "shortcircuit.hit_threshold",
            "shortcircuit.degrade_hours",
            "l1.low_l0_threshold",
            "l1.candidate_min_score",
            "metrics.input_price_per_million",
            "metrics.cached_input_price_per_million",
            "metrics.output_price_per_million",
        }:
            return float(raw)
        if path == "l3.host_context_mode":
            value = str(raw)
            if value not in ("replace", "prepend", "off"):
                raise ValueError
            return value
        return raw
