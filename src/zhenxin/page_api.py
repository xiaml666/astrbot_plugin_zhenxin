"""AstrBot plugin-page backend.

AstrBot discovers ``pages/<name>/index.html`` and lists it under the left
"扩展功能" rail.  This module backs that page with curated JSON endpoints:
config save, realtime status, the memory garden, style learning, proactive
messages, a plugin-conflict scan and a feedback channel.  It intentionally
exposes only allow-listed fields, so the page cannot change the database path
or model endpoints at runtime.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sqlite3
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
    "proactive.enabled",
    "proactive.min_idle_minutes",
    "proactive.min_interval_hours",
    "proactive.max_per_day",
    "proactive.active_window_hours",
    "proactive.quiet_start",
    "proactive.quiet_end",
    "proactive.only_private",
    "proactive.use_llm",
    "proactive.max_chars",
    "proactive.template",
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
    "memory.enabled",
    "memory.top_k",
    "memory.min_score",
    "memory.weight_semantic",
    "memory.weight_importance",
    "memory.weight_recency",
    "memory.weight_usage",
    "memory.weight_kind",
    "memory.exact_boost",
    "memory.address_boost",
    "memory.cycle_days",
    "memory.decay_half_life_days",
    "memory.treasure_threshold",
    "memory.shallow_threshold",
    "memory.trash_ttl_days",
    "style_learning.enabled",
    "style_learning.learn_user",
    "style_learning.online",
    "style_learning.interval_days",
    "style_learning.max_chars",
    "metrics.enabled",
    "metrics.input_price_per_million",
    "metrics.cached_input_price_per_million",
    "metrics.output_price_per_million",
}

_BOOL_PATHS = {p for p in ALLOWED_PATHS if p.endswith((".enabled", "shadow_mode"))} | {
    "l0.tail_injection",
    "aux.style",
    "proactive.only_private",
    "proactive.use_llm",
    "style_learning.learn_user",
    "style_learning.online",
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
    "memory.top_k",
    "style_learning.max_chars",
    "proactive.min_idle_minutes",
    "proactive.max_per_day",
    "proactive.quiet_start",
    "proactive.quiet_end",
    "proactive.max_chars",
}
_FLOAT_PATHS = {
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
    "memory.min_score",
    "memory.weight_semantic",
    "memory.weight_importance",
    "memory.weight_recency",
    "memory.weight_usage",
    "memory.weight_kind",
    "memory.exact_boost",
    "memory.address_boost",
    "memory.cycle_days",
    "memory.decay_half_life_days",
    "memory.treasure_threshold",
    "memory.shallow_threshold",
    "memory.trash_ttl_days",
    "style_learning.interval_days",
    "proactive.min_interval_hours",
    "proactive.active_window_hours",
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

# "Recommended" is the tested combination: real takeover + stable prefix.
# It is adapted at runtime when other memory/context plugins are installed.
RECOMMENDED_VALUES = {
    "shadow_mode": False,
    "l3.host_context_mode": "replace",
    "l0.inject": 3,
    "l0.max_inject_chars": 260,
    "l0.tail_injection": True,
    "l3.recent_turns": 8,
    "l3.window_grow_turns": 4,
    "l3.recent_char_low_water": 0.7,
    "l0.hysteresis": 2,
    "l0.min_dwell_rounds": 8,
    "l0.rebalance_interval": 8,
    "l1.enabled": True,
    "l2.enabled": True,
    "aux.topic_stack": True,
    "aux.compression": True,
    "aux.rhythm": True,
    "aux.style": True,
    "aux.lexicon": True,
    "aux.open_loops": True,
    "l1.wake_cooldown_rounds": 8,
    "memory.enabled": True,
    "memory.top_k": 3,
    "memory.min_score": 0.3,
    "style_learning.enabled": True,
    "style_learning.learn_user": True,
    "style_learning.online": True,
    "proactive.enabled": False,
    "shortcircuit.enabled": False,
}
TAKEOVER_VALUES = {"shadow_mode": False, "l3.host_context_mode": "replace"}


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
        self._scan_cache: tuple[float, dict[str, Any]] | None = None

    def register(self) -> None:
        register = getattr(self.plugin.context, "register_web_api", None)
        if not callable(register):
            return
        routes = (
            ("status", self.status, ["GET"], "Zhenxin page status"),
            ("config", self.save_config, ["POST"], "Zhenxin page config"),
            ("preset", self.preset, ["POST"], "Zhenxin apply a preset"),
            ("reset", self.reset, ["POST"], "Zhenxin page reset"),
            ("memory", self.memory_list, ["GET"], "Zhenxin memory list"),
            ("memory/rebuild", self.memory_rebuild, ["POST"], "Zhenxin memory rebuild"),
            ("memory/boost", self.memory_boost, ["POST"], "Zhenxin memory boost"),
            ("memory/undo", self.memory_undo, ["POST"], "Zhenxin undo memory boost"),
            ("memory/trash", self.memory_trash, ["POST"], "Zhenxin memory trash"),
            ("memory/restore", self.memory_restore, ["POST"], "Zhenxin memory restore"),
            ("memory/delete", self.memory_delete, ["POST"], "Zhenxin memory delete"),
            ("memory/clear", self.memory_clear, ["POST"], "Zhenxin memory clear"),
            ("style", self.style_status, ["GET"], "Zhenxin style knowledge"),
            ("style/learn", self.style_learn, ["POST"], "Zhenxin style learn"),
            ("proactive", self.proactive_status, ["GET"], "Zhenxin proactive status"),
            ("proactive/test", self.proactive_test, ["POST"], "Zhenxin proactive test"),
            ("health", self.health_scan, ["GET"], "Zhenxin plugin conflict scan"),
            ("health/ack", self.health_ack, ["POST"], "Zhenxin acknowledge conflict alert"),
        )
        for suffix, handler, methods, desc in routes:
            register(f"{PAGE_PREFIX}/{suffix}", handler, methods, desc)

    # --------------------------------------------------------------- requests
    async def _json_body(self) -> dict[str, Any]:
        override = getattr(self, "_payload_override", None)
        if override is not None:
            return dict(override)
        try:
            from astrbot.api.web import request as web_request  # type: ignore

            data = await web_request.json({})
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001
            pass
        try:
            from quart import request  # type: ignore

            return await request.get_json(silent=True) or {}
        except Exception:  # noqa: BLE001
            return {}

    def _query(self, name: str, default: str = "") -> str:
        override = getattr(self, "_query_override", None)
        if override is not None:
            return str(override.get(name, default))
        try:
            from astrbot.api.web import request as web_request  # type: ignore

            value = web_request.query.get(name, default)
            return str(value) if value is not None else default
        except Exception:  # noqa: BLE001
            pass
        try:
            from quart import request  # type: ignore

            return str(request.args.get(name, default))
        except Exception:  # noqa: BLE001
            return default

    # ---------------------------------------------------------------- basics
    async def _persist_config(self) -> None:
        for name in ("save_config_async", "save_config"):
            save = getattr(self.config, name, None)
            if not callable(save):
                continue
            result = save()
            if inspect.isawaitable(result):
                await result
            return

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
        elif path == "l0.inject":
            # Never ask for more clues than the slot table can hold.
            self.engine.cfg.l0.inject = max(1, min(int(value), int(self.engine.cfg.l0.slots)))
            if self.config is not None:
                try:
                    _set_path(self.config, "l0.inject", self.engine.cfg.l0.inject)
                except Exception:
                    pass
        elif path == "memory.enabled":
            self.engine.facts.cfg.enabled = bool(value)
        elif path == "style_learning.enabled":
            self.engine.style_learning.cfg.enabled = bool(value)

    # ------------------------------------------------------------- presets
    @staticmethod
    def _enabled_conflicts(scan: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            item
            for item in scan.get("plugins", [])
            if item.get("enabled", True) and item.get("severity") in ("direct", "warning")
        ]

    def _health_issues(self) -> list[dict[str, Any]]:
        health = self._health()
        issues: list[dict[str, Any]] = []
        if not health.get("db_ok"):
            issues.append(
                {
                    "name": "__database__",
                    "display": "记忆数据库",
                    "severity": "direct",
                    "reasons": ["数据库不可用，接管后可能丢记忆"],
                }
            )
        elif int(health.get("errors") or 0) > 0:
            issues.append(
                {
                    "name": "__errors__",
                    "display": "运行错误",
                    "severity": "warning",
                    "reasons": [
                        f"已累计 {health.get('errors')} 次错误"
                        + (f"：{health.get('last_error')}" if health.get("last_error") else "")
                    ],
                }
            )
        return issues

    async def _apply_preset(self, values: dict[str, Any]) -> dict[str, Any]:
        clean = {
            path: value
            for path, value in values.items()
            if path in ALLOWED_PATHS
        }
        for path, value in clean.items():
            self._apply(path, value)
        await self._persist_config()
        return clean

    async def preset(self) -> dict[str, Any]:
        """Apply takeover / recommended / factory defaults in one click.

        ``takeover`` is a dry run unless ``force`` is true: the page first gets
        the conflict list, asks the user, then retries with ``force``.
        ``recommended`` adapts to the installed plugins and applies silently.
        ``defaults`` restores the shipped fresh-install values.
        """
        payload = await self._json_body()
        mode = str(payload.get("mode") or "").strip()
        force = bool(payload.get("force", False))
        if mode not in ("takeover", "recommended", "defaults"):
            return {"ok": False, "message": "未知的设置模式"}
        try:
            scan = await asyncio.to_thread(self._scan_plugins, True)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"插件检查失败：{exc}"}
        conflicts = self._enabled_conflicts(scan)
        health_issues = self._health_issues()

        if mode == "takeover":
            issues = conflicts + health_issues
            if issues and not force:
                return {
                    "ok": True,
                    "needs_confirm": True,
                    "issues": issues,
                    "values": dict(TAKEOVER_VALUES),
                    "message": "检测到可能冲突的插件，请确认是否继续接管",
                }
            values = dict(TAKEOVER_VALUES)
            messages = ["已由真心接管最近窗口（观察模式已关闭）"]
            if issues:
                messages.append("你已确认在冲突环境下继续接管")
        elif mode == "recommended":
            values = dict(RECOMMENDED_VALUES)
            messages = []
            direct = [item for item in conflicts if item.get("severity") == "direct"]
            if direct:
                values.update(
                    {
                        "shadow_mode": True,
                        "l3.host_context_mode": "off",
                        "l2.enabled": False,
                        "aux.compression": False,
                    }
                )
                messages.append(
                    "检测到直接冲突（"
                    + "、".join(item.get("display") or item.get("name") for item in direct)
                    + "），推荐改为只观察，并关闭摘要/压缩，避免重复接管上下文"
                )
            else:
                messages.append("未发现直接冲突，推荐由真心接管最近窗口")
            proactive = [
                item for item in conflicts if "proactive" in str(item.get("name") or "").lower()
            ]
            if proactive:
                values["proactive.enabled"] = False
                messages.append("已有主动消息插件，推荐关闭真心的主动消息")
            memoryish = [item for item in conflicts if item.get("memoryish")]
            if memoryish:
                values["memory.top_k"] = 2
                values["memory.min_score"] = 0.35
                messages.append("有记忆功能相近的插件，推荐每轮只带 2 条高相关记忆")
            if health_issues:
                values["shadow_mode"] = True
                values["l3.host_context_mode"] = "off"
                messages.append("检测到运行错误/数据库异常，推荐先只观察")
        else:
            values = schema_defaults()
            messages = ["已恢复出厂默认：观察模式，只启用必要功能"]

        try:
            applied = await self._apply_preset(values)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"应用设置失败：{exc}"}
        return {
            "ok": True,
            "applied": True,
            "mode": mode,
            "values": applied,
            "config": self.engine.cfg.to_dict(),
            "messages": messages,
            "conflicts": [
                {
                    "name": item.get("name"),
                    "display": item.get("display"),
                    "severity": item.get("severity"),
                }
                for item in conflicts
            ],
        }

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
        if path in _FLOAT_PATHS:
            return float(raw)
        if path == "l3.host_context_mode":
            value = str(raw)
            if value not in ("replace", "prepend", "off"):
                raise ValueError
            return value
        if path == "proactive.template":
            return str(raw)[:200]
        return raw

    # --------------------------------------------------------------- status
    async def status(self) -> dict[str, Any]:
        from ._version import build_info

        stats = self.engine.stats(None)
        cfg = self.engine.cfg.to_dict()
        pools = stats.get("memory_pools", {})
        usage_rounds = int(stats.get("actual_usage_rounds", 0) or 0)
        series = self._series(30)
        estimated_cost = sum(float(item.get("cost", 0)) for item in series if item.get("estimated"))
        actual_cost = float(stats.get("actual_cost", 0) or 0)
        # Keep the payload small and avoid exposing anything secret-like.
        return {
            "ok": True,
            "plugin": PLUGIN_NAME,
            "version": build_info()["version"],
            "build": build_info(),
            "db_path": self.engine.db.path,
            "scorer": self.engine.scorer.name,
            "config": cfg,
            "health": self._health(),
            "memory": {
                "pools": pools,
                "total": int(sum(pools.values())) if pools else self.engine.facts.count_all(),
                "distinct": self.engine.facts.distinct_count(),
            },
            "proactive": {
                "enabled": bool(self.engine.cfg.proactive.enabled),
                **stats.get("proactive", {}),
            },
            "conflict_alert": self.engine.db.get_state("__global__", "conflict_alert", None),
            "feedback_issue_url": str(self.engine.cfg.feedback.github_issues or ""),
            "usage": {
                "available": usage_rounds > 0,
                "rounds": usage_rounds,
                "source": "provider" if usage_rounds else "estimate",
            },
            "series": series,
            "metrics": {
                "rounds": stats.get("rounds", 0),
                "wake_rate": stats.get("wake_rate", 0),
                "last10_wake_rate": stats.get("last10_wake_rate", 0),
                "prefix_stable_last20": stats.get("prefix_stable_last20", 0),
                "prefix_stable_ratio": stats.get("prefix_stable_ratio", 0),
                "l0_chars_max": stats.get("l0_chars_max", 0),
                "solid_hits": stats.get("solid_hits", 0),
                "solid_hit_rate": stats.get("solid_hit_rate", 0),
                "actual_usage_rounds": usage_rounds,
                "actual_total_input_tokens": stats.get("actual_total_input_tokens", 0),
                "actual_cached_tokens": stats.get("actual_cached_tokens", 0),
                "actual_output_tokens": stats.get("actual_output_tokens", 0),
                "actual_cost": actual_cost,
                "estimated_cost": round(estimated_cost, 6),
                "cost_display": actual_cost if usage_rounds else round(estimated_cost, 6),
                "actual_currency": stats.get("actual_currency", "CNY"),
                "l2_llm_calls": stats.get("l2_llm_calls", 0),
                "injected_tokens_mean": stats.get("injected_tokens_mean", 0),
            },
        }

    def _health(self) -> dict[str, Any]:
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
            health["facts"] = self.engine.facts.distinct_count()
            health["fact_rows"] = self.engine.facts.count_all()
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
                " output_tokens_actual, injected_tokens, reply_chars,"
                " actual_usage_available, created_at FROM metrics"
                " ORDER BY id DESC LIMIT ?",
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
            estimated = not (fresh or cached or output)
            if estimated:
                # The provider did not return usage.  Show the plugin's own
                # prompt estimate instead of an empty chart.
                fresh = int(row["injected_tokens"] or 0)
                output = max(1, int(row["reply_chars"] or 0) // 2)
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
                    "estimated": estimated,
                }
            )
        return out

    # --------------------------------------------------------------- memory
    async def memory_list(self) -> dict[str, Any]:
        try:
            # Recovery pass first: pulls "叫我…/你叫…/记住…" out of the stored
            # recent turns, then runs one forgetting cycle.
            self.engine.facts.backfill_all()
            self.engine.facts.sweep(None, max_archive=50)
            user = self._query("user")
            view = self._query("view", "day")
            pool = self._query("pool")
            date = self._query("date")
            if view == "all" or pool:
                date = ""
            if view == "day" and not date:
                date = time.strftime("%Y-%m-%d")
            facts = self.engine.facts.list_facts(
                user or None, date=date, pool=pool, limit=400
            )
            dates = self.engine.facts.dates(user or None)
            pools = self.engine.facts.pool_counts(user or None)
            traces = self.engine.facts.traces(user or None, limit=100)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"读取记忆库失败：{exc}", "facts": []}
        return {
            "ok": True,
            "view": view,
            "date": date,
            "pool": pool,
            "total": len(facts),
            "pools": pools,
            "dates": dates[:60],
            "facts": facts,
            "traces": traces,
            "archived_total": self.engine.facts.trace_count(user or None),
            "today": time.strftime("%Y-%m-%d"),
        }

    async def memory_rebuild(self) -> dict[str, Any]:
        """Re-extract identity and explicit memories from conversation history."""
        try:
            counts = self.engine.facts.backfill_all()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"重建记忆失败：{exc}"}
        total = sum(counts.values())
        return {
            "ok": True,
            "added": total,
            "users": len(counts),
            "detail": counts,
            "message": f"从历史对话补回 {total} 条记忆",
        }

    async def memory_boost(self) -> dict[str, Any]:
        payload = await self._json_body()
        fact_id = payload.get("id")
        delta = payload.get("delta", 0.12)
        if fact_id is None:
            return {"ok": False, "message": "缺少记忆 id"}
        try:
            row = self.engine.facts.boost(int(fact_id), float(delta))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"调整权重失败：{exc}"}
        if row is None:
            return {"ok": False, "message": "这条记忆不存在"}
        return {"ok": True, "fact": row}

    async def memory_undo(self) -> dict[str, Any]:
        """Restore the values captured before the last click-to-deepen."""
        payload = await self._json_body()
        if payload.get("id") is None:
            return {"ok": False, "message": "缺少记忆 id"}
        try:
            row = self.engine.facts.set_boost(
                int(payload["id"]),
                float(payload.get("manual_boost", 0)),
                float(payload.get("importance", 0.5)),
                str(payload.get("pool") or "") or None,
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"撤销失败：{exc}"}
        if row is None:
            return {"ok": False, "message": "这条记忆不存在"}
        return {"ok": True, "fact": row}

    async def memory_trash(self) -> dict[str, Any]:
        payload = await self._json_body()
        if payload.get("id") is None:
            return {"ok": False, "message": "缺少记忆 id"}
        try:
            ok = self.engine.facts.trash(int(payload["id"]), "manual")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"移入回收池失败：{exc}"}
        return {"ok": bool(ok)}

    async def memory_restore(self) -> dict[str, Any]:
        payload = await self._json_body()
        if payload.get("id") is None:
            return {"ok": False, "message": "缺少记忆 id"}
        try:
            ok = self.engine.facts.restore(int(payload["id"]))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"恢复失败：{exc}"}
        return {"ok": bool(ok)}

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
                    "DELETE FROM facts WHERE user_key LIKE ?", (f"%{user}%",)
                )
                self.engine.db.execute(
                    "DELETE FROM fact_traces WHERE user_key LIKE ?", (f"%{user}%",)
                )
            else:
                self.engine.db.execute("DELETE FROM facts")
                self.engine.db.execute("DELETE FROM fact_traces")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"清空失败：{exc}"}
        return {"ok": True}

    # ---------------------------------------------------------------- style
    async def style_status(self) -> dict[str, Any]:
        try:
            users = self.engine.style_learning.recent_users(limit=12)
            user = self._query("user")
            if not user and users:
                user = users[0]["user"]
            status = self.engine.style_learning.status(user) if user else {
                "enabled": self.engine.cfg.style_learning.enabled,
                "user": "",
                "knowledge": "",
                "chars": 0,
                "max_chars": self.engine.cfg.style_learning.max_chars,
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"读取说话习惯失败：{exc}"}
        return {
            "ok": True,
            "users": users,
            "selected": user,
            "status": status,
            "profile": self.engine.style.describe(user) if user else "还在观察",
        }

    async def style_learn(self) -> dict[str, Any]:
        payload = await self._json_body()
        user = str(payload.get("user") or "").strip()
        online = bool(payload.get("online", False))
        if not user:
            users = self.engine.style_learning.recent_users(limit=1)
            user = users[0]["user"] if users else ""
        if not user:
            return {"ok": False, "message": "还没有可学习的对话"}
        try:
            result = await asyncio.to_thread(
                self.engine.style_learning.refresh, user, online=online
            )
        except Exception as exc:  # noqa: BLE001
            self.engine.record_error("style_learn", str(exc))
            return {"ok": False, "message": f"学习失败：{exc}"}
        return result

    # ------------------------------------------------------------ proactive
    async def proactive_status(self) -> dict[str, Any]:
        cfg = self.engine.cfg.proactive
        try:
            candidates = self.engine.proactive.candidates()
            previews = []
            for item in candidates[:5]:
                preview = self.engine.proactive.preview(item["user"])
                ok, reason = self.engine.proactive.gate(item["user"], item["last_user_at"])
                preview.update({"allowed": ok, "reason": reason})
                previews.append(preview)
            stats = self.engine.proactive.stats(limit=10)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"读取主动消息状态失败：{exc}"}
        return {
            "ok": True,
            "config": {
                "enabled": cfg.enabled,
                "min_idle_minutes": cfg.min_idle_minutes,
                "min_interval_hours": cfg.min_interval_hours,
                "max_per_day": cfg.max_per_day,
                "quiet_start": cfg.quiet_start,
                "quiet_end": cfg.quiet_end,
                "only_private": cfg.only_private,
                "use_llm": cfg.use_llm,
            },
            "stats": stats,
            "previews": previews,
        }

    async def proactive_test(self) -> dict[str, Any]:
        payload = await self._json_body()
        user = str(payload.get("user") or "").strip()
        text = str(payload.get("text") or "").strip()
        try:
            if not user:
                candidates = self.engine.proactive.candidates()
                user = candidates[0]["user"] if candidates else ""
            if not user:
                return {"ok": False, "message": "还没有可发送的会话"}
            if not text:
                text, kind = self.engine.proactive.compose(user)
            else:
                kind = "manual"
            sender = getattr(self.plugin, "_send_text", None)
            sent = False
            if callable(sender):
                sent = bool(await sender(user, text))
            if sent:
                self.engine.proactive.mark_sent(user, text, kind)
            return {"ok": True, "sent": sent, "user": user, "text": text, "kind": kind}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"发送失败：{exc}"}

    # --------------------------------------------------------------- health
    async def health_scan(self) -> dict[str, Any]:
        try:
            payload = await asyncio.to_thread(self._scan_plugins, True)
            alert = await asyncio.to_thread(self._update_conflict_alert, payload)
            return {**payload, "alert": alert}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"插件体检失败：{exc}", "plugins": []}

    def _scan_plugins(self, force: bool = False) -> dict[str, Any]:
        now = time.time()
        if not force and self._scan_cache and now - self._scan_cache[0] < 30:
            return dict(self._scan_cache[1])
        plugins_dir = Path(__file__).resolve().parents[3]
        if plugins_dir.name != "plugins":
            guess = Path.cwd() / "data" / "plugins"
            if guess.is_dir():
                plugins_dir = guess
        results: list[dict[str, Any]] = []
        inactive = self._inactive_plugins(plugins_dir)
        disabled = 0
        if plugins_dir.is_dir():
            for child in sorted(plugins_dir.iterdir()):
                if not child.is_dir() or child.name.startswith((".", "_")):
                    continue
                if child.name == PLUGIN_NAME:
                    continue
                if child.name in inactive:
                    # Disabled plugins are not part of the live conflict set.
                    disabled += 1
                    continue
                item = self._inspect_plugin(child, inactive)
                if item:
                    results.append(item)
        summary = {
            "direct": sum(1 for item in results if item["severity"] == "direct"),
            "warning": sum(1 for item in results if item["severity"] == "warning"),
            "info": sum(1 for item in results if item["severity"] == "info"),
            "disabled": disabled,
        }
        payload = {
            "ok": True,
            "plugins_dir": str(plugins_dir),
            "scanned": len(results),
            "summary": summary,
            "plugins": results,
            "advice": (
                "只让一个插件接管上下文最稳。只有会改写上下文窗口、系统提示或每轮摘要/压缩的插件才算直接冲突；"
                "表情包、B 站等媒体插件不影响真心的上下文寿命和预算，不计入冲突。"
            ),
        }
        self._scan_cache = (now, payload)
        return payload

    def _update_conflict_alert(
        self, payload: dict[str, Any], *, baseline: bool = False
    ) -> dict[str, Any] | None:
        """Compare enabled direct/warning plugins with the last scan.

        The first scan only records a baseline, so an existing installation
        does not get an alert storm.  Both the background poll and the health
        page call this, so opening 体检与反馈 catches a plugin enabled since the
        last poll instead of waiting for the next 60s tick.
        """
        current = [
            item
            for item in payload.get("plugins", [])
            if item.get("enabled", True) and item.get("severity") in ("direct", "warning")
        ]
        names = [item["name"] for item in current]
        known_raw = self.engine.db.get_state("__global__", "health_known_plugins", None)
        if known_raw is None or baseline:
            self.engine.db.set_state("__global__", "health_known_plugins", names)
            return None
        known = set(str(x) for x in (known_raw or []))
        fresh = [item for item in current if item["name"] not in known]
        self.engine.db.set_state("__global__", "health_known_plugins", names)
        if not fresh:
            return None
        alert = {
            "at": time.time(),
            "items": [
                {
                    "name": item["name"],
                    "display": item.get("display") or item["name"],
                    "severity": item["severity"],
                    "reasons": item.get("reasons", [])[:2],
                }
                for item in fresh
            ],
        }
        self.engine.db.set_state("__global__", "conflict_alert", alert)
        return alert

    def poll_conflicts(self) -> dict[str, Any] | None:
        """Background entry point: alert when a conflicting plugin appears."""
        try:
            return self._update_conflict_alert(self._scan_plugins(True))
        except Exception:  # noqa: BLE001 - health polling is best effort
            return None

    async def health_ack(self) -> dict[str, Any]:
        self.engine.db.set_state("__global__", "conflict_alert", None)
        return {"ok": True}

    @staticmethod
    def _inactive_plugins(plugins_dir: Path) -> set[str]:
        """Read AstrBot's persisted disabled-plugin list (best effort).

        A disabled memory plugin is not a live conflict.  AstrBot stores the
        list in ``data/data_v4.db`` under ``preferences.inactivated_plugins``
        as module paths; map those back to directory names.
        """
        out: set[str] = set()
        db_path = plugins_dir.parent / "data_v4.db"
        if not db_path.is_file():
            return out
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
            row = conn.execute(
                "SELECT value FROM preferences WHERE scope='global' AND scope_id='global'"
                " AND key='inactivated_plugins' LIMIT 1"
            ).fetchone()
            conn.close()
            if not row:
                return out
            payload = json.loads(row[0])
            values = payload.get("val", payload) if isinstance(payload, dict) else payload
            for item in values or []:
                parts = str(item).split(".")
                if len(parts) >= 3 and parts[-1] == "main":
                    out.add(parts[-2])
        except (OSError, sqlite3.Error, ValueError, TypeError):
            return set()
        return out

    @classmethod
    def _inspect_plugin(cls, path: Path, inactive: set[str] | None = None) -> dict[str, Any] | None:
        meta = path / "metadata.yaml"
        if not meta.is_file():
            return None
        name = display = version = desc = ""
        try:
            for line in meta.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or ":" not in stripped:
                    continue
                key, value = stripped.split(":", 1)
                value = value.strip().strip("'\"")
                if key == "name":
                    name = value
                elif key == "display_name":
                    display = value
                elif key == "version":
                    version = value
                elif key in ("desc", "short_desc", "description"):
                    desc = (desc + " " + value).strip()
        except OSError:
            return None
        name = name or path.name
        text_parts: list[str] = []
        hooks = {"on_llm_request": 0, "on_llm_response": 0, "contexts": 0, "system_prompt": 0}
        scanned = 0
        for source in path.rglob("*.py"):
            if scanned >= 160:
                break
            if any(part in ("tests", "test", "__pycache__", ".venv", "venv") for part in source.parts):
                continue
            try:
                content = source.read_text(encoding="utf-8", errors="ignore")[:220_000]
            except OSError:
                continue
            scanned += 1
            text_parts.append(content)
            for key in hooks:
                hooks[key] += content.count(key)
        haystack = (name + " " + display + " " + desc).lower()
        name_l = name.lower()
        # Media / platform-specific plugins only send links, images or stats.
        # They do not touch the context window or prompt budget, so counting
        # them as conflicts is noise (the user explicitly called this out).
        if any(
            key in name_l
            for key in (
                "bilibili",
                "biliread",
                "meme",
                "parser",
                "qqprofile",
                "matsuko",
                "music",
                "video",
                "tts",
            )
        ):
            return None
        memoryish = any(
            word in haystack
            for word in (
                "记忆",
                "memory",
                "上下文",
                "context",
                "摘要",
                "summary",
                "压缩",
                "compact",
                "知识库",
                "rag",
                "人设",
                "persona",
            )
        )
        reasons: list[str] = []
        severity = "info"
        if "proactive" in name_l:
            # A second proactive-message plugin duplicates behaviour, but it
            # does not shorten the context window; warn, do not call it direct.
            severity = "warning"
            reasons.append("也提供主动消息，和真心的主动消息功能重复（可只留一个）")
        elif memoryish and hooks["on_llm_request"] and (hooks["contexts"] or hooks["system_prompt"]):
            severity = "direct"
            if hooks["contexts"]:
                reasons.append("也会改写 req.contexts（最近对话窗口），与真心的窗口接管重叠")
            if hooks["system_prompt"]:
                reasons.append("也会改写 system_prompt，可能每轮改变稳定前缀")
            reasons.append("会实际影响上下文寿命/预算，属于直接冲突")
        elif hooks["on_llm_request"] and hooks["system_prompt"]:
            severity = "warning"
            reasons.append("在 on_llm_request 里改写 system_prompt，可能每轮改变稳定前缀")
        elif hooks["on_llm_request"]:
            severity = "warning"
            reasons.append("也挂了 on_llm_request，注入顺序可能互相影响")
        if memoryish and severity == "info":
            severity = "warning"
            reasons.append("功能标签与真心相近（记忆/上下文/摘要/知识库），可能重复注入")
        if not reasons:
            return None
        enabled = path.name not in (inactive or set())
        if not enabled:
            reasons = ["当前已停用，不会生效"] + reasons
            severity = "info"
        return {
            "name": name,
            "display": display or name,
            "version": version,
            "severity": severity,
            "reasons": reasons,
            "hooks": hooks,
            "memoryish": memoryish,
            "enabled": enabled,
        }

    # ---------------------------------------------------------------- reset
    async def reset(self) -> dict[str, Any]:
        """Clear learned state without touching AstrBot's own data.

        ``conversation`` is what "换模型 / 新开对话" needs: recent window, hot
        clues, topic stack, open loops and solidified paths.  Long-term facts
        and the cost ledger survive unless explicitly included.
        """
        payload = await self._json_body()
        scope = str(payload.get("scope") or "").strip()
        user = str(payload.get("user") or "").strip()
        if scope not in ("conversation", "memory", "metrics", "style", "all"):
            return {"ok": False, "message": "未知的重置范围"}
        deleted: dict[str, int] = {}
        try:
            if scope in ("conversation", "all"):
                deleted["conversation"] = self._reset_conversation(user)
            if scope in ("memory", "all"):
                deleted["memory"] = self._delete_rows("facts", user)
                deleted["traces"] = self._delete_rows("fact_traces", user)
            if scope in ("style", "all"):
                deleted["style"] = self._delete_rows("style_knowledge", user)
                deleted["proactive"] = self._delete_rows("proactive_log", user)
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
