"""Metrics and shadow-mode reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import MetricsConfig
from .db import Database
from .llm import LLMCallLedger
from .models import TurnTrace


class MetricsRecorder:
    def __init__(self, db: Database, cfg: MetricsConfig, ledger: LLMCallLedger | None = None) -> None:
        self.db = db
        self.cfg = cfg
        self.ledger = ledger

    def record(self, trace: TurnTrace) -> None:
        if not self.cfg.enabled:
            return
        self.db.execute(
            "INSERT INTO metrics(user_key,round_no,mode,wake,wake_trigger,prefix_hash,prefix_stable,"
            "l0_chars,l0_balls,recent_chars,ephemeral_chars,injected_tokens,shortcircuit_hit,"
            "solid_signature,solid_confidence,l2_llm_calls,main_llm_calls,decision_llm_calls,"
            "reply_chars,input_tokens_actual,cached_tokens_actual,output_tokens_actual,"
            "actual_usage_available,created_at,trace_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                trace.user_key,
                trace.round_no,
                trace.mode,
                int(trace.wake),
                trace.wake_trigger,
                trace.prefix_hash,
                None if trace.prefix_stable is None else int(trace.prefix_stable),
                trace.l0_chars,
                trace.l0_balls,
                trace.recent_chars,
                trace.ephemeral_chars,
                trace.injected_tokens,
                int(trace.shortcircuit_hit),
                trace.solid_signature,
                trace.solid_confidence,
                trace.l2_llm_calls,
                trace.main_llm_calls,
                trace.decision_llm_calls,
                trace.reply_chars,
                trace.input_tokens_actual,
                trace.cached_tokens_actual,
                trace.output_tokens_actual,
                int(trace.actual_usage_available),
                trace.created_at,
                trace.to_json(),
            ),
        )
        if self.cfg.jsonl_path and self.cfg.log_traces:
            try:
                path = Path(self.cfg.jsonl_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(trace.to_json() + "\n")
            except OSError:
                # Metrics must never break the chat turn.
                pass

    def report(self, user_key: str | None = None) -> dict[str, Any]:
        where = "WHERE user_key=?" if user_key else ""
        params = (user_key,) if user_key else ()
        rows = [
            dict(r)
            for r in self.db.fetchall(
                f"SELECT * FROM metrics {where} ORDER BY round_no ASC,id ASC", params
            )
        ]
        if not rows:
            return {"rounds": 0, "wake_rate": 0.0}
        rounds = len(rows)
        wakes = sum(int(r["wake"]) for r in rows)
        stable_rows = [r for r in rows if r["prefix_stable"] is not None]
        stable = sum(int(r["prefix_stable"]) for r in stable_rows)
        l2_calls = sum(int(r["l2_llm_calls"]) for r in rows)
        main_calls = sum(int(r["main_llm_calls"]) for r in rows)
        decision_calls = sum(int(r["decision_llm_calls"]) for r in rows)
        solid_hits = sum(int(r["shortcircuit_hit"]) for r in rows)
        actual_rounds = sum(1 for r in rows if int(r.get("actual_usage_available", 0)))
        actual_input = sum(int(r.get("input_tokens_actual", 0)) for r in rows)
        actual_cached = sum(int(r.get("cached_tokens_actual", 0)) for r in rows)
        actual_output = sum(int(r.get("output_tokens_actual", 0)) for r in rows)
        actual_cost = (
            actual_input / 1_000_000 * self.cfg.input_price_per_million
            + actual_cached / 1_000_000 * self.cfg.cached_input_price_per_million
            + actual_output / 1_000_000 * self.cfg.output_price_per_million
        )
        l0_chars = [int(r["l0_chars"]) for r in rows]
        tokens = [int(r["injected_tokens"]) for r in rows]
        buckets: dict[str, dict[str, float]] = {}
        for r in rows:
            b = ((int(r["round_no"]) - 1) // 10) * 10 + 1
            key = f"{b}-{b + 9}"
            bucket = buckets.setdefault(key, {"rounds": 0, "wakes": 0, "wake_rate": 0.0})
            bucket["rounds"] += 1
            bucket["wakes"] += int(r["wake"])
            bucket["wake_rate"] = bucket["wakes"] / bucket["rounds"]
        # Last 20 prefix stability, the acceptance metric.
        last20 = stable_rows[-20:]
        last20_ratio = (sum(int(r["prefix_stable"]) for r in last20) / len(last20)) if last20 else 1.0
        longest = 0
        current = 0
        for r in stable_rows:
            if int(r["prefix_stable"]):
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return {
            "rounds": rounds,
            "turns_alive": max(int(r["round_no"]) for r in rows),
            "wake_count": wakes,
            "wake_rate": wakes / rounds,
            "wake_buckets": buckets,
            "prefix_stable_ratio": (stable / len(stable_rows)) if stable_rows else 1.0,
            "prefix_stable_last20": last20_ratio,
            "prefix_longest_streak": longest,
            "l2_llm_calls": l2_calls,
            "main_llm_calls": main_calls,
            "decision_llm_calls": decision_calls,
            "solid_hits": solid_hits,
            "solid_hit_rate": solid_hits / rounds,
            "actual_usage_rounds": actual_rounds,
            "actual_input_tokens": actual_input,
            "actual_cached_tokens": actual_cached,
            "actual_total_input_tokens": actual_input + actual_cached,
            "actual_output_tokens": actual_output,
            "actual_cost": actual_cost,
            "actual_currency": self.cfg.currency,
            "l0_chars_max": max(l0_chars) if l0_chars else 0,
            "l0_chars_mean": sum(l0_chars) / len(l0_chars) if l0_chars else 0.0,
            "injected_tokens_mean": sum(tokens) / len(tokens) if tokens else 0.0,
            "shadow_rounds": sum(1 for r in rows if r["mode"].startswith("shadow")),
            "modes": _counts(r["mode"] for r in rows),
            "wake_triggers": _counts(r["wake_trigger"] for r in rows if r["wake"]),
            "created_span": (rows[-1]["created_at"] - rows[0]["created_at"]) if rounds > 1 else 0.0,
        }

    def recent(self, user_key: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT * FROM metrics WHERE user_key=? ORDER BY id DESC LIMIT ?",
            (user_key, limit),
        )
        return [dict(r) for r in reversed(rows)]


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        out[str(value)] = out.get(str(value), 0) + 1
    return out
