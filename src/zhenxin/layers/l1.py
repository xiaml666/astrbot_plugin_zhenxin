"""L1 on-demand wake-up.

This is the only retrieval-ish path.  It is deliberately hard to trigger:

* deictic/question wording AND weak L0 relevance, or
* a contradiction with a hot clue, or
* an open loop being touched.

Candidates are scored locally.  A cold seed already represented in L0 is not
"recalled" again, and a recently woken seed has a cooldown.  Both rules make
the wake rate fall as L0 absorbs recurring topics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..config import L1Config
from ..db import Database
from ..embeddings import LocalScorer
from ..normalize import content_tokens, has_strong_deictic, is_ack, is_negation


@dataclass
class Candidate:
    row_id: int
    seed_tokens: str
    thin_context: str
    score: float
    last_wake_round: int
    wake_count: int


@dataclass
class L1Result:
    wake: bool
    trigger: str
    recall: str
    candidates: list[Candidate] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


class L1Layer:
    def __init__(
        self,
        db: Database,
        scorer: LocalScorer,
        cfg: L1Config,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db = db
        self.scorer = scorer
        self.cfg = cfg
        self.clock = clock

    def candidates(
        self,
        user_key: str,
        text: str,
        round_no: int,
        injected_seeds: Sequence[str] = (),
    ) -> list[Candidate]:
        rows = self.db.fetchall(
            "SELECT id,seed_tokens,thin_context,last_wake_round,wake_count,access_count "
            "FROM cold_seeds WHERE user_key=? ORDER BY id DESC LIMIT 200",
            (user_key,),
        )
        out: list[Candidate] = []
        for row in rows:
            last_wake = int(row["last_wake_round"] or 0)
            if last_wake and round_no - last_wake < self.cfg.wake_cooldown_rounds:
                continue
            score = self.scorer.seed_similarity(text, row["seed_tokens"], row["thin_context"])
            if score < self.cfg.candidate_min_score:
                continue
            # If L0 already carries this seed, waking adds nothing.
            if any(self.scorer.similarity(row["seed_tokens"], seed) >= 0.72 for seed in injected_seeds if seed):
                continue
            out.append(
                Candidate(
                    row_id=int(row["id"]),
                    seed_tokens=row["seed_tokens"],
                    thin_context=row["thin_context"],
                    score=score,
                    last_wake_round=last_wake,
                    wake_count=int(row["wake_count"] or 0),
                )
            )
        out.sort(key=lambda c: c.score, reverse=True)
        return out[: max(1, self.cfg.rerank_top_k if self.cfg.rerank else self.cfg.max_recalls)]

    def decide(
        self,
        user_key: str,
        text: str,
        round_no: int,
        *,
        l0_best: float,
        injected_seeds: Sequence[str] = (),
        injected_slots: Sequence[int] = (),
        loop_touched: Sequence[str] = (),
        reranker: Callable[[str, list[Candidate]], list[int]] | None = None,
    ) -> L1Result:
        if not self.cfg.enabled or is_ack(text):
            return L1Result(False, "off_or_ack", "")
        cands = self.candidates(user_key, text, round_no, injected_seeds)
        if not cands:
            return L1Result(False, "no_candidate", "", detail={"l0_best": l0_best})

        trigger = ""
        strong_ref = has_strong_deictic(text)
        contradiction = self._contradiction(text, injected_seeds) and l0_best >= 0.1
        if contradiction:
            trigger = "contradiction"
        elif loop_touched and strong_ref:
            trigger = "open_loop"
        elif l0_best < self.cfg.low_l0_threshold and strong_ref:
            trigger = "deictic_low_l0"
        if not trigger:
            return L1Result(False, "l0_sufficient", "", cands, {"l0_best": l0_best})

        # Optional small-model rerank.  It only reorders local candidates and
        # is never used for retrieval or memory updates.
        if self.cfg.rerank and reranker is not None and len(cands) > 1:
            try:
                order = reranker(text, cands)
                if order:
                    by_id = {c.row_id: c for c in cands}
                    cands = [by_id[i] for i in order if i in by_id] + [c for c in cands if c.row_id not in order]
            except Exception:
                pass

        selected = cands[: max(1, self.cfg.max_recalls)]
        recall = self._assemble(selected)
        now = self.clock()
        for cand in selected:
            self.db.execute(
                "UPDATE cold_seeds SET access_count=access_count+1,last_access=?,"
                "last_wake_round=?,wake_count=wake_count+1 WHERE id=?",
                (now, round_no, cand.row_id),
            )
        return L1Result(
            True,
            trigger,
            recall,
            selected,
            {"l0_best": l0_best, "candidates": len(cands), "trigger": trigger},
        )

    def _assemble(self, selected: list[Candidate]) -> str:
        if not selected:
            return ""
        bits = [c.thin_context for c in selected if c.thin_context]
        text = "【想起】" + "；".join(bits)
        if len(text) > self.cfg.recall_max_chars:
            text = text[: self.cfg.recall_max_chars - 1].rstrip("；") + "…"
        return text

    @staticmethod
    def _contradiction(text: str, injected_seeds: Sequence[str]) -> bool:
        if not is_negation(text):
            return False
        neg_free = text
        for neg in ("不是", "不对", "没有", "不", "没", "别", "不要"):
            neg_free = neg_free.replace(neg, "")
        q = set(content_tokens(neg_free))
        if not q:
            return False
        for seed in injected_seeds:
            shared = q & set(content_tokens(seed))
            if len(shared) >= 1 and any(len(tok) >= 2 for tok in shared):
                return True
        return False
