"""Local similarity backends.

Two implementations:

* ``OnnxEncoder`` - optional bge-small-zh / multilingual MiniLM ONNX model.
* ``HashingEncoder`` - deterministic char/word feature hashing fallback.

The fallback is intentionally dependency-free and is the default on a bare
install.  Neither implementation calls a remote API, and the scorer is never
given an LLM client.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

from .config import EmbeddingConfig
from .normalize import content_tokens, overlap_ratio, tokens

try:  # AstrBot plugin runtime: use the host logger (log level / tag aware).
    from astrbot.api import logger as log
except Exception:  # Standalone library / tests: stdlib fallback.
    from ._aux.stdlog import log


def _l2(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 1e-12:
        return [0.0 for _ in vec]
    return [x / norm for x in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return max(-1.0, min(1.0, sum(a[i] * b[i] for i in range(n))))


class HashingEncoder:
    name = "hash"

    def __init__(self, dims: int = 384) -> None:
        self.dims = max(64, int(dims))

    def encode(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        toks = tokens(text)
        # Word/bigram features carry most of the signal.
        for tok in toks:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, "little") % self.dims
            vec[idx] += 1.0
        # Character unigrams make short Chinese messages less brittle.
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                digest = hashlib.blake2b(("u:" + ch).encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest, "little") % self.dims
                vec[idx] += 0.35
        return _l2(vec)

    def encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]


class OnnxEncoder:
    name = "onnx"

    def __init__(self, model_path: str, tokenizer_path: str, dims: int = 384, max_length: int = 256) -> None:
        import onnxruntime as ort  # type: ignore
        from tokenizers import Tokenizer  # type: ignore

        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding()
        self._input_names = {i.name for i in self._session.get_inputs()}
        self.dims = dims
        self.name = "onnx"

    def encode(self, text: str) -> list[float]:
        enc = self._tokenizer.encode(text or "")
        feed: dict[str, Any] = {}
        if "input_ids" in self._input_names:
            feed["input_ids"] = [enc.ids]
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = [enc.attention_mask]
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = [enc.type_ids]
        outputs = self._session.run(None, feed)
        hidden = outputs[0][0]
        mask = enc.attention_mask
        pooled = [0.0] * len(hidden[0])
        denom = max(1, sum(mask))
        for row, keep in zip(hidden, mask):
            if not keep:
                continue
            for i, val in enumerate(row):
                pooled[i] += float(val)
        return _l2([v / denom for v in pooled])

    def encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]


class LocalScorer:
    """Blend embedding cosine with lexical overlap.

    The blend matters for Chinese short text: embeddings find related wording,
    lexical overlap keeps concrete entities (names, dates, project codes) from
    being washed out.
    """

    def __init__(self, cfg: EmbeddingConfig | None = None) -> None:
        cfg = cfg or EmbeddingConfig()
        self.cfg = cfg
        self.encoder = self._build_encoder(cfg)
        self.name = getattr(self.encoder, "name", "hash")
        self._cache: dict[str, list[float]] = {}

    @staticmethod
    def _build_encoder(cfg: EmbeddingConfig) -> Any:
        backend = (cfg.backend or "auto").lower()
        if backend in ("hash", "hashing", "off", "none"):
            return HashingEncoder(cfg.dims)
        model = Path(cfg.model_path)
        tok = Path(cfg.tokenizer_path)
        if backend == "onnx" or (backend == "auto" and model.exists() and tok.exists()):
            if not (model.exists() and tok.exists()):
                if backend == "onnx":
                    log.warning("ONNX embedding requested but model/tokenizer missing; using hashing fallback")
                return HashingEncoder(cfg.dims)
            try:
                return OnnxEncoder(str(model), str(tok), dims=cfg.dims, max_length=cfg.max_length)
            except Exception as exc:  # pragma: no cover - depends on optional runtime
                log.warning("ONNX embedding unavailable (%s); using hashing fallback", exc)
                return HashingEncoder(cfg.dims)
        return HashingEncoder(cfg.dims)

    def embed(self, text: str) -> list[float]:
        key = text or ""
        cached = self._cache.get(key)
        if cached is None:
            if len(self._cache) > 4096:
                self._cache.clear()
            cached = self.encoder.encode(key)
            self._cache[key] = cached
        return cached

    def cosine(self, a: str, b: str) -> float:
        return cosine(self.embed(a), self.embed(b))

    def similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        # Use the better directional coverage: a long question should still
        # match a very short seed ("截稿日..." vs "周五 交稿").
        coverage = max(overlap_ratio(a, b), 0.6 * overlap_ratio(b, a))
        return max(0.0, min(1.0, 0.58 * self.cosine(a, b) + 0.42 * coverage))

    def seed_similarity(self, query: str, seed_tokens: str, thin_context: str = "") -> float:
        target = (seed_tokens + " " + thin_context).strip()
        return self.similarity(query, target)


def keyword_set(text: str, limit: int = 8) -> list[str]:
    seen: list[str] = []
    for tok in content_tokens(text):
        if len(tok) < 2:
            continue
        if tok not in seen:
            seen.append(tok)
        if len(seen) >= limit:
            break
    return seen
