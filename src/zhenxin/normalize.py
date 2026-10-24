"""Local text helpers: normalisation, signatures, lexical scoring, rule cues.

Nothing in this module performs network I/O or imports a model client.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\s\u3000-\u303f\u2000-\u206f!-/:-@\[-`{-~，。！？、；：“”‘’（）《》…—·~]+")
_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\u2b00-\u2bff\ufe0f\u200d]+"
)
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_.+\-]{0,30}|\d+(?:\.\d+)?")

STOP_CHARS = set("的了呢吧啊呀哦嘛哈嗯是我你他她它们这那有个一不太就都也还很和在与吗么什怎")
STOPWORDS = {
    "今天", "现在", "然后", "就是", "还是", "可以", "这个", "那个", "什么", "怎么", "因为",
    "所以", "但是", "不过", "觉得", "真的", "有点", "一下", "已经", "自己", "我们", "你们",
    "他们", "一个", "没有", "不是", "这样", "那样", "时候", "东西", "事情", "感觉",
}

# Backward-looking references only.  "这个/那个" alone is too common to justify
# a memory retrieval.
STRONG_DEICTIC_RE = re.compile(r"上次|上回|之前|以前|前面说|你说过|还记得|记得|后来|那次|当时|刚才说")
QUESTION_RE = re.compile(r"[?？]|吗|呢|么|是不是|有没有|要不要|能不能|什么|谁|哪|几点|多少|怎么样|如何")
NEGATION_RE = re.compile(r"不是|不对|没有|不喜欢|讨厌|别这样|不要|不用|记错|瞎说|胡扯|才不是")
CORRECTION_RE = re.compile(r"不是.{0,8}(?:是|而是)|记错|说错|不对|我说的是|其实是|纠正一下|你搞错")
COLD_RE = re.compile(r"^(哦|噢|行吧|好吧|随便|算了|没事|不用了|不聊了|先这样|就这样)[。.!！?？]*$")
ACK_RE = re.compile(r"^(嗯+|哦+|噢+|好+|行+|可以|收到|懂了|知道了|对|是的|没错|谢谢|辛苦了|哈哈+|笑死|好家伙|👌|ok|okay)[。.!！?？~～]*$")
TIME_RE = re.compile(
    r"(今天|今晚|明天|明早|明晚|后天|大后天|昨天|昨晚|前天|周末|这周|本周|下周|下个月|月底|月初|"
    r"周[一二三四五六日天]|星期[一二三四五六日天]|\d{1,2}[号日]|\d{1,2}[:：]\d{2}|"
    r"上午|下午|中午|晚上|早上|凌晨|半夜|一会儿|晚点|回头)"
)
EMOTION_WORDS = {
    "累": "疲惫", "困": "疲惫", "疲惫": "疲惫", "烦": "烦躁", "气": "生气", "焦虑": "焦虑",
    "难过": "低落", "想哭": "低落", "开心": "开心", "高兴": "开心", "爽": "开心",
    "委屈": "委屈", "害怕": "害怕", "孤单": "孤单", "想": "想念", "崩": "崩溃", "无语": "无语",
}
_EMOTION_KEYS = sorted((k for k in EMOTION_WORDS if k != "想"), key=len, reverse=True)
# "想" is only an emotion when it points at a person/place, not "想看电影".
EMOTION_RE = re.compile(
    "|".join([re.escape(k) for k in _EMOTION_KEYS] + [r"想(?=你|他|她|它|家|妈|爸)"])
)
RELATION_RE = re.compile(r"喜欢|讨厌|不喜欢|爱吃|不吃|习惯|总是|每次|最怕|最烦|离不开|上瘾")
FUTURE_ACTION_RE = re.compile(
    r"(?:明天|后天|下周|周末|月底|周五|周六|周日|今晚|明早|待会|回头|下次|改天|"
    r"要|得|准备|打算|计划|约了|答应|截止|交稿|面试|考试|出差|看医生|复诊|搬家|旅行|生日|开会)"
)
COMPLETION_RE = re.compile(r"(做完了|搞定了|弄完了|交了|交稿了|结束了|完成了|处理好了|看完了|去过了|已经.*了)")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return _WS_RE.sub(" ", text).strip()


def compact(text: str) -> str:
    """Normalised string used for exact signatures and regex anchors."""
    text = normalize(text).lower()
    text = _EMOJI_RE.sub("", text)
    text = _PUNCT_RE.sub("", text)
    return text


def canonical_short(text: str) -> str:
    text = compact(text)
    # Collapse runs of the same character ("好好好" / "哈哈哈" / "累累累").
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"(.)\1+", r"\1\1", text)
    # Trailing discourse particles are not part of the habit pattern.
    text = re.sub(r"(啊|呀|呢|吧|哦|喔|嘛|哈|啦|咯|嘞)+$", "", text)
    return text


def cjk_runs(text: str) -> list[str]:
    return [m.group(0) for m in re.finditer(r"[\u3400-\u4dbf\u4e00-\u9fff]+", text)]


def tokens(text: str) -> list[str]:
    text = normalize(text).lower()
    out: list[str] = []
    out.extend(_WORD_RE.findall(text))
    for run in cjk_runs(text):
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i : i + 2] for i in range(len(run) - 1))
    return out


def content_tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in tokens(text):
        if tok in STOPWORDS:
            continue
        if len(tok) == 2 and all(ch in STOP_CHARS for ch in tok):
            continue
        out.append(tok)
    return out


def overlap_ratio(query: str, target: str) -> float:
    q = set(content_tokens(query))
    t = set(content_tokens(target))
    if not q or not t:
        return 0.0
    return len(q & t) / len(q)


def is_ack(text: str) -> bool:
    c = compact(text)
    if ACK_RE.match(c):
        return True
    # Short confirmations often carry a trailing phrase ("嗯，先这样").
    return len(c) <= 8 and bool(re.match(r"^(嗯|哦|噢|好|行|对|是|收到|谢谢|哈哈|👌|ok)", c))


def is_cold(text: str) -> bool:
    return bool(COLD_RE.match(compact(text)))


def is_negation(text: str) -> bool:
    # "是不是/要不要" are questions, not disagreement.
    if re.search(r"是不是|要不要|能不能|有没有", text) and not CORRECTION_RE.search(text):
        return False
    return bool(NEGATION_RE.search(text))


def is_correction(text: str) -> bool:
    return bool(CORRECTION_RE.search(text))


def has_strong_deictic(text: str) -> bool:
    return bool(STRONG_DEICTIC_RE.search(text))


def is_question(text: str) -> bool:
    return bool(QUESTION_RE.search(normalize(text)))


@dataclass(frozen=True)
class Signature:
    key: str
    intent: str
    literal: str


_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("tired", re.compile(r"^(今天|今天也|真的|最近|又)?(好|很|太|超级|特别)?(累|困|疲惫|心累)(了|死|坏|死了)*$")),
    ("overtime", re.compile(r"^(又|今天又|还在)?加班(了|中|到很晚|到半夜)?$")),
    ("morning", re.compile(r"^(早|早安|早上好|早啊)$")),
    ("night", re.compile(r"^(晚安|睡了|先睡|去睡了|睡觉了)$")),
    ("angry", re.compile(r"^(烦|烦死|气死|无语|服了|崩溃|裂开)(了|死|啊|呀)*$")),
    ("sad", re.compile(r"^(难过|想哭|emo|低落|不开心|委屈)(了|啊|呀)*$")),
    ("happy", re.compile(r"^(开心|高兴|太好了|爽|笑死)(了|啊|呀)*$")),
    ("meal", re.compile(r"^(吃了吗|吃饭了吗|饿|好饿|饿了)$")),
    ("thanks", re.compile(r"^(谢谢|谢了|谢啦|辛苦了|麻烦你了)$")),
    ("ack", re.compile(r"^(嗯|哦|好|行|可以|收到|懂了|知道了|对|是的|没错|哈哈|👌|ok)$")),
]

_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tired": ("累", "困", "疲惫", "熬夜", "没睡"),
    "overtime": ("加班", "通宵", "赶稿", "截止", "交稿"),
    "morning": ("早", "早安"),
    "night": ("晚安", "睡了"),
    "angry": ("烦", "气", "无语", "服了", "崩溃"),
    "sad": ("难过", "想哭", "低落", "委屈", "不开心"),
    "happy": ("开心", "高兴", "太好了", "爽"),
    "meal": ("吃", "饿", "饭"),
    "thanks": ("谢谢", "辛苦"),
    "ack": ("嗯", "哦", "好", "行", "收到", "对"),
}

SHAPES: dict[str, dict[str, object]] = {
    "tired": {"min_chars": 12, "max_chars": 45, "ask_followup": True, "emotion": "warm", "use_emoji": False},
    "overtime": {"min_chars": 12, "max_chars": 50, "ask_followup": False, "emotion": "warm", "use_emoji": False},
    "morning": {"min_chars": 4, "max_chars": 20, "ask_followup": False, "emotion": "bright", "use_emoji": True},
    "night": {"min_chars": 4, "max_chars": 20, "ask_followup": False, "emotion": "soft", "use_emoji": True},
    "angry": {"min_chars": 10, "max_chars": 45, "ask_followup": False, "emotion": "steady", "use_emoji": False},
    "sad": {"min_chars": 10, "max_chars": 45, "ask_followup": True, "emotion": "soft", "use_emoji": False},
    "happy": {"min_chars": 6, "max_chars": 35, "ask_followup": False, "emotion": "bright", "use_emoji": True},
    "meal": {"min_chars": 6, "max_chars": 30, "ask_followup": True, "emotion": "bright", "use_emoji": False},
    "thanks": {"min_chars": 4, "max_chars": 24, "ask_followup": False, "emotion": "warm", "use_emoji": False},
    "ack": {"min_chars": 2, "max_chars": 16, "ask_followup": False, "emotion": "neutral", "use_emoji": False},
    "generic": {"min_chars": 6, "max_chars": 40, "ask_followup": False, "emotion": "neutral", "use_emoji": False},
}

TEMPLATES: dict[str, list[str]] = {
    "tired": [
        "先靠一会儿吧，别硬撑着。",
        "听起来今天被榨干了，先歇十分钟。",
        "那先别管别的，喝口水缓一缓。",
        "累了就先停一下，活是干不完的。",
    ],
    "overtime": [
        "又加班啊，今晚别熬太狠。",
        "这节奏有点顶，能摸鱼就摸十分钟。",
        "先把手头这段收个尾，剩下的明天再说。",
    ],
    "morning": ["早。", "早呀，今天慢慢来。", "早，先吃点东西。"],
    "night": ["晚安，早点睡。", "晚安，今天到这儿吧。", "睡吧，明天再说。"],
    "angry": [
        "确实挺让人无语的，先别急着回。",
        "这换谁都得烦，先缓一下。",
        "先别跟他们较劲，气坏的是你。",
    ],
    "sad": ["先不用撑着，想说就说。", "听着不太好受，我在。", "那先缓一缓，别逼自己。"],
    "happy": ["挺好，值得开心一下。", "不错啊，今天算没白过。", "那就好好享受一下。"],
    "meal": ["吃了吗？别又拖到很晚。", "先去吃点，别空着。"],
    "thanks": ["客气了。", "没事，顺手的事。"],
    "ack": ["嗯。", "好。", "收到。"],
    "generic": ["嗯，我在听。", "然后呢？", "这事儿可以慢慢说。"],
}


def infer_intent(text: str) -> str:
    lit = canonical_short(text)
    for name, pattern in _INTENT_PATTERNS:
        if pattern.match(lit):
            return name
    scores: dict[str, int] = {}
    for name, words in _INTENT_KEYWORDS.items():
        scores[name] = sum(1 for w in words if w in text)
    best = max(scores.items(), key=lambda kv: kv[1]) if scores else ("generic", 0)
    return best[0] if best[1] > 0 else "generic"


def pattern_signature(text: str, max_literal_chars: int = 24) -> Signature:
    lit = canonical_short(text)
    intent = infer_intent(text)
    for name, pattern in _INTENT_PATTERNS:
        if pattern.match(lit):
            return Signature(f"re:{name}", name, lit)
    if len(lit) <= max_literal_chars:
        return Signature(f"lit:{lit}", intent, lit)
    bag = sorted(set(content_tokens(text)))
    digest = hashlib.sha1("\x1f".join(bag).encode("utf-8")).hexdigest()[:16]
    return Signature(f"bag:{intent}:{digest}", intent, lit)


def response_shape(intent: str, signature: str) -> dict[str, object]:
    shape = dict(SHAPES.get(intent, SHAPES["generic"]))
    shape["signature"] = signature
    shape["intent"] = intent
    return shape
