"""Rule-based detection of zero-value memory.

The memory bank should not wait 30 days to forget something the user explicitly
retracted.  These patterns catch four common cases without an LLM call:

* "不用再提醒我了 / 别老提这个"        -> stop resurfacing a fact
* "稿子已经交了 / 早就搞定了"          -> completed open loop
* "算了，不做了 / 取消吧"              -> cancelled plan
* "别叫我阿岚 / 不要这样称呼"          -> rejected address

Detection is intentionally conservative: a cue alone is not enough, at least
one content token must also match the candidate fact.  Otherwise "不用了"
would wipe unrelated memories.
"""

from __future__ import annotations

import re

from ..normalize import content_tokens

CANCEL_RE = re.compile(
    r"(?:不|别|不要|不用|无需|不想)(?:再)?(?:提醒|提|记|管|说|叫|弄|做|搞)"
    r"|(?:取消|算了|放弃了|不做了|不弄了|不想做了|不想弄了|不用了|别提了)"
)
DONE_RE = re.compile(
    r"(?:已经|早就|都|刚)(?:做完|完成|搞定|弄完|交了|结束|处理完|办完)"
    r"|(?:做完|完成|搞定|交了|结束)了"
)
ADDRESS_CANCEL_RE = re.compile(
    r"(?:别|不要|不用|不许)(?:再)?(?:这样|这么)?(?:叫我|喊我|称呼我|称呼)"
    r"|(?:不叫|别叫)(?:我)?"
)
ADDRESS_RE = re.compile(
    r"(?:叫我|喊我|称呼我(?:为|作)?|我的名字(?:是|叫)|我叫)"
    r"([\u4e00-\u9fffA-Za-z0-9_]{1,12})"
)
# The user may also give *the assistant* a name ("你以后叫小满").  That is a
# different identity fact and must not be rendered as "call the user X".
ASSISTANT_NAME_RE = re.compile(
    r"(?:你(?:以后|就)(?:叫|叫做)(?!我)|你的名字(?:是|叫)|以后叫你)"
    r"([\u4e00-\u9fffA-Za-z0-9_]{1,12})"
)
_EXPLICIT_MEMORY_RE = re.compile(r"(?:记住|记一下|帮我记|别忘了|提醒我)[，,:：\s]*(.{2,40})")
# Words that carry no identity by themselves; never use them as the only
# overlap when deciding what to forget.
_STOP = {
    "这个",
    "那个",
    "这样",
    "那样",
    "事情",
    "东西",
    "什么",
    "怎么",
    "我们",
    "你们",
    "他们",
    "就是",
    "真的",
    "已经",
    "一下",
    "现在",
    "以后",
    "今天",
    "明天",
}


def cue(text: str) -> str:
    """Return ``cancelled|done|address_cancel|`` for a message."""
    raw = (text or "").strip()
    if not raw:
        return ""
    if ADDRESS_CANCEL_RE.search(raw):
        return "address_cancel"
    if DONE_RE.search(raw):
        return "done"
    if CANCEL_RE.search(raw):
        return "cancelled"
    return ""


def address_in(text: str) -> str:
    match = ADDRESS_RE.search(text or "")
    if not match:
        return ""
    return _clean_name(match.group(1))


def _clean_name(raw: str) -> str:
    name = re.sub(r"[了啦吧呀呢啊哦嘛哈]+$", "", (raw or "").strip()).strip()
    if len(name) < 1 or name in _STOP:
        return ""
    if any(word in name for word in ("什么", "啥", "哪", "怎么", "多少", "吗", "呢")):
        return ""
    return name


def assistant_name_in(text: str) -> str:
    match = ASSISTANT_NAME_RE.search(text or "")
    if not match:
        return ""
    name = _clean_name(match.group(1))
    # "你叫我老大" is the user's address, not the assistant's name.
    if name.startswith("我"):
        return ""
    return name


def identity_facts(text: str) -> list[tuple[str, str]]:
    """Return durable identity facts in ``(fact_text, kind)`` form."""
    out: list[tuple[str, str]] = []
    name = address_in(text)
    if name:
        out.append((f"称呼我“{name}”", "address"))
    ai_name = assistant_name_in(text)
    if ai_name and not any(ai_name == item[0] for item in out):
        out.append((f"你叫“{ai_name}”", "assistant_name"))
    return out


def explicit_memory(text: str) -> str:
    """The clause after "记住…", if the user explicitly asked to remember."""
    match = _EXPLICIT_MEMORY_RE.search(text or "")
    if not match:
        return ""
    clause = match.group(1).strip().strip("。！？!?，,；; ")
    if len(clause) < 2:
        return ""
    # "记住我叫阿岚" is an identity, already handled above; keep it once.
    return clause[:40]


def meaningful_tokens(text: str) -> set[str]:
    return {t for t in content_tokens(text or "") if t and t not in _STOP and len(t) >= 1}


def overlap_ratio(a: str, b: str) -> float:
    ta, tb = meaningful_tokens(a), meaningful_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def matches(fact_text: str, fact_kind: str, message: str, reason: str) -> bool:
    """Whether ``message`` retracts ``fact_text``."""
    if not reason:
        return False
    if reason == "address_cancel":
        if fact_kind != "address" and "称呼" not in fact_text and "叫" not in fact_text:
            return False
        name = address_in(message)
        if name:
            return name in fact_text
        # "不要这样称呼我了" -> any address fact is a candidate.  A vague
        # "不要叫外卖" must not wipe an address, so require the word itself.
        return fact_kind == "address" and any(
            word in message for word in ("称呼", "名字", "叫我", "喊我")
        )
    ratio = overlap_ratio(fact_text, message)
    if reason == "done":
        if fact_kind not in ("unclosed", "fact", "habit"):
            return False
        # "周五交稿" vs "稿子已经交了" shares only one bigram; an unfinished
        # action is the strongest signal, so it gets the lower bar.
        return ratio >= 0.25 or (fact_kind == "unclosed" and ratio >= 0.18)
    # cancelled
    if ratio >= 0.34:
        return True
    # Address-like memory can be retracted by a short explicit name mention.
    name = address_in(message)
    return bool(name and fact_kind == "address" and name in fact_text)
