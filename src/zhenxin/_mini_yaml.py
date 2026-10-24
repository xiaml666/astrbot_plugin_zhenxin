"""A deliberately small YAML subset loader.

The plugin only needs configuration-shaped YAML: nested mappings, lists of
scalars/mappings, inline lists, quoted strings and block scalars.  Keeping this
fallback in-tree means the plugin still starts on a bare Python install when
PyYAML is absent.  If PyYAML is installed, ``config.py`` prefers it.
"""

from __future__ import annotations

import json
import re
from typing import Any

_BOOL = {"true": True, "false": False, "yes": True, "no": False, "on": True, "off": False}


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and quote:
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text[1:-1]
    if text.startswith("'") and text.endswith("'") and len(text) >= 2:
        return text[1:-1].replace("''", "'")
    low = text.lower()
    if low in _BOOL:
        return _BOOL[low]
    if low in ("null", "none", "~"):
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        # Split on commas that are not inside quotes.
        parts: list[str] = []
        buf = ""
        quote: str | None = None
        for ch in inner:
            if quote:
                buf += ch
                if ch == quote:
                    quote = None
            elif ch in ("'", '"'):
                quote = ch
                buf += ch
            elif ch == ",":
                parts.append(buf)
                buf = ""
            else:
                buf += ch
        parts.append(buf)
        return [_parse_scalar(p) for p in parts if p.strip()]
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?", text):
        try:
            return float(text)
        except ValueError:
            pass
    return text


class MiniYamlError(ValueError):
    pass


def loads(text: str) -> Any:
    lines: list[tuple[int, str]] = []
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(raw_lines):
        raw = raw_lines[i]
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise MiniYamlError(f"tabs are not allowed for indentation (line {i + 1})")
        stripped = _strip_comment(raw).rstrip()
        if not stripped.strip():
            i += 1
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        content = stripped[indent:]
        # Block scalar: key: |
        block = re.match(r"^([^:#][^:]*):\s*[|>]\s*$", content)
        if block:
            key = block.group(1).strip()
            block_lines: list[str] = []
            i += 1
            base_indent = None
            while i < len(raw_lines):
                nxt = raw_lines[i]
                if nxt.strip() == "":
                    block_lines.append("")
                    i += 1
                    continue
                nxt_indent = len(nxt) - len(nxt.lstrip(" "))
                if nxt_indent <= indent:
                    break
                if base_indent is None:
                    base_indent = nxt_indent
                block_lines.append(nxt[min(base_indent, len(nxt)):])
                i += 1
            lines.append((indent, key + ": " + json.dumps("\n".join(block_lines).strip("\n"), ensure_ascii=False)))
            continue
        lines.append((indent, content))
        i += 1

    if not lines:
        return {}

    def parse_block(pos: int, indent: int) -> tuple[Any, int]:
        if pos >= len(lines):
            return {}, pos
        is_list = lines[pos][1].startswith("- ")
        if is_list:
            items: list[Any] = []
            while pos < len(lines):
                cur_indent, content = lines[pos]
                if cur_indent < indent:
                    break
                if cur_indent > indent:
                    raise MiniYamlError(f"unexpected indentation near: {content}")
                if not content.startswith("- "):
                    break
                item_text = content[2:].strip()
                pos += 1
                if not item_text:
                    value, pos = parse_block(pos, indent + 2)
                    items.append(value)
                    continue
                # Mapping item: "- key: value" plus following indented keys.
                kv = re.match(r"^([^:#][^:]*):(.*)$", item_text)
                if kv:
                    item: dict[str, Any] = {}
                    key = kv.group(1).strip()
                    rest = kv.group(2).strip()
                    if rest:
                        item[key] = _parse_scalar(rest)
                    else:
                        value, pos = parse_block(pos, indent + 2)
                        item[key] = value
                    while pos < len(lines):
                        sub_indent, sub_content = lines[pos]
                        if sub_indent <= indent:
                            break
                        if sub_content.startswith("- "):
                            break
                        sub_kv = re.match(r"^([^:#][^:]*):(.*)$", sub_content)
                        if not sub_kv:
                            raise MiniYamlError(f"invalid mapping line: {sub_content}")
                        sub_key = sub_kv.group(1).strip()
                        sub_rest = sub_kv.group(2).strip()
                        pos += 1
                        if sub_rest:
                            item[sub_key] = _parse_scalar(sub_rest)
                        else:
                            value, pos = parse_block(pos, sub_indent + 2)
                            item[sub_key] = value
                    items.append(item)
                else:
                    items.append(_parse_scalar(item_text))
            return items, pos

        mapping: dict[str, Any] = {}
        while pos < len(lines):
            cur_indent, content = lines[pos]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise MiniYamlError(f"unexpected indentation near: {content}")
            if content.startswith("- "):
                break
            kv = re.match(r"^([^:#][^:]*):(.*)$", content)
            if not kv:
                raise MiniYamlError(f"invalid mapping line: {content}")
            key = kv.group(1).strip()
            rest = kv.group(2).strip()
            pos += 1
            if rest:
                mapping[key] = _parse_scalar(rest)
            else:
                # Empty value followed by a less-indented line -> null.
                if pos < len(lines) and lines[pos][0] > cur_indent:
                    value, pos = parse_block(pos, lines[pos][0])
                    mapping[key] = value
                else:
                    mapping[key] = {}
        return mapping, pos

    value, pos = parse_block(0, lines[0][0])
    if pos != len(lines):
        raise MiniYamlError(f"could not parse line: {lines[pos][1]}")
    return value
