#!/usr/bin/env python
"""Build a blind A/B human-evaluation sheet from collected reply pairs.

Input JSONL (one object per case):
    {"id":"HE01","category":"short_ack","user":"嗯",
     "baseline_reply":"...","plugin_reply":"..."}

Usage:
    python scripts/quality_sheet.py --pairs data/quality_pairs.jsonl
    # open the html, click numbers, then "导出" -> copy the JSON

Design notes:
* click-only scoring: every score is a visible 1-5 button;
* state is persisted in localStorage, so a refresh never loses work;
* export never depends on a download: it opens a textarea with the JSON and a
  copy button, because sandboxed preview iframes silently block downloads.
"""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

NUMBERS = "".join(
    f'<button class="num" type="button" data-v="{i}">{i}</button>' for i in range(1, 6)
)
STORE_KEY = "zhenxin_quality_v1"


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _case(idx: int, pair: dict, swap: bool) -> str:
    baseline = str(pair.get("baseline_reply", ""))
    plugin = str(pair.get("plugin_reply", ""))
    a, b = (baseline, plugin) if swap else (plugin, baseline)
    plugin_side = "B" if swap else "A"
    case_id = _e(pair.get("id", idx + 1))
    return f"""
<section class="case" data-id="{case_id}" data-plugin-side="{plugin_side}">
  <div class="head"><span class="cid">{case_id}</span><span class="cat">{_e(pair.get('category', ''))}</span></div>
  <p class="user">用户：{_e(pair.get('user', ''))}</p>
  <div class="replies">
    <div class="reply">
      <div class="who">A</div>
      <p>{_e(a)}</p>
      <div class="strip" data-side="A">{NUMBERS}</div>
    </div>
    <div class="reply">
      <div class="who">B</div>
      <p>{_e(b)}</p>
      <div class="strip" data-side="B">{NUMBERS}</div>
    </div>
  </div>
  <div class="pref">
    <span>更像真人在聊：</span>
    <button class="pbtn" type="button" data-v="A">A</button>
    <button class="pbtn" type="button" data-v="tie">打平</button>
    <button class="pbtn" type="button" data-v="B">B</button>
  </div>
</section>"""


def render_sheet(pairs: list[dict], seed: int = 20261023) -> str:
    rng = random.Random(seed)
    cases = [_case(idx, pair, rng.random() < 0.5) for idx, pair in enumerate(pairs)]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>真心 · 盲测评分表</title>
<style>
:root{{--ink:#111827;--mut:#6b7280;--line:#e5e7eb;--acc:#0f766e}}
*{{box-sizing:border-box}}
body{{margin:0;background:#f6f7fb;color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}}
.wrap{{max-width:1020px;margin:0 auto;padding:26px 16px 130px}}
h1{{font-size:25px;margin:0 0 4px}} .sub{{color:var(--mut);margin:0 0 16px}}
.case{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:12px 0}}
.case.done{{border-color:#99f6e4}}
.head{{display:flex;gap:8px;align-items:center;margin-bottom:2px}}
.cid{{font-size:12px;background:#f3f4f6;padding:2px 8px;border-radius:999px}} .cat{{font-size:12px;color:var(--mut)}}
.user{{font-weight:600;margin:6px 0 10px}}
.replies{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.reply{{border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:#fcfcfd;display:flex;flex-direction:column}}
.who{{font-size:12px;color:var(--mut);margin-bottom:4px}} .reply p{{margin:0 0 10px;flex:1}}
.strip{{display:flex;gap:6px}}
.num,.pbtn{{font:inherit;cursor:pointer;border:1px solid #d1d5db;background:#fff;color:var(--ink);
  border-radius:8px;min-width:38px;height:34px;padding:0 10px;transition:.12s}}
.num:hover,.pbtn:hover{{border-color:var(--acc);color:var(--acc)}}
.num.on,.pbtn.on{{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}}
.pref{{display:flex;gap:8px;align-items:center;margin-top:10px;font-size:13px;color:#374151}}
.bar{{position:fixed;left:0;right:0;bottom:0;background:#fffffff2;backdrop-filter:blur(6px);
  border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:10px;align-items:center;justify-content:center;flex-wrap:wrap}}
.bar .prog{{font-size:13px;color:var(--mut)}}
.rubric{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin:0 0 14px}}
.rubric .rt{{font-weight:600;margin-bottom:4px}}
.rubric ol{{margin:6px 0 6px 18px;padding:0;font-size:13.5px;line-height:1.7}}
.rubric .rule{{margin:6px 0 0;font-size:13px;color:#374151;background:#f9fafb;border-radius:8px;padding:6px 10px}}
.bar button{{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:10px 18px;font:inherit;cursor:pointer}}
.ghost{{background:#fff!important;color:var(--acc)!important;border:1px solid var(--acc)!important}}
.modal{{position:fixed;inset:0;background:#111827aa;display:none;align-items:center;justify-content:center;padding:20px;z-index:9}}
.modal.show{{display:flex}}
.panel{{background:#fff;border-radius:14px;max-width:820px;width:100%;padding:16px}}
.panel h2{{margin:0 0 8px;font-size:17px}}
.panel textarea{{width:100%;height:240px;font:12px/1.5 ui-monospace,Consolas,monospace;border:1px solid var(--line);border-radius:8px;padding:8px}}
.panel .row{{display:flex;gap:8px;margin-top:10px;justify-content:flex-end}}
@media(max-width:720px){{.replies{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<h1>真心 · 盲测评分表</h1>
<p class="sub">A/B 顺序已随机化，别猜来源。点一下就算打分，会自动记住（刷新不丢）。</p>
<section class="rubric">
  <div class="rt">评判标准（两边各打一个 1–5 总分）</div>
  <ol>
    <li><b>事实和记忆</b>：用对前文信息（项目/人/偏好/时间）；说错、张冠李戴 = 1–2；老实说不知道 = 4</li>
    <li><b>指代承接</b>：接住“那本 / 他们 / 上次说的”；答非所问、换话题 = 1–2</li>
    <li><b>不编造</b>：编出细节、把猜测当事实 = 1；只讲已知 = 4–5</li>
    <li><b>人味</b>：模板腔、客服腔、排比句、说教 = 1–3；像朋友 = 4–5</li>
    <li><b>长度节奏</b>：一句打发或三四段大道理 = 1–3；一两段刚好 = 4–5</li>
  </ol>
  <p class="rule">二选一：差距 ≤0.5 分算平；一边事实错、另一边对，直接判对的那边；<b>更长不等于更好</b>。</p>
</section>
{''.join(cases)}
</div>
<div class="bar"><span class="prog" id="prog">已评 0 / {len(pairs)}</span>
<button type="button" onclick="exportScores()">导出</button>
<button type="button" class="ghost" onclick="resetAll()">清空重来</button></div>

<div class="modal" id="modal"><div class="panel">
  <h2>评分 JSON（全选复制，发回给我）</h2>
  <textarea id="out" readonly></textarea>
  <div class="row">
    <button type="button" class="ghost" onclick="copyOut()">复制</button>
    <button type="button" onclick="document.getElementById('modal').classList.remove('show')">关闭</button>
  </div>
</div></div>

<script>
const STORE = "{STORE_KEY}";
const cases = [...document.querySelectorAll('.case')];
let state = {{}};
try {{ state = JSON.parse(localStorage.getItem(STORE) || "{{}}"); }} catch (e) {{ state = {{}}; }}

function save() {{ try {{ localStorage.setItem(STORE, JSON.stringify(state)); }} catch (e) {{}} }}
function cell(el, side) {{ return el.querySelector('.strip[data-side="' + side + '"]'); }}
function pref(el) {{ return el.querySelector('.pref'); }}

function paint(el) {{
  const row = state[el.dataset.id] || {{}};
  ['A', 'B'].forEach(side => {{
    const c = cell(el, side);
    const v = row['s' + side];
    c.dataset.value = v || '';
    c.querySelectorAll('.num').forEach(b => b.classList.toggle('on', String(b.dataset.v) === String(v)));
  }});
  const p = pref(el);
  p.dataset.value = row.p || '';
  p.querySelectorAll('.pbtn').forEach(b => b.classList.toggle('on', b.dataset.v === row.p));
  el.classList.toggle('done', Boolean(row.p) || Boolean(row.sA && row.sB));
}}

function updateProgress() {{
  const n = cases.filter(el => el.classList.contains('done')).length;
  document.getElementById('prog').textContent = '已评 ' + n + ' / ' + cases.length;
}}

function touch(el) {{
  state[el.dataset.id] = state[el.dataset.id] || {{}};
  return state[el.dataset.id];
}}

cases.forEach(el => {{
  el.querySelectorAll('.strip').forEach(strip => {{
    strip.querySelectorAll('.num').forEach(btn => {{
      btn.addEventListener('click', () => {{
        touch(el)['s' + strip.dataset.side] = Number(btn.dataset.v);
        paint(el); save(); updateProgress();
      }});
    }});
  }});
  pref(el).querySelectorAll('.pbtn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      touch(el).p = btn.dataset.v;
      paint(el); save(); updateProgress();
    }});
  }});
  paint(el);
}});
updateProgress();

function rows() {{
  return cases.map(el => {{
    const row = state[el.dataset.id] || {{}};
    return {{
      id: el.dataset.id,
      plugin_side: el.dataset.pluginSide,
      score_a: row.sA || null,
      score_b: row.sB || null,
      prefer: row.p || null
    }};
  }});
}}

function exportScores() {{
  const text = JSON.stringify(rows());
  document.getElementById('out').value = text;
  document.getElementById('modal').classList.add('show');
  // best effort: also offer a file for normal browser tabs
  try {{
    const blob = new Blob([JSON.stringify(rows(), null, 2)], {{type: 'application/json'}});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'quality_scores.json';
    document.body.appendChild(a); a.click(); a.remove();
  }} catch (e) {{}}
}}

function copyOut() {{
  const ta = document.getElementById('out');
  ta.select();
  let ok = false;
  try {{ ok = document.execCommand('copy'); }} catch (e) {{ ok = false; }}
  if (!ok && navigator.clipboard) {{
    navigator.clipboard.writeText(ta.value).catch(() => {{}});
  }}
  ta.focus(); ta.select();
}}

function resetAll() {{
  if (!confirm('清空所有打分？')) return;
  state = {{}}; save(); cases.forEach(paint); updateProgress();
}}
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", default="reports/quality_sheet.html")
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    pairs = []
    with Path(args.pairs).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                pairs.append(json.loads(line))
    if not pairs:
        raise SystemExit("no pairs")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_sheet(pairs, args.seed), encoding="utf-8")
    print(f"wrote {out} ({len(pairs)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
