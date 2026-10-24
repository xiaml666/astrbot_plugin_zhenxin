# 0.1.1 缓存布局改动：上线与回滚

> 这份文档只讲 P0（缓存命中稳定性）的改动。质量盲测、分窗口预算、200k 长跑仍未做，
> 不要因为“缓存变好了”就宣称整体省钱比例。

## 改了什么

| 项 | 0.1.0 | 0.1.1 |
|---|---|---|
| L0 线索块位置 | `system_prompt` 末尾（宿主系统提示 + 人格 + L0） | 本轮用户消息末尾的临时区（`extra_user_content_parts`） |
| 最近窗口 | 每轮都按 `recent_turns` 滑动，最旧一轮每轮被丢 | 先只追加，涨到 `recent_turns + window_grow_turns` 才一次性回收到 `recent_turns` |
| 字符预算 | 超预算就从最旧端删到刚好不超 | 超预算一次回收到 `recent_max_chars × recent_char_low_water` |

新增配置（都有默认值，旧配置文件不用改）：

- `l0.tail_injection`（默认 `true`）
- `l3.window_grow_turns`（默认 `4`）
- `l3.recent_char_low_water`（默认 `0.7`）

## 为什么（实测依据）

4.29.1 上回查 100 轮 32k 会话的真实 `usage`：

- L0 未变的轮次：缓存 10,752 / 非缓存约 2,100（92 轮）
- L0 变化的那一轮：缓存 **2,816** / 非缓存约 10,000（8 轮）
- 其中 7 次造成真正的缓存断档，每次多付 7,936 非缓存 token，合计 55,552，
  占该会话非缓存输入（279,356）的 **19.9%**，按价约 ¥0.11 / ¥1.13

AstrBot 在插件 hook 之后还会往 `system_prompt` 追加技能/工具说明（实机约 7.9k token），
所以线索块放在系统提示里，一换线索就把后面这一整段也作废了。线索块本身只有 30–40 字。

`python scripts/cache_prefix_sim.py` 的布局模拟（60 轮仓库对话 + 7.9k 宿主后缀）：
非缓存输入 -52.9%、估算价格 -11.1%。**这是模拟，不是账单。**

## 上线步骤（需要重启 AstrBot）

1. 本地构建安装包：
   ```bash
   cd E:\ai相关\DSH\zhenxin
   python scripts/package_plugin.py
   # dist/astrbot_plugin_zhenxin-0.1.1.zip
   ```
2. 面板 → 插件 → 安装插件 → 从文件安装，选上面这个 zip（会覆盖旧版本目录，`plugin_data` 里的 SQLite 不受影响）。
   或直接替换服务器上的 `/opt/bots/astrbot/data/plugins/astrbot_plugin_zhenxin/` 内容。
3. 重启 AstrBot（容器 `astrbot`）。
4. 启动后确认插件版本是 0.1.1：面板插件卡片，或 `docker logs astrbot | grep zhenxin`。

不需要手动改 `astrbot_plugin_zhenxin_config.json`：缺失的新键会取默认值。
如果想保持 0.1.0 的行为做对照，在页面“专业模式 → 上下文预算”里把“长期线索放在请求末尾”关掉即可（即时生效）。

## 验证（跑 50 轮，看缓存台阶是否消失）

同一个 provider、同一组消息、新 WebChat 会话，只开 shadow 也能看到 `usage` 记录。

在容器里跑：

```bash
docker exec -it astrbot python3 - <<'PY'
import glob, sqlite3
p = glob.glob('/AstrBot/data/plugin_data/*/zhenxin*.sqlite3')[0]
c = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
for r in c.execute("""
  SELECT round_no, input_tokens_actual, cached_tokens_actual, prefix_stable, l0_chars
  FROM metrics WHERE user_key=? ORDER BY round_no DESC LIMIT 50
""", (input('user_key: '),)):
    tot = (r[1] or 0) + (r[2] or 0)
    print(r[0], f'hit={(r[2] or 0)/tot:.1%}' if tot else '-', 'stable' if r[3] else 'CHG', f'l0={r[4]}')
PY
```

判据：

- 换线索的轮次不再出现“缓存输入掉到 ~2.8k、非缓存跳到 ~10k”的台阶；
- `prefix_stable` 基本恒为 1（除非宿主系统提示自己变了）；
- 50 轮总 `actual_cached / (actual_cached + actual_input)` 高于改动前。

改动前的对照数字：50 轮 shadow 非缓存 104,593 / 缓存 739,968；50 轮 live 非缓存 143,437 / 缓存 497,280。

### 已实测（本地 4.29.1，2026-10-23，50 轮）

| | shadow 基线 | live 0.1.1 |
|---|---:|---:|
| 非缓存输入 | 104,491 | 103,274 |
| 缓存输入 | 573,056 | 331,776 |
| 总输入 | 677,547 | 435,050（-35.8%） |
| 估算价格 | ¥0.55896 | ¥0.39820（-28.8%） |
| 线索块变化 / 缓存回落 | 4 / 0 | 4 / 8（全部 256–384 token） |

- 换线索再也没有引起缓存台阶：live 的 8 次回落全部发生在 `recent_chars` 骤降的轮次（窗口回收）。
- 平均回复从 40.4 字降到 31.6 字（-21.8%），质量仍未盲测。
- `replace` 模式会把 AstrBot 自己会话记录里的历史裁到窗口大小（实测 152 条 → 23 条）；
  与宿主自带记忆共存时改 `off`。插件自己的 `turns` 表不受影响。

## 回滚

全部是运行期配置，页面改完下一轮生效，不用重启：

| 想回到 | 设置 |
|---|---|
| 0.1.0 的线索位置 | `l0.tail_injection = false` |
| 0.1.0 的窗口滑动 | `l3.window_grow_turns = 0` |
| 完全旧行为 | 上面两条都改 |

代码回滚：重装 `dist/astrbot_plugin_zhenxin-0.1.0.zip` 并重启。

## 副作用与已知风险

- 线索块从“系统提示”变成“本轮用户消息末尾”，模型对它的服从度可能略降。这是**待验证**项：
  同一组 50 轮做隐藏来源 A/B，重点看“接不接得住指代”“会不会把线索当成用户刚说的话”。
- 窗口会在 8–12 轮之间波动，峰值比原来多 4 轮；200k 窗口下可忽略，32k 下也要重新测撞窗轮次。
- 若宿主每轮往 `system_prompt` 追加时间戳/状态栏，前缀仍会断在那里；这部分不在插件控制范围内。
