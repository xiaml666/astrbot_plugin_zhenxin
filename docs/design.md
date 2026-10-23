# 真心 · 设计说明

本文解释每个模块与“默认不检索”原则的对应关系，以及为什么所有更新路径都可以做到零 LLM 调用。

## 1. 目标与非目标

**目标**

- 每一轮只为模型准备“足够好”的上下文，而不是“尽可能多”的上下文。
- 大部分轮次不检索：稳定前缀 + 最近窗口 + 少量动态提示。
- 对话越长，注入量接近常数；回复的拟人感不因轮次增加而衰减。
- 写路径、亮度、置信度、节奏全部规则化，零 LLM 调用。
- 稳定前缀逐字节稳定，尽量命中服务端提示缓存。

**非目标**

- 不做通用知识库问答。
- 不把十个线索槽位全部塞给模型。
- 不用 LLM 做记忆抽取、摘要、打分或冲突判断。
- 不替代宿主框架的会话存储；真心的 `turns` 表是最近窗口的本地副本。

## 2. 一轮的时序

```text
begin_turn
  1. round_no += 1
  2. 归一化 / 长消息压缩
  3. 首轮冷启动：最多 2 条种子（本地）
  4. 上一轮固化路径的反馈结算（+1 / +0.5 / -1）
  5. 节奏状态、话题栈、未闭合队列（本地）
  6. 线索亮度衰减
  7. L0 重排检查（只有换球/纠正才改字节）
  8. 短路器匹配（本地）
     ├─ 命中：生成形状直答，跳过 L1 与主模型
     └─ 未命中：L1 触发判断（本地）
  9. 组装 STABLE_PREFIX / RECENT_WINDOW / EPHEMERAL_TAIL
complete_turn
  10. 节奏后处理 / 词汇替换
  11. L2 正则蒸馏、亮度/置信度结算、物理淘汰
  12. 指标落库（含 L2 LLM 调用数，必须为 0）
```

顺序上有两个刻意的选择：

- **反馈先于本轮匹配**：上一轮固化回复的确认/否定，必须先结算，才能决定本轮是否继续命中同一条路径。
- **L0 重排先于短路器**：短路器不需要上下文，但指标里的前缀哈希要反映本轮真实的前缀；重排只影响下一轮请求末尾的线索块，不碰系统提示。

## 3. 为什么默认不检索

检索的成本不只是向量查询，还包括：

- 每次检索都要把结果拼进 prompt，增加输入 token；
- 检索结果每轮不同，破坏前缀缓存；
- 召回噪声会让模型“看起来记得很多，但接不住眼前的话”；
- 真实聊天里，大多数轮次只需要“当前情绪 + 最近窗口 + 一两条长期线索”。

真心的默认路径是：

```text
STABLE_PREFIX（系统 + 人格；逐字节稳定，不随线索变化）
+ RECENT_WINDOW（最近 8–12 轮，先追加、涨满再一次性回收）
+ EPHEMERAL_TAIL（最多 3 条 ≤30 字线索 + 本轮回忆/节奏/话题栈/未闭合提示，不落历史）
```

L0 的十条线索中，模型永远只看到 2–3 条。冷存储只保存 `seed_tokens` 和规则生成的 `thin_context`，不保存原始长文。

## 4. 短路器：习惯固化

### 4.1 数据结构

```text
(user_id, pattern_signature) -> {
  response_shape, confidence, last_hit, neg_streak, degraded_until
}
```

`pattern_signature`：

- 命中内置正则锚点（`tired`、`overtime`、`morning`、`night`、`angry`、`sad`、`happy`、`meal`、`thanks`、`ack`）→ `re:<intent>`
- 短输入（≤ `max_literal_chars`）→ `lit:<canonical>`
- 长输入 → `bag:<intent>:<sha1(排序后的 content tokens)>`

`response_shape` 只存形状：长度区间、是否追问、情绪倾向、是否表情。不存用户原话。

### 4.2 置信度更新（零 LLM）

| 用户下一轮 | 更新 |
|---|---|
| 继续同一话题 / 重复同一模式 | `+1`（重复模式在未命中时也会加） |
| 短确认（嗯/好/哈哈/收到） | `+0.5` |
| 纠正 / 反驳 / 冷淡 / 模板抱怨 | `-1`，`neg_streak += 1` |
| 正向反馈 | `neg_streak = max(0, neg_streak-1)` |
| `neg_streak >= 2` | `degraded_until = now + 24h`，立即回退完整推理 |

命中条件：

```text
confidence >= 3.0
and neg_streak == 0
and now >= degraded_until
and 输入长度 <= max_hit_chars（默认 8）
and 不是疑问/纠正
```

`max_hit_chars` 是防止把“周五交稿，周六想睡一天”这类计划陈述误判成习惯。默认只对“今天好累”“又加班”“早”“晚安”这类短模式命中。

### 4.3 固化生成

- 高置信度（或未配置小模型）：模板直答，零 LLM。
- 中等置信度且 `use_small_model=true`：调用 `decision_model`，带形状约束；**绝不调用主模型**。
- 模板按 `sha1(user|signature) + hits` 稳定轮换，避免每次同一句。

### 4.4 降级

降级不是删除路径：保留行，`degraded_until` 生效期间 `_can_hit` 恒为假；到期后 `neg_streak` 归零，允许重新学习。这样负反馈不会永久封死一个用户习惯。

## 5. L0：长期线索槽位与稳定前缀

### 5.1 槽位

- `hot_slots` 固定 10 行/用户，`slot` 1..10。
- 每个球：`content_seed`（≤30 字）、`brightness`、`last_touched`、`type`。
- 备用线索 `warm_slots` 上限 30；冷种 `cold_seeds` 只存 `seed_tokens` + `thin_context`。

### 5.2 亮度

```text
每轮：brightness *= 0.99（可配）
命中关键词：+0.5
新事实：1.6–2.2
纠正：覆盖 seed，brightness = 1.0
备用线索亮度超过最暗活跃线索 + 0.35：交换槽位
```

### 5.3 注入位置：请求末尾，而不是系统提示

线索块的内容是：

```text
【长期线索·L0】
- ...
- ...
- ...
```

它默认拼在本轮用户消息的末尾（`l0.tail_injection=true`），不写进 `system_prompt`。理由见 §8.2：AstrBot 会在插件 hook 之后把技能/工具说明追加到 `system_prompt` 末尾，线索块若留在系统提示里，每次换线索都会把这之后的 7.9k token 标记为未命中缓存。线索块本身只有 30–40 字，放末尾每轮多付的也就是这几十个 token。

槽位编号只用于内部排序；模型看到的是自然条目，不是数据库行号。

按 **slot 编号升序**，不按本轮相关度排序。相关度只在“重排检查点”作为很小的 tie-breaker（0.25 权重）参与选球，
不参与每轮字符串生成；字符串本身仍由 slot + seed 唯一决定。

换球规则：

```text
if 当前成员数 < k: 立即补足（允许一次早期变化）
elif 未到 min_dwell_rounds: 不改
elif 未到 rebalance_interval: 不改
else:
    挑战者分数 > 最弱在任者分数 + hysteresis 才替换
```

默认 `hysteresis=2.0`，远大于一轮 `+0.5` 的亮度变化，因此“亮度变化”几乎不会改线索块；只有新事实/纠正/备用线索交换这种材料性事件才可能改。0.1.1 之后换线索不再影响系统提示哈希，`prefix_stable` 的下降只会来自宿主系统提示本身的变化。

`current_block()` 每次都会用当前成员和 seed 重新渲染并比对；如果内容没变，不写状态、不改字节。

### 5.4 为什么不是每轮按消息相关度选球

那会让 L0 块每轮变化，直接破坏缓存前缀，也违背“每轮只注入极小线索”的稳定性目标。真心的折中：

- 长期 salience 决定“哪些球有资格被注入”；
- 消息相关度决定“这轮触摸/强化哪个球”；
- 换球有 dwell + hysteresis，避免抖动。

## 6. L1：按需唤醒

### 6.1 触发矩阵

| 条件 | 本地判断 |
|---|---|
| 指代 + 低 L0 相关 | `has_strong_deictic()` 且 `l0_best < low_l0_threshold` |
| 矛盾 | 否定词 + 与注入球有 ≥1 个双字词重叠 |
| 未闭合事务被触及 | 当前消息命中 open loop 关键词，且有强指代 |
| 冷候选 | 与冷种的 `seed_similarity >= candidate_min_score` |

只有“强指代”（上次/上回/之前/记得/后来/那次/当时/刚才说）才允许低 L0 唤醒；“这个/那个”单独出现不算，避免每轮都唤醒。普通问题（“有什么推荐吗”）不会唤醒。

### 6.2 冷却与去重

- 同一冷种子在 `wake_cooldown_rounds`（默认 8）内不重复唤醒。
- 如果冷种子已经被 L0 表示（相似度 ≥0.72），不唤醒——模型已经能从 L0 看到。
- 这两条让唤醒率随轮次自然下降：早期事实还没进入 L0 前三，晚期已经被吸收。

### 6.3 唤醒内容

```text
【想起】<thin_context 1>；<thin_context 2>
```

- 每条 `thin_context` 是规则拼接（类型 + 种子），不是原文；
- 总长 ≤100 字；
- 只进入 `EPHEMERAL_TAIL`，下一轮构建时整体剥离，不写入 `turns`。

可选 `l1.rerank=true`：只对已经取出的本地候选调用小模型重排，不做检索，不更新记忆。

## 7. L2：零调用写入

### 7.1 抽取规则

- 引号原话：`“...” / "..." / 「...」`
- 标识符：`CODE123`、版本号、英文项目名
- 中文偏好：`喜欢/不喜欢/爱喝/不吃/最怕/最烦 + 宾语`
- 时间 + 行动：`周五/明天/月底 + 交稿/面试/看医生/搬家/...`
- 情绪：`累/烦/难过/开心/...` → 情绪基线球
- 领域实体：海盐、团子、阿哲、冰美式、香菜、中药等
- 纠正：`不是 X，是 Y` → 用 Y 覆盖含 X（或其情绪映射）的球

### 7.2 物理淘汰

- 活跃线索被覆盖：旧 seed 先折叠进备用线索，再覆盖当前行。
- 备用线索溢出：选亮度最低/最旧的一条，写冷种，然后 `DELETE`。
- 冷种超过上限：删除最旧行。
- 未闭合队列溢出/超时：写冷种，然后 `DELETE`。
- 没有“标记忽略”的墓碑行；被淘汰的节点在 hot/warm 表中不存在。

### 7.3 为什么可以零 LLM

- 所有抽取都是正则/词典/字符重叠；
- 所有更新都是 `+0.5 / -1 / *0.99 / 交换 / DELETE`；
- `L2Writer` 的构造函数根本没有 LLM 客户端参数；
- `LLMCallLedger` 在 `record()` 入口拒绝 `l2_write/l2_distill/brightness/...`；
- `complete_turn` 统计本轮 `l2_llm_calls`，非 0 直接抛 `PolicyViolation`。

这是“结构上不可能”+“运行时审计”的双保险。

## 8. L3：缓存感知修剪

### 8.1 三段

```text
STABLE_PREFIX   system_prompt + persona             （逐字节不变，含 L0 时除外）
RECENT_WINDOW   最近 N 轮原文，先只追加，涨到高水位再一次性回收
EPHEMERAL_TAIL  L0 线索块 + L1 回忆 + 话题栈 + 节奏 + 词汇 + 未闭合 + 负面模式
```

### 8.2 规则

- `STABLE_PREFIX` 不参与任何修剪；`refresh_prefix()` 只在适配器首次拿到宿主 system prompt 时重绑一次。
- L0 线索块默认不进 `STABLE_PREFIX`（`l0.tail_injection=true`）：它是“偶尔会变”的内容，放进前缀等于让前缀缓存在它之后整段失效。实机 100 轮回查里，每次换线索多花约 7.9k 非缓存 token（AstrBot 在插件 hook 之后追加的技能/工具说明都排在线索块之后）。
- `RECENT_WINDOW` 先只追加：窗口起点 `window_first_id` 存在 `state` 里，只有涨到 `recent_turns + window_grow_turns` 才一次回收到 `recent_turns`；超字符预算时回收到 `recent_max_chars × recent_char_low_water`。这样 3/4 的轮次历史前缀逐字节不变。
- `EPHEMERAL_TAIL` 只拼进当前轮 `prompt`，不写入 `turns`；下一轮 `build()` 不会读到它。
- `cache_aware_pruning=false`：退化为普通滑动窗口，不按字符预算、保留上一轮 ephemeral，便于 A/B。

### 8.3 缓存命中估算

`metrics.prefix_stable` 直接比较本轮 `sha256(STABLE_PREFIX)` 与上一轮：

```text
prefix_stable = (hash_t == hash_{t-1})
```

这比“按 token 估算缓存命中”更保守：哈希一变就记为未命中。L0 移到末尾后，这个哈希只反映宿主系统提示与人格锚点，60 轮模拟中保持在 100%（冷启动第一轮除外）。窗口起点移动的轮次用 `window_moves` 单独观察。

## 9. 辅助模块

| 模块 | 作用 | 默认开关 |
|---|---|---|
| 话题栈 | 深度 3–5；切话题压栈不注入，回话题弹栈；栈顶一句话进 ephemeral | `aux.topic_stack` |
| 长消息压缩 | >150 字时保留情绪短句/指令原文/问句，压缩叙述；不调用模型 | `aux.compression` |
| 节奏控制器 | 情绪基线 × 关系温度 × 话题类型 → 字数/追问/表情软约束 + 轻量后处理 | `aux.rhythm` |
| 用户词汇表 | `X（Y）` / `不是X，是Y` 学习；生成后把标准词替换成用户词 | `aux.lexicon` |
| 未闭合事务 | 上限 5；时间/情绪指向弹出；超轮次落冷种并 `DELETE` | `aux.open_loops` |
| 负面模式 | 冷淡/终止/“模板/太长/别追问” → 回溯上轮回复特征计数，注入规避提示 | `aux.negative_patterns` |

每个模块都在 `begin_turn`/`complete_turn` 外侧有显式 `if` 开关，关闭后不会写库、不会注入。

## 10. 策略门禁

`src/zhenxin/llm.py`：

```python
FORBIDDEN_STAGES = {"l2_write", "l0_score", "brightness", "confidence", ...}
ALLOWED_STAGES = {"main_generate", "solidify_small", "l1_rerank"}
```

任何调用在发出网络请求前先 `ledger.record(stage)`；禁止 stage 直接 `PolicyViolation`。这保证“新增代码不小心把 LLM 带进写路径”会在单测/首轮就炸掉，而不是悄悄产生账单。

## 11. AstrBot 适配

| 阶段 | 接口 | 处理 |
|---|---|---|
| 消息阶段 | `filter.event_message_type(ALL, priority=100)` | `begin_turn()`；固化命中 → `yield plain_result` + `stop_event()` |
| LLM 前 | `filter.on_llm_request()` | 取/建 plan；`req.system_prompt = stable_prefix`；`req.contexts = recent_messages`；`req.prompt = compressed_text`；`extra_user_content_parts.append(TextPart(ephemeral).mark_as_temp())` |
| LLM 后 | `filter.on_llm_response()` | `complete_turn()`，L2 写入与指标 |

宿主 system prompt 在第一次 `on_llm_request` 时缓存到 adapter，后续 `begin_turn` 直接使用；`_base_system()` 会剥离上次追加的 `【人格锚点】/【长期线索·L0】` 标记，避免递归堆叠。

低版本 AstrBot 没有 `TextPart` 时，ephemeral 会退化为拼进 `req.prompt`，不影响主流程。

`on_llm_response` 会读取 `LLMResponse.usage` 的 `input_other / input_cached / output`，写入 `metrics` 的 `actual_*` 字段；这是账单口径。provider 不返回 usage 时这些字段为 0，`validate_metrics.py` 会明确提示“只能看估算”。

`l3.host_context_mode` 提供三种共存策略：`replace`（默认，接管最近窗口）、`prepend`（保留宿主 contexts 再追加）、`off`（完全不碰宿主 contexts，只注入稳定前缀/临时尾部）。它是冲突排查开关，不是“免费兼容”承诺。`replace` 下窗口本身按 §8.2 粗粒度回收，所以“接管”不再等于“每轮滑动、缓存全丢”。

## 12. 验收指标如何得到

`scripts/run_shadow_sim.py`：

1. 用同一份 60 轮数据跑 live 与 shadow；
2. 每轮写入 `metrics`（wake、prefix_hash、l0_chars、solid、LLM calls）；
3. `wake_rate_at_50 = 前 50 轮唤醒数 / 50`；
4. `prefix_stable_last20 = 末 20 轮 hash 不变比例`；
5. `l2_llm_calls = sum(metrics.l2_llm_calls)`；
6. `solid_hits` 来自 live；
7. `degrade_events` 来自固化器状态计数（即使 24h 后恢复，也能证明曾降级）。

## 13. 已知取舍

- **稳定 vs 相关**：L0 不按每轮消息重排，相关性略低于“每轮检索”，换来缓存命中和恒定注入量。L1 负责补足真正需要的例外。
- **少唤醒 vs 不漏**：强指代门槛会漏掉一部分“它”类指代；这是刻意选择。启用 ONNX 语义模型后可以降低漏召回，但仍应保留冷却。
- **模板 vs 变化**：默认模板直答的拟人感有上限；`use_small_model` 可改善，但会引入额外调用。建议用 20 条人工评测决定是否开启。
- **物理删除 vs 可追溯**：冷种只保留种子和规则摘要，旧原文不可恢复。这是隐私与成本取舍。

## 14. A/B 对比工具

`scripts/compare_baseline.py` 用同一份对话跑三条管线：

1. **装插件**：`STABLE_PREFIX + recent_messages + prompt`；如果本轮被短路器命中，输入 token 记为 0（没有发请求）。
2. **不装·全量**：`system_prompt + 全部历史 + 当前输入`。
3. **不装·滑窗**：`system_prompt + 最近 N 轮 + 当前输入`，默认 N=20。

三条线使用同一个字符估算器，默认生成器是本地固定回复，因此差异只来自上下文，不来自模型随机性。`--mode adaptive` 会让插件分支使用 L1 回忆，便于观察回复差异。报告输出 `comparison.html`（并排聊天 + 曲线 + 逐轮上下文）、`comparison.md`、`comparison.json`。

注意：这是结构成本，不是账单。真实费用还取决于服务商 tokenizer、缓存命中、输出 token 和工具调用。

## 参考

- AstrBot 插件开发指南：<https://docs.astrbot.app/dev/star/plugin-new.html>
- AstrBot 消息事件与 `on_llm_request` / `extra_user_content_parts`：<https://docs.astrbot.app/dev/star/guides/listen-message-event.html>
- AstrBot 插件配置 `_conf_schema.json`：<https://docs.astrbot.app/dev/star/guides/plugin-config.html>
- BAAI bge-small-zh-v1.5 模型卡：<https://huggingface.co/BAAI/bge-small-zh-v1.5>




