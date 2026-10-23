# 真心（Zhenxin）

<p align="center"><img src="logo.png" width="112" alt="真心 logo"></p>

> 不改变底层对话模型，只接管“每一轮为模型准备什么上下文”。

真心是一个面向 QQ / 微信聊天机器人的上下文层插件。它默认**不检索**：大部分轮次只注入 2–3 条极短线索；只有出现指代、矛盾或未闭合事务时，才按需唤醒冷记忆。记忆写入、亮度更新、置信度更新、节奏计算全部走规则/计数，**零 LLM 调用**。上下文修剪只从尾部剥离临时片段，稳定前缀逐字节保持，以命中服务端提示缓存。

## 为什么是真心（和市面插件的不同）

市面上的记忆插件大多在解决“怎么检索得更准”：每轮把用户消息做 embedding、召回 top-k、拼进系统提示。真心解决的是另一个问题：**每轮到底该让模型看到多少东西，才能又记得住、又不把上下文和缓存撑爆。**

| 维度 | 常见记忆 / 上下文插件 | 真心 |
|---|---|---|
| 召回方式 | 每轮向量检索 top-k，或把记忆全量注入 | **默认不检索**；只带 2–3 条极短线索，出现指代 / 矛盾 / 未闭事务才唤醒 |
| 模型调用 | 抽取、摘要、重排可能多次调用模型 | 写路径 **零 LLM 调用**，一轮对话仍然只付一次 chat 的钱 |
| 提示缓存 | 每轮改系统提示，prefix cache 容易整段失效 | 线索放请求**末尾**，稳定前缀逐字节不变，缓存持续命中 |
| 成本曲线 | 对话越长，注入越多越贵 | 100 轮实测 ¥1.32 → ¥0.82（**-38%**），总输入 -50% |
| 记忆治理 | 只增不减，越攒越多 | 按**重要性 + 时间线**治理，零价值条目自动淡出，只留关键词痕迹 |
| 数据依赖 | 向量库 / 外部 embedding API | SQLite 单文件；可选本地 ONNX，没有就退化为哈希向量，**不联网** |
| 可控性 | 改配置文件、看日志 | 自带控制中心：概览 / 记忆 / 花费 / 行为设置 / 高级 / 重置，可先 shadow 再接管 |

**独有亮点：**

- **缓存感知**：不是简单“截断历史”，而是让每轮请求的稳定前缀保持字节一致，专门服务 prompt cache。
- **零调用写入**：L2 用正则/规则蒸馏线索，`strict_zero_llm` 会直接拦截写路径里的模型调用。
- **问句不当事实**：“上次说的阿哲是不是搞摄影的”不会被记成“阿哲是摄影师”，减少模型自问自答。
- **真实账单**：读取模型返回的真实 usage（非缓存/缓存/输出），页面直接显示每轮花了多少钱。
- **可回退**：推荐设置、恢复默认、分范围重置（对话/记忆/统计），换模型或新开对话后不用手动清库。
- **两种显式工具**：`zhenxin_recall` / `zhenxin_remember`，需要时模型可主动回忆或记住；平时不占调用。

- **主适配目标：AstrBot 4.x**
- 兼容层：NoneBot2（薄适配，宿主自行接入 LLM）
- 运行环境：Python 3.11+
- 持久化：SQLite 单文件
- 本地嵌入：`onnxruntime` + bge-small-zh（可选）；没有 ONNX 时自动退化到纯 Python 哈希向量

> **状态：0.3.0。** 74 个 unittest、真实 AstrBot 4.29.1 实机验证、100 轮成本 A/B。先开 shadow 跑真实数据，再决定是否 live。省 token 是条件性收益，不是对所有配置的保证；验证方法见 [`docs/validation.md`](docs/validation.md)。

## 0. 验证状态（请先读）

- **已做**：74 个 unittest；60 轮 live/shadow 模拟；装/不装上下文对比；本地 ONNX/哈希回退；
  用假 AstrBot event/request 验证 hook 注入、短路阻断、shadow 不改请求、`LLMResponse.usage` 落库。
- **未做**：在真实 AstrBot 实例 + 真实 QQ/微信适配器 + 其他插件组合下的长期回归。当前环境没有 AstrBot 依赖，
  所以仓库不能声称“已对所有 AstrBot 版本验证通过”。
- **发布前必须做**：在目标 AstrBot 版本运行 `scripts/astrbot_preflight.py`，再开 shadow 跑至少 50 轮，
  用 `scripts/validate_metrics.py` 看真实 usage；最后按 [`docs/validation.md`](docs/validation.md) 做冲突和盲测。实机 68 轮结果见 [`docs/astrbot-live-report.md`](docs/astrbot-live-report.md)。
- **收益口径**：`98.5%` 只是“128k + 全量历史 + 无缓存折扣”的模型估算；8k 下约 60%，有缓存或宿主已截断历史时会更低。
  对外描述请带上前提，不要写“所有用户省 98%”。
- **实机 50 轮干净 A/B（4.29.1）**：shadow 总输入 844,561 / ¥0.62636；live 总输入 640,717 / ¥0.57529。
  输入下降 **24.1%**，价格下降 **8.2%**；非缓存输入 +37%、缓存输入 -33%，所以 token 降幅没有等比例变成价格降幅。
- **实机 32k 撞窗测试**：基线第 **91 轮后**撞到 32k 窗口；截至撞窗前 shadow ¥1.24643、live ¥1.02339，
  总输入 **-34.4%**、价格 **-17.9%**。这同样不支持“98%”的普遍说法。详情见
  [`docs/astrbot-live-report.md`](docs/astrbot-live-report.md)。
- **更早的混合插件对照**：59 轮 shadow 总输入 1,288,474 / ¥1.10；9 轮 live 反而贵 24.8%，因为其他插件每轮注入约 18k token。
  说明“装上就省”不成立，必须先做干净对照。

---

## 1. 已验收的指标（60 轮模拟，影子模式）

数据：`data/sim_dialogue.jsonl`；命令：`python scripts/run_shadow_sim.py`。

| 指标 | 结果 | 目标 |
|---|---:|---|
| 轮次 | 60 | ≥ 50 |
| L1 累计唤醒率（第 50 轮） | **10.0%** | ≤ 20% |
| 末 10 轮唤醒率 | **10.0%** | 下降 |
| STABLE_PREFIX 稳定率（末 20 轮） | **100%** | ≥ 95% |
| STABLE_PREFIX 稳定率（整体，含冷启动） | **95%** | 参考 |
| L0 最大字符数 | **39** | ≤ 150 |
| L2 LLM 调用 | **0** | = 0 |
| 固化命中 | **4** | > 0 |
| 负反馈降级 | **1 次** | 能正确降级 |
| 每轮注入 token 均值（含最近窗口） | **~340** | 近常数 |

完整报告：[`reports/shadow_report.md`](reports/shadow_report.md)。唤醒率曲线：10 轮 20% → 20 轮 10% → 50 轮 10% → 60 轮 10%。

> 说明：上述是仓库内固定数据集的复现结果，不代表所有真实对话的分布。接入真实机器人后请先开 `shadow_mode` 观察自己的曲线。

### 1.1 直观对比：装插件 vs 不装插件

```bash
python scripts/compare_baseline.py
# 打开 reports/comparison.html
```

它用同一段 60 轮对话跑三条上下文管线：

- **装插件**：STABLE_PREFIX + 最近 8 轮 + 临时尾部；固化轮次完全不调用模型，输入记为 0；
- **不装 · 全量历史**：system + 从第 1 轮开始的所有消息；
- **不装 · 滑动窗口**：system + 最近 20 轮（`--baseline-turns` 可调）。

HTML 会并排展示每轮“模型实际看到的上下文”、插件/基线回复、逐轮 token 和累计曲线。默认 `--mode fixed` 让两条分支使用同一套固定回复，从而只比较上下文；`--mode adaptive` 会让插件分支使用 L1 回忆，便于看聊天差异。

当前数据集结果（字符估算）：

| 指标 | 装插件 | 全量 | 滑窗 20 轮 |
|---|---:|---:|---:|
| 总输入 token | **20,101** | 45,071 | 26,497 |
| 平均每轮 | **335** | 751 | 442 |
| 峰值单轮 | **402** | 1,405 | 533 |
| 末轮 | **353** | 1,405 | 506 |

相对全量历史节省 **55.4%**，相对 20 轮滑窗节省 **24.1%**。滑窗方案可能更接近“只保留最近对话”的低成本，但它没有 L0 长期线索、不会在固化轮次免掉请求；强度对比见 [`reports/comparison.md`](reports/comparison.md) 和 HTML 曲线。

> **这不是“所有用户省 98%”的承诺。** 它依赖“不装方案保留全量历史、128k 窗口、无缓存折扣”等前提；8k 窗口下估算只有 60% 左右，开启缓存或 AstrBot 本来就会截断历史时，比例会更低。真实数字必须用自己的 `metrics` 实测。

> token 是字符估算，不是账单。它的价值在于三条曲线的相对形状；真实账单请用模型服务商返回的 usage 校准。AstrBot 的 `LLMResponse.usage` 已接入：`validate_metrics.py` 会显示 `actual_*` 字段。

### 1.2 对话寿命与价格：50 轮/天

```bash
python scripts/lifetime_cost.py
# 可改：--context 32768 131072 --turn-tokens 80 --input-price 2
```

假设：一轮 = 一次用户输入 + 一次助手回复；每轮新增 60 token；插件每轮上下文预算 700 token；
输出预留 4k、每轮输出 80 token；输入 ¥2/百万、输出 ¥8/百万；固化免调用比例 5%；不含缓存折扣。

| 上下文窗口 | 不装寿命 | 装插件寿命 | 不装总价 | 装插件总价 | 价格降低 |
|---:|---:|---|---:|---:|---:|
| 8k | 1.4 天 | 不因历史增长耗尽 | ¥0.34 | ¥0.14 | 60.1% |
| 32k | 9.6 天 | 不因历史增长耗尽 | ¥14.09 | ¥0.94 | 93.3% |
| 128k | 42.3 天 | 不因历史增长耗尽 | ¥270.59 | ¥4.17 | 98.5% |

“不因历史增长耗尽”指插件输入有上界（稳定前缀 + 最近窗口 + 临时尾部）；单条超长消息、宿主额外注入、工具结果仍可能撞窗口。价格是在“不装方案耗尽”的同一轮数上比较；之后不装方案无法继续，插件仍可继续。完整表：[`reports/lifetime_cost.md`](reports/lifetime_cost.md)。

### 1.3 上线前怎么验证，而不是只信介绍

完整流程见 [`docs/validation.md`](docs/validation.md)。最小步骤：

```bash
# 1) 在 AstrBot 的 Python 环境里检查真实 API 面
python scripts/astrbot_preflight.py

# 2) 开 shadow_mode，真实聊天至少 50 轮，然后看真实 usage
python scripts/validate_metrics.py --db data/zhenxin.sqlite3

# 3) 确认冲突/人味/成本
#    - 只装真心 → 再加一个真实依赖插件，各 20 轮
#    - 20 条 human_eval_cases 盲测，平均分差 ≤1
#      python scripts/quality_sheet.py --pairs data/quality_pairs.example.jsonl
#    - 用 actual_* 字段重新算 lifetime_cost
```

需要明确的边界：

- **真实省钱**：只有 `LLMResponse.usage` 返回数据时，`actual_cost` 才是账单口径；否则只能看估算，不应对外承诺具体金额。
- **真实寿命**：插件不因历史增长耗尽，但单条超长消息、宿主额外注入、工具结果仍可能撞窗口。
- **真实兼容**：默认会接管 `req.contexts`；与其他记忆/知识库插件共存时，先把 `l3.host_context_mode` 设为 `off` 或 `prepend` 做对照。
- **真实人味**：模板固化率过高会机械；先看 `solid_hit_rate`，再用 20 条盲测决定是否关掉 `shortcircuit`。

### 1.4 缓存布局：线索为什么放在请求末尾

2026-10 在 4.29.1 实机上回查 100 轮 32k 会话的真实 `LLMResponse.usage`：

| 轮次类型 | 缓存输入 | 非缓存输入 | 轮次数 |
|---|---:|---:|---:|
| L0 线索块未变 | 10,752 | 约 2,100 | 92 |
| L0 线索块变化 | 2,816 | 约 10,000 | 8 |

缓存断档 7 次（另有 1 轮是连续第二次变化，断档已在上轮付过），每次多付 **7,936** 非缓存 token，
合计 **55,552**，占该会话非缓存输入（279,356）的 **19.9%**，按价约 ¥0.11 / ¥1.13。
原因不是 L0 本身大（30–40 字），而是 AstrBot 在插件 hook 之后还会把技能/工具说明（约 7.9k token）
追加到 `system_prompt` 末尾：前缀缓存在 L0 处一断，后面这 7.9k 就全部按原价重算。

0.1.1 起：

- `l0.tail_injection=true`：线索块改走请求末尾的临时区，系统提示只剩“宿主系统提示 + 人格锚点”；
- `l3.window_grow_turns=4`：最近窗口先只追加，涨到 `recent_turns + 4` 才一次回收到下限，避免每轮滑动把整段历史前缀作废。

`python scripts/cache_prefix_sim.py` 用仓库 60 轮对话，按“前缀缓存 + 64 token 分块”复算四种布局：

| 布局（宿主系统后缀按 7.9k token 模拟） | 非缓存输入 | 估算价格 |
|---|---:|---:|
| L0 在系统提示 + 每轮滑动（0.1.0 默认） | 45,719 | ¥0.3157 |
| L0 在系统提示 + 粗粒度回收 | 42,537 | ¥0.3116 |
| L0 在末尾 + 每轮滑动 | 26,583 | ¥0.2876 |
| L0 在末尾 + 粗粒度回收（0.1.1 默认） | **21,544** | **¥0.2808** |

非缓存输入 **-52.9%**，估算价格 **-11.1%**。这是布局模拟，不是账单；真实收益要在 4.29.1 上用 `actual_*` 实测。

**本地 50 轮实测（0.1.2，2026-10-23，公平基线）**：同一份 `validation_50.jsonl`，两个 WebChat 会话，
shadow 总输入 614,240 / ¥0.5056；live 435,233 / ¥0.3955。总输入 **-29.1%**、价格 **-21.8%**；
线索块变化 4 次，**缓存回落 0 次**。
30 对人工盲测：真心 **18:5**（平 7），均分 **4.63 vs 3.73**——这一版在更短的前提下更受偏好。
早先 0.1.1 报的 -35.8% / -28.8% 与 21:10 是“基线回复被节奏后处理截断”造成的虚高，已作废；
实验过程见 [`docs/quality-experiments.md`](docs/quality-experiments.md)。

宿主本来就没有大段系统提示时（纯库 / NoneBot），末尾布局的差异在几分钱以内，可设 `l0.tail_injection=false` 做对照。
上线步骤、验证 SQL 与回滚开关见 [`docs/cache-layout-rollout.md`](docs/cache-layout-rollout.md)。

---

## 2. 架构

```text
入站消息
  │
  ▼
[短路器] 固化路径匹配（本地，零 LLM）
  │ 命中高置信度 → 模板/形状约束直答，结束
  │ 未命中
  ▼
[L0 线索层] 长期线索槽位中只注入 2–3 条，放进请求末尾的临时区
  │
  ▼
[L1 按需唤醒] 仅指代 + 低 L0 相关 / 矛盾 / 未闭合事务（稀有）
  │
  ▼
[最终生成] 主模型（STABLE_PREFIX + 可选回忆片段 + 最近窗口）
  │
  ▼
[L2 零调用写入] 正则蒸馏，更新线索/备用线索/冷种/队列
  │
  ▼
[L3 缓存感知修剪] 下一轮构建时剥离 EPHEMERAL_TAIL
```

关键边界：

- **L0 / L1 / L2 / 修剪 / 节奏 / 置信度**：没有任何模型客户端。
- 只有 `main_generate`（最终回复）、可选的 `solidify_small`（中等置信度固化）、可选的 `l1_rerank` 能接触模型。
- `LLMCallLedger` 会在调用发生前拒绝写路径 stage，并抛 `PolicyViolation`。
- `STABLE_PREFIX = 宿主系统提示 + 人格锚点`，修剪永远不碰它；线索块与回忆/节奏提示一起放在本轮用户消息末尾，换线索不会让系统提示之后的整段上下文缓存失效（见 §1.4）。

---

## 3. AstrBot 安装（推荐）

AstrBot 会加载插件目录下的 `main.py`，读取 `metadata.yaml`，并把 `_conf_schema.json` 渲染成 WebUI 配置。当前安装包按 **AstrBot 4.29.1** 制作，`metadata.yaml` 声明 `>=4.16,<5`；其他版本先跑 `astrbot_preflight.py`。

**方式一：插件市场（发布后）**

AstrBot 面板 → 插件 → 安装插件 → 搜索「真心」或 `astrbot_plugin_zhenxin`。

**方式二：从 GitHub 安装**

```bash
cd AstrBot/data/plugins
git clone https://github.com/xiaml666/astrbot_plugin_zhenxin.git
pip install -r astrbot_plugin_zhenxin/requirements.txt
```

**方式三：从 Release 安装**

下载 [最新 Release](https://github.com/xiaml666/astrbot_plugin_zhenxin/releases/latest) 里的
`astrbot_plugin_zhenxin-0.3.1.zip`，AstrBot 面板 → 插件 → 安装插件 → 从文件安装。插件卡片会使用 [`logo.png`](logo.png)。

> 0.3.1 起包目录为 `src/zhenxin/_aux`（`aux` 是 Windows 保留名），从 zip 安装不会再被系统改写。


重启 AstrBot，或在 WebUI 插件页重载。插件名为 `astrbot_plugin_zhenxin`。

要求 AstrBot 4.16+；`TextPart.mark_as_temp()` 在 4.24+ 可用，低版本会自动把本轮提示并入 `req.prompt`，不会崩。

### 3.1 它如何接进 AstrBot

| AstrBot 接口 | 真心做什么 |
|---|---|
| `@filter.event_message_type(ALL, priority=100)` | 先算本轮计划；固化命中时 `yield event.plain_result(...)` + `event.stop_event()`，**不进入 LLM** |
| `@filter.on_llm_request()` | 改写 `req.system_prompt`（稳定前缀 = 宿主系统提示 + 人格锚点）、`req.contexts`（最近窗口）、`req.prompt`（压缩后的当前输入）；L0 线索块、L1 回忆、节奏、话题栈都通过 `req.extra_user_content_parts` 的临时 `TextPart` 追加到本轮用户消息末尾 |
| `@filter.on_llm_response()` | 拿到真实回复与 `LLMResponse.usage`，执行 L2 写入、真实 token/费用记录；不调用任何模型 |

AstrBot 官方的提示词注入建议是“稳定内容进 system_prompt，每轮变化的内容进 `extra_user_content_parts`”，本插件遵循这一点：长期线索块按更换线索事件才变，L1 回忆只进临时尾部。

装好后可以在聊天里发送 `/zhenxin` 查看本用户的轮次、唤醒率、前缀稳定率、L0 字符、固化命中与真实 usage 覆盖情况；这个命令只读本地 SQLite，不调用模型。

针对 4.29.1 面板、已有 LivingMemory 的验证顺序见 [`docs/astrbot-install.md`](docs/astrbot-install.md)。

### 3.2 左侧「扩展功能 → 真心」页面

安装后 AstrBot 会在左侧 **扩展功能** 中注册「真心」页面（`pages/zhenxin/index.html`）。
0.3.0 起是带左侧导航的控制中心，只保留用户真正会用到的内容：

| 分页 | 内容 |
|---|---|
| **概览** | 轮次 / 缓存命中 / 记忆条数 / 累计花费 四张卡；最近 30 轮消耗柱状图（绿=命中缓存、橙=未命中）；运行健康行（版本、向量后端、数据库大小、累计错误、最近一次错误与时间） |
| **记忆** | 记忆库按“重要性 / 最近出现 / 最早出现”排序，显示类型、出现时间、用过几次、出现在几个会话；长期没被用到又不重要的记忆会自动淡出调用库，只保留关键词痕迹 |
| **花费** | 非缓存 / 缓存 / 输出 token 与总花费（只读；单价在 AstrBot 插件配置里） |
| **行为设置** | 只有 6 项真的会改变回复：只观察、是否接管最近对话、每轮线索条数、保留几轮、短句直答、线索字数上限。每项下面都有白话说明；另有“恢复推荐设置 / 恢复默认配置” |
| **高级** | 缓存保护、窗口回收、换线索节奏，以及“按需回忆 / 写入 / 辅助模块”折叠区 |
| **重置** | 换模型、新开对话后清掉短期状态：对话状态 / 长期记忆 / 统计分别可重置，也可一键全部恢复初始 |

- 保存后在下一轮生效，不需要重启 AstrBot（数据库路径等需要重启的字段没有放进页面）。
- “恢复默认配置”读取插件自带的 `_conf_schema.json`，和刚装好时的配置一致（`host_context_mode` 默认为接管 `replace`；与其他记忆插件共存时手动选“不接管”）。
- 记忆清理是规则式的：写入时记录 `importance`，每 20 轮（以及打开记忆页时）做一次小范围清扫；30 天以上、没被调用过、价值分低于阈值的条目会从 `facts` 删除，只在 `fact_traces` 留一行关键词，不再进入模型。
- “重置”只动真心自己的 SQLite：对话状态、`facts`、`metrics` 分范围删除；“全部恢复初始”同时把配置写回默认值。
- 错误会自动落库：`begin_turn` / `llm_request` / `llm_response` / 固化回合任一环节抛异常，
  都会写进 `__global__` 状态，页面健康行直接红字显示，不用翻日志。
- 配色跟随系统深浅色；窄屏自动单列。

页面后端只暴露白名单字段，不会让页面改数据库路径或模型 API；页面入口由 AstrBot `pages/<name>/index.html` 自动发现。

### 3.3 模型可调用的两个工具

插件会在 AstrBot 的「人格 → 工具」里出现两个函数工具，默认启用，可随时在人格里勾掉：

| 工具 | 作用 | 何时调用 |
|---|---|---|
| `zhenxin_recall` | 从该用户的长期记忆里按相关度找最多 3 条 | 用户明确问起过去（“你还记得…”“上次说的…”） |
| `zhenxin_remember` | 把用户明确要求记住的一句话写入记忆（重要度 0.9） | 用户说“记住…”“帮我记一下…” |

两者都是本地 SQLite + 哈希向量，不联网、不额外调用模型；`zhenxin_remember` 写入的记忆也会参与之后的按需回忆和自动淡出。

### 3.4 AstrBot 配置

WebUI 里可改的项见 `_conf_schema.json`：数据库路径、影子模式、长期线索开关/注入条数、短路器阈值、L1 冷却、L2 备用线索上限、L3 修剪、各辅助模块开关。完整嵌套配置仍可用 YAML/环境变量覆盖：

```bash
export ZHENXIN_CONFIG=/path/to/config.example.yaml
export ZHENXIN_DB_PATH=/path/to/data/zhenxin.sqlite3
export ZHENXIN_SHADOW_MODE=true
```

默认数据库：`data/zhenxin.sqlite3`（相对 AstrBot 根目录）。插件更新/重装不会覆盖它。

---

## 4. NoneBot2 接入

NoneBot2 没有内置 LLM 管线，因此这里提供的是两阶段薄适配，不假装存在统一 hook：

```python
from nonebot import on_message
from nonebot.adapters import Message
from zhenxin.adapters.nonebot2 import NoneBotAdapter

adapter = NoneBotAdapter(engine)
matcher = on_message(priority=1, block=False)

@matcher.handle()
async def _(event):
    plan = adapter.begin(event)
    if plan.shortcircuit:
        await matcher.finish(adapter.finish_shortcircuit(plan))
    reply = await my_llm(adapter.messages(plan))   # 宿主自己的模型
    adapter.finish(plan, reply)
```

`adapter.messages(plan)` 返回 `[{"role":"system","content":stable_prefix}, *recent_window+prompt]`。如果你的框架只接受一个 system 字符串，直接使用 `plan.stable_prefix`。

---

## 5. 独立运行 / 二次开发

```bash
python -m pip install -e .
python -c "from zhenxin import Config, ZhenxinEngine; ..."
```

无第三方依赖也能跑；`requirements.txt` 只包含可选的 PyYAML。ONNX 后端：

```bash
pip install -r requirements-onnx.txt
optimum-cli export onnx --model BAAI/bge-small-zh-v1.5 models/bge-small-zh
```

然后在配置里指定：

```yaml
embedding:
  backend: onnx
  model_path: models/bge-small-zh/model.onnx
  tokenizer_path: models/bge-small-zh/tokenizer.json
```

找不到模型时自动回退到哈希向量，不会下载、不会调用远程 embedding API。

---

## 6. 配置与模块开关

完整示例见 [`config.example.yaml`](config.example.yaml)。所有模块默认开启；每个都有独立开关，便于影子模式对比。

| 配置 | 默认 | 说明 |
|---|---:|---|
| `shadow_mode` | `false` | 只记录决策，不改上下文、不短路回复 |
| `strict_zero_llm` | `true` | 写路径出现 LLM 调用直接抛错 |
| `l0.enabled` | `true` | L0 长期线索槽位 |
| `l0.inject` | `3` | 注入条数（1–3） |
| `l0.max_inject_chars` | `150` | L0 字符上限 |
| `l0.hysteresis` | `2.0` | 更换线索需要超过最弱在任线索多少亮度；越高前缀越稳 |
| `l0.tail_injection` | `true` | 线索块放请求末尾临时区；关闭则回写系统提示，仅供对照 |
| `shortcircuit.enabled` | `true` | 习惯固化短路器 |
| `shortcircuit.hit_threshold` | `3.0` | 命中置信度 |
| `shortcircuit.max_hit_chars` | `8` | 超过这个长度不走固化，避免把计划陈述当习惯 |
| `shortcircuit.use_small_model` | `false` | 中等置信度用小模型；默认模板直答 |
| `l1.enabled` | `true` | 按需唤醒 |
| `l1.low_l0_threshold` | `0.3` | L0 最高相关低于此值才考虑唤醒 |
| `l1.wake_cooldown_rounds` | `8` | 同一冷种子唤醒冷却 |
| `l1.rerank` | `false` | 可选小模型 rerank（只重排本地候选） |
| `l2.enabled` | `true` | 零调用写入 |
| `l2.warm_limit` | `30` | 备用线索上限；溢出物理删除并落冷种 |
| `l3.enabled` | `true` | 三段上下文 |
| `l3.cache_aware_pruning` | `true` | 关闭后退化为普通滑动窗口，便于对比 |
| `l3.window_grow_turns` | `4` | 窗口追加到 `recent_turns+4` 再一次性回收；`0` = 旧版每轮滑动 |
| `l3.recent_char_low_water` | `0.7` | 超过字符预算时一次回收到 70%，避免每轮都切 |
| `l3.cache_ttl_seconds` | `300` | 缓存 TTL 假设（仅用于指标解释） |
| `l3.host_context_mode` | `replace` | `replace` 接管最近窗口；`off` 最兼容；`prepend` 保留宿主上下文 |
| `aux.topic_stack` | `true` | 话题栈（深度 3–5） |
| `aux.compression` | `true` | 长消息规则压缩 |
| `aux.rhythm` | `true` | 节奏软约束 + 轻量后处理 |
| `aux.lexicon` | `true` | 用户词汇表 |
| `aux.open_loops` | `true` | 未闭合事务队列（≤5） |
| `aux.negative_patterns` | `true` | 负面模式黑名单 |
| `metrics.enabled` | `true` | SQLite + JSONL 指标 |
| `metrics.input_price_per_million` | `2.0` | 实际 usage 的输入单价（¥/百万） |
| `metrics.cached_input_price_per_million` | `0.5` | 缓存命中单价（¥/百万） |
| `metrics.output_price_per_million` | `8.0` | 输出单价（¥/百万） |

环境变量：`ZHENXIN__L1__ENABLED=false`、`ZHENXIN__L3__CACHE_AWARE_PRUNING=false` 这类双下划线路径可覆盖任意字段；`ZHENXIN_MAIN_API_KEY`/`ZHENXIN_DECISION_API_KEY` 只从环境读，不写进配置。

> AstrBot 首次安装的 WebUI 默认是**安全配置**：`shadow_mode=true`、`shortcircuit.enabled=false`、`host_context_mode=replace`（已接管但 shadow 下不改请求）。与别的记忆插件共存时，先把 `host_context_mode` 改成 `off`；确认不冲突后再逐项打开。

---

## 7. 影子模式

目标：先验证“唤醒率是否随轮次下降”，不改变真实回复。

1. AstrBot WebUI 打开 `shadow_mode`，或设置 `ZHENXIN_SHADOW_MODE=true`。
2. 正常聊天；插件仍计算 L0/L1/短路计划并写入本地记忆，但：
   - 不 `stop_event`，不替换回复；
   - 不修改 `req.system_prompt/contexts/prompt`；
   - 只记录 `shadow_*` 指标。
3. 跑仓库数据集：

```bash
python scripts/run_shadow_sim.py
```

该脚本同时跑 live 与 shadow，输出：

- `reports/shadow_report.json`
- `reports/shadow_report.md`
- 检查项全部通过时退出码 0。

真实流量的唤醒率可在 SQLite 里查看：

```sql
SELECT round_no, wake, wake_trigger, prefix_stable FROM metrics ORDER BY round_no;
```

---

## 8. 复现验收

```bash
# 全部单测（74 个，含 60 轮验收）
python -m unittest discover -s tests -t . -v

# 仅跑验收
python scripts/run_shadow_sim.py

# 查看报告
cat reports/shadow_report.md

# AstrBot 真实环境：API 预检 + 真实 usage 校验
python scripts/astrbot_preflight.py
python scripts/validate_metrics.py --db data/zhenxin.sqlite3
```

测试覆盖：

- 10 个活跃线索、L0 ≤150 字、亮度-only 不改前缀
- 前缀逐字节稳定、临时片段不落历史、最近窗口字符预算
- 固化命中 → 负反馈 → `neg_streak>=2` → 24h 降级
- L2 零 LLM 调用、冷存储不含原文、备用线索/未闭合物物理淘汰
- L1 唤醒/冷却/普通问题不唤醒
- AstrBot 假对象上的拦截、上下文注入、shadow 不修改请求、重复 hook 不重复追加
- `LLMResponse.usage` → `input_other/input_cached/output` 落库
- 策略门禁：写路径调用 LLM 直接抛错
- 60 轮 live + shadow 全指标
- 装/不装 A/B 图表、寿命/价格估算、盲测评分表

### 人工评测（20 条）

[`data/human_eval_cases.jsonl`](data/human_eval_cases.jsonl) 覆盖短确认、情绪表达、话题切回、未闭合事追问、纠正、长消息压缩。

建议流程：

1. 同一组用例分别用“完整推理路径”和“线索/固化路径”生成回复；
2. 去掉路径标签，随机打乱；
3. 两名评分者按 5 分制打分（1=明显机械/跑题，5=自然且接得住）；
4. 计算每条用例两条路径的均分差；验收标准：平均差 ≤1 分，且没有“固化路径明显更差”的系统性模式。

把收集到的 A/B 回复写成 JSONL 后，可生成随机化、隐藏来源的评分页：

```bash
python scripts/quality_sheet.py --pairs data/quality_pairs.example.jsonl
# 替换成你自己的 data/quality_pairs.jsonl
```

---

## 9. 目录结构

```text
.
├── main.py                     # AstrBot 入口（Star + 三个钩子）
├── logo.png                    # AstrBot 插件卡片 logo
├── metadata.yaml               # AstrBot 插件元数据
├── _conf_schema.json           # AstrBot WebUI 配置
├── requirements.txt            # 可选 PyYAML
├── requirements-onnx.txt       # 可选 ONNX 后端
├── pyproject.toml
├── config.example.yaml
├── pages/zhenxin/index.html    # 左侧「扩展功能」可视化配置页
├── src/zhenxin/
│   ├── engine.py               # 五层编排
│   ├── config.py / db.py / llm.py / metrics.py / compare.py
│   ├── layers/                 # l0 l1 l2 l3 shortcircuit
│   ├── _aux/                   # 话题栈/压缩/节奏/词汇/队列/负面模式
│   └── adapters/               # astrbot / nonebot2
├── tests/                      # 74 个 unittest
├── data/sim_dialogue.jsonl     # 60 轮模拟数据
├── data/human_eval_cases.jsonl # 20 条人工评测
├── data/quality_pairs.example.jsonl # 盲测输入示例
├── scripts/run_shadow_sim.py   # 验收统计
├── scripts/compare_baseline.py # 装/不装 A/B 可视化
├── scripts/lifetime_cost.py    # 50 轮/天寿命与价格估算
├── scripts/astrbot_preflight.py # AstrBot API 面预检
├── scripts/validate_metrics.py # 真实 usage/指标校验
├── scripts/quality_sheet.py    # 盲测 A/B 评分表
├── scripts/cache_prefix_sim.py # 缓存布局模拟（前缀缓存 + 64 token 分块）
├── reports/shadow_report.md    # 已生成报告
├── reports/comparison.html     # 装/不装并排对比（浏览器打开）
├── reports/lifetime_cost.md    # 寿命与价格估算
├── docs/design.md              # 设计说明
├── docs/validation.md          # 上线前验证手册
├── docs/astrbot-install.md     # 4.29.1 面板安装/新建验证对话
├── docs/astrbot-live-report.md # 4.29.1 实机验证记录
├── docs/cache-layout-rollout.md # 0.1.1 缓存布局改动：上线/验证/回滚
└── dist/astrbot_plugin_zhenxin-0.3.1.zip
```

---

## 10. 上传 GitHub / 作为 AstrBot 插件发布

仓库地址：[`xiaml666/astrbot_plugin_zhenxin`](https://github.com/xiaml666/astrbot_plugin_zhenxin)。
发布到 AstrBot 插件市场：

1. 仓库根目录保留 `main.py`、`metadata.yaml`、`_conf_schema.json`、`requirements.txt`、`logo.png`。
2. `metadata.yaml` 的 `name` 用 `astrbot_plugin_zhenxin`，`repo` 指向本仓库，版本号与 Release 一致。
3. 打 tag（如 `v0.3.0`）并上传 Release zip；zip 不超过 16MB（当前约 150KB）。
4. 登录 [AstrBot Cloud](https://cloud.astrbot.app/) 的插件发布页，提交本仓库；审核通过后即可在
   AstrBot 面板「插件 → 安装插件」里搜索到。
5. 不要提交 `data/*.sqlite3`、`logs/`、`.venv/`、`models/`、`reports/`、`.astrbot_cookies.txt`。
6. 在 AstrBot 的 `data/plugins/` 下 clone 本仓库即可加载；不需要 `pip install` 本包（`main.py` 会把 `src/` 加入路径）。
7. 建议先开 `shadow_mode` 跑一段真实对话，再切 live。

---

## 11. 设计边界与已知限制

- **宿主系统提示**：AstrBot 会在插件 hook 之后把技能/工具说明追加到 `system_prompt` 末尾（实机约 7.9k token）。真心只保证自己写入的部分逐字节稳定，并把它放在请求末尾以免换线索时连带作废这段后缀；但其他插件若每轮往 `system_prompt` 追加时间、状态栏等**动态**内容，前缀仍会在那里断开。
- **哈希向量是词面匹配**：`它/那个` 这类纯指代在无 ONNX 模型时可能无法命中语义；生产建议启用 bge-small-zh。L1 因此要求“强指代 + 低 L0 相关”，宁可少唤醒。
- **AstrBot Agent/工具链**：一轮可能多次调用 LLM；适配器按 request id 分别注入，但 `on_llm_response` 在首个响应后就会完成本轮 L2 写入，Agent 的后续工具调用/最终文本可能没有被计入。简单 QQ/微信对话不受影响；Agent 深度用户建议先用 shadow 对照。
- **真实 token**：`on_llm_response` 会读取 `LLMResponse.usage` 的 `input_other/input_cached/output`；如果当前 provider 不返回 usage，`validate_metrics.py` 会显示 `actual_usage_rounds=0`，此时只能看估算，不能对外承诺账单节省。
- **兼容模式**：默认 `l3.host_context_mode=replace` 会接管最近窗口；与另一个记忆/知识库插件共存时，先改 `off` 或 `prepend` 做对照。
- **隐私**：SQLite 会保存最近对话原文（用于最近窗口）与规则蒸馏后的种子；冷存储不保存原始长文。`metrics` 默认只记录哈希、长度、模式，不记录 prompt 全文。生产环境请自行做数据保留期和访问控制。
- **这不是“记忆越多越好”**：长期线索只有 10 个槽位、注入 3 条；被覆盖/淘汰的节点会物理删除。默认不检索是刻意选择，不是能力缺失。

---

## License

MIT。见 [`LICENSE`](LICENSE)。






















