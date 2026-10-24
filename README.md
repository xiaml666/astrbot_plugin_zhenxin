# 真心（Zhenxin）

> 不换模型，只接管“每一轮给模型看什么”。

**v0.6.0 · 109 个测试 · AstrBot 4.29.1 实测 · MIT**

真心是给 AstrBot 用的长期记忆与上下文插件：

- **默认不检索**：平时只带 3 条极短线索，出现“你还记得…”才翻旧记忆；
- **写记忆零 LLM 调用**：规则蒸馏，不额外花 token；
- **缓存友好**：线索放请求末尾，稳定前缀逐字节不变；
- **会遗忘**：记忆按重要度自动分池，零价值信息进回收池，避免上下文越聊越长；
- **会学人话**：模仿你本人的句长/语气词/表情，并周期性从网络补充真人聊天知识（≤500 字）；
- **可主动**：可选主动消息，只在私聊、非静默时段、满足间隔和每日上限时发一条。

---

## 和其他记忆插件有什么不同

| 常见做法 | 真心 |
|---|---|
| 每轮向量检索 top-k，或把记忆全量注入 | 默认不检索，只带 3 条线索；指代/矛盾/未完成事务才唤醒 |
| 抽取、摘要、重排多次调用模型 | 写路径零 LLM 调用，一轮仍然只付一次 chat 的钱 |
| 改系统提示，prefix cache 容易整段失效 | 线索放请求末尾，稳定前缀不变，缓存持续命中 |
| 记忆只增不减 | 周期结束后分入珍藏池/浅忆池；回收池定期彻底删除 |
| 只记事实，不记称呼和说话方式 | “叫我阿岚”每轮必带；说话习惯单独学习并限长 |
| 装完不知道和谁冲突 | 内置插件体检，扫描也会改写上下文的已安装插件 |

**适合**：想让它记住长期偏好/关系，又不想每轮多花 token；长对话越聊越贵；想先观察再接管。

---

## 0.6.0 / 0.5.0 新增

0.6.0 在 0.5.0 基础上：

- **运行总览**：侧栏状态卡、健康分、指标卡、关键链路、运行节奏，配置页沿用卡片分组。
- **记忆图谱**：节点带标签、按池子着色、按权重变大，节点间自动连线；点一下加深。
- **冲突逻辑重做**：只扫描已启用插件；媒体/B站/表情包/解析类不算冲突；
  只有真正改写上下文窗口、系统提示、摘要/压缩的才算直接冲突；新启用冲突插件会页面提醒 + AI 下一轮提一次。
- **自动学习**：说话知识库按设定周期后台联网更新；知识库默认收起。
- **反馈**：去掉本地保存，一键打开 GitHub 预填 Issue。
- **记忆恢复**：从历史对话补回“叫我…/你叫…/记住…”；首次对话和打开记忆页自动执行，也可手动“从历史对话重建”。

0.5.0 起：

1. **版本单一来源**：页面、`@register`、打包全部读 `metadata.yaml`，不会再出现“插件列表是最新、配置页还是 v0.1.0”。
2. **记忆花园**：点阵可视化，颜色越深越重要；默认只看今天，可切日期或看全部。
3. **四个池子**：周期池 / 珍藏池 / 浅忆池 / 回收池；艾宾浩斯式保持度，越高越难忘。
4. **手动加深**：点一下记忆点即可提高权重；支持移入回收池、恢复、彻底删除。
5. **零价值剔除**：明确说“不用提醒了 / 别叫我 X / 已经做完了”会自动进回收池，默认 7 天后删除。
6. **称呼记忆**：识别“叫我… / 我的名字是…”，作为高权重 identity 每轮注入。
7. **检索权重可调**：语义、重要度、保持度、使用次数、类型、原文命中、称呼加分。
8. **说话习惯**：模仿用户 + 周期性联网学习，知识库限 500 字，每轮只注入约 110 字。
9. **主动消息**：本地模板（未完成事项/高价值记忆）或可选模型润色；静默时段、每日上限、私聊限制。
10. **插件体检 + 反馈通道**：扫描已安装插件冲突，反馈可先存本地再跳 GitHub Issue。
11. **实时状态条**：概览/花费合并到页面顶部，带刷新按钮；模型不返回 usage 时显示估算而不是空白。
12. **长期线索上限**：每轮注入从最多 3 条放宽到 8 条。
13. **推荐设置即接管**：一键关闭观察模式并由真心接管最近窗口。

---

## 安装

**方式一：AstrBot 插件市场**

面板 → 插件 → 安装插件 → 搜索「真心」。

**方式二：Release zip**

下载 `astrbot_plugin_zhenxin-0.6.0.zip` → 面板 → 插件 → 从文件安装。

**方式三：GitHub**

```bash
cd AstrBot/data/plugins
git clone https://github.com/xiaml666/astrbot_plugin_zhenxin.git
```

重启 AstrBot。要求 AstrBot `>=4.16,<5`；`TextPart.mark_as_temp()` 在 4.24+ 可用，低版本会自动把本轮提示并入 `req.prompt`。

---

## 第一次用

1. 保持默认的**观察模式**，正常聊 50 轮；页面顶部能看到轮次、缓存命中、记忆和花费。
2. 看「记忆花园」是否符合预期，必要时点深/删除。
3. 点「由真心接管对话」→ 底部「保存并生效」。下一轮生效，不用重启。
4. 如果还装了别的记忆/摘要插件，先去「体检与反馈」扫描，或把 `l3.host_context_mode` 设为 `off` 做对照。

---

## 控制中心

| 页面 | 内容 |
|---|---|
| **行为与状态** | 实时状态、接管开关、长期线索（1–8 条）、最近窗口、主动消息、推荐设置 |
| **记忆** | 图谱 / 点阵、日期切换、搜索、周期/珍藏/浅忆/回收池、关键词痕迹 |
| **说话习惯** | 模仿用户、按周期自动联网学习、知识库默认收起、字数进度 |
| **体检与反馈** | 插件冲突扫描、GitHub Issue / 本地反馈 |
| **高级** | 换线索节奏、记忆检索权重、遗忘周期、辅助模块开关 |
| **重置** | 对话状态、记忆库、说话学习、统计、全部恢复初始 |

页面只暴露白名单字段，不能改数据库路径或模型 API。

---

## 记忆怎么工作

写入时先用规则抽出候选（偏好、关系、未完成事项、称呼、情绪），不调用模型。之后每轮只把**最相关的 1–3 条**放进请求末尾。

每 `cycle_days`（默认 14 天）评估一次：

| 池子 | 含义 | 行为 |
|---|---|---|
| **周期池** | 刚记住、还在观察 | 正常检索，按保持度衰减 |
| **珍藏池** | 重要/常被提到/手动加深 | 衰减更慢，优先注入 |
| **浅忆池** | 低价值、长期没用到 | 只有强相关才想起，优先遗忘 |
| **回收池** | 被明确撤回或手动移入 | 不进入模型；默认 7 天后彻底删除，只留关键词痕迹 |

“重要”不是拍脑袋：`语义相似 + 重要度 + 艾宾浩斯保持度 + 被使用次数 + 类型 + 原文命中 + 称呼加分`。这些权重都能在「高级」里调。

---

## 说话习惯与主动消息

**说话习惯**会从你最近的消息里统计：平均句长、常用语气词、笑声、省略号、表情比例、提问频率、常用口头禅/网络用语。它只收紧回复（更短、少表情、少追问），不会让回复变长，也不额外调用模型。

**联网学习**默认 7 天一次，搜索公开的聊天习惯资料，用规则提炼成短句；失败会保留上一次结果。所有内容硬限制在 500 字，每轮只取约 110 字。

**主动消息**默认关闭。开启后需要同时满足：最近活跃、私聊、非静默时段、对方空闲超过 `min_idle_minutes`、离上次超过 `min_interval_hours`、当天未超过 `max_per_day`。发送记录持久化，重启不会重置每日上限。

> 主动消息是产品选择，不是技术默认。群聊默认不发；不确定时先用「立即试发一条」测试目标会话。

---

## 兼容与冲突

真心默认接管 `req.contexts`。同时开启另一个“记忆/上下文/摘要”插件时，常见结果是重复注入、互相覆盖或缓存收益被抵消。

- 页面「体检与反馈」会扫描已安装插件，标出也会改写 `req.contexts`/`system_prompt` 的插件，并区分“已停用”；
- 最稳的共存方式：**只让一个插件接管上下文**；
- 需要对照时把 `l3.host_context_mode` 设为 `off`（最兼容）或 `prepend`（保留宿主上下文但更费 token）。

---

## 配置

WebUI 里能改的都在 [`_conf_schema.json`](_conf_schema.json)。常用项：

| 配置 | 默认 | 说明 |
|---|---|---|
| `shadow_mode` | `true` | 只记录，不改回复 |
| `l0.inject` | `3` | 每轮长期线索条数（1–8） |
| `l0.tail_injection` | `true` | 线索放请求末尾，保护缓存 |
| `l3.host_context_mode` | `replace` | 接管 / off / prepend |
| `memory.top_k` | `3` | 每轮最多带几条记忆 |
| `memory.cycle_days` | `14` | 遗忘周期 |
| `memory.trash_ttl_days` | `7` | 回收池保留天数 |
| `style_learning.max_chars` | `500` | 说话知识库上限 |
| `proactive.enabled` | `false` | 主动消息开关 |

独立运行或二次开发可用环境变量覆盖任意字段，例如 `ZHENXIN__L1__ENABLED=false`；`ZHENXIN_MAIN_API_KEY` / `ZHENXIN_DECISION_API_KEY` 只从环境读。

---

## 验证

```bash
python -m unittest discover -s tests -t .   # 109 个测试
python scripts/package_plugin.py            # 生成 dist zip
```

历史实测（AstrBot 4.29.1 + deepseek-flash，100 轮）：基线 ¥1.32 / 真心 ¥0.82（-38%），总输入 -50%；第 101 轮仍记得第 5 轮说的细节。真实账单以 provider 返回的 usage 为准；没有 usage 时页面会明确标成“估算”。

---

## 仓库文件

Each file appears once; no duplicated or doubled suffixes.

| File | English description |
|---|---|
| `main.py` | AstrBot entry point: Star registration, LLM hooks, background scheduler, two local tools. |
| `metadata.yaml` | Plugin manifest read by AstrBot; its `version` is the single source of truth. |
| `_conf_schema.json` | WebUI configuration schema and defaults. |
| `config.example.yaml` | Standalone/NoneBot-style YAML example. |
| `pages/zhenxin/index.html` | Control center page: status, memory, style, health, feedback, reset. |
| `src/zhenxin/engine.py` | Five-layer pipeline and turn orchestration. |
| `src/zhenxin/config.py` | Typed configuration model. |
| `src/zhenxin/db.py` | SQLite schema, migrations and helpers. |
| `src/zhenxin/layers/l0.py` | Hot-clue slots and cache-stable clue block. |
| `src/zhenxin/layers/l1.py` | On-demand recall. |
| `src/zhenxin/layers/l2.py` | Zero-LLM write path and rule distillation. |
| `src/zhenxin/layers/l3.py` | Cache-aware recent-window pruning. |
| `src/zhenxin/layers/memory.py` | Weighted retrieval, pools, forgetting cycle and recycle bin. |
| `src/zhenxin/_aux/` | Small local helpers: style, style learning, zero-value, rhythm, topic stack, loops. |
| `src/zhenxin/proactive.py` | Opt-in proactive message scheduler. |
| `src/zhenxin/page_api.py` | Plugin page backend and conflict scanner. |
| `src/zhenxin/adapters/` | AstrBot and NoneBot2 adapters. |
| `tests/` | Unit and regression tests. |

---

## 边界

- AstrBot 会在插件 hook 之后追加技能/工具说明；真心只保证自己写入的前缀稳定。
- 哈希向量是词面匹配；纯指代在无 ONNX 模型时可能命中不到。
- `LLMResponse.usage` 不返回时只能估算，不能对外承诺账单节省。
- SQLite 会保存最近对话原文与规则蒸馏的种子；生产环境请自行设置保留期和访问控制。
- 长期线索只有 10 个槽位；“默认不检索”是刻意选择，不是能力缺失。

---

## License

MIT。见 [`LICENSE`](LICENSE)。







