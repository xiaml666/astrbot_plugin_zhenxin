# 上线前验证手册

这份手册解决三个问题：

1. 装到 AstrBot 上会不会和其他插件冲突？
2. 省钱/延长上下文寿命是不是真的，而不是模型估算？
3. 回复的“人味”有没有被模板或上下文压缩破坏？

结论先说：**没有任何插件能保证“在所有 AstrBot 配置、所有模型、所有上下文窗口下都省 98%”**。能保证的是：给出可测量的指标、可回滚的开关、以及一套在真实环境里跑出你自己数字的流程。

## 0. 先修正预期

仓库里的 `98.5%` 来自：

- 不装方案保留**全部历史**
- 128k 上下文
- 每轮约 60 token
- 不含缓存折扣
- 固定模拟数据

换成 8k、短会话、或者 AstrBot 本来就会自动截断历史，节省会显著变小；如果开启 prompt caching，绝对金额也会下降。正确说法是：

> 在“全量历史会不断增长”的基线里，真心把输入上下文变成有界；窗口越大、聊得越久，节省越明显。具体数值用你自己的 `metrics` 实测。

## 1. 技术预检：先确认 API 面

在 **AstrBot 的 Python 环境**里运行：

```bash
cd AstrBot/data/plugins/astrbot_plugin_zhenxin
python scripts/astrbot_preflight.py
```

它会检查：

- `filter.event_message_type` / `on_llm_request` / `on_llm_response`
- `EventMessageType.ALL`
- `ProviderRequest` 的 `prompt/system_prompt/contexts/extra_user_content_parts`
- `LLMResponse.usage` 与 `TokenUsage.input_other/input_cached/output`
- `TextPart.mark_as_temp`（没有则走 prompt 回退）
- `main.ZhenxinPlugin` 是否能导入
- 本地三轮回放是否 L2=0、L0≤150

任何 ❌ 都先别开 live。⚠️ 表示该版本缺字段，但插件有回退路径。

## 2. 影子模式：不改变回复，先测真实请求

1. WebUI 打开 `shadow_mode`，或设置 `ZHENXIN_SHADOW_MODE=true`。
2. 正常聊天至少 **50 轮**（最好跨 2~3 天）。
3. 运行：

```bash
python scripts/validate_metrics.py --db data/zhenxin.sqlite3
```

重点看：

| 指标 | 健康信号 | 说明 |
|---|---|---|
| `actual_usage_rounds` | 接近 `rounds` | AstrBot 是否真的返回了 usage |
| `actual_total_input_tokens` | 有值 | 这是账单口径，不是字符估算 |
| `actual_cost` | 有值 | 按你在 `metrics` 里配置的单价计算 |
| `l2_llm_calls` | 0 | 写路径没有偷偷调模型 |
| `prefix_last20` | ≥95% | 稳定前缀没有被其他插件/动态内容打碎 |
| `wake_rate` | 越低越好，第 50 轮 ≤20% | L1 是否过于频繁 |
| `solid_hits` | 少量 | 太多说明模板化风险 |

如果 `actual_usage_rounds=0`：不是插件坏了，而是当前 AstrBot/provider 没有把 usage 暴露出来；这时只能看估算，**不能对外宣称真实省了多少钱**。

一次 AstrBot 4.29.1 的实机记录（含“其他插件导致短期不省反贵”的结果）见 [`astrbot-live-report.md`](astrbot-live-report.md)。

## 3. 冲突排查：一次只加一个插件

真心的默认行为会**接管 `req.contexts` 和 `req.system_prompt` 的稳定部分**。这天然会和“另一个记忆/知识库插件”竞争。按下面顺序验证：

1. **只装真心**：跑 20 轮，确认回复正常、`validate_metrics` 通过。
2. **加一个插件**：只加你实际会用的那个，再跑 20 轮。
3. 看日志/数据库里的 `prefix_last20`、`l0_chars`、回复长度是否突变。
4. 如果冲突，先改配置而不是改代码：

```yaml
l3:
  host_context_mode: "off"    # 不碰 req.contexts，只注入 L0/尾部
  # off: 最兼容，但 token 节省变小
  # prepend: 保留宿主 contexts 再追加最近窗口，兼容但更费 token
  # replace: 默认，省钱但接管最近窗口
```

其他检查点：

- 真心在 `on_llm_request` 使用 `priority=-100`，尽量晚于其他注入插件；它会保留宿主 system prompt 的原文（只剥离自己上次追加的标记）。
- 同一个 `req` 对象重复触发 hook 时不会重复追加（按 request id 去重）；Agent 多次请求会分别注入。
- 不要同时开两个“接管上下文”的插件；如果必须共存，先用 `off`/`prepend`，再看实际 token。

建议做一个 **canary 群/小号**：先跑 1~2 天，再扩到主号。

## 4. 人味评测：盲测，不看插件标签

不要用“感觉还行”判断。用仓库里的 20 条 [`data/human_eval_cases.jsonl`](../data/human_eval_cases.jsonl)：

1. 固定模型、temperature、system prompt、人格。
2. 同一批用例跑两次：
   - A：不装插件（全量历史）
   - B：装插件（live 或 shadow 的上下文）
3. 收集每条的 A/B 回复，随机打乱，隐藏哪条来自插件。
4. 两名评分者独立按 5 分制打分：

| 分数 | 标准 |
|---:|---|
| 5 | 自然、接得住情绪/事实，像真人朋友 |
| 4 | 基本自然，有小瑕疵 |
| 3 | 能聊，但明显机械/略跑题 |
| 2 | 明显模板化、重复、答非所问 |
| 1 | 破坏人设/冒犯/完全跑题 |

通过标准建议：

- 平均分差 **≤1 分**；
- 没有“插件路径在某一类（短确认/情绪/话题切回/未闭合追问）系统性低 1 分以上”；
- `solid_hit_rate` 超过 20% 时，人工重点看模板轮；
- 负面反馈/降级路径至少抽 10 条。

如果做不到人工双评，至少做**单评 + 盲选**，并保留原始 A/B 文本，别只留平均分。

把收集到的回复写成 JSONL（每行一条）：

```json
{"id":"HE01","category":"short_ack","user":"嗯","baseline_reply":"...","plugin_reply":"..."}
```

然后生成盲测表：

```bash
python scripts/quality_sheet.py --pairs data/quality_pairs.example.jsonl
# 打开 reports/quality_sheet.html，评分后导出 JSON
```

表格会随机交换 A/B 顺序，隐藏哪条来自插件。

## 5. 成本与寿命：用真实 usage 对齐

`validate_metrics.py` 的 `actual_*` 是 AstrBot `LLMResponse.usage` 的账单口径。把真实 50 轮的：

- `actual_total_input_tokens`
- `actual_cached_tokens`
- `actual_output_tokens`
- `actual_cost`

代入 [`scripts/lifetime_cost.py`](../scripts/lifetime_cost.py) 的 `--turn-tokens`、`--plugin-tokens` 假设，重新算寿命。理论表只是起点，**你自己的 50 轮实测**才是对外的数字。

建议对外只写：

> 在 128k、全量历史、无缓存折扣的基准下，固定数据集的估算节省为 55.4%（60 轮总输入）/ 98.5%（到基线耗尽）。实际值以你的 `metrics` 为准。

不要写“所有用户省 98%”。

## 6. 回滚开关

出问题按顺序退回：

1. `shadow_mode: true`（不改回复）
2. `shortcircuit.enabled: false`（禁用模板直答，先保人味）
3. `l1.enabled: false`（禁用唤醒）
4. `l3.host_context_mode: "off"`（不接管 contexts）
5. 移除插件，SQLite 文件保留，可回滚

## 7. 最小上线门槛

- [ ] `astrbot_preflight.py` 无 ❌
- [ ] shadow 50 轮，`l2_llm_calls=0`
- [ ] `actual_usage_rounds > 0`（否则不承诺真实省钱）
- [ ] `prefix_last20 ≥ 95%`
- [ ] 20 条盲测平均分差 ≤1
- [ ] 只装真心 + 一个真实依赖插件各 20 轮无冲突
- [ ] canary 群 1~2 天无异常
- [ ] README 的收益描述带有上述前提

做到这些，再上传 GitHub 才比较稳。

