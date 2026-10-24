# 在你截图里的 AstrBot 4.29.1 上安装与验证

目标环境：`your-server` 的 AstrBot 面板（4.29.1，Ubuntu，`/opt/bots/astrbot/data`），面板里已有 LivingMemory 等记忆插件。

## 1. 上传安装包

本地已生成：

```text
dist/astrbot_plugin_zhenxin-0.1.0.zip
```

在 AstrBot 面板：

1. 左侧 **插件**
2. 右上 **安装插件**
3. **从文件安装**
4. 选择上面的 zip
5. 安装后重载插件

插件卡片应显示：logo、**真心**、`astrbot_plugin_zhenxin`。如果安装器要求目录结构，zip 内已经是 `astrbot_plugin_zhenxin/` 根目录。

安装后左侧 **扩展功能** 会出现「真心」页面；如果没有，先在插件页点一次重载。页面分小白/专业两档，直接点进去改配置，不需要用插件齿轮里的原始 JSON 表单。

如果页面能打开但按钮没反应：先确认是从 AstrBot 的「扩展功能」打开，而不是直接双击 HTML；然后强制刷新一次（Ctrl+F5）。页面依赖 AstrBot 注入的 bridge SDK，直接打开本地文件不会工作。

## 2. 第一次配置：兼容优先，不追求省钱

进入插件配置页，先这样设：

| 配置 | 值 | 原因 |
|---|---|---|
| `shadow_mode` | `true` | 只记录，不改回复，不短路 |
| `shortcircuit.enabled` | `false` | 先排除模板化对“人味”的影响 |
| `l3.host_context_mode` | `off` | 你已装 LivingMemory，先不抢 `req.contexts` |
| `l1.enabled` | `true` | 只读本地 SQLite，不调用模型 |
| `strict_zero_llm` | `true` | 写路径强校验 |
| `db_path` | 默认 | AstrBot 内会落到插件专属 data 目录 |

保存后，先只装真心跑一轮，确认机器人还能正常回复、面板无报错。再打开 LivingMemory 跑 20 轮，看是否冲突。

## 3. 新建一个干净的验证对话

AstrBot 的会话通常按 `unified_msg_origin` 隔离。用以下任一种方式拿一个干净会话：

- WebUI 聊天页新建一个对话/新会话；
- 用测试小号私聊机器人；
- 如果当前版本支持，在聊天里发送 `/new`（不同版本命令可能不同）。

关键是：**不要在你和真实朋友的历史会话里做第一次验证**。用新用户/新会话，让真心的 `user_key` 从零开始。

注意：AstrBot 的“新对话”如果仍复用同一个 `unified_msg_origin`，真心的 L0 记忆会延续，不是干净会话。稳妥做法是：

- 用测试小号私聊；或
- 先停插件，删除 `data/plugins/.../zhenxin.sqlite3`（或插件专属 data 目录下的 `zhenxin.sqlite3`），再启动；或
- 在配置里临时把 `db_path` 指到一个新的空文件。

发 `/zhenxin` 时如果轮次不是 0，就说明不是干净会话。

## 4. 10 轮冒烟消息

直接依次发送：

```text
今天好累
嗯，海盐那本还没画完
周五就要交稿了，感觉画不完
最近还在喝中药，苦死了
上次说的那个中药，还要喝多久？
团子今天一直趴我键盘上
你还记得团子叫什么吗？
算了先不管，画稿要紧
又加班
晚安
```

然后发送：

```text
/zhenxin
```

应看到：轮次、唤醒率、前缀末20稳定率、L0 字符、固化命中、真实 usage 覆盖率。`shadow=true` 时回复应和没装插件基本一致。

## 5. 50 轮验证

继续聊到 50 轮以上（可以隔天继续）。然后在本机/服务器 AstrBot 的 Python 环境运行：

```bash
cd /opt/bots/astrbot
python data/plugins/astrbot_plugin_zhenxin/scripts/astrbot_preflight.py
python data/plugins/astrbot_plugin_zhenxin/scripts/validate_metrics.py \
  --db data/plugins/astrbot_plugin_zhenxin/data/zhenxin.sqlite3
```

如果实际数据目录不同，以 `/zhenxin` 或日志中的路径为准。重点看：

- `actual_usage_rounds` 是否接近总轮次；
- `actual_cost` 是否是真实账单口径；
- `prefix_last20 ≥95%`；
- `l2_llm_calls=0`；
- `wake_rate` 是否随轮次下降；
- `solid_hit_rate` 是否过高。

## 5.1 50 轮 A/B（推荐用 `data/validation_50.jsonl`）

只跑一组 50 轮不能证明省钱。需要两个干净会话：

- A：`shadow=true`、`shortcircuit=false`、`host_context_mode=off`
- B：`shadow=true` + `live_allowlist` 只放 B 的 `user_key`，`host_context_mode=replace`

两组发同一组 50 条用户消息，比较 `actual_total_input_tokens` / `actual_cost`。本机实测 50 轮为输入 **-24.1%**、价格 **-8.2%**；50 轮还不足以验证“上下文耗尽”寿命。

## 6. 从 shadow 切 live

只有在第 5 步和 20 条盲测都通过后：

1. `shadow_mode=false`
2. 先保持 `shortcircuit.enabled=false` 跑 20 轮，确认人味
3. 想省调用再打开 `shortcircuit.enabled=true`
4. 与 LivingMemory 共存时保持 `host_context_mode=off`；如果确认不需要它的上下文，再切 `replace` 做 A/B

每次只改一个开关，跑 20 轮再改下一个。

## 7. 直接让我代操作？

上面的安装/新建对话需要面板登录态或 SSH。当前工作区没有该服务器的凭据，所以我不能隔空点击你的 `your-server:6185` 面板。你可以：

- 自己按上面 4 步操作；或
- 提供临时 SSH/面板访问方式，我继续代装并跑验证。

