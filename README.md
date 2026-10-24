# 真心（Zhenxin）

> 虚拟不代表冰冷，每一个ta都带着真心。

**v0.6.1 · 110 个测试 · AstrBot 4.29.1 实测 · MIT**

真心是给 AstrBot 用的长期记忆与上下文插件：不换模型，只决定每一轮让模型看到什么。

- **默认不检索**：平时只带 3 条极短线索，出现“你还记得…”才翻旧记忆；
- **写记忆零 LLM 调用**：规则蒸馏，不额外花 token；线索放请求末尾，保护缓存；
- **会遗忘**：记忆按重要度分入周期池 / 珍藏池 / 浅忆池，零价值信息进回收池；
- **会学人话，也能主动**：模仿你的说话习惯并周期性联网学习；可选主动消息和插件体检。

## 安装

**方式一：AstrBot 插件市场**

面板 → 插件 → 安装插件 → 搜索「真心」。

**方式二：Release zip**

下载 `astrbot_plugin_zhenxin-0.6.1.zip` → 面板 → 插件 → 从文件安装。

**方式三：GitHub**

```bash
cd AstrBot/data/plugins
git clone https://github.com/xiaml666/astrbot_plugin_zhenxin.git
```

重启 AstrBot。要求 AstrBot `>=4.16,<5`；`TextPart.mark_as_temp()` 在 4.24+ 可用，低版本会自动把本轮提示并入 `req.prompt`。
