# Cabinet

一个以私域信源为基础的“透明研究导览助手”。
它不替你下结论，而是带你找到原作、标出关键证据，让你在完整语境里做判断。

## 愿景

在信息过载与AI幻觉的时代，为用户提供一个**绝对可信**、**过程透明**、**还原语境**的思考代理人。它不生产内容，它用你信任的智慧来审视你的难题。

## 解决的问题
- 传统搜索：关键词匹配不足，召回率低，容易漏掉关键文章。
- 碎片化 RAG：上下文断裂，逻辑链条丢失。
- Chatbot：黑盒生成，结论难以信任。

## 产品隐喻
不是“做饭的厨师”，而是“博物馆的专业导览员（带你找到原作，打上高光，让你自己看）”：
- 给你 5-10 篇必读，而不是海量链接。
- 关键内容必须来自原文引用，且可跳转溯源。

## MVP 核心体验
- **Project Based**：输入一个问题，生成一个研究项目并可持久化。
- **透明工作流**：展示检索、筛选、阅读过程。
- **Split View**：左侧证据卡片，右侧原文全文，点击联动高亮。
- **Quote-Centric**：只输出原文引用与理由，杜绝改写。
- **可复制的“证据回答”**：一键生成适合发布的纯文本答案。

## 如何体验
1) 启动服务（UI 默认端口 8002）：

```
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

2) 打开界面：

```
http://127.0.0.1:8002/
```

## 两种检索模式

- `Quick Search (cabinet)`：本地加权检索 + 并发审阅，适合快速找证据。
- `Deep Research (codex ...)`：启动独立 `codex exec --json` 进程做迭代深度检索，并把思考/命令/消息实时流式输出。
  - 深度模式只允许基于本地 `data/ps_2026-01-07.json` 做检索与引用；如果 Codex 尝试任何网络/下载命令会直接中止。

## CLI 直连（推荐先用）

先不经过前端，直接在终端跑 deep research：

```bash
cd /home/yy/project/ai_arch_lesson/cabinet_repo
./scripts/codexr "和女朋友异地两年，怎么减少关系风险" --context "希望给可执行步骤+原句证据"
```

输出是 JSON Lines，核心字段是：
- `event`: `start|progress|complete|error`
- `id`: 任务 id
- `status`: `running|heartbeat|event|thought|call|response|turn_completed|success|failed|timeout`

可选：做成短命令（接近 `codex xxxx`）

```bash
alias codexx='/home/yy/project/ai_arch_lesson/cabinet_repo/scripts/codexr'
codexx "请给我一份异地恋冷静期行动清单"
```

如果本机访问模型需要代理（例如 7890），仅对本次命令生效：

```bash
./scripts/codexr "你的问题" --proxy http://127.0.0.1:7890
```

对应接口：

- `GET /stream_research`
- `GET /stream_codex_research`

深度模式示例：

```bash
curl -N "http://127.0.0.1:8002/stream_codex_research?query=恋人在争吵后的冷静期适合做些什么&context=希望有可执行步骤"
```

深度模式可观测性（Streaming JSON）：
- 事件 `stream_log`：结构化 JSON（字段含 `event` / `id` / `status`），用于机器解析与监控。
- 事件 `codex_reasoning` / `codex_command` / `codex_message`：默认实时显示 thought / call / response。
- 事件 `codex_event`：保留 Codex 原始事件，便于调试。
- 自动心跳日志：无新输出时会周期性推送 heartbeat，避免“假死”感。

深度模式额外参数：
- `privilege_mode=default|full-auto|danger`
  - `default`：`--sandbox ... --ask-for-approval never`
  - `full-auto`：`--full-auto`
  - `danger`：`--dangerously-bypass-approvals-and-sandbox`
  - 推荐/默认：`danger`（避免 Codex 的 `LandlockRestrict` 导致无法执行本地检索命令）
  - 如果你的环境在 sandbox 下能正常执行本地命令，可考虑用 `full-auto` 代替 `danger`
- CLI 脚本也支持同名参数：`--privilege-mode`
- 超时重试：`--timeout-sec 300 --retries 1`

提示词安全：
- 服务端会在启动 Codex 前自动移除 `\0` 字符（NUL byte），并记录清洗计数，避免 CLI 因 `nul byte found` 崩溃。

## 致谢
当前数据来自 sooon.ai，源自知乎用户 https://www.zhihu.com/people/kvxjr369f 的开放版权与无私奉献，特此致谢。

## 开源协议
本项目采用最宽松的开源协议。

## 了解更多
产品定义与理念：`docs/product-definition-cabinet.md`
