# Agent 记忆系统 - 架构设计文档

> 最后更新: 2026-06-06

## 一、三层服务架构

```
┌──────────┐  HTTP    ┌──────────┐  HTTP    ┌──────────┐
│  Agent   │ ──────► │ MCP Svr  │ ──────► │ RAG Svc  │
│  CLI     │         │ SSE:8001 │         │ :8002    │
│ LangGraph│         │ FastMCP  │         │ FastAPI  │
└──────────┘         └──────────┘         └──────────┘
```

| 层 | 端口 | 职责 |
|------|------|------|
| RAG | 8002 | FAISS 向量存储 + JSON 元数据 + LLM 提取管道 |
| MCP | 8001 | 标准 MCP SSE 协议暴露 RAG，供外部客户端使用 |
| Agent | CLI | LangGraph ReAct Agent，工具调 RAG HTTP API |

Agent 不 import rag/，MCP 是可选网关。

## 二、Agent 核心设计原则

### 2.1 无状态

Agent 本身 = LLM + 推理循环 = 纯函数。状态全在外部：

```
Agent（不存数据）           外部服务（有状态）
─────────────               ────────────────
不知道你是谁                  Redis 知道 user_id → session
不知道你聊到哪了              Redis 知道 session_id → messages
不知道你记忆是什么            FAISS 知道 user_id → memories
只管: 收到 user_id → 透传给工具   只管: 校验权限 → 返回该用户数据
```

每次对话结束 Agent 可销毁，下次重建靠 `user_id + session_id` 恢复。

### 2.2 工具自治（非流程控制）

```
之前（流程控制）                      现在（工具自治）
─────────────────                    ────────────────
chat() {                             Agent 收到消息
  if redis.exists(sid):                  → 调 is_new_conversation()
    inject_memories()                    → 返回 true
  else:                                  → 自己决定调 search_memory()
    continue                             → 拿到记忆，生成回复
  agent.invoke()                     
}                                    系统只提供工具，不替 Agent 做决策
系统替 Agent 做决策
```

决策权从系统交还 Agent，系统只负责把真实世界信息准确暴露。

### 2.3 Skill 模式

```
memory_skill = {
    "prompt":   "你是记忆助手，新对话查记忆，续聊不主动查...",
    "tools":    [is_new_conversation, search_memory, add_memory, list_memories],
    "backend":  RAG (FAISS + embedding)
}
```

`prompt` + `tools` + `backend` 三件套捆在一起就是一个 skill。任何 Agent 挂上这个 skill 就有记忆能力。写提示词时发现自己其实在写 skill——这是正常的，工具解耦后 prompt 自然就变成抽象的行为规则。

## 三、记忆系统

### 3.1 短期记忆（Redis + 磁盘）

| 项目 | 说明 |
|------|------|
| 存储 | Redis 热数据 + `data/sessions/{id}.json` 兜底 |
| 写入时机 | 每轮对话实时写入 |
| 内容 | user、assistant、tool 完整消息（含 tool_calls 和 tool_call_id） |
| 裁剪 | 滑动窗口 10 轮（`max_turns=10`） |

### 3.2 长期记忆（FAISS + JSON）

| 项目 | 说明 |
|------|------|
| 向量 | BAAI/bge-small-zh-v1.5（512维，国产中文模型，HF镜像下载） |
| 元数据 | `data/memories.json` |
| 触发方式 | `/end` 指令（客户端主动触发），无定时任务 |
| 管道 | 候选提取 → 向量去重 → LLM 冲突判断 → 落库 |

### 3.3 Embedding

```
BAAI/bge-small-zh-v1.5 (512维)
├── 国产中文优化模型
├── 本地运行，无需 API Key
├── HF 镜像下载（HF_ENDPOINT=https://hf-mirror.com）
└── 启动时预加载，避免首次搜索卡顿
```

### 3.4 记忆注入策略

| 场景 | 行为 |
|------|------|
| 新对话（Redis key 不存在或为空） | Agent 调 `is_new_conversation` → true → 主动搜长期记忆 |
| 续聊（Redis 有消息） | Agent 调 `is_new_conversation` → false → 只在用户提"上次/之前/昨天"时搜 |
| 提示词 | 不写死工具名，只写行为规则。工具定义由 LangGraph function calling 自动传给 LLM |

## 四、对话生命周期

```
启动 → [系统] 新对话开始
  [0] 你:聊天                       ← 短期记忆实时存 Redis
  [1] 你: /end                      ← LLM 提取长期记忆 → FAISS，对话继续
  [2] 你: 继续聊
  [3] 你: /new                      ← 提取长期记忆 → 新对话（模拟关闭重开聊天页面）
  [系统] 新对话开始
  [0] 你: 新对话首条消息             ← Agent 判断 is_new=true → 主动搜记忆
  [1] 你: /quit                     ← 退出
```

| 指令 | 提取长期记忆 | 开新对话 |
|------|:----------:|:------:|
| `/end` | ✅ | ❌ 对话继续 |
| `/new` | ✅ | ✅ |
| `/quit` | ❌ | 退出 |

### 新对话判断

```
Redis key 不存在  ─┐
                   ├─→ is_new = true → Agent 主动搜长期记忆
Redis key 值为空  ─┘
Redis key 有消息   ──→ is_new = false → Agent 只在用户提及时搜
```

## 五、工具系统

### 5.1 当前工具列表

| 工具 | 类型 | 作用 |
|------|------|------|
| `is_new_conversation` | 基础 | 查 Redis 判断是否新对话 |
| `search_memory` | 基础 | HTTP → RAG `/search` 检索长期记忆 |
| `add_memory` | 基础 | HTTP → RAG `/add` 写入长期记忆 |
| `list_memories` | 基础 | HTTP → RAG `/memories` 列出全部记忆 |

### 5.2 工具分类设计

```
基础工具（始终加载，不进向量库）              动态工具（进向量库，按需语义检索）
────────────────────────────               ──────────────────────────
is_new_conversation → 判断新对话              search_memory → 记忆检索
search_tools        → 元工具，搜其他工具的     plan_route → 高德路线规划
add_memory          → 写记忆                  search_around → 高德周边搜索
list_memories       → 列记忆                  geocode → 高德地址转坐标
                                             ... 后续接入的 MCP 工具
```

### 5.3 工具定义

```python
# Pydantic Model = 给 LLM 看的参数说明书，三方传递：

# 1. 你写的
class SearchInput(BaseModel):
    query: str
    top_k: int = None

# 2. LangChain 提取为 JSON Schema 发给 API
# {"query": {"type": "string"}, "top_k": {"type": "integer"}}

# 3. API 约束解码 + Function Calling 保证输出正确
# → 工具名 ✅  参数名 ✅  类型 ✅  个数 ✅
```

### 5.4 工具注册中心

```
启动时（一次性）                          运行时（每次请求）
───────────                              ──────────
all_tools = load_all()                   relevant = registry.search(query)
for t in all_tools:                      agent = create(llm, relevant)
    if t.name not in existing:           agent.invoke(msg)
        registry.register(t)
```

- 工具索引独立文件：`data/tool_index.faiss` + `data/tool_metadata.json`
- 与记忆索引物理隔离
- 同名工具自动跳过不重复注册
- `search_tools` 是元工具，不进向量库（避免递归）

## 六、System Prompt 设计

```python
# 不写死工具名，只写行为规则
# 工具定义由 LangGraph function calling 自动传给 LLM
SYSTEM_PROMPT = """你是一个有记忆的助手。

## 核心规则
1. 收到消息后先检查是否为新对话。如果是，主动检索与用户消息相关的历史记忆
2. 如果是续聊，不要主动检索，只在用户明确提到"上次""之前""昨天""以前聊过"等时才检索
3. 永远不要捏造用户没说过的话
4. 搜索结果为空时诚实告知
5. 用户透露新的重要信息（偏好、事实、决策）时，主动记录
"""
```

- 加新工具、删工具、改工具名 → 不用动提示词
- LLM 从 function calling 拿到精确的工具名和参数，提示词只描述何时用哪类能力

## 七、多用户隔离

### 7.1 核心方案：user_id 分区

```
方案 A（当前，元数据过滤）：
  FAISS 一个 index → 搜 top_k*3 → 按 user_id 过滤 → 返回 top_k

方案 B（生产，物理隔离）：
  Qdrant collection = f"user_{user_id}" / Pinecone namespace = user_id
```

两个方案接口相同：`search(query, user_id)`。换 B 只改 store.py 内部实现。

### 7.2 请求链路

```
用户A 请求 (user_id=A)           用户B 请求 (user_id=B)
        │                                │
   FastAPI 独立协程                  FastAPI 独立协程
        │                                │
   Agent(user_id=A)                 Agent(user_id=B)
        │                                │
   search(query, "A")               search(query, "B")
        │                                │
   RAG: WHERE user_id=A            RAG: WHERE user_id=B
```

### 7.3 需要加的东西

| 现在 | 缺的 |
|------|------|
| `user_id = "demo_user"` 写死 | 每个请求带 user_id |
| FAISS 一个索引所有人 | 元数据加 user_id 字段过滤 |
| Redis session key 已有 user 前缀 | ✅ 这层已隔离 |
| CLI 交互 | FastAPI `/chat` 路由 |

## 八、MCP 集成架构

### 8.1 当前架构

```
Agent 工具 ──HTTP──► RAG REST API (8002)
外部客户端 ──MCP──► MCP Server (8001) ──HTTP──► RAG (8002)
```

Agent 和外部 MCP 客户端是平级的 RAG 消费者。

### 8.2 接入第三方 MCP（如高德）

```
Agent
  ├── 连接 Memory MCP (localhost:8001)  → memory_tools
  ├── 连接 Gaode MCP  (gaode:8003)      → gaode_tools
  └── 合并: all_tools = memory + gaode
```

```python
from langchain_mcp_adapters.client import load_mcp_tools

async with sse_client("gaode-mcp-url") as (read, write):
    gaode_tools = await load_mcp_tools(session)  # 0 行 Pydantic，自动发现
```

### 8.3 自己的工具 vs MCP 工具

| | 自建工具 | MCP 工具 |
|------|------|------|
| schema 在哪 | 你代码里 Pydantic | MCP Server 的 inputSchema |
| 谁写的 | 你 | MCP 作者 |
| 你要写 Pydantic 吗 | 要 | 不要（adapter 自动转） |
| 原理 | 都是告诉 LLM 工具参数格式 | 完全一样 |

## 九、生产部署改造要点

| 层 | 当前（Demo） | 生产 |
|------|------|------|
| Agent 接口 | CLI `input()` | FastAPI `/chat`（Spring Boot Controller 同行） |
| 向量库 | FAISS 单进程 | Qdrant / Pinecone（并发安全） |
| 会话 | Redis + 磁盘 | Redis 集群 |
| LLM 提取 | 同步调用 | Celery 异步队列（避免阻塞） |
| 启动 | `python run_all.py` | Uvicorn `--workers 4` |

## 十、项目文件结构

```
agent_memory/
├── .env                          # DeepSeek API Key + 配置
├── config.py                     # 统一配置
├── requirements.txt
│
├── rag/                          # RAG 服务 (port 8002)
│   ├── embedder.py               #   BAAI/bge-small-zh-v1.5 (512维)
│   ├── store.py                  #   FAISS + JSON CRUD
│   ├── extractor.py              #   LLM 提取/去重/冲突
│   └── server.py                 #   FastAPI REST API
│
├── mcp_server/                   # MCP 网关 (port 8001)
│   └── server.py                 #   FastMCP SSE，4 个 tools
│
├── agent/                        # Agent (CLI)
│   ├── session.py                #   短期记忆 Redis
│   ├── prompt.py                 #   System prompt（行为规则）
│   ├── agent.py                  #   LangGraph Agent + 工具定义
│   └── main.py                   #   CLI 入口 + 会话生命周期
│
└── data/                         # 持久化数据
    ├── index.faiss               #   长期记忆向量索引
    ├── memories.json             #   长期记忆元数据
    └── sessions/                 #   短期记忆磁盘备份
```
