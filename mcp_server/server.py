"""
MCP Server - SSE 传输
=====================
独立进程，通过 MCP SSE 协议暴露 RAG 能力。
端口: 8001
底层通过 httpx 调用 RAG REST API (8002)。
使用 FastMCP 简化实现。
"""

import json
import logging

import httpx

from mcp.server.fastmcp import FastMCP

from config import CFG

logger = logging.getLogger("mcp.server")

# ── RAG 服务地址 ──
RAG_BASE = f"http://{CFG['rag_host']}:{CFG['rag_port']}"

# ── 创建 FastMCP Server ──
mcp = FastMCP(
    name="agent-memory-mcp",
    instructions="Agent Memory MCP Server - 暴露 RAG 记忆能力",
    host=CFG["mcp_host"],
    port=CFG["mcp_port"],
    sse_path="/sse",
    message_path="/messages",
)


# ═══════════════════════════════════════
# MCP Tools
# ═══════════════════════════════════════

@mcp.tool()
async def search_memory(query: str, top_k: int = 5) -> str:
    """搜索用户历史对话中的长期记忆。
    当用户提到'上次'、'之前'、'以前'、'之前聊过'、历史偏好或决策时，使用此工具查询。

    Args:
        query: 搜索查询，描述要查找的记忆内容
        top_k: 返回结果数量，默认 5
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(
                f"{RAG_BASE}/search",
                params={"query": query, "top_k": top_k},
            )
            resp.raise_for_status()
            data = resp.json()
            if not data["results"]:
                return "未找到相关历史记忆。"
            lines = []
            for r in data["results"]:
                date = r.get("created_at", "")[:10]
                lines.append(f"- [{r['category']}, {date}] {r['content']}")
            return "\n".join(lines)
        except httpx.ConnectError:
            return "错误: RAG 记忆服务不可用，请检查服务是否启动。"
        except Exception as e:
            return f"错误: {e}"


@mcp.tool()
async def add_memory(content: str, category: str, session_id: str = "") -> str:
    """添加一条新的长期记忆。当需要记住用户的新信息、偏好、事实或决策时使用。

    Args:
        content: 要记住的记忆内容
        category: 记忆分类: preference（偏好）、fact（事实/信息）、decision（决策）
        session_id: 关联的会话 ID，可选
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                f"{RAG_BASE}/add",
                json={
                    "content": content,
                    "category": category,
                    "session_id": session_id,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return f"记忆已保存 [{data['mem_id']}]: {content}"
        except httpx.ConnectError:
            return "错误: RAG 记忆服务不可用，请检查服务是否启动。"
        except Exception as e:
            return f"错误: {e}"


@mcp.tool()
async def list_memories() -> str:
    """列出所有已存储的长期记忆。用于查看用户的全部历史记录。"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(f"{RAG_BASE}/memories")
            resp.raise_for_status()
            data = resp.json()
            if not data["memories"]:
                return "（暂无长期记忆）"
            lines = [f"长期记忆 ({data['count']} 条):"]
            for m in data["memories"]:
                lines.append(
                    f"  [{m['category']}] {m['content']}  "
                    f"(v{m['version']}, {m['stability']})"
                )
            return "\n".join(lines)
        except httpx.ConnectError:
            return "错误: RAG 记忆服务不可用，请检查服务是否启动。"
        except Exception as e:
            return f"错误: {e}"


@mcp.tool()
async def extract_memories(messages_json: str, session_id: str = "", auto_commit: bool = True) -> str:
    """从对话历史中提取需要长期记住的用户信息。
    auto_commit=True: 全部自动写入。
    auto_commit=False: 冲突项不自动写，返回 JSON 格式的待确认列表。

    Args:
        messages_json: 对话消息的 JSON 字符串
        session_id: 关联的会话 ID，可选
        auto_commit: 是否自动写入冲突项，默认 true
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            try:
                messages = json.loads(messages_json)
            except json.JSONDecodeError:
                return "错误: messages_json 不是有效的 JSON"

            resp = await client.post(
                f"{RAG_BASE}/extract",
                json={
                    "messages": messages,
                    "session_id": session_id,
                    "auto_commit": auto_commit,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("pending"):
                # 有冲突 → 返回 JSON 让前端展示
                pending_list = [
                    f"[{p['action']}] target={p.get('target_id','')} content={p.get('new_content','')}"
                    for p in data["pending"]
                ]
                return f"PENDING_CONFLICTS: {json.dumps(data['pending'], ensure_ascii=False)}"
            return f"记忆提取完成，处理了 {data.get('committed', data.get('added_or_updated', 0))} 条记忆"
        except httpx.ConnectError:
            return "错误: RAG 记忆服务不可用，请检查服务是否启动。"
        except Exception as e:
            return f"错误: {e}"


# ── 启动入口 ──

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info(f"[MCP Server] 启动在 {CFG['mcp_host']}:{CFG['mcp_port']}")
    logger.info(f"[MCP Server] RAG 后端: {RAG_BASE}")
    logger.info("[MCP Server] 端点:")
    logger.info(f"  SSE:  http://{CFG['mcp_host']}:{CFG['mcp_port']}/sse")
    logger.info(f"  Messages: http://{CFG['mcp_host']}:{CFG['mcp_port']}/messages")
    logger.info("[MCP Server] Tools: search_memory, add_memory, list_memories, extract_memories")

    import asyncio
    asyncio.run(mcp.run_sse_async())
