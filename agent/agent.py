"""
Agent 系统 - LangGraph Agent + MCP 客户端
=========================================
通过 MCP 协议发现和调用工具，不直接 import rag/。
"""

import json
import logging
from typing import Optional

from mcp.client.session import ClientSession
from mcp.types import CallToolResult

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import CFG

logger = logging.getLogger("agent.core")

# ── 全局 MCP Session + 工具列表（由 main 设置） ──
_mcp_session: Optional[ClientSession] = None
_active_tools: list = []


def set_active_tools(tools: list):
    global _active_tools
    _active_tools = list(tools)


def _extract_text(result: CallToolResult) -> str:
    """从 MCP CallToolResult 提取文本"""
    texts = []
    for c in result.content:
        if hasattr(c, "text"):
            texts.append(c.text)
        elif hasattr(c, "type") and c.type == "text":
            texts.append(c.text)
    return "\n".join(texts)


# ═══════════════════════════════════════
# 本地基础工具（不进 MCP）
# ==========================================
# 记忆工具（search_memory/add_memory/list_memories）从 MCP 加载，
# 这里只保留 Agent 内部的本地工具。
# ═══════════════════════════════════════

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# ── search_tools：元工具，始终加载，不进向量库 ──

def _search_tools(query: str) -> str:
    """语义检索可用工具。当任务可能用到外部工具（路线规划、周边搜索等）时调用。"""
    from agent.tool_registry import search_text
    return search_text(query, top_k=5)


class SearchToolsInput(BaseModel):
    query: str = Field(description="任务描述，如'路线规划'、'周边搜索'")


search_tools_tool = StructuredTool.from_function(
    name="search_tools",
    description="搜索可用的外部工具。当用户提出不确定是否有对应工具的任务（如路线规划、周边搜索、地址查询）时，先调用此工具查找。",
    func=_search_tools,
    args_schema=SearchToolsInput,
)




# ── MCP Session 管理 ──

_mcp_session: Optional[ClientSession] = None


def set_mcp_session(session: ClientSession):
    """设置全局 MCP session（由 main 调用）"""
    global _mcp_session
    _mcp_session = session


def get_mcp_session() -> Optional[ClientSession]:
    """获取全局 MCP session"""
    return _mcp_session


# ── Agent 创建 ──

def create_langgraph_agent(tools=None):
    """创建 LangGraph ReAct Agent。tools 为 None 时使用基础工具列表。"""
    llm = ChatOpenAI(
        model=CFG["llm_model"],
        api_key=CFG["api_key"],
        base_url=CFG["base_url"],
        temperature=0.1,
    )
    if tools is None:
        tools = _active_tools if _active_tools else [is_new_conversation, search_tools_tool]
    return create_react_agent(llm, tools)


# ── 对话处理 ──

async def run_chat(
    messages: list[dict],
    user_input: str,
    injected_memories: str = "",
) -> dict:
    """运行一轮 Agent 对话。

    两阶段模式：
    1. Agent 用基础工具执行，可能调 search_tools 发现动态工具
    2. 如果 search_tools 返回了匹配工具，重建 Agent 加入这些工具再执行

    injected_memories: 规则预筛注入的长期记忆，拼入 system prompt
    """
    import time
    from agent.prompt import build_system_prompt
    from langchain_core.messages import ToolMessage

    t0 = time.time()
    base_tools = _active_tools if _active_tools else [search_tools_tool]

    # ── 阶段 1：拼接 prompt ──
    agent = create_langgraph_agent(base_tools)
    prompt = build_system_prompt(injected_memories=injected_memories) if injected_memories else build_system_prompt()

    lc_messages = [SystemMessage(content=prompt)]
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            ai_msg = AIMessage(content=content)
            if m.get("tool_calls"):
                ai_msg.tool_calls = m["tool_calls"]
            lc_messages.append(ai_msg)
        elif role == "tool":
            lc_messages.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "")))

    lc_messages.append(HumanMessage(content=user_input))

    logger.info(f"[Agent] 阶段1: {user_input[:60]}...")
    try:
        result = await agent.ainvoke({"messages": lc_messages})
    except Exception as e:
        logger.error(f"[Agent] 阶段1 失败: {e}")
        return {
            "messages": messages + [{"role": "user", "content": user_input}],
            "reply": f"抱歉，调用 LLM 时出错了：{e}",
            "elapsed": time.time() - t0,
        }

    # ── 阶段 2：检查 search_tools 是否被调用，动态加载工具 ──
    search_query = None
    for msg in result["messages"]:
        if getattr(msg, "type", None) == "ai" and hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                if tc.get("name") == "search_tools":
                    search_query = tc.get("args", {}).get("query", "")
                    break

    if search_query:
        from agent.tool_registry import search as registry_search
        matched = registry_search(search_query, top_k=5)
        if matched:
            logger.info(f"[Agent] search_tools 匹配到 {len(matched)} 个工具，重建 Agent 重跑")
            # 合并工具列表
            all_tools = list(base_tools)
            for t in matched:
                if t not in all_tools:
                    all_tools.append(t)

            agent2 = create_langgraph_agent(all_tools)
            try:
                result = await agent2.ainvoke({"messages": lc_messages})
            except Exception as e:
                logger.error(f"[Agent] 阶段2 失败: {e}")
                # 阶段2失败，继续用阶段1的结果
            logger.info(f"[Agent] 阶段2完成")

    elapsed = time.time() - t0
    logger.info(f"[Agent] 完成，耗时 {elapsed:.1f}s")

    # 提取完整消息历史
    full_messages = []
    for msg in result["messages"]:
        msg_type = getattr(msg, "type", None)
        if msg_type == "system":
            continue
        content = getattr(msg, "content", "") or ""

        if msg_type == "human":
            full_messages.append({"role": "user", "content": content})
        elif msg_type == "ai":
            entry = {"role": "assistant", "content": content}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = [
                    {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
                    for tc in msg.tool_calls
                ]
            full_messages.append(entry)
        elif msg_type == "tool":
            full_messages.append({
                "role": "tool",
                "content": content,
                "tool_call_id": getattr(msg, "tool_call_id", ""),
            })
        else:
            full_messages.append({"role": "user", "content": content})

    last_ai = result["messages"][-1]
    reply = last_ai.content if hasattr(last_ai, "content") else str(last_ai)

    return {
        "messages": full_messages,
        "reply": reply,
        "elapsed": elapsed,
    }


# ── 记忆提取（通过 MCP） ──

async def extract_memories_via_mcp(messages: list[dict], session_id: str, auto_commit: bool = True) -> str:
    """通过 MCP 调用 RAG 的记忆提取管道。auto_commit=False 时冲突项返回 PENDING_CONFLICTS JSON。"""
    messages_json = json.dumps(messages, ensure_ascii=False, default=str)
    result = await _mcp_session.call_tool(
        "extract_memories",
        {
            "messages_json": messages_json,
            "session_id": session_id,
            "auto_commit": auto_commit,
        },
    )
    return _extract_text(result)


async def search_memories_via_mcp(query: str = "", top_k: int = None) -> str:
    """通过 MCP 预搜索记忆（用于构建 system prompt）"""
    args = {"query": query or ""}
    if top_k is not None:
        args["top_k"] = top_k
    result = await _mcp_session.call_tool("search_memory", args)
    return _extract_text(result)
