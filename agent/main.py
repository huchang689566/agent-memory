"""
Agent 系统 - CLI 入口
====================
独立的 Agent 进程，通过 MCP 协议与 MCP Server 通信。
短期记忆以 user_id 为维度，10 轮窗口 + 30 分钟 TTL，无 session 概念。
"""

import json
import logging
import sys
import time
from typing import Optional

import anyio

# ── Windows CMD GBK 编码兼容 ──
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding=sys.stdout.encoding, errors="replace"
    )

from config import CFG
from agent.session import ShortMemory
from agent.extract_worker import ExtractWorker
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from security import init_guards, get_input_guard, get_output_guard

from agent.agent import (
    set_mcp_session,
    set_active_tools,
    run_chat,
    extract_memories_via_mcp,
    search_memories_via_mcp,
)

logger = logging.getLogger("agent.main")

memory = ShortMemory()
MCP_URL = f"http://{CFG['mcp_host']}:{CFG['mcp_port']}/sse"

# ── 后台记忆提取 Worker ──
extract_worker = ExtractWorker(memory.redis)


def _filter_for_mcp(messages: list[dict]) -> list[dict]:
    """过滤消息，只保留 user text 和 assistant text"""
    cleaned = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "user":
            if isinstance(content, str):
                cleaned.append(m)
            elif isinstance(content, list):
                if any(b.get("type") == "text" for b in content):
                    cleaned.append(m)
        elif role == "assistant":
            if isinstance(content, str):
                cleaned.append(m)
            elif isinstance(content, list):
                if any(b.get("type") == "text" for b in content):
                    cleaned.append(m)
    return cleaned


async def _handle_extract(user_id: str, messages: list[dict]):
    """Worker 消费回调：执行提取管道"""
    filtered = _filter_for_mcp(messages)
    if len(filtered) < 4:
        logger.info("[Worker] 消息不足，跳过提取")
        return
    try:
        result = await extract_memories_via_mcp(filtered, f"auto_{user_id}", auto_commit=True)
        logger.info(f"[Worker] 提取完成: {result}")
    except Exception as e:
        logger.warning(f"[Worker] 提取失败: {e}")


# ── 检索规则预筛：轻量判断是否需要预取长期记忆 ──

RETRIEVAL_KEYWORDS = [
    "上次", "之前", "以前", "昨天", "上回", "过去",
    "还记得", "记不记得", "你记得", "回忆",
    "偏好", "喜欢", "爱好", "习惯",
]


def _should_prefetch(short_mem: list[dict], user_input: str) -> bool:
    """规则层：零成本判断是否需要预取长期记忆"""
    if not short_mem:
        return True  # 新用户/短期记忆过期
    if any(kw in user_input for kw in RETRIEVAL_KEYWORDS):
        return True
    return False


async def chat(user_id: str, user_input: str) -> dict:
    """一轮对话。返回: {"reply": str, "has_history": bool}"""

    # 1. 加载短期记忆
    messages = memory.load(user_id)
    has_history = len(messages) > 0

    logger.info(f"[记忆] 加载 {len(messages)} 条短期记忆 (user={user_id})")

    # ── 安全防护：输入检测 ──
    input_guard = get_input_guard()
    guard_result = await input_guard.check(user_input)
    if guard_result.blocked:
        logger.warning(f"[安全] 输入被拦截: {guard_result.reason}")
        return {
            "reply": f"输入被安全策略拦截（{guard_result.layer}层）。",
            "has_history": has_history,
        }

    # ── 检索规则预筛：符合条件自动预取长期记忆注入上下文 ──
    pre_memories = ""
    if _should_prefetch(messages, user_input):
        try:
            pre_memories = await search_memories_via_mcp(
                query=user_input, top_k=CFG["top_k_memories"]
            )
            if pre_memories and pre_memories.strip() != "暂无记忆":
                logger.info(f"[记忆] 规则预筛触发，注入长期记忆")
        except Exception as e:
            logger.warning(f"[记忆] 预取失败: {e}")

    # 2. 执行 Agent
    result = await run_chat(messages, user_input, injected_memories=pre_memories)

    # 3. 保存短期记忆
    full_messages = result["messages"]
    memory.save(user_id, full_messages)

    # 4. 判断是否需要后台提取（窗口溢出 / 超时），入队异步消费
    if extract_worker.should_extract(user_id, full_messages):
        logger.info("[提取] 触发条件满足，消息入队")
        new_msgs = extract_worker.get_new_messages(
            full_messages,
            extract_worker.get_cursor(user_id),
        )
        if new_msgs:
            extract_worker.enqueue(user_id, new_msgs)

    return {
        "reply": result["reply"],
        "has_history": has_history,
    }


async def extract_now(user_id: str, auto_commit: bool = True) -> dict:
    """立即提取当前短期记忆到长期记忆"""
    messages = memory.load(user_id)
    if not messages:
        logger.info("[提取] 短期记忆为空，跳过")
        return {"status": "empty", "pending": []}

    logger.info(f"[提取] 共 {len(messages)} 条消息")
    filtered = _filter_for_mcp(messages)

    try:
        result = await extract_memories_via_mcp(
            filtered, f"extract_{user_id}", auto_commit=auto_commit
        )
        logger.info(f"[提取] 完成: {result}")
        if result.startswith("PENDING_CONFLICTS:"):
            json_str = result[len("PENDING_CONFLICTS:"):].strip()
            return {"status": "pending", "pending": json.loads(json_str)}
        return {"status": "done", "pending": []}
    except Exception as e:
        logger.error(f"[提取] 失败: {e}")
        return {"status": "error", "pending": []}


# ── CLI ──

def print_banner():
    print("=" * 60)
    print("  Agent 记忆系统")
    print("  Agent ─MCP SSE─► MCP Server ─HTTP─► RAG Service")
    print("=" * 60)
    print()
    print("  /end  提取长期记忆")
    print("  /mem  查看所有长期记忆")
    print("  /quit 退出")
    print()


async def cli_loop():
    """异步 CLI 主循环"""
    print_banner()

    user_id = "demo_user"
    turn_count = 0

    # 先看有没有历史记忆
    recent = memory.load(user_id)
    if recent:
        print(f"[系统] 恢复最近的 {len(recent)} 条对话记录")
    print()

    while True:
        try:
            user_input = input(f"[{turn_count}] 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        cmd = user_input.lower().strip()

        if cmd == "/quit":
            print("再见！")
            break

        if cmd == "/end":
            print("[系统] 正在提取长期记忆...")
            result = await extract_now(user_id, auto_commit=False)
            if result["status"] == "pending" and result["pending"]:
                print(f"[系统] 检测到 {len(result['pending'])} 条冲突，请确认:")
                for i, p in enumerate(result["pending"]):
                    print(f"  [{i}] [{p.get('category','')}] {p.get('new_content','')}")
                    ans = input(f"      是否更新？Y/N/q: ").strip().lower()
                    if ans == "y":
                        from agent.agent import _mcp_session
                        await _mcp_session.call_tool("add_memory", {
                            "content": p.get("new_content", ""),
                            "category": p.get("category", "fact"),
                            "session_id": f"extract_{user_id}",
                        })
                        print(f"      已更新")
                    elif ans == "q":
                        break
            else:
                print("[系统] 长期记忆已归档")
            print()
            continue

        if cmd == "/mem":
            try:
                from agent.agent import _mcp_session
                result = await _mcp_session.call_tool("list_memories", {})
                texts = []
                for c in result.content:
                    if hasattr(c, "text"):
                        texts.append(c.text)
                print("  " + "\n  ".join("\n".join(texts).split("\n")))
            except Exception as e:
                print(f"  获取记忆失败: {e}")
            continue

        turn_count += 1
        try:
            result = await chat(user_id, user_input)
            reply = result['reply']

            # ── 安全防护：输出检测 ──
            output_guard = get_output_guard()
            out_result = await output_guard.check(reply)
            if out_result.blocked:
                logger.warning(f"[安全] 输出被拦截: {out_result.reason}")
                reply = f"回复被安全策略拦截（{out_result.layer}层）。"

            try:
                print(f"助手: {reply}")
            except UnicodeEncodeError:
                print(f"助手: {reply.encode('gbk', errors='replace').decode('gbk')}")
        except Exception as e:
            logger.error(f"[CLI] 对话失败: {e}")
            print(f"助手: 出错了... {e}")
        print()


async def main():
    """入口：在 MCP SSE 连接上下文内运行 CLI"""
    logger.info("=" * 50)
    logger.info(f"[Agent] 启动，MCP Server: {MCP_URL}")

    # 1. 初始化 ToolRegistry + 基础工具
    from agent.tool_registry import register, set_base_tools, init as init_registry
    from agent.agent import search_tools_tool
    from rag.embedder import Embedder

    shared = Embedder(local_files_only=True)
    _ = shared.dim
    init_registry(embedder=shared)
    base = [search_tools_tool]
    set_base_tools(base)

    # ── 初始化安全守卫 ──
    from agent.prompt import build_system_prompt
    init_guards(build_system_prompt())
    logger.info("安全守卫初始化完成")

    # 2. 健康检查：Redis
    try:
        memory.redis.ping()
        logger.info("Redis 连接正常")
    except Exception as e:
        logger.error(f"Redis 连接失败: {e}")
        print("请先启动 Redis")
        return

    # 3. 连接 MCP Server
    try:
        async with sse_client(MCP_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                set_mcp_session(session)

                tools_result = await session.list_tools()
                tool_names = [t.name for t in tools_result.tools]
                logger.info(f"[Agent] Memory MCP: {tool_names}")

                from langchain_mcp_adapters.client import load_mcp_tools
                mcp_tools = await load_mcp_tools(session)
                logger.info(f"[Agent] MCP adapter 加载: {[t.name for t in mcp_tools]}")

                # ── 记忆写入工具不给 Agent，由后台自动管理 ──
                AGENT_BLOCKED_TOOLS = {"add_memory", "extract_memories"}
                agent_mcp_tools = [
                    t for t in mcp_tools
                    if t.name not in AGENT_BLOCKED_TOOLS
                ]
                logger.info(
                    f"[Agent] 过滤后: {[t.name for t in agent_mcp_tools]}"
                    f" (已屏蔽: {AGENT_BLOCKED_TOOLS})"
                )

                from agent.tools.gaode import load_gaode_tools
                gaode_tools = await load_gaode_tools()
                if gaode_tools:
                    register(gaode_tools)

                all_tools = base + agent_mcp_tools + (gaode_tools or [])
                set_active_tools(all_tools)
                logger.info(f"[Agent] 工具总数: {len(all_tools)}")

                # ── 启动后台提取 Worker ──
                import asyncio as _asyncio
                worker_task = _asyncio.create_task(
                    extract_worker.consume(_handle_extract)
                )
                logger.info("[Worker] 后台消费已启动")

                await cli_loop()

                # 清理
                worker_task.cancel()
                try:
                    await worker_task
                except _asyncio.CancelledError:
                    pass
    except BaseExceptionGroup:
        pass
    except Exception as e:
        logger.error(f"无法连接 MCP Server: {e}")
        print(f"请先启动 MCP Server (python -m mcp_server.server)")
        return


if __name__ == "__main__":
    anyio.run(main)
