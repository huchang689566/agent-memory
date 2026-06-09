"""
Agent 系统 - CLI 入口
====================
独立的 Agent 进程，通过 MCP 协议与 MCP Server 通信。
不直接 import rag/ 的任何模块。
"""

import json
import logging
import sys
import uuid
import time
from typing import Optional

import anyio

# ── Windows CMD GBK 编码兼容：emoji 等字符用 ? 替代而非崩溃 ──
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding=sys.stdout.encoding, errors="replace"
    )

from config import CFG
from agent.session import SessionManager
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from security import init_guards, get_input_guard, get_output_guard

from agent.agent import (
    set_mcp_session,
    set_session_context,
    set_active_tools,
    run_chat,
    extract_memories_via_mcp,
    search_memories_via_mcp,
)

logger = logging.getLogger("agent.main")

session_mgr = SessionManager()

MCP_URL = f"http://{CFG['mcp_host']}:{CFG['mcp_port']}/sse"


def _filter_for_mcp(messages: list[dict]) -> list[dict]:
    """过滤消息，只保留 user text 和 assistant text，减小传输体积"""
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


async def chat(
    user_id: str,
    session_id: Optional[str],
    user_input: str,
) -> dict:
    """一轮对话。返回: {"reply": str, "session_id": str, "is_new_session": bool}"""

    # 1. 会话判断：Redis key 不存在 或 值为空 → 新对话
    is_new = False
    messages = []
    if session_id:
        messages = session_mgr.load(session_id)

    if not messages:
        # key 不存在或值为空 → 新对话
        is_new = True
        # ── 惰性提取：补上次未提取的会话 ──
        last_sid = session_mgr.get_last_session(user_id)
        extracted_sid = session_mgr.get_last_extracted(user_id)
        if last_sid and last_sid != extracted_sid:
            logger.info(
                f"[惰性提取] 发现未提取会话 {last_sid[:20]}...，正在补提取"
            )
            await end_session(user_id, last_sid)
        # ──────────────────────────────────
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        messages = []

    logger.info(
        f"[会话] {'新' if is_new else '续'} session={session_id[:20]}..."
    )

    # ── 安全防护：输入检测 ──
    input_guard = get_input_guard()
    guard_result = await input_guard.check(user_input)
    if guard_result.blocked:
        logger.warning(f"[安全] 输入被拦截: {guard_result.reason}")
        return {
            "reply": f"输入内容被安全策略拦截（{guard_result.layer}层）。如有疑问请联系管理员。",
            "session_id": session_id,
            "is_new_session": is_new,
        }

    # 2. 设置会话上下文（供 Agent 的 is_new_conversation 工具读取）
    set_session_context(is_new, session_id)

    # 3. 执行 Agent（system prompt 由 run_chat 根据 tools 自动生成）
    result = await run_chat(messages, user_input)

    # 5. 保存短期记忆 + 更新指针
    session_mgr.save(session_id, result["messages"])
    session_mgr.set_last_session(user_id, session_id)

    return {
        "reply": result["reply"],
        "session_id": session_id,
        "is_new_session": is_new,
    }


async def end_session(user_id: str, session_id: str, auto_commit: bool = True) -> dict:
    """结束会话，通过 MCP 触发长期记忆提取。返回 {"status": str, "pending": list}"""
    messages = session_mgr.load(session_id)
    if not messages:
        logger.info("[结束] 会话为空，跳过提取")
        session_mgr.set_last_extracted(user_id, session_id)
        return {"status": "empty", "pending": []}

    logger.info(f"[结束] 会话 {session_id[:20]}... 共 {len(messages)} 条消息")
    filtered = _filter_for_mcp(messages)

    try:
        result = await extract_memories_via_mcp(filtered, session_id, auto_commit=auto_commit)
        logger.info(f"[结束] 提取完成: {result}")

        # 检查是否有待审批的冲突
        pending = []
        if result.startswith("PENDING_CONFLICTS:"):
            json_str = result[len("PENDING_CONFLICTS:"):].strip()
            pending = json.loads(json_str)
            return {"status": "pending", "pending": pending}

        return {"status": "done", "pending": []}
    except Exception as e:
        logger.error(f"[结束] 提取失败: {e}")
        return {"status": "error", "pending": []}
    finally:
        session_mgr.set_last_extracted(user_id, session_id)


# ── CLI ──

def print_banner():
    print("=" * 60)
    print("  Agent 记忆系统 Demo（三系统独立架构）")
    print("  Agent ─MCP SSE─► MCP Server ─HTTP─► RAG Service")
    print("=" * 60)
    print()
    print("  /end  提取长期记忆（对话继续）")
    print("  /new  开始新对话（注入长期记忆库）")
    print("  /mem  查看所有长期记忆")
    print("  /quit 退出")
    print()


async def cli_loop():
    """异步 CLI 主循环。模拟聊天页面：打开=新会话，关闭=提取记忆"""
    print_banner()

    user_id = "demo_user"
    turn_count = 0

    # ── 打开聊天页面 → 全新会话 ──
    # 惰性提取：如果上次会话未提取，先补上
    last_sid = session_mgr.get_last_session(user_id)
    extracted_sid = session_mgr.get_last_extracted(user_id)
    if last_sid and last_sid != extracted_sid:
        print(f"[系统] 检测到上次对话未归档，正在提取记忆...")
        await end_session(user_id, last_sid)

    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    print(f"[系统] 新对话开始 ({session_id[:12]}...)")
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
            # 提取长期记忆，冲突时弹确认
            print("[系统] 正在提取长期记忆...")
            result = await end_session(user_id, session_id, auto_commit=False)
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
                            "session_id": session_id,
                        })
                        print(f"      已更新")
                    elif ans == "q":
                        break
            elif result["status"] == "done":
                print("[系统] 长期记忆已归档到向量库，当前对话继续")
            print()
            continue

        if cmd == "/new":
            # 提取长期记忆 → 开新对话
            await end_session(user_id, session_id)
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
            turn_count = 0
            print(f"[系统] 长期记忆已提取，新对话开始 ({session_id[:12]}...)")
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
            result = await chat(user_id, session_id, user_input)
            session_id = result["session_id"]
            if result["is_new_session"]:
                print("[系统] 新会话")
            reply = result['reply']

            # ── 安全防护：输出检测 ──
            output_guard = get_output_guard()
            out_result = await output_guard.check(reply)
            if out_result.blocked:
                logger.warning(f"[安全] 输出被拦截: {out_result.reason}")
                reply = f"回复内容被安全策略拦截（{out_result.layer}层）。"

            try:
                print(f"助手: {reply}")
            except UnicodeEncodeError:
                # Windows CMD GBK 编码无法显示某些字符（如 emoji）
                print(f"助手: {reply.encode('gbk', errors='replace').decode('gbk')}")
        except Exception as e:
            logger.error(f"[CLI] 对话失败: {e}")
            print(f"助手: 出错了... {e}")
        print()


async def main():
    """入口：在 MCP SSE 连接上下文内运行 CLI"""
    logger.info("=" * 50)
    logger.info(f"[Agent] 启动，MCP Server: {MCP_URL}")

    # 1. 初始化 ToolRegistry + 本地基础工具
    from agent.tool_registry import register, set_base_tools, init as init_registry
    from agent.agent import is_new_conversation, search_tools_tool
    from rag.embedder import Embedder

    shared = Embedder(local_files_only=True)
    _ = shared.dim
    init_registry(embedder=shared)
    base = [is_new_conversation, search_tools_tool]
    set_base_tools(base)

    # ── 初始化安全守卫（注入系统提示词作为输出侧锚点） ──
    from agent.prompt import build_system_prompt
    init_guards(build_system_prompt())
    logger.info("安全守卫初始化完成")

    # 2. 健康检查：Redis
    try:
        session_mgr.redis.ping()
        logger.info("Redis 连接正常")
    except Exception as e:
        logger.error(f"Redis 连接失败: {e}")
        print("请先启动 Redis")
        return

    # 3. 连接 MCP Server（连接生命周期覆盖整个 CLI）
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

                from agent.tools.gaode import load_gaode_tools
                gaode_tools = await load_gaode_tools()
                if gaode_tools:
                    register(gaode_tools)

                all_tools = base + mcp_tools + (gaode_tools or [])
                set_active_tools(all_tools)
                logger.info(f"[Agent] 工具总数: {len(all_tools)}")

                await cli_loop()
    except BaseExceptionGroup:
        pass
    except Exception as e:
        logger.error(f"无法连接 MCP Server: {e}")
        print(f"请先启动 MCP Server (python -m mcp_server.server)")
        return


if __name__ == "__main__":
    anyio.run(main)
