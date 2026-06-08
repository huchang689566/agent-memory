"""
一键启动三个系统（开发用）
========================
启动顺序: RAG → MCP → Agent
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

ROOT = Path(__file__).parent

processes = []


def start_rag():
    print("[启动] RAG Service (port 8002)...")
    p = subprocess.Popen(
        [sys.executable, "-m", "rag.server"],
        cwd=ROOT,
    )
    processes.append(("RAG", p))
    time.sleep(2)  # 等 RAG 初始化
    print("[启动] RAG Service 已启动")


def start_mcp():
    print("[启动] MCP Server (port 8001)...")
    p = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        cwd=ROOT,
    )
    processes.append(("MCP", p))
    time.sleep(2)  # 等 MCP 初始化
    print("[启动] MCP Server 已启动")


def start_agent():
    print("[启动] Agent CLI...")
    print("=" * 60)
    p = subprocess.Popen(
        [sys.executable, "-m", "agent.main"],
        cwd=ROOT,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    processes.append(("Agent", p))
    # Agent 是交互式的，等它自己结束
    p.wait()


def cleanup():
    print("\n[清理] 关闭所有服务...")
    for name, p in reversed(processes):
        if p.poll() is None:
            print(f"[清理] 关闭 {name}...")
            # Windows: 用 CTRL_BREAK_EVENT 优雅关闭，给进程 fsync 的机会
            if sys.platform == "win32":
                try:
                    import signal as _sig
                    p.send_signal(_sig.CTRL_BREAK_EVENT)
                    p.wait(timeout=5)
                except Exception:
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        p.kill()
            else:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
    print("[清理] 完成")


def main():
    signal.signal(signal.SIGINT, lambda s, f: cleanup())
    signal.signal(signal.SIGTERM, lambda s, f: cleanup())

    try:
        start_rag()
        start_mcp()
        start_agent()
    except Exception as e:
        print(f"[错误] 启动失败: {e}")
        cleanup()
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
