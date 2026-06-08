"""
Agent 系统 - SessionManager
===========================
短期记忆管理：Redis 热数据 + 磁盘 JSON 兜底，滑动窗口按轮裁剪。
Agent 自己管理短期记忆，不经过 MCP/RAG。
"""

import json
import logging
from pathlib import Path
from typing import Optional

import redis

from config import CFG, SESSIONS_DIR

logger = logging.getLogger("agent.session")


class SessionManager:
    """短期记忆：Redis + 磁盘双写，滑动窗口按轮裁剪"""

    def __init__(self):
        self.redis = redis.Redis(
            host=CFG["redis_host"],
            port=CFG["redis_port"],
            db=CFG["redis_db"],
            decode_responses=True,
        )
        self.sessions_dir = SESSIONS_DIR
        self.sessions_dir.mkdir(exist_ok=True)

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def _disk_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    # ── 消息轮次判断 ──

    @staticmethod
    def is_new_turn(msg: dict) -> bool:
        """user 消息里包含 text 类型内容 → 新轮次起点"""
        if msg.get("role") != "user":
            return False
        content = msg.get("content")
        if isinstance(content, str):
            return True
        if isinstance(content, list):
            return any(block.get("type") == "text" for block in content)
        return False

    @staticmethod
    def count_turns(messages: list[dict]) -> int:
        return sum(1 for m in messages if SessionManager.is_new_turn(m))

    # ── 读写 ──

    def load(self, session_id: str) -> list[dict]:
        """优先 Redis → 回源磁盘"""
        raw = self.redis.get(self._key(session_id))
        if raw:
            return json.loads(raw)
        disk_path = self._disk_path(session_id)
        if disk_path.exists():
            logger.info(
                f"[Session] Redis 未命中，从磁盘加载 {session_id[:20]}..."
            )
            return json.loads(disk_path.read_text(encoding="utf-8"))
        return []

    def save(self, session_id: str, messages: list[dict], max_turns: int = None):
        if max_turns is None:
            max_turns = CFG["max_turns"]
        messages = self._trim(messages, max_turns)
        payload = json.dumps(messages, ensure_ascii=False, default=str)
        # 双写：Redis + 磁盘
        self.redis.setex(self._key(session_id), 3600, payload)
        self._disk_path(session_id).write_text(payload, encoding="utf-8")

    def _trim(self, messages: list[dict], max_turns: int) -> list[dict]:
        turn_indices = [i for i, m in enumerate(messages) if self.is_new_turn(m)]
        if len(turn_indices) <= max_turns:
            return messages
        cut_from = turn_indices[-max_turns]
        return messages[cut_from:]

    def exists(self, session_id: str) -> bool:
        if self.redis.exists(self._key(session_id)):
            return True
        return self._disk_path(session_id).exists()

    # ── 会话生命周期指针 ──

    def _last_session_key(self, user_id: str) -> str:
        return f"user:{user_id}:last_session"

    def _last_extracted_key(self, user_id: str) -> str:
        return f"user:{user_id}:last_extracted"

    def set_last_session(self, user_id: str, session_id: str):
        self.redis.set(self._last_session_key(user_id), session_id)

    def get_last_session(self, user_id: str) -> Optional[str]:
        val = self.redis.get(self._last_session_key(user_id))
        return val if val else None

    def set_last_extracted(self, user_id: str, session_id: str):
        self.redis.set(self._last_extracted_key(user_id), session_id)

    def get_last_extracted(self, user_id: str) -> Optional[str]:
        val = self.redis.get(self._last_extracted_key(user_id))
        return val if val else None
