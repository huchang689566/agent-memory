"""
Agent 系统 - ShortMemory
=======================
短期记忆：以 user_id 为 key，Redis 热数据 + 磁盘双写，滑动窗口保留最近 N 轮。
无 session 概念——上下线自然管理，Agent 自主决定是否检索长期记忆。
"""

import json
import logging
from pathlib import Path
from typing import Optional

import redis

from config import CFG, SESSIONS_DIR

logger = logging.getLogger("agent.session")


class ShortMemory:
    """短期记忆：user_id 维度，滑动窗口 + TTL 自动过期"""

    # TTL：30 分钟不说话自动清空，下次回来 Agent 自行判断是否检索长期记忆
    TTL_SECONDS = 1800

    def __init__(self):
        self.redis = redis.Redis(
            host=CFG["redis_host"],
            port=CFG["redis_port"],
            db=CFG["redis_db"],
            decode_responses=True,
        )
        self.disk_dir = SESSIONS_DIR
        self.disk_dir.mkdir(exist_ok=True)
        self.max_turns = CFG["max_turns"]

    def _key(self, user_id: str) -> str:
        return f"short_mem:{user_id}"

    def _disk_path(self, user_id: str) -> Path:
        return self.disk_dir / f"{user_id}.json"

    # ── 消息轮次判断 ──

    @staticmethod
    def is_new_turn(msg: dict) -> bool:
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
        return sum(1 for m in messages if ShortMemory.is_new_turn(m))

    # ── 读写 ──

    def load(self, user_id: str) -> list[dict]:
        """优先 Redis → 回源磁盘"""
        raw = self.redis.get(self._key(user_id))
        if raw:
            return json.loads(raw)
        disk_path = self._disk_path(user_id)
        if disk_path.exists():
            logger.info(f"[Memory] Redis 未命中，从磁盘加载 user={user_id}")
            return json.loads(disk_path.read_text(encoding="utf-8"))
        return []

    def save(self, user_id: str, messages: list[dict]):
        messages = self._trim(messages)
        payload = json.dumps(messages, ensure_ascii=False, default=str)
        self.redis.setex(self._key(user_id), self.TTL_SECONDS, payload)
        self._disk_path(user_id).write_text(payload, encoding="utf-8")

    def _trim(self, messages: list[dict]) -> list[dict]:
        turn_indices = [i for i, m in enumerate(messages) if self.is_new_turn(m)]
        if len(turn_indices) <= self.max_turns:
            return messages
        return messages[turn_indices[-self.max_turns]:]
