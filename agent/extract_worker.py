"""
后台记忆提取 Worker
===================
独立的后台消费进程，从 Redis 队列拉取提取任务，异步执行提取管道。
生产环境可替换为 Kafka/RabbitMQ，接口不变。

架构：
  chat() 触发 → 消息写入 Redis 队列 → Worker 消费 → 管道提取 → 落库
                                            ↑
  窗口裁剪触发 ──────────────────────────────┤
  定时触发 ─────────────────────────────────┘
"""

import json
import hashlib
import logging
import asyncio
from typing import Optional

import redis

from config import CFG

logger = logging.getLogger("agent.extract_worker")

EXTRACT_QUEUE = "extract_queue"  # Redis list key


class ExtractWorker:
    """
    后台记忆提取 Worker。

    生产替换指南：
      Redis Queue → Kafka Topic: 天然有序、支持消费者组、持久化
      asyncio.sleep → 独立进程 + supervisor: 进程级容灾
      单 Worker → 消费者组多实例: 水平扩展
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    # ── 游标管理 ──

    @staticmethod
    def _cursor_key(user_id: str) -> str:
        return f"extract_cursor:{user_id}"

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def get_cursor(self, user_id: str) -> Optional[dict]:
        raw = self.redis.get(self._cursor_key(user_id))
        return json.loads(raw) if raw else None

    def set_cursor(self, user_id: str, last_msg: dict):
        cursor = {
            "last_hash": self._content_hash(str(last_msg.get("content", ""))),
            "last_time": int(asyncio.get_event_loop().time() * 1000
                           if asyncio.get_event_loop().is_running()
                           else __import__("time").time() * 1000),
            "message_count": last_msg.get("_index", 0),
        }
        self.redis.set(self._cursor_key(user_id), json.dumps(cursor))

    def get_new_messages(self, messages: list[dict], cursor: Optional[dict]) -> list[dict]:
        """只返回游标之后的新消息，找不到游标则全量返回"""
        if not cursor or not messages:
            return messages

        target_hash = cursor.get("last_hash", "")
        for i, msg in enumerate(messages):
            if self._content_hash(str(msg.get("content", ""))) == target_hash:
                return messages[i + 1:]
        return messages  # 游标被裁掉了，全量兜底

    # ── 触发条件判断 ──

    @staticmethod
    def _count_turns(messages: list[dict]) -> int:
        return sum(1 for m in messages
                   if m.get("role") == "user"
                   and isinstance(m.get("content"), str))

    def should_extract(self, user_id: str, messages: list[dict],
                       max_turns: int = 10, max_minutes: int = 10) -> bool:
        """两个触发条件，任一满足即返回 True"""
        turns = self._count_turns(messages)

        # 条件 1：窗口溢出——旧消息要丢了
        if turns >= max_turns:
            return True

        # 条件 2：时间触发——太久没提取
        cursor = self.get_cursor(user_id)
        if cursor:
            import time
            elapsed = time.time() - cursor["last_time"] / 1000
            if elapsed > max_minutes * 60:
                return True

        return False

    # ── 队列操作 ──

    def enqueue(self, user_id: str, messages: list[dict]):
        """将提取任务推入 Redis 队列"""
        payload = json.dumps({
            "user_id": user_id,
            "messages": messages,
        }, ensure_ascii=False, default=str)
        self.redis.rpush(EXTRACT_QUEUE, payload)
        logger.info(f"[Worker] 任务入队: user={user_id}, {len(messages)} msgs")

    async def consume(self, handler):
        """
        消费循环：阻塞式轮询 Redis 队列，有任务就交给 handler 处理。
        生产环境可替换为 Kafka consumer。
        """
        logger.info("[Worker] 消费循环启动")
        while True:
            try:
                # BRPOP 阻塞等待，超时 5 秒后继续循环
                result = self.redis.blpop(EXTRACT_QUEUE, timeout=5)
                if result is None:
                    continue

                _, payload = result
                task = json.loads(payload)
                user_id = task["user_id"]
                messages = task["messages"]

                logger.info(f"[Worker] 消费任务: user={user_id}")
                await handler(user_id, messages)

                # 更新游标
                if messages:
                    last_msg = messages[-1]
                    last_msg["_index"] = len(messages) - 1
                    self.set_cursor(user_id, last_msg)

            except Exception as e:
                logger.error(f"[Worker] 消费异常: {e}")
                await asyncio.sleep(1)
