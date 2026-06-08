"""
RAG 系统 - MemoryExtractor
==========================
从对话中提取长期记忆：候选提取 → 向量去重 → 冲突判断 → 提交。
纯 Python，通过依赖注入使用 Embedder + MemoryStore + LLM client。
"""

import json
import logging
import numpy as np
from typing import Optional

from openai import OpenAI

from config import CFG

logger = logging.getLogger("rag.extractor")

# ── Prompt 模板 ──

EXTRACTION_PROMPT = """从以下对话中提取需要长期记住的用户信息。

规则:
1. 只提取**新信息**——偏好、决策、事实、用户背景
2. 不要提取闲聊内容（问候、随口评论）
3. 每条记忆标注类别: preference（偏好）、fact（事实/信息）、decision（决策）
4. 如果对话没有值得记住的信息，返回空数组 []

对话:
{dialogue}

返回 JSON 数组（不要其他文字）:
[{{"content": "...", "category": "preference|fact|decision"}}]"""


CONFLICT_PROMPT = """判断新信息与已有记忆的关系。

已有记忆:
{existing}

新信息: "{new_info}"

返回 JSON（不要其他文字）:
- 重复（只是换个说法）-> {{"action": "duplicate"}}
- 完全新增（之前没有相关信息）-> {{"action": "add"}}
- 冲突/更新（和已有记忆矛盾）-> {{"action": "update", "target_id": "mem_xxx", "new_content": "修正后的内容"}}"""


class MemoryExtractor:
    """从对话中提取、去重、解决冲突，然后提交给 MemoryStore"""

    def __init__(self, store, embedder, llm_client: OpenAI = None):
        self.store = store          # MemoryStore 实例
        self.embedder = embedder    # Embedder 实例
        self.llm = llm_client or OpenAI(
            base_url=CFG["base_url"],
            api_key=CFG["api_key"],
        )

    # ── 工具方法 ──

    @staticmethod
    def _filter_messages(messages: list[dict]) -> list[dict]:
        """只保留 user text 和 assistant text，跳过 tool_use/tool_result"""
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

    @staticmethod
    def _extract_text(msg: dict) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(parts)
        return str(content)

    # ── 提取管道 ──

    def extract_candidates(self, messages: list[dict]) -> list[dict]:
        """调用 LLM 从对话中提取候选长期记忆"""
        filtered = self._filter_messages(messages)
        if len(filtered) < 2:
            return []

        dialogue = "\n".join(
            f"{m['role']}: {self._extract_text(m)}" for m in filtered
        )
        try:
            resp = self.llm.chat.completions.create(
                model=CFG["llm_model"],
                messages=[{
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(dialogue=dialogue),
                }],
                temperature=0.1,
            )
            text = resp.choices[0].message.content.strip()
            # 处理 markdown 代码块包裹
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
            candidates = json.loads(text)
            logger.info(f"[Extractor] 候选记忆 {len(candidates)} 条")
            return candidates
        except Exception as e:
            logger.warning(f"[Extractor] 提取失败: {e}")
            return []

    def deduplicate(self, candidates: list[dict]) -> list[dict]:
        """向量相似度去重：跳过与已有记忆高度相似的候选"""
        existing = self.store.get_all()
        if not existing:
            return candidates

        existing_contents = [m["content"] for m in existing]
        existing_vectors = self.embedder.embed(existing_contents)

        threshold = CFG["duplicate_threshold"]
        kept = []
        for c in candidates:
            cand_vec = self.embedder.embed([c["content"]])
            scores = np.dot(cand_vec, existing_vectors.T)[0]
            max_score = float(scores.max()) if len(scores) > 0 else 0
            if max_score > threshold:
                logger.info(
                    f"[Extractor] 跳过重复: {c['content'][:60]}... ({max_score:.3f})"
                )
            else:
                kept.append(c)
        return kept

    def check_conflicts(self, candidates: list[dict], session_id: str) -> list[dict]:
        """对每条候选，在同类别 top 3 中判断冲突"""
        actions = []
        all_memories = self.store.get_all()

        for c in candidates:
            same_cat = [m for m in all_memories if m["category"] == c["category"]]
            if not same_cat:
                actions.append({
                    "action": "add",
                    "content": c["content"],
                    "category": c["category"],
                })
                continue

            cand_vec = self.embedder.embed([c["content"]])
            cat_vecs = self.embedder.embed([m["content"] for m in same_cat])
            scores = np.dot(cand_vec, cat_vecs.T)[0]
            top_indices = np.argsort(scores)[-3:][::-1]
            top_memories = [same_cat[i] for i in top_indices if scores[i] > 0.5]

            if not top_memories:
                actions.append({
                    "action": "add",
                    "content": c["content"],
                    "category": c["category"],
                })
                continue

            existing_text = "\n".join(
                f"[{m['id']}] {m['content']}" for m in top_memories
            )
            try:
                resp = self.llm.chat.completions.create(
                    model=CFG["llm_model"],
                    messages=[{
                        "role": "user",
                        "content": CONFLICT_PROMPT.format(
                            existing=existing_text, new_info=c["content"],
                        ),
                    }],
                    temperature=0.1,
                )
                text = resp.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    if text.endswith("```"):
                        text = text[:-3]
                result = json.loads(text)
                logger.info(
                    f"[Extractor] {c['content'][:40]}... -> {result['action']}"
                )
                # 保证 content 始终存在（LLM 返回的 add action 可能不带 content）
                actions.append({
                    "content": c["content"],
                    **result,
                    "category": c["category"],
                })
            except Exception as e:
                logger.warning(f"[Extractor] 冲突判断失败: {e}，默认新增")
                actions.append({
                    "action": "add",
                    "content": c["content"],
                    "category": c["category"],
                })

        return actions

    def commit(self, actions: list[dict], session_id: str, auto: bool = True) -> dict:
        """执行 add/update。auto=True 全部自动写入，auto=False 冲突项返回待确认。"""
        pending = []
        for a in actions:
            action = a["action"]
            if action == "add":
                self.store.add(a["content"], a["category"], session_id)
            elif action == "update":
                if auto:
                    self.store.update(a["target_id"], a["new_content"])
                else:
                    # 冲突更新不自动写，返回给用户确认
                    pending.append(a)
        return {"committed": len(actions) - len(pending), "pending": pending}

    def run(self, messages: list[dict], session_id: str, auto_commit: bool = True) -> dict:
        """
        完整提取管道：候选 → 去重 → 冲突 → 提交。

        auto_commit=True: 全部自动写入（默认行为）
        auto_commit=False: 冲突项返回待确认，不自动写
        返回: {"committed": int, "pending": [{"action":"update", ...}]}
        """
        logger.info("=" * 50)
        logger.info("[Extractor] 开始长期记忆提取...")

        candidates = self.extract_candidates(messages)
        if not candidates:
            logger.info("[Extractor] 无候选，跳过")
            return {"committed": 0, "pending": []}

        fresh = self.deduplicate(candidates)
        if not fresh:
            logger.info("[Extractor] 全部重复，跳过")
            return {"committed": 0, "pending": []}

        actions = self.check_conflicts(fresh, session_id)
        result = self.commit(actions, session_id, auto=auto_commit)

        all_mems = self.store.get_all()
        logger.info(f"[Extractor] 当前共 {len(all_mems)} 条记忆")
        for m in all_mems:
            logger.info(f"  [{m['category']}] {m['content']}")
        return result
