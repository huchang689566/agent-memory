"""
RAG 系统 - MemoryStore
=====================
FAISS 向量索引 + JSON 元数据文件。纯 Python，零外部框架依赖。
"""

import json
import logging
import numpy as np
import faiss
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import METADATA_PATH, FAISS_INDEX_PATH
from rag.embedder import Embedder

logger = logging.getLogger("rag.store")


class MemoryStore:
    """FAISS 向量索引 + JSON 元数据，提供 CRUD 操作"""

    def __init__(self, embedder: Embedder = None):
        self.embedder = embedder or Embedder()
        self.index: Optional[faiss.IndexIDMap] = None
        self.metadata: dict[str, dict] = {}
        self._next_id = 0
        self._load()

    # ── 持久化 ──

    def _load(self):
        if METADATA_PATH.exists():
            self.metadata = json.loads(
                METADATA_PATH.read_text(encoding="utf-8")
            )
            if self.metadata:
                self._next_id = (
                    max(int(k.split("_")[1]) for k in self.metadata.keys()) + 1
                )
        if FAISS_INDEX_PATH.exists():
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            logger.info(f"[Store] 加载 {self.index.ntotal} 条向量索引")
        if self.metadata:
            logger.info(f"[Store] 加载 {len(self.metadata)} 条元数据")

    def _save_metadata(self):
        payload = json.dumps(self.metadata, ensure_ascii=False, indent=2)
        tmp_path = METADATA_PATH.with_suffix(".tmp")
        # 写入临时文件 + fsync，然后原子替换
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            import os as _os
            _os.fsync(f.fileno())
        tmp_path.replace(METADATA_PATH)

    def _save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, str(FAISS_INDEX_PATH))
            import os as _os
            fd = _os.open(str(FAISS_INDEX_PATH), _os.O_RDWR)
            _os.fsync(fd)
            _os.close(fd)

    def _ensure_index(self, dim: int):
        if self.index is not None:
            return
        base_index = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap(base_index)
        logger.info(f"[Store] 创建新 FAISS 索引，维度: {dim}")

    # ── CRUD ──

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """搜索最相似的 top_k 条记忆"""
        if self.index is None or self.index.ntotal == 0:
            return []
        vec = self.embedder.embed([query])
        scores, ids = self.index.search(vec, min(top_k, self.index.ntotal))
        results = []
        for score, mem_id_int in zip(scores[0], ids[0]):
            if mem_id_int < 0:
                continue
            mem_id = f"mem_{mem_id_int}"
            if mem_id in self.metadata:
                results.append({**self.metadata[mem_id], "score": float(score)})
        return results

    def add(self, content: str, category: str, session_id: str = "") -> str:
        """添加新记忆，返回 mem_id"""
        dim = self.embedder.dim
        self._ensure_index(dim)

        mem_id = f"mem_{self._next_id}"
        self._next_id += 1

        vec = self.embedder.embed([content])
        mem_id_int = int(mem_id.split("_")[1])
        self.index.add_with_ids(vec, np.array([mem_id_int], dtype=np.int64))

        now = datetime.now(timezone.utc).isoformat()
        self.metadata[mem_id] = {
            "id": mem_id,
            "content": content,
            "category": category,
            "created_at": now,
            "updated_at": now,
            "session_id": session_id,
            "source": "user_stated",
            "stability": "tentative",
            "version": 1,
        }

        self._save_metadata()
        self._save_index()
        logger.info(f"[Store] 新增 {mem_id}: {content[:60]}...")
        return mem_id

    def update(self, mem_id: str, new_content: str) -> bool:
        """更新已有记忆的内容，重新 embedding"""
        if mem_id not in self.metadata:
            return False

        mem_id_int = int(mem_id.split("_")[1])
        self.index.remove_ids(np.array([mem_id_int], dtype=np.int64))

        vec = self.embedder.embed([new_content])
        self.index.add_with_ids(vec, np.array([mem_id_int], dtype=np.int64))

        self.metadata[mem_id]["content"] = new_content
        self.metadata[mem_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata[mem_id]["version"] += 1
        self.metadata[mem_id]["stability"] = "confirmed"

        self._save_metadata()
        self._save_index()
        logger.info(f"[Store] 更新 {mem_id}: {new_content[:60]}...")
        return True

    def delete(self, mem_id: str) -> bool:
        """删除记忆"""
        if mem_id not in self.metadata:
            return False
        mem_id_int = int(mem_id.split("_")[1])
        self.index.remove_ids(np.array([mem_id_int], dtype=np.int64))
        del self.metadata[mem_id]
        self._save_metadata()
        self._save_index()
        logger.info(f"[Store] 删除 {mem_id}")
        return True

    def get(self, mem_id: str) -> Optional[dict]:
        """获取单条记忆"""
        return self.metadata.get(mem_id)

    def get_all(self) -> list[dict]:
        """获取全部记忆"""
        return list(self.metadata.values())

    def __len__(self) -> int:
        return len(self.metadata)
