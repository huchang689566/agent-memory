"""
工具注册中心
============
独立 FAISS 索引（data/tool_index.faiss），与记忆索引物理隔离。
启动时一次性注册，运行时通过 search_tools 语义检索。
"""

import json
import logging
from pathlib import Path

import numpy as np
import faiss

from config import DATA_DIR

logger = logging.getLogger("agent.tool_registry")

# 独立的工具索引路径
TOOL_INDEX_PATH = DATA_DIR / "tool_index.faiss"
TOOL_META_PATH = DATA_DIR / "tool_metadata.json"

# 工具名 → 工具对象
_tool_map: dict[str, object] = {}

# 基础工具列表（始终加载，不进向量库）
_base_tools = []

# 自身的 embedder 和索引
_embedder = None
_index = None
_metadata = {}
_next_id = 0


def init(embedder=None):
    global _embedder, _index, _metadata, _next_id
    if embedder is not None:
        _embedder = embedder
    else:
        from rag.embedder import Embedder
        _embedder = Embedder()

    if TOOL_META_PATH.exists():
        _metadata = json.loads(TOOL_META_PATH.read_text(encoding="utf-8"))
        if _metadata:
            _next_id = max(int(k.split("_")[1]) for k in _metadata.keys()) + 1

    if TOOL_INDEX_PATH.exists():
        _index = faiss.read_index(str(TOOL_INDEX_PATH))

    logger.info(f"[ToolRegistry] 已初始化，{len(_metadata)} 条工具定义")


def _ensure_index(dim: int):
    global _index
    if _index is not None:
        return
    base = faiss.IndexFlatIP(dim)
    _index = faiss.IndexIDMap(base)


def _save():
    if _index is not None:
        faiss.write_index(_index, str(TOOL_INDEX_PATH))
    TOOL_META_PATH.write_text(json.dumps(_metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def register(tools: list):
    """注册工具列表。同名跳过。仅启动时调用。"""
    global _tool_map, _next_id

    if _embedder is None:
        init()

    for t in tools:
        if t.name in _tool_map:
            continue

        dim = _embedder.dim
        _ensure_index(dim)

        tid = f"t_{_next_id}"
        _next_id += 1

        content = f"{t.name}: {t.description}"
        vec = _embedder.embed([content])
        tid_int = int(tid.split("_")[1])
        _index.add_with_ids(vec, np.array([tid_int], dtype=np.int64))

        _metadata[tid] = {"id": tid, "content": content, "tool_name": t.name}
        _tool_map[t.name] = t
        logger.info(f"[ToolRegistry] 注册: {t.name}")

    _save()
    logger.info(f"[ToolRegistry] 共 {len(_tool_map)} 个工具")


def set_base_tools(tools: list):
    global _base_tools
    _base_tools = list(tools)


def search(query: str, top_k: int = 5) -> list:
    """语义检索工具，返回工具对象列表"""
    global _embedder, _index, _metadata, _tool_map

    if _index is None or _index.ntotal == 0:
        return []

    vec = _embedder.embed([query])
    fetch_k = min(top_k, _index.ntotal)
    scores, ids = _index.search(vec, fetch_k)

    result = []
    for score, tid_int in zip(scores[0], ids[0]):
        if tid_int < 0:
            continue
        tid = f"t_{tid_int}"
        if tid in _metadata:
            name = _metadata[tid]["tool_name"]
            if name in _tool_map:
                result.append(_tool_map[name])

    return result


def search_text(query: str, top_k: int = 5) -> str:
    """语义检索工具，返回可读文本"""
    results = search(query, top_k=top_k)
    if not results:
        return "未找到相关工具。"
    lines = [f"找到 {len(results)} 个相关工具:"]
    for t in results:
        lines.append(f"  - {t.name}: {t.description}")
    return "\n".join(lines)
