"""
RAG 系统 - FastAPI REST 服务
============================
独立进程，暴露检索和写入接口。
端口: 8002
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from config import CFG
from rag.embedder import Embedder
from rag.store import MemoryStore
from rag.extractor import MemoryExtractor

logger = logging.getLogger("rag.server")

# ── 全局实例 ──
embedder: Embedder = None
store: MemoryStore = None
extractor: MemoryExtractor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, store, extractor
    # 禁止 HF 每次验证缓存
    import os
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    embedder = Embedder()
    store = MemoryStore(embedder=embedder)
    extractor = MemoryExtractor(store=store, embedder=embedder)

    # 预加载 embedding 模型，避免首次搜索卡顿
    _ = embedder.dim
    logger.info(f"[RAG] 就绪，{len(store)} 条记忆，{embedder.dim} 维向量")
    yield
    logger.info("[RAG Server] 关闭")


app = FastAPI(
    title="RAG Memory Service",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Request/Response Models ──

class AddMemoryRequest(BaseModel):
    content: str
    category: str
    session_id: str = ""


class UpdateMemoryRequest(BaseModel):
    mem_id: str
    new_content: str


class ExtractRequest(BaseModel):
    messages: list[dict]
    session_id: str = ""
    auto_commit: bool = True   # True=全部自动写入，False=冲突项返回待确认


# ── API Routes ──

@app.get("/health")
def health():
    """健康检查"""
    return {
        "status": "ok",
        "total_memories": len(store) if store else 0,
        "embed_dim": embedder.dim if embedder else None,
    }


@app.get("/search")
def search_memory(
    query: str = Query(..., description="搜索查询"),
    top_k: int = Query(None, description="返回结果数量，默认使用配置值"),
):
    """搜索最相似的历史记忆"""
    k = top_k or CFG["top_k_memories"]
    results = store.search(query, top_k=k)
    return {"results": results, "count": len(results)}


@app.post("/add")
def add_memory(req: AddMemoryRequest):
    """添加一条新记忆"""
    if not req.content.strip():
        raise HTTPException(400, "content 不能为空")
    mem_id = store.add(req.content, req.category, req.session_id)
    return {"mem_id": mem_id, "status": "created"}


@app.put("/update")
def update_memory(req: UpdateMemoryRequest):
    """更新已有记忆"""
    ok = store.update(req.mem_id, req.new_content)
    if not ok:
        raise HTTPException(404, f"记忆 {req.mem_id} 不存在")
    return {"mem_id": req.mem_id, "status": "updated"}


@app.delete("/delete/{mem_id}")
def delete_memory(mem_id: str):
    """删除一条记忆"""
    ok = store.delete(mem_id)
    if not ok:
        raise HTTPException(404, f"记忆 {mem_id} 不存在")
    return {"mem_id": mem_id, "status": "deleted"}


@app.get("/memories")
def list_memories():
    """获取所有长期记忆"""
    memories = store.get_all()
    return {"memories": memories, "count": len(memories)}


@app.post("/extract")
def extract_memories(req: ExtractRequest):
    """从对话中提取长期记忆（完整管道：候选 → 去重 → 冲突 → 提交）"""
    if not req.messages:
        raise HTTPException(400, "messages 不能为空")
    result = extractor.run(req.messages, req.session_id, auto_commit=req.auto_commit)
    return {"status": "done", **result}


# ── 启动入口 ──

if __name__ == "__main__":
    import uvicorn
    logger.info(f"[RAG Server] 启动在 {CFG['rag_host']}:{CFG['rag_port']}")
    uvicorn.run(
        "rag.server:app",
        host=CFG["rag_host"],
        port=CFG["rag_port"],
        log_level="info",
    )
