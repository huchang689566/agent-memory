"""
文档预处理工具 —— 一行调用，输入 mineru JSON，输出带页码的语义块
"""

import re
import numpy as np
from typing import Optional


def process_mineru_output(
    mineru_result: dict,
    embedder,
    max_chunk_size: int = 800,
    min_chunk_size: int = 100,
) -> list[dict]:
    """
    输入 mineru 完整输出（blocks + 顶层元数据），输出语义合并后的 chunks。

    每项 chunk: {
        "content": str,            # 文本
        "pages": [int, ...],       # 原始页码
        "source": str,             # 文件名
        "char_count": int,         # 字符数
    }
    """

    # ── 1. 展平 blocks（带页码标签） ──
    flat = _flatten_blocks(mineru_result.get("blocks", []))
    if not flat:
        return []

    # ── 2. 语义分块（在带页码标签的句子列表上操作） ──
    chunks = _semantic_chunk_with_pages(flat, embedder, max_chunk_size, min_chunk_size)

    # ── 3. 组装结果 ──
    source = mineru_result.get("title", "unknown")
    result = []
    for content, pages in chunks:
        result.append({
            "content": content.strip(),
            "pages": sorted(set(pages)),
            "source": source,
            "char_count": len(content.strip()),
        })
    return result


# ═══════════════════════════════════════
# 内部函数
# ═══════════════════════════════════════

def _flatten_blocks(blocks: list[dict]) -> list[dict]:
    """将 block 展平为统一格式：{text, page}"""
    flat = []
    for b in blocks:
        t = b.get("type")
        page = b.get("page_idx", 0)

        if t == "text":
            flat.append({"text": b.get("text", ""), "page": page})

        elif t == "image":
            parts = []
            cap = b.get("img_caption", []) or []
            if cap:
                cap_text = " ".join(cap) if isinstance(cap, list) else str(cap)
                parts.append(cap_text)
            fn = b.get("img_footnote", []) or []
            if fn:
                fn_text = " ".join(fn) if isinstance(fn, list) else str(fn)
                parts.append(f"({fn_text})")
            flat.append({"text": f"[图片: {'; '.join(parts)}]" if parts else "[图片]", "page": page})

        elif t == "table":
            parts = []
            cap = b.get("table_caption", "") or ""
            if cap:
                parts.append(cap)
            body = b.get("table_body", []) or []
            if body:
                rows = [" | ".join(row) for row in body]
                parts.append("; ".join(rows))
            fn = b.get("table_footnote", []) or []
            if fn:
                fn_text = " ".join(fn) if isinstance(fn, list) else str(fn)
                parts.append(f"({fn_text})")
            flat.append({"text": f"[表格: {' / '.join(parts)}]" if parts else "[表格]", "page": page})

    return flat


def _semantic_chunk_with_pages(
    flat_blocks: list[dict], embedder, max_size: int, min_size: int
) -> list[tuple[str, list[int]]]:
    """
    在带 page 标签的文本块上做语义合并。
    返回 [(text, [pages]), ...]
    """
    # 先把每个 block 拆成句子，保留页码
    sentences = []   # [(text, page), ...]
    for item in flat_blocks:
        for sent in _split_sentences(item["text"]):
            sentences.append((sent, item["page"]))

    if not sentences:
        return []
    if len(sentences) == 1:
        return [(sentences[0][0], [sentences[0][1]])]

    # embedding
    texts = [s[0] for s in sentences]
    vecs = embedder.embed(texts)

    # 相邻句子相似度
    similarities = []
    for i in range(1, len(vecs)):
        sim = float(np.dot(vecs[i], vecs[i - 1]))
        similarities.append(sim)

    # 动态阈值
    threshold = float(np.mean(similarities) - 0.5 * np.std(similarities)) if similarities else 0.5

    # 合并
    chunks = []
    current_texts = [sentences[0][0]]
    current_pages = {sentences[0][1]}
    current_len = len(sentences[0][0])

    for i in range(1, len(sentences)):
        sim = similarities[i - 1]
        text, page = sentences[i]

        should_split = False
        if sim < threshold and current_len >= min_size:
            should_split = True
        elif current_len + len(text) > max_size and current_len >= min_size:
            should_split = True

        if should_split:
            chunks.append((" ".join(current_texts), list(current_pages)))
            current_texts = [text]
            current_pages = {page}
            current_len = len(text)
        else:
            current_texts.append(text)
            current_pages.add(page)
            current_len += len(text)

    if current_texts:
        chunks.append((" ".join(current_texts), list(current_pages)))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """按中英文标点拆句子"""
    raw = re.split(r'(?<=[。！？；\n])', text)
    return [s.strip() for s in raw if s.strip() and len(s.strip()) > 1]


# ═══════════════════════════════════════
# 批量处理（Dify 多文件场景）
# ═══════════════════════════════════════

def process_batch(
    mineru_results: list[dict],
    embedder,
    max_chunk_size: int = 800,
    min_chunk_size: int = 100,
) -> list[dict]:
    """输入 mineru 返回的多个文件的 JSON 列表，输出全部文件的合并 chunks"""
    all_chunks = []
    for doc in mineru_results:
        chunks = process_mineru_output(doc, embedder, max_chunk_size, min_chunk_size)
        all_chunks.extend(chunks)
    return all_chunks


# ═══════════════════════════════════════
# 测试
# ═══════════════════════════════════════
if __name__ == "__main__":
    sample = {
        "title": "系统设计文档.pdf",
        "blocks": [
            {"type": "text", "text": "系统设计概述", "page_idx": 0},
            {"type": "text", "text": "本系统采用三层架构。RAG负责向量检索，Agent负责推理决策。", "page_idx": 0},
            {"type": "image", "img_caption": ["架构图", "三层示意"], "page_idx": 0},
            {"type": "text", "text": "部署时需注意内存分配，建议最低8GB。", "page_idx": 1},
            {"type": "text", "text": "Redis配置主从模式，建议开启AOF持久化。", "page_idx": 1},
            {"type": "table", "table_caption": "端口配置", "table_body": [
                ["服务", "端口", "用途"], ["RAG", "8002", "向量检索"], ["MCP", "8001", "工具网关"]
            ], "page_idx": 2},
            {"type": "text", "text": "安全方面需要配置防火墙规则和API鉴权。", "page_idx": 2},
        ]
    }

    import sys; sys.path.insert(0, '.')
    from rag.embedder import Embedder
    e = Embedder(local_files_only=True)
    _ = e.dim  # 预加载

    chunks = process_mineru_output(sample, embedder=e)
    for i, c in enumerate(chunks):
        print(f"--- Chunk {i} | pages: {c['pages']} | chars: {c['char_count']} ---")
        print(c["content"][:200])
        print()
