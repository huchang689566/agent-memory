"""
Marker 文档处理工具
==================
输入 PDF/Word/图片文件路径 → marker 转 Markdown → 语义分块 → 返回带元数据的 chunks

用法:
    from utils.marker_processor import process_with_marker
    from rag.embedder import Embedder

    e = Embedder(local_files_only=True); _ = e.dim
    chunks = process_with_marker("doc.pdf", embedder=e)
    for c in chunks:
        print(c["content"], c["metadata"])
"""

import re
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("marker_processor")


def process_with_marker(
    file_path: str,
    embedder,
    max_chunk_size: int = 800,
    min_chunk_size: int = 100,
) -> list[dict]:
    """
    输入文件路径，输出语义合并后的 chunks。

    每项: {
        "content": str,
        "metadata": {"source": str, "section": str}  # section 可能为空
        "char_count": int,
    }
    """

    # ── 1. marker 转换 ──
    markdown_text = _convert_to_markdown(file_path)
    if not markdown_text or not markdown_text.strip():
        logger.warning(f"[MarkerProcessor] {file_path} 转换结果为空")
        return []

    # ── 2. 按 # 标题边界预切（不管层级对不对，就切个大概） ──
    sections = _split_by_header_boundary(markdown_text)

    # ── 3. 每个区域内语义合并 ──
    source = Path(file_path).name
    chunks = []
    for section_title, section_text in sections:
        sub_chunks = _semantic_chunk(embedder, section_text, max_chunk_size, min_chunk_size)
        for sc in sub_chunks:
            if sc.strip():
                chunks.append({
                    "content": sc.strip(),
                    "metadata": {"source": source, "section": section_title or ""},
                    "char_count": len(sc.strip()),
                })
    return chunks


# ═══════════════════════════════════════
# 内部函数
# ═══════════════════════════════════════

def _convert_to_markdown(file_path: str) -> str:
    """用 marker 将文件转为 Markdown"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    try:
        if suffix == ".pdf":
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            converter = PdfConverter(artifact_dict=create_model_dict())
            rendered = converter(str(path))
            return rendered.markdown
        elif suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
            from marker.converters.image import ImageConverter
            converter = ImageConverter()
            rendered = converter(str(path))
            return rendered.markdown
        elif suffix in (".docx", ".doc", ".pptx"):
            # 非 PDF/图片格式先尝试 markitdown 转，再走语义分块
            try:
                from markitdown import MarkItDown
                md = MarkItDown()
                result = md.convert(str(path))
                return result.text_content
            except ImportError:
                logger.warning("markitdown 未安装，跳过转换")
                return ""
        elif suffix in (".txt", ".md"):
            return Path(file_path).read_text(encoding="utf-8")
        else:
            logger.warning(f"不支持的文件格式: {suffix}")
            return ""
    except Exception as e:
        logger.error(f"marker 转换失败: {e}")
        return ""


def _split_by_header_boundary(text: str) -> list[tuple[str, str]]:
    """
    按 # / ## / ### 边界切开。
    返回 [(标题, 正文), ...]。没有标题时返回 [("", 全篇)]。
    """
    # 匹配标题行
    header_pattern = re.compile(r'^#{1,3}\s+(.+)$', re.MULTILINE)
    matches = list(header_pattern.finditer(text))

    if not matches:
        return [("", text)]

    sections = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((title, body))

    # 标题之前的引言部分
    if matches and matches[0].start() > 0:
        intro = text[:matches[0].start()].strip()
        if intro:
            sections.insert(0, ("", intro))

    return sections


def _semantic_chunk(
    embedder, text: str, max_size: int, min_size: int
) -> list[str]:
    """语义分块"""
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [text.strip()] if text.strip() else []

    vecs = embedder.embed(sentences)

    similarities = []
    for i in range(1, len(vecs)):
        similarities.append(float(np.dot(vecs[i], vecs[i - 1])))

    if not similarities:
        return [text.strip()] if text.strip() else []

    threshold = float(np.mean(similarities) - 0.5 * np.std(similarities))

    chunks = []
    current = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = similarities[i - 1]
        sent_len = len(sentences[i])

        should_split = (
            (sim < threshold and current_len >= min_size)
            or (current_len + sent_len > max_size and current_len >= min_size)
        )

        if should_split:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_len = sent_len
        else:
            current.append(sentences[i])
            current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[。！？；\n])', text)
    return [s.strip() for s in raw if s.strip() and len(s.strip()) > 1]


# ═══════════════════════════════════════
# 测试
# ═══════════════════════════════════════
if __name__ == "__main__":
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from rag.embedder import Embedder

    sample_md = """
# 系统设计

本系统采用三层架构。RAG负责向量检索，Agent负责推理决策。

## 部署方案

部署时需注意内存分配，建议最低8GB。
Redis配置主从模式，建议开启AOF持久化。

安全方面需要配置防火墙规则和API鉴权。
这跟之前的架构部分没有直接关联。
"""
    e = Embedder(local_files_only=True)
    _ = e.dim

    # 模拟输入：把 Markdown 写入临时文件
    tmp = Path(__file__).parent / "_test_sample.md"
    tmp.write_text(sample_md, encoding="utf-8")

    chunks = process_with_marker(str(tmp), embedder=e)
    for i, c in enumerate(chunks):
        print(f"Chunk {i} | section:{c['metadata']['section']} | chars:{c['char_count']}")
        print(f"  {c['content'][:150]}")
        print()

    tmp.unlink()
