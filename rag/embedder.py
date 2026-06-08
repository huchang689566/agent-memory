"""
RAG 系统 - Embedder
==================
纯 Python 类，零框架依赖。
支持 API embedding（OpenAI/DeepSeek）和本地 TF-IDF。
"""

import logging
from typing import Optional

import numpy as np
import faiss

from config import CFG

logger = logging.getLogger("rag.embedder")


class Embedder:
    """Embedding 服务：API 或本地 TF-IDF"""

    def __init__(self, model_name: str = None, local_files_only: bool = True):
        self.model_name = model_name or CFG["embedding_model"]
        self._dim: Optional[int] = None
        self._local_model = None
        self._all_texts: list[str] = []
        self._use_local = self.model_name.startswith("local:")
        self._local_type = self.model_name.split(":", 1)[1] if self._use_local else ""
        self._local_files_only = local_files_only
        self._api_client = None
        if not self._use_local:
            from openai import OpenAI
            self._api_client = OpenAI(
                base_url=CFG["base_url"],
                api_key=CFG["api_key"],
            )

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["test"])[0])
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """批量 embedding，返回 L2 归一化后的 ndarray"""
        if self._use_local:
            return self._sentence_transformers_embed(texts)
        return self._api_embed(texts)

    def _sentence_transformers_embed(self, texts: list[str]) -> np.ndarray:
        if self._local_model is None:
            import os
            # 国内自动使用 HuggingFace 镜像
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from sentence_transformers import SentenceTransformer
            # 国产中文 embedding 模型，通过 HF 镜像下载
            model = "BAAI/bge-small-zh-v1.5"
            logger.info(f"[Embedder] loading {model}...")
            self._local_model = SentenceTransformer(
                model, local_files_only=self._local_files_only
            )
        arr = self._local_model.encode(texts, normalize_embeddings=True)
        return np.array(arr, dtype=np.float32)

    def _api_embed(self, texts: list[str]) -> np.ndarray:
        resp = self._api_client.embeddings.create(
            model=self.model_name, input=texts,
        )
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        faiss.normalize_L2(arr)
        return arr
