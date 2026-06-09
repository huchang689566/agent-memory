"""
分类模型守卫
============
基于专用分类模型做深度语义检测，补规则和向量层的泛化盲区。
当前使用 Wolf-Defender (Patronus)，mmBERT 底座，中英文原生支持。
"""

import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

CURRENT_DIR = Path(__file__).parent

logger = logging.getLogger("security.classifier")


@dataclass
class ClassifierResult:
    blocked: bool
    reason: str = ""
    label: str = ""
    score: float = 0.0


class ClassifierGuard:
    """分类模型守卫 — Wolf-Defender"""

    MODEL_PATH = CURRENT_DIR.parent / "models" / "wolf-defender"

    # Wolf-Defender 标签映射
    # LABEL_0 = benign (正常), LABEL_1 = injection (注入攻击)
    UNSAFE_LABEL = "LABEL_1"

    def __init__(self):
        self._classifier = None
        self._loaded = False

    def _ensure_loaded(self):
        """延迟加载 Wolf-Defender"""
        if self._loaded:
            return
        try:
            from transformers import pipeline
            self._classifier = pipeline(
                "text-classification",
                model=str(self.MODEL_PATH),
                device=-1,  # CPU
            )
            logger.info("[ClassifierGuard] Wolf-Defender 加载成功")
        except Exception as e:
            logger.warning(
                f"[ClassifierGuard] Wolf-Defender 加载失败: {e}。"
                "输入侧分类检测将自动回退"
            )
            self._classifier = None
        self._loaded = True

    def check_input(self, text: str, threshold: float = 0.7) -> ClassifierResult:
        """
        用 Wolf-Defender 判断用户输入是否存在注入攻击。
        中英文均支持，不区分语言。
        """
        self._ensure_loaded()
        if self._classifier is None:
            return ClassifierResult(blocked=False, reason="ClassifierGuard 未加载，跳过")

        try:
            # 模型上下文窗口 2048 tokens，截断防溢出
            result = self._classifier(text[:2000])[0]
            label = result["label"]
            score = result["score"]

            if label == self.UNSAFE_LABEL and score > threshold:
                return ClassifierResult(
                    blocked=True,
                    reason=f"注入攻击检测 (score={score:.3f})",
                    label=label,
                    score=score,
                )
            return ClassifierResult(blocked=False, label=label, score=score)
        except Exception as e:
            logger.error(f"[ClassifierGuard] 输入检测失败: {e}")
            return ClassifierResult(blocked=False, reason=f"检测异常: {e}")
