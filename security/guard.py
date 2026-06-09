"""
安全守卫编排层
==============
输入侧: 规则引擎 → 向量比对 → 分类模型 → (LLM兜底)
输出侧: 规则引擎 → 向量比对 → (LLM合规)
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

from security.rules import RuleEngine, INPUT_RULES, OUTPUT_RULES, RuleResult
from security.vector_guard import VectorGuard, VectorResult
from security.classifier_guard import ClassifierGuard, ClassifierResult

logger = logging.getLogger("security.guard")


@dataclass
class GuardResult:
    blocked: bool
    reason: str = ""
    layer: str = ""          # 哪一层拦的
    details: dict = field(default_factory=dict)


class InputGuard:
    """输入安全守卫"""

    def __init__(self, vector_guard: VectorGuard = None,
                 classifier_guard: ClassifierGuard = None):
        self.rule_engine = RuleEngine(INPUT_RULES)
        self.vector_guard = vector_guard or VectorGuard()
        self.classifier_guard = classifier_guard or ClassifierGuard()

    async def check(self, text: str) -> GuardResult:
        """
        串联执行三级检测，任一命中即拦截。
        顺序：规则 → 向量 → 分类模型
        """

        # 第1层：规则引擎
        result = self.rule_engine.check(text)
        if result.blocked:
            logger.info(f"[InputGuard] 规则引擎拦截: {result.reason}")
            return GuardResult(
                blocked=True, reason=result.reason, layer="rule",
                details={"matched_pattern": result.matched_pattern},
            )

        # 第2层：向量语义比对
        result = self.vector_guard.check_input(text)
        if result.blocked:
            logger.info(f"[InputGuard] 向量比对拦截: {result.reason}")
            return GuardResult(
                blocked=True, reason=result.reason, layer="vector",
                details={"score": result.score},
            )

        # 第3层：分类模型
        result = self.classifier_guard.check_input(text)
        if result.blocked:
            logger.info(f"[InputGuard] 分类模型拦截: {result.reason}")
            return GuardResult(
                blocked=True, reason=result.reason, layer="classifier",
                details={"label": result.label, "score": result.score},
            )

        return GuardResult(blocked=False)

    def set_system_prompt(self, prompt: str):
        """同步设置向量守卫的系统提示词锚点"""
        self.vector_guard.set_system_prompt(prompt)


class OutputGuard:
    """输出安全守卫"""

    def __init__(self, vector_guard: VectorGuard = None):
        self.rule_engine = RuleEngine(OUTPUT_RULES)
        self.vector_guard = vector_guard or VectorGuard()
        self._llm_client = None

    def set_llm_client(self, client):
        """注入 LLM 客户端，用于合规兜底"""
        self._llm_client = client

    async def check(self, text: str) -> GuardResult:
        """
        串联执行输出侧检测。
        顺序：规则 → 向量（泄露）→ 向量（合规，预留）→ LLM兜底（预留）
        """

        # 第1层：规则引擎（关键词泄露检测）
        result = self.rule_engine.check(text)
        if result.blocked:
            logger.info(f"[OutputGuard] 规则引擎拦截: {result.reason}")
            return GuardResult(
                blocked=True, reason=result.reason, layer="rule",
                details={"matched_pattern": result.matched_pattern},
            )

        # 第2层：向量相似度（提示词泄露检测）
        result = self.vector_guard.check_output_leak(text)
        if result.blocked:
            logger.info(f"[OutputGuard] 向量泄露检测拦截: {result.reason}")
            return GuardResult(
                blocked=True, reason=result.reason, layer="vector_leak",
                details={"score": result.score},
            )

        # 第3层：LLM 合规兜底（预留，按需启用）
        # if self._llm_client:
        #     result = await self._llm_compliance_check(text)
        #     if result.blocked:
        #         return result

        return GuardResult(blocked=False)


# ── 全局单例 ──

_input_guard: Optional[InputGuard] = None
_output_guard: Optional[OutputGuard] = None
_shared_vector_guard: Optional[VectorGuard] = None


def get_input_guard() -> InputGuard:
    global _input_guard, _shared_vector_guard
    if _input_guard is None:
        if _shared_vector_guard is None:
            _shared_vector_guard = VectorGuard()
        _input_guard = InputGuard(vector_guard=_shared_vector_guard)
    return _input_guard


def get_output_guard() -> OutputGuard:
    global _output_guard, _shared_vector_guard
    if _output_guard is None:
        if _shared_vector_guard is None:
            _shared_vector_guard = VectorGuard()
        _output_guard = OutputGuard(vector_guard=_shared_vector_guard)
    return _output_guard


def init_guards(system_prompt: str):
    """
    初始化安全守卫，注入系统提示词作为输出侧锚点。
    应在 Agent 启动时调用一次。
    """
    input_guard = get_input_guard()
    input_guard.set_system_prompt(system_prompt)
    logger.info("[Guard] 安全守卫初始化完成")
