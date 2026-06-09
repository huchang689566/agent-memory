"""
LLM 安全防护模块
================
输入输出纵深防御体系：

输入侧: 规则引擎 → 向量语义比对 → 分类模型 (Prompt-Guard-86M)
输出侧: 规则引擎 → 向量泄露检测 → (LLM 合规兜底, 预留)
"""

from security.guard import (
    InputGuard,
    OutputGuard,
    GuardResult,
    get_input_guard,
    get_output_guard,
    init_guards,
)

__all__ = [
    "InputGuard",
    "OutputGuard",
    "GuardResult",
    "get_input_guard",
    "get_output_guard",
    "init_guards",
]
