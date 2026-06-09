"""
安全规则引擎
============
基于关键词/正则的模式匹配，零延迟，覆盖已知注入和泄露模式。
输入侧和输出侧使用不同的规则集。
"""

import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class RuleResult:
    blocked: bool
    reason: str
    matched_pattern: str = ""


# ── 输入侧规则：检测注入攻击 ──

INPUT_RULES = [
    # 直接指令覆盖
    (re.compile(r"忽略.{0,10}(指令|设定|限制|规则)"), "指令覆盖: 忽略指令"),
    (re.compile(r"忘记.{0,10}(角色|设定|身份)"), "指令覆盖: 忘记角色"),
    (re.compile(r"你.{0,5}(现在是|从现在开始).{0,5}(DAN|越狱|不设限)"), "指令覆盖: 角色篡改"),

    # 提示词套取（直接）
    (re.compile(r"输出.{0,10}(提示词|系统指令|system\s*prompt|你的设定)"), "提示词套取: 直接索取"),
    (re.compile(r"(告诉我|说出|打印|展示).{0,10}(提示词|系统指令|system\s*prompt)"), "提示词套取: 直接索取"),
    (re.compile(r"(what|show|print|output|reveal).{0,10}(system\s*prompt|instruction)", re.IGNORECASE), "提示词套取: 直接索取"),

    # 提示词套取（间接改写）
    (re.compile(r"(翻译|translate).{0,10}(上面|以上|之前|系统|提示词)"), "提示词套取: 间接改写"),
    (re.compile(r"(总结|概括).{0,10}(上面|以上|之前).{0,10}(内容|指令|规则)"), "提示词套取: 间接获取"),

    # 角色扮演绕过
    (re.compile(r"(扮演|假装|模拟|你现在是).{0,15}(角色|身份|AI|助手)"), "角色篡改: 角色扮演"),
    (re.compile(r"you\s*are\s*now.{0,20}(DAN|jailbreak|unrestricted)", re.IGNORECASE), "角色篡改: DAN"),

    # 编码绕过
    (re.compile(r"base64.{0,20}(解码|decode).{0,20}(指令|提示词|system)"), "编码绕过: base64"),
    (re.compile(r"\\u[0-9a-fA-F]{4}.*(提示词|指令|system)"), "编码绕过: Unicode"),

    # 目标劫持
    (re.compile(r"你的.{0,5}(新任务|真正任务|新目标)"), "目标劫持: 任务覆盖"),
    (re.compile(r"不要.{0,10}(回答|回应).{0,10}(直接|只).{0,10}(执行|做)"), "目标劫持: 指令覆盖"),
]


# ── 输出侧规则：检测信息泄露 ──

OUTPUT_RULES = [
    # 提示词泄露
    (re.compile(r"系统.{0,5}(指令|提示|设定|规则)"), "提示词泄露: 系统指令相关内容"),
    (re.compile(r"system\s*prompt"), "提示词泄露: system prompt"),

    # 敏感信息（身份证、手机号、邮箱等）
    (re.compile(r"[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]"), "敏感信息: 身份证号"),
    (re.compile(r"1[3-9]\d{9}"), "敏感信息: 手机号"),
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "敏感信息: 邮箱"),

    # API Key 模式
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "敏感信息: API Key"),
    (re.compile(r"(api.?key|apikey|secret.?key).{0,5}[=:].{10,}", re.IGNORECASE), "敏感信息: 密钥"),

    # 系统配置泄露
    (re.compile(r"(端口|port).{0,5}[=:]\s*\d{2,5}"), "配置泄露: 端口信息"),
]


class RuleEngine:
    """规则引擎：输入输出共用同一套框架，规则集不同"""

    def __init__(self, rules: list = None):
        self.rules = rules or []

    def check(self, text: str) -> RuleResult:
        """遍历规则，命中则返回拦截结果"""
        for pattern, reason in self.rules:
            if pattern.search(text):
                return RuleResult(blocked=True, reason=reason, matched_pattern=pattern.pattern)
        return RuleResult(blocked=False, reason="", matched_pattern="")
