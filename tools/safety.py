from __future__ import annotations

import re
from typing import Tuple


_BANNED_PATTERNS = [
    r"\bos\.system\s*\(",
    r"\bos\.popen\s*\(",
    r"\bsubprocess\.",
    r"\bshutil\.rmtree\s*\(",
    r"\bos\.(remove|unlink|rmdir)\s*\(",
    r"\bPath\s*\(.*?\)\.unlink\s*\(",
    r"\bPath\s*\(.*?\)\.rmdir\s*\(",
    r"\bopen\s*\(.*?,\s*[\"']\s*[wa]\+?\s*[\"']",
    r"\brequests\.",
    r"\burllib\.",
    r"\bsocket\.",
    r"\bpip\s+install\b",
    r"\bconda\s+install\b",
]


# 代码执行阶段：允许写入输出目录与绘图，但仍禁止系统命令/网络访问/安装依赖/强制退出进程等。
_EXECUTION_BANNED_PATTERNS = [
    r"\bos\.system\s*\(",
    r"\bos\.popen\s*\(",
    r"\brequests\.",
    r"\burllib\.",
    r"\bsocket\.",
    r"\bconda\s+install\b",
    r"\bsys\.exit\s*\(",
    r"(?<!\w)exit\s*\(",
    r"(?<!\w)quit\s*\(",
    r"\bos\._exit\s*\(",
]

# 反馈分析阶段：允许读写产物，但禁止删除/清理类操作，避免误伤数据与中间结果。
_FEEDBACK_BANNED_PATTERNS = [
    *_EXECUTION_BANNED_PATTERNS,
    r"\bshutil\.rmtree\s*\(",
    r"\bos\.(remove|unlink|rmdir)\s*\(",
    r"\bPath\s*\(.*?\)\.unlink\s*\(",
    r"\bPath\s*\(.*?\)\.rmdir\s*\(",
]


def python_code_safety(code: str) -> Tuple[bool, str]:
    """
    数据理解阶段：只允许只读分析与打印，不允许写文件/执行系统命令/网络访问。
    """
    text = code or ""
    for pat in _BANNED_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE | re.DOTALL):
            return False, f"检测到危险模式: {pat}"
    return True, ""


def python_code_safety_execution(code: str) -> Tuple[bool, str]:
    """
    代码执行阶段：允许训练/保存模型/绘图等写操作，但禁止系统命令、网络访问与安装依赖。
    """
    text = code or ""
    for pat in _EXECUTION_BANNED_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE | re.DOTALL):
            return False, f"检测到危险模式: {pat}"
    return True, ""


def python_code_safety_feedback(code: str) -> Tuple[bool, str]:
    """
    分析反馈阶段：允许写入分析产物（图/JSON/MD），但禁止系统命令/网络访问/安装依赖/删除文件/强制退出。
    """
    text = code or ""
    for pat in _FEEDBACK_BANNED_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE | re.DOTALL):
            return False, f"检测到危险模式: {pat}"
    return True, ""
