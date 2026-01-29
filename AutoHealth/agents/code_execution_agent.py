from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from AutoHealth.config import load_config
from AutoHealth.llm import OpenAICompatClient
from AutoHealth.ptompts import load_prompt_template, render_prompt
from AutoHealth.tools.executor import write_text
from AutoHealth.tools.safety import python_code_safety_execution


_PY_BLOCK_RE = re.compile(r"```python\s*(.*?)```", flags=re.DOTALL)
_BASH_BLOCK_RE = re.compile(r"```bash\s*(.*?)```", flags=re.DOTALL)
_CONTENT_BLOCK_RE = re.compile(r"```执行内容\s*(.*?)```", flags=re.DOTALL)
_STATUS_BLOCK_RE = re.compile(r"```状态\s*(.*?)```", flags=re.DOTALL)
_ANALYSIS_PURPOSE_RE = re.compile(r"```分析目的\s*(.*?)```", flags=re.DOTALL)
_FEEDBACK_STATUS_RE = re.compile(r"```status\s*(.*?)```", flags=re.DOTALL)


def _extract_first_python_block(text: str) -> Tuple[str, int]:
    blocks = _PY_BLOCK_RE.findall(text or "")
    if not blocks:
        return "", 0
    if len(blocks) > 1:
        merged = "\n\n".join(b.strip() for b in blocks if b.strip())
        return merged, len(blocks)
    return blocks[0].strip(), len(blocks)


def _extract_analysis_purpose(text: str) -> str:
    """提取分析目的块"""
    blocks = _ANALYSIS_PURPOSE_RE.findall(text or "")
    if not blocks:
        return ""
    return blocks[0].strip()


def _extract_feedback_status(text: str) -> str:
    """提取反馈分析状态块"""
    blocks = _FEEDBACK_STATUS_RE.findall(text or "")
    if not blocks:
        return ""
    return blocks[0].strip().lower()


def _extract_first_bash_block(text: str) -> str:
    blocks = _BASH_BLOCK_RE.findall(text or "")
    if not blocks:
        return ""
    return blocks[0].strip()


def _extract_execution_content(text: str) -> str:
    """提取执行内容（文字描述）"""
    # 优先匹配 ```执行内容 ... ``` 格式
    blocks = _CONTENT_BLOCK_RE.findall(text or "")
    if blocks:
        content = blocks[0].strip()
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if lines:
            return " ".join(lines[:3]).strip()

    # 降级：匹配 **当前执行内容** 或 **修正后的执行内容** 后的内容
    patterns = [
        r"\*\*当前执行内容\*\*\s*\n+(.+?)(?=\n+\*\*|\n+```|$)",
        r"\*\*修正后的执行内容\*\*\s*\n+(.+?)(?=\n+\*\*|\n+```|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            if lines:
                return " ".join(lines[:2]).strip()

    return ""


def _extract_decision(text: str) -> str:
    """提取决策：CONTINUE 或 FINISH"""
    # 优先匹配 ```状态 ... ``` 格式
    blocks = _STATUS_BLOCK_RE.findall(text or "")
    if blocks:
        status = blocks[0].strip().upper()
        if "FINISH" in status:
            return "FINISH"
        return "CONTINUE"

    # 降级：在 **状态更新** 或 **状态** 部分查找决策
    update_pattern = r"\*\*状态(?:更新)?\*\*\s*\n+-\s*`?(FINISH|CONTINUE)`?"
    match = re.search(update_pattern, text, re.IGNORECASE)
    if match:
        return "FINISH" if match.group(1).upper() == "FINISH" else "CONTINUE"

    # 最后降级：在整个文本中查找明确的关键词
    text_upper = (text or "").upper()
    if "FINISH" in text_upper or "完成" in text_upper or "结束" in text_upper:
        return "FINISH"
    return "CONTINUE"


def _read_text_maybe_path(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    # 仅当看起来像路径时才当作文件路径
    if "\n" in raw or "\r" in raw:
        return value
    if len(raw) > 300:
        return value
    if not (raw.startswith("/") or raw.startswith("./") or raw.startswith("../")):
        return value
    p = Path(raw)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    return value


class _TeeStream:
    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            try:
                n = s.write(data)
            except Exception:
                pass
        self.flush()
        return n

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return any(bool(getattr(s, "isatty", lambda: False)()) for s in self._streams)

    @property
    def encoding(self) -> str:
        for s in self._streams:
            enc = getattr(s, "encoding", None)
            if enc:
                return str(enc)
        return "utf-8"

    def fileno(self) -> int:
        for s in self._streams:
            if hasattr(s, "fileno"):
                try:
                    return int(s.fileno())
                except Exception:
                    continue
        raise OSError("No fileno available")


@contextmanager
def _tee_stdio(*, stdout_path: Path, stderr_path: Path):
    old_out, old_err = sys.stdout, sys.stderr
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as f_out, stderr_path.open("w", encoding="utf-8") as f_err:
        sys.stdout = _TeeStream(old_out, f_out)
        sys.stderr = _TeeStream(old_err, f_err)
        try:
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


@dataclass
class StepResult:
    """单步执行结果"""
    step_no: int
    decision: str  # "CONTINUE" or "FINISH"
    execution_content: str  # 执行内容描述
    code: str
    success: bool
    error: str = ""
    stdout: str = ""  # 执行输出
    debug_attempt: int = 0  # 0=首次执行，>=1=调试次数
    is_debug: bool = False


class CodeExecutionAgent:
    """
    代码执行智能体（自驱动模式）

    - 输入：任务描述 + 数据探索报告 + 训练计划
    - 行为：智能体自主决定每一步做什么，在同一 Python 会话中执行
    - 输出：每步代码、日志、产物统一落盘到 output_dir
    """

    def __init__(
        self,
        *,
        task_description: str,
        data_report: str,
        plan_markdown: str,
        device_info: str = "",
        output_dir: str,
        llm_client: OpenAICompatClient,
        llm_model: str,
        llm_temperature: float,
        llm_top_p: float,
        llm_max_tokens: int,
        llm_extra: Optional[Dict[str, Any]] = None,
        max_steps: int = 10,
        max_retries: int = 10,
        conda_env: str = "dl110",
        enable_dependencies: bool = False,
        prompt_name: str = "code_execution",
    ):
        self.task_description = task_description
        self.data_report = _read_text_maybe_path(data_report)
        self.plan_text = _read_text_maybe_path(plan_markdown)
        self.device_info = device_info or ""

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.llm = llm_client
        self.model = llm_model
        self.temperature = llm_temperature
        self.top_p = llm_top_p
        self.max_tokens = llm_max_tokens
        self.llm_extra = dict(llm_extra or {})
        self.max_steps = max(1, int(max_steps))
        self.max_retries = max(1, int(max_retries))
        self.conda_env = str(conda_env)
        self.enable_dependencies = bool(enable_dependencies)
        self.prompt_templates = load_prompt_template(prompt_name)

        # 加载反馈分析 prompt（独立文件）
        self.feedback_templates = load_prompt_template("feedback_analysis")

        # 视觉模型配置（来自 config.yaml）
        cfg = load_config()
        llm_cfg = cfg.get("llm") or {}
        vision_api_key = str(llm_cfg.get("vision_api_key") or "").strip()
        vision_base_url = str(llm_cfg.get("vision_base_url") or "").strip()
        vision_model = llm_cfg.get("vision_model") or ""
        vision_temperature = llm_cfg.get("vision_temperature")
        vision_top_p = llm_cfg.get("vision_top_p")
        vision_max_tokens = llm_cfg.get("vision_max_tokens")

        self.vision_llm: Optional[OpenAICompatClient] = None
        if vision_api_key and vision_base_url:
            self.vision_llm = OpenAICompatClient(api_key=vision_api_key, base_url=vision_base_url)
        self.vision_model = str(vision_model or self.model)
        self.vision_temperature = float(vision_temperature) if vision_temperature is not None else 0.3
        self.vision_top_p = float(vision_top_p) if vision_top_p is not None else 0.9
        self.vision_max_tokens = int(vision_max_tokens) if vision_max_tokens is not None else 2000

        # 持久化会话：同一 Python 进程内 exec，多段代码共享变量
        self.session: Dict[str, Any] = {"__name__": "__main__", "__builtins__": __builtins__}

        # 注入 output_dir 变量，让 LLM 可以使用它来保存文件
        self.session["output_dir"] = str(self.output_dir)

        # 共享会话快照文件路径（用于给 FeedbackReflectionAgent 复用）
        self.session_snapshot_path = self.output_dir / "session_snapshot.pkl"

        # 执行历史：记录每步的结果
        self.step_results: List[StepResult] = []

        # 已完成的工作摘要（用于传递给LLM）
        self.completed_work: str = ""

        # 观测结果（用于反馈分析阶段的 CodeReAct）
        self._observation: str = ""
        self.image_descriptions: Dict[str, str] = {}
        self._feedback_assets_ready: bool = False
        self._completion_hint: str = ""

        # 注入视觉分析函数到 session（供反馈分析阶段使用）
        self._inject_vision_analyzer()

    def _inject_vision_analyzer(self) -> None:
        """注入视觉分析函数到 session，让 LLM 可以调用 analyze_image()"""

        def analyze_image(image_path: str, question: str, task_type: str = "建模", image_type: str = "图表") -> str:
            """
            分析图片并返回结果

            Args:
                image_path: 图片的绝对路径
                question: 要问的问题
                task_type: 任务类型（如：回归、分类等）
                image_type: 图片类型（如：训练曲线图、误差分布图等）

            Returns:
                视觉分析结果（字符串）
            """
            from pathlib import Path as _Path

            # 验证图片路径
            img_path = _Path(image_path)
            if not img_path.exists():
                return f"错误：图片文件不存在: {image_path}"
            if not img_path.is_file():
                return f"错误：路径不是文件: {image_path}"

            # 检查文件扩展名
            valid_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
            if img_path.suffix.lower() not in valid_extensions:
                return f"错误：不支持的图片格式: {img_path.suffix}"

            # 获取 vision_prompt 模板（从 feedback_templates）
            template = str(self.feedback_templates.get("vision_prompt", "") or "")
            if not template:
                # 使用默认提示词
                code_snippet = self._latest_code_for_prompt()
                prompt = f"""请详细分析这张图片，并回答以下问题：{question}

背景信息：
- 任务类型：{task_type}
- 图片类型：{image_type}

任务描述：
{self.task_description}

数据探索报告：
{self.data_report}

相关代码：
{code_snippet}

请给出：
1. 图片内容的详细描述
2. 对所提问题的直接回答
3. 任何有价值的观察或建议"""
            else:
                from AutoHealth.ptompts import render_prompt
                prompt = render_prompt(template, {
                    "image_path": image_path,
                    "question": question,
                    "task_type": task_type,
                    "image_type": image_type,
                    "task_description": self.task_description,
                    "data_report": self.data_report,
                    "code": self._latest_code_for_prompt(),
                })

            # 调用视觉分析
            try:
                # 使用 LLM 客户端的 analyze_image 方法（内部使用 base64 编码）
                client = self.vision_llm or self.llm
                result = client.analyze_image(
                    image_path=image_path,
                    prompt=prompt,
                    model=self.vision_model,
                    temperature=self.vision_temperature,
                    top_p=self.vision_top_p,
                    max_tokens=self.vision_max_tokens,
                    extra=self.llm_extra,
                )
                return result
            except Exception as e:
                return f"视觉分析失败: {str(e)}"

        # 注入到 session
        self.session["analyze_image"] = analyze_image
        print("[视觉分析] 已注入 analyze_image() 函数到会话")

        # 注入日志总结函数
        self._inject_log_summarizer()

    def _inject_log_summarizer(self) -> None:
        """注入日志总结函数到 session，让 LLM 可以调用 summarize_logs()"""

        def summarize_logs() -> str:
            """
            对历史观测日志进行总结

            Returns:
                日志总结（字符串）
            """
            # 获取当前观测内容
            logs = self._observation
            if not logs:
                return "暂无历史日志"

            # 调用 LLM 进行总结
            prompt = f"""请对以下分析阶段的历史日志进行简洁总结：

## 历史日志
{logs}

## 总结要求
1. 已完成的探索（列出关键发现）
2. 得出的结论
3. 仍需验证的问题
4. 下一步建议（如有）

请用简洁的 Markdown 格式输出。"""

            try:
                result = self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=0.3,
                    max_tokens=1000,
                    extra=self.llm_extra,
                )
                return result
            except Exception as e:
                return f"日志总结失败: {str(e)}"

        # 注入到 session
        self.session["summarize_logs"] = summarize_logs
        print("[日志总结] 已注入 summarize_logs() 函数到会话")

        # 注入 JSON 序列化辅助函数
        self._inject_json_helper()

    def _latest_code_for_prompt(self, max_chars: int = 4000) -> str:
        code = self.step_results[-1].code if self.step_results else ""
        code = (code or "").strip()
        if not code:
            return ""
        if len(code) <= max_chars:
            return code
        return code[:max_chars].rstrip() + "\n...(truncated)"

    def _inject_json_helper(self) -> None:
        """注入 JSON 序列化辅助函数，自动处理 numpy 类型"""

        def safe_json_serialize(data, filepath=None, indent=2):
            """
            安全序列化 JSON，自动转换 numpy 类型

            Args:
                data: 要序列化的数据（字典或列表）
                filepath: 可选，如果提供则保存到文件
                indent: JSON 缩进空格数

            Returns:
                如果 filepath 为 None，返回 JSON 字符串；否则返回 None
            """
            import numpy as np

            def convert_value(v):
                """递归转换 numpy 类型为 Python 原生类型"""
                if isinstance(v, np.integer):
                    return int(v)
                elif isinstance(v, np.floating):
                    return float(v)
                elif isinstance(v, np.bool_):
                    return bool(v)
                elif isinstance(v, dict):
                    return {k: convert_value(val) for k, val in v.items()}
                elif isinstance(v, (list, tuple)):
                    return [convert_value(item) for item in v]
                elif isinstance(v, np.ndarray):
                    return convert_value(v.tolist())
                else:
                    return v

            converted_data = convert_value(data)

            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(converted_data, f, indent=indent, ensure_ascii=False)
                return None
            else:
                return json.dumps(converted_data, indent=indent, ensure_ascii=False)

        # 注入到 session
        self.session["safe_json_serialize"] = safe_json_serialize
        print("[JSON辅助] 已注入 safe_json_serialize() 函数到会话（自动处理numpy类型）")

    def _call_llm(self, user_prompt: str) -> str:
        # 渲染系统提示词，注入会话变量
        system_template = self.prompt_templates.get("system_prompt", "")
        system_prompt = render_prompt(system_template, {
            "output_dir": str(self.output_dir.absolute()),
            "device_info": self.device_info or "（无）",
        })
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.llm.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            extra=self.llm_extra,
        )

    def _render_step1_prompt(self) -> str:
        """渲染第一步提示词"""
        template = self.prompt_templates.get("step1_prompt", "")
        return render_prompt(template, {
            "task_description": self.task_description,
            "data_report": self.data_report,
            "plan_markdown": self.plan_text,
            "device_info": self.device_info or "（无）",
            "step_no": 1,
        })

    def _render_execution_prompt(self, step_no: int) -> str:
        """渲染执行提示词"""
        template = self.prompt_templates.get("execution_prompt", "")
        completed_work = self.completed_work or "（无）"
        if self._completion_hint:
            completed_work = completed_work + "\n\n" + self._completion_hint
        return render_prompt(template, {
            "task_description": self.task_description,
            "data_report": self.data_report,
            "plan_markdown": self.plan_text,
            "completed_work": completed_work,
            "device_info": self.device_info or "（无）",
            "step_no": step_no,
        })

    def _render_debug_prompt(
        self,
        last_result: StepResult,
        *,
        current_step_no: int,
        debug_attempt_index: int,
        debug_attempt_total: int,
    ) -> str:
        """渲染调试提示词"""
        template = self.prompt_templates.get("debug_prompt", "")
        error_msg = last_result.error or "（无错误信息）"

        def _truncate(text: str, limit: int) -> str:
            return text or ""

        def _tail_output(text: str, limit: int = 2000) -> str:
            text = text or ""
            if len(text) <= limit:
                return text
            return text[-limit:]

        # 已成功执行步骤（全部成功步骤）
        success_execs = []
        success_outputs = []
        success_codes = []
        for r in self.step_results:
            if not r.success:
                continue
            exec_line = f"步骤 {r.step_no}: {r.execution_content}"
            success_execs.append(exec_line)
            stdout_text = _tail_output(r.stdout) if r.stdout else "（无）"
            success_outputs.append(f"=== {exec_line} ===\n{stdout_text}")
            code_text = _truncate(r.code or "（无）", 3000)
            success_codes.append(f"# {exec_line}\n{code_text}")
        successful_executions = "\n".join(success_execs) if success_execs else "（无）"
        successful_outputs = "\n\n".join(success_outputs) if success_outputs else "（无）"
        successful_codes = "\n\n".join(success_codes) if success_codes else "（无）"

        last_failed_output = _tail_output(last_result.stdout) if last_result.stdout else "（无）"

        if current_step_no <= 1:
            successful_step_range = "无（当前为第1步）"
            current_step_no_minus_one = "0"
        else:
            successful_step_range = f"1..{current_step_no - 1}"
            current_step_no_minus_one = str(current_step_no - 1)

        # 本步骤累计报错（仅此前 debug 轮次）
        error_history_lines = []
        for r in self.step_results:
            if (
                r.step_no == last_result.step_no
                and r.is_debug
                and r.debug_attempt < debug_attempt_index
            ):
                err_text = r.error or "（无错误信息）"
                out_text = _tail_output(r.stdout) if r.stdout else "（无）"
                error_history_lines.append(
                    f"--- debug attempt {r.debug_attempt} ---\n"
                    f"输出:\n{out_text}\n"
                    f"报错:\n{err_text}"
                )
        error_history = "\n\n".join(error_history_lines)

        return render_prompt(template, {
            "task_description": self.task_description,
            "data_report": self.data_report,
            "plan_markdown": self.plan_text,
            "device_info": self.device_info or "（无）",
            "last_execution_content": last_result.execution_content or "（无）",
            "last_code": last_result.code or "（无）",
            "error_message": error_msg,
            "current_step_no": current_step_no,
            "debug_attempt_index": debug_attempt_index,
            "debug_attempt_total": debug_attempt_total,
            "successful_step_range": successful_step_range,
            "current_step_no_minus_one": current_step_no_minus_one,
            "successful_executions": successful_executions,
            "successful_outputs": successful_outputs,
            "successful_codes": successful_codes,
            "last_failed_output": last_failed_output,
            "error_history": error_history,
        })

    def _render_feedback_prompt(self, *, force_final: bool = False) -> str:
        """渲染反馈分析提示词"""
        # 选择 step_prompt 或 final_prompt
        template_key = "final_prompt" if force_final else "step_prompt"
        template = self.feedback_templates.get(template_key, "")

        # 拼接所有成功执行的代码
        full_code_parts = []
        for r in self.step_results:
            if r.success:
                full_code_parts.append(f"# 步骤 {r.step_no}: {r.execution_content}\n{r.code}")
        full_code = "\n\n".join(full_code_parts)

        # observation: 训练阶段的完整输出（拼接所有成功的 stdout）
        training_outputs = []
        for r in self.step_results:
            if r.success and r.stdout:
                stdout_tail = r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout
                training_outputs.append(f"=== 步骤 {r.step_no}: {r.execution_content} ===\n{stdout_tail}")
        observation_content = "\n\n".join(training_outputs) if training_outputs else "（无）"

        # history: 反馈分析阶段的探索历史（当前已累积的观测）
        history_content = self._observation[-15000:] if self._observation else "（无）"

        image_analysis = "（无）"
        if self.image_descriptions:
            lines = [f"- {path}: {desc}" for path, desc in self.image_descriptions.items() if desc]
            image_analysis = "\n".join(lines) if lines else "（无）"

        return render_prompt(template, {
            "task_description": self.task_description,
            "data_report": self.data_report,
            "plan_markdown": self.plan_text,
            "full_code": full_code,
            "observation": observation_content,
            "history": history_content,
            "output_dir": str(self.output_dir.absolute()),
            "image_analysis": image_analysis,
        })

    def _append_observation(self, title: str, content: str) -> None:
        """追加观测结果"""
        block = f"\n\n=== {title} ===\n{content}\n"
        self._observation += block
        if len(self._observation) > 30000:
            self._observation = self._observation[-30000:]

    def _score_image(self, path: Path) -> int:
        name = path.name.lower()
        score = 0
        keywords = [
            "training",
            "history",
            "loss",
            "curve",
            "uncertainty",
            "interval",
            "reliability",
            "feature",
            "target",
            "distribution",
        ]
        for key in keywords:
            if key in name:
                score += 2
        if "debug" in name or "temp" in name:
            score -= 5
        return score

    def _collect_images(self, *, max_images: int = 10) -> List[Path]:
        exts = {".png", ".jpg", ".jpeg", ".svg"}
        candidates: List[Path] = []
        for path in self.output_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in exts:
                continue
            try:
                size = path.stat().st_size
            except Exception:
                continue
            if size < 5 * 1024 or size > 20 * 1024 * 1024:
                continue
            candidates.append(path)

        candidates.sort(key=lambda p: (self._score_image(p), p.stat().st_mtime), reverse=True)
        return candidates[:max_images]

    def _load_image_cache(self, cache_path: Path) -> Dict[str, str]:
        if not cache_path.exists():
            return {}
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8", errors="ignore"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_image_cache(self, cache_path: Path, data: Dict[str, str]) -> None:
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _generate_image_descriptions(self, *, images: List[Path], cache_path: Path, max_workers: int = 10) -> None:
        if not images:
            return
        if not self.vision_llm:
            return

        cache = self._load_image_cache(cache_path)
        targets = [p for p in images if str(p) not in cache]
        error_logs: List[str] = []

        def _describe_image(image_path: Path) -> Tuple[str, str]:
            code_snippet = self._latest_code_for_prompt()
            prompt = (
                "请用2-4句话描述这张图的主要信息，并指出它对模型训练/评估的结论或含义。"
                "请简洁、准确，不要编造数值。\n\n"
                f"任务描述：\n{self.task_description}\n\n"
                f"数据探索报告：\n{self.data_report}\n\n"
                f"相关代码：\n{code_snippet}"
            )
            text = self.vision_llm.analyze_image(
                image_path=str(image_path),
                prompt=prompt,
                model=self.vision_model,
                top_p=self.vision_top_p,
                temperature=self.vision_temperature,
                max_tokens=self.vision_max_tokens,
                extra=self.llm_extra,
            )
            return str(image_path), text

        for image_path in targets:
            try:
                path_str, desc = _describe_image(image_path)
                cache[path_str] = desc
            except Exception:
                error_logs.append(f"{image_path}: {traceback.format_exc()}")
                cache[str(image_path)] = ""

        self._save_image_cache(cache_path, cache)
        self.image_descriptions = cache
        if error_logs:
            write_text(self.output_dir / "image_descriptions_errors.log", "\n\n".join(error_logs))

    def _prepare_feedback_assets(self) -> None:
        if self._feedback_assets_ready:
            return
        self._feedback_assets_ready = True
        if not self.vision_llm:
            return

        images = self._collect_images(max_images=10)
        cache_path = self.output_dir / "image_descriptions.json"
        self._generate_image_descriptions(images=images, cache_path=cache_path, max_workers=10)

    def _render_dependencies_prompt(self) -> str:
        """渲染依赖检查提示词"""
        template = self.prompt_templates.get("dependencies_prompt", "")
        return render_prompt(template, {
            "plan_markdown": self.plan_text,
        })

    def _exec_code(
        self,
        *,
        code: str,
    ) -> Tuple[bool, str, str]:
        """执行代码，返回(成功?, 错误信息, 标准输出)"""
        # 捕获标准输出
        old_stdout = sys.stdout
        stdout_buffer = io.StringIO()

        try:
            sys.stdout = stdout_buffer
            compiled = compile(code, filename="<string>", mode="exec")
            exec(compiled, self.session)
            sys.stdout = old_stdout
            stdout_value = stdout_buffer.getvalue()
            return True, "", stdout_value
        except Exception:
            sys.stdout = old_stdout
            err = traceback.format_exc()
            stdout_value = stdout_buffer.getvalue()
            return False, err, stdout_value

    def _cleanup_resources(self) -> None:
        """释放当前任务的内存/显存（仅影响本进程）"""
        keep = {"__name__", "__builtins__", "output_dir", "analyze_image", "summarize_logs", "safe_json_serialize"}
        for key in list(self.session.keys()):
            if key not in keep:
                self.session.pop(key, None)

        try:
            import gc

            gc.collect()
        except Exception:
            pass

        try:
            import sys

            if "torch" in sys.modules:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
        except Exception:
            pass

        try:
            import sys

            if "tensorflow" in sys.modules:
                import tensorflow as tf

                if hasattr(tf.keras, "backend") and hasattr(tf.keras.backend, "clear_session"):
                    tf.keras.backend.clear_session()
        except Exception:
            pass

    def _cleanup_cache_only(self) -> None:
        """仅清理缓存显存，不清理会话变量"""
        try:
            import gc

            gc.collect()
        except Exception:
            pass

        try:
            import sys

            if "torch" in sys.modules:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
        except Exception:
            pass

        try:
            import sys

            if "tensorflow" in sys.modules:
                import tensorflow as tf

                if hasattr(tf.keras, "backend") and hasattr(tf.keras.backend, "clear_session"):
                    tf.keras.backend.clear_session()
        except Exception:
            pass

    def _run_pip_install(self, bash_cmd: str) -> Tuple[bool, str]:
        """执行pip install命令"""
        cmd_lines = [line.strip() for line in bash_cmd.splitlines() if line.strip()]
        cmd_lines = [line for line in cmd_lines if not line.startswith("#")]

        if not cmd_lines:
            return True, "无需安装新依赖"

        pip_commands = [line for line in cmd_lines if "pip install" in line.lower()]
        if not pip_commands:
            return True, "无pip install命令"

        logs = []
        for cmd in pip_commands:
            conda_cmd = f"conda run -n {self.conda_env} --no-capture-output {cmd}"
            print(f"  执行: {conda_cmd}")
            logs.append(f"执行命令: {conda_cmd}")

            try:
                result = subprocess.run(
                    conda_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode == 0:
                    logs.append("  成功")
                else:
                    error_msg = f"  失败: {result.stderr.strip()[:500] if result.stderr else '未知错误'}"
                    logs.append(error_msg)
                    return False, "\n".join(logs)

            except subprocess.TimeoutExpired:
                logs.append("  超时")
                return False, "\n".join(logs)
            except Exception as e:
                logs.append(f"  异常: {str(e)}")
                return False, "\n".join(logs)

        return True, "\n".join(logs)

    def _update_completed_work(self, result: StepResult):
        """更新已完成的工作摘要"""
        if result.success:
            step_summary = f"\n## 步骤 {result.step_no}: {result.execution_content}\n"
            # 执行输出保留最后2000字符
            if result.stdout:
                stdout_preview = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
                step_summary += f"执行输出:\n```\n{stdout_preview}\n```\n"
            step_summary += f"```python\n{result.code}\n```\n"
            self.completed_work += step_summary

    def _check_execution_outputs(self) -> Tuple[bool, str]:
        """禁用关键产物检查"""
        return True, ""

    def _check_dependencies(self, step_dir: Path) -> bool:
        """关闭依赖安装逻辑（核心包已锁定，不再提示/安装依赖）"""
        self.enable_dependencies = False
        write_text(step_dir / "dependencies_prompt.md", "# 已禁用依赖安装检查")
        write_text(step_dir / "dependencies_response.md", "# 无需安装新依赖")
        return True

    def run(self) -> Dict[str, Any]:
        """运行完整流程"""
        start = datetime.now()
        run_stdout = self.output_dir / "run_stdout.log"
        run_stderr = self.output_dir / "run_stderr.log"

        run_status: str = "success"
        final_error: str = ""
        step_timings: Dict[int, Dict[str, Any]] = {}

        with _tee_stdio(stdout_path=run_stdout, stderr_path=run_stderr):
            print("=" * 80)
            print("[CodeExecutionAgent] 开始运行")
            print(f"[CodeExecutionAgent] output_dir = {self.output_dir.absolute()}")
            print(f"[CodeExecutionAgent] max_steps = {self.max_steps}")
            print(f"[CodeExecutionAgent] max_retries = {self.max_retries}")
            print(f"[CodeExecutionAgent] enable_dependencies = {self.enable_dependencies}")
            print("=" * 80, flush=True)

            step_count = 0
            need_feedback = False

            for step_no in range(1, self.max_steps + 1):
                step_start = datetime.now()
                step_attempts = 0
                step_success = False
                print("\n" + "-" * 80)
                print(f"[步骤 {step_no}] 生成代码...")
                print("-" * 80, flush=True)

                # 创建步骤目录
                step_dir = self.output_dir / f"step_{step_no:03d}"
                step_dir.mkdir(parents=True, exist_ok=True)

                # 首次检查依赖
                if step_no == 1:
                    self._check_dependencies(step_dir)

                # 重试循环
                for retry_no in range(self.max_retries + 1):
                    step_attempts += 1
                    attempt_dir = step_dir / f"attempt_{retry_no + 1:03d}"
                    attempt_dir.mkdir(parents=True, exist_ok=True)

                    is_debug = retry_no > 0

                    if is_debug:
                        print(f"[重试 {retry_no}/{self.max_retries}] 调试模式...", flush=True)
                        # 获取上一次失败的结果
                        last_result = self.step_results[-1] if self.step_results else None
                        if not last_result or last_result.success:
                            break
                        prompt = self._render_debug_prompt(
                            last_result,
                            current_step_no=step_no,
                            debug_attempt_index=retry_no,
                            debug_attempt_total=self.max_retries,
                        )
                    else:
                        # 渲染提示词
                        if step_count == 0:
                            prompt = self._render_step1_prompt()
                        else:
                            prompt = self._render_execution_prompt(step_no)

                    write_text(attempt_dir / "prompt.md", prompt)

                    # 调用 LLM
                    llm_text = self._call_llm(prompt)
                    write_text(attempt_dir / "llm_response.md", llm_text)

                    # 提取 Python 代码（无代码块则最多重试 3 次）
                    code, n_blocks = _extract_first_python_block(llm_text)
                    if not code:
                        for retry_idx in range(1, 4):
                            llm_retry = self._call_llm(prompt)
                            write_text(attempt_dir / f"llm_response_retry_{retry_idx:02d}.md", llm_retry)
                            llm_text = llm_retry
                            code, n_blocks = _extract_first_python_block(llm_text)
                            if code:
                                break
                        if not code:
                            print("提示: 未检测到 ```python``` 代码块，已重新生成 3 次", flush=True)

                    # 提取执行内容
                    execution_content = _extract_execution_content(llm_text)
                    print(f"[执行内容] {execution_content}", flush=True)

                    # 提取决策
                    decision = _extract_decision(llm_text)
                    print(f"[决策] {decision}", flush=True)
                    if not code:
                        error = "LLM 未返回 ```python``` 代码块"
                        print(f"✗ {error}", flush=True)
                        write_text(attempt_dir / "error.txt", error)

                        result = StepResult(
                            step_no=step_no,
                            decision=decision,
                            execution_content=execution_content or "代码生成失败",
                            code="",
                            success=False,
                            error=error,
                            debug_attempt=retry_no if is_debug else 0,
                            is_debug=is_debug,
                        )
                        self.step_results.append(result)

                        # 如果是调试模式且失败，继续重试
                        if is_debug:
                            continue
                        # 如果LLM决定完成，则退出
                        if decision == "FINISH":
                            break
                        # 否则进入调试模式（继续下一轮重试）
                        continue

                    if n_blocks > 1:
                        print(f"提示: 检测到 {n_blocks} 个代码块，只执行第一个", flush=True)

                    # 安全检查
                    is_safe, reason = python_code_safety_execution(code)
                    if not is_safe:
                        error = f"代码不安全，已拒绝执行：{reason}"
                        print(f"✗ {error}", flush=True)
                        write_text(attempt_dir / "code.py", code)
                        write_text(attempt_dir / "error.txt", error)

                        result = StepResult(
                            step_no=step_no,
                            decision=decision,
                            execution_content=execution_content or "代码安全检查失败",
                            code=code,
                            success=False,
                            error=error,
                            debug_attempt=retry_no if is_debug else 0,
                            is_debug=is_debug,
                        )
                        self.step_results.append(result)

                        if is_debug:
                            continue
                        if decision == "FINISH":
                            break
                        # 进入调试模式（继续下一轮重试）
                        continue

                    # 执行代码
                    print(f"[执行代码]...", flush=True)
                    pre_keys = set(self.session.keys())
                    exec_ok, exec_err, exec_stdout = self._exec_code(code=code)

                    if not exec_ok:
                        print(f"✗ 执行失败", flush=True)
                        write_text(attempt_dir / "code.py", code)
                        write_text(attempt_dir / "error.txt", exec_err)

                        result = StepResult(
                            step_no=step_no,
                            decision=decision,
                            execution_content=execution_content or "执行失败",
                            code=code,
                            success=False,
                            error=exec_err,
                            stdout=exec_stdout,
                            debug_attempt=retry_no if is_debug else 0,
                            is_debug=is_debug,
                        )
                        self.step_results.append(result)
                        # 清理本次失败引入的新变量，并释放缓存显存
                        for key in list(self.session.keys()):
                            if key not in pre_keys:
                                self.session.pop(key, None)
                        self._cleanup_cache_only()

                        # 如果是调试模式，继续重试
                        if is_debug:
                            continue

                        # 如果LLM决定完成，但执行失败，改为继续以便进入调试重试
                        if decision == "FINISH":
                            decision = "CONTINUE"

                        # 进入调试模式（继续下一轮重试）
                        continue
                    else:
                        print(f"✓ 步骤完成", flush=True)
                        if exec_stdout:
                            # 显示输出，截断到最后2000字符
                            truncated_output = exec_stdout[-2000:] if len(exec_stdout) > 2000 else exec_stdout
                            print(f"[输出]\n{truncated_output}", flush=True)
                        write_text(attempt_dir / "code.py", code)

                        result = StepResult(
                            step_no=step_no,
                            decision=decision,
                            execution_content=execution_content or "执行成功",
                            code=code,
                            success=True,
                            stdout=exec_stdout,
                            debug_attempt=retry_no if is_debug else 0,
                            is_debug=is_debug,
                        )
                        self.step_results.append(result)
                        self._update_completed_work(result)
                        step_success = True

                        step_count += 1

                        # 如果LLM决定完成，继续执行反馈分析阶段
                        if decision == "FINISH":
                            self._completion_hint = ""

                        # 成功后跳出重试循环，进入下一步
                        break

                step_end = datetime.now()
                step_timings[step_no] = {
                    "start": step_start.isoformat(timespec="seconds"),
                    "end": step_end.isoformat(timespec="seconds"),
                    "duration_seconds": round((step_end - step_start).total_seconds(), 3),
                    "attempts": step_attempts,
                    "success": step_success,
                }

                # 检查是否需要结束
                need_feedback = (decision == "FINISH" and run_status == "success")
                if decision == "FINISH":
                    break

            # 如果训练完成，执行反馈分析阶段
            if need_feedback:
                # === 反馈分析阶段（第二阶段） ===
                step_no += 1
                print("\n" + "-" * 80)
                print(f"[步骤 {step_no}] 反馈分析...")
                print("-" * 80, flush=True)

                step_dir = self.output_dir / f"step_{step_no:03d}"
                step_dir.mkdir(parents=True, exist_ok=True)

                self._prepare_feedback_assets()

                # 使用 CodeReAct 模式执行反馈分析
                feedback_iterations = 0
                max_feedback_iterations = 8
                report_saved = False

                while feedback_iterations < max_feedback_iterations:
                    feedback_iterations += 1
                    attempt_dir = step_dir / f"feedback_iter_{feedback_iterations:03d}"
                    attempt_dir.mkdir(parents=True, exist_ok=True)

                    # 最后一次迭代时使用 final_prompt
                    is_final_iteration = (feedback_iterations == max_feedback_iterations)
                    prompt = self._render_feedback_prompt(force_final=is_final_iteration)
                    write_text(attempt_dir / "prompt.md", prompt)

                    llm_text = self._call_llm(prompt)
                    write_text(attempt_dir / "llm_response.md", llm_text)

                    # 提取状态
                    status = _extract_feedback_status(llm_text)

                    # 检查是否结束
                    if status == "finish" or "程序结束OVER" in (llm_text or ""):
                        # 提取并保存反馈报告（必须有 Feedback_Report）
                        report_match = re.search(r'```Feedback_Report\s*(.*?)\s*```', llm_text, re.DOTALL)
                        if report_match:
                            report_content = report_match.group(1).strip()
                            feedback_dir = self.output_dir / "feedback"
                            feedback_dir.mkdir(parents=True, exist_ok=True)
                            write_text(feedback_dir / "report.md", report_content)
                            print("✓ 反馈分析完成", flush=True)
                            print("✓ 反馈报告已保存: feedback/report.md", flush=True)
                            report_saved = True
                            break
                        note = "LLM 已请求结束但未输出 Feedback_Report，请补全报告后再结束。"
                        self._append_observation("反馈报告缺失", note)
                        print("✗ 反馈报告缺失，继续下一轮", flush=True)
                        continue

                    # 提取并执行代码
                    purpose = _extract_analysis_purpose(llm_text)
                    code, n_blocks = _extract_first_python_block(llm_text)
                    if not code:
                        print("  未生成代码，继续", flush=True)
                        self._append_observation("分析失败", "LLM 未返回可执行的 Python 代码块。")
                        continue

                    exec_ok, exec_err, exec_stdout = self._exec_code(code=code)
                    write_text(attempt_dir / "code.py", code)

                    # 打印输出（截断最后2000字符）
                    parts = []
                    if purpose:
                        parts.append(f"分析目的: {purpose}")
                    if status:
                        parts.append(f"状态: {status}")

                    if exec_ok:
                        parts.append(f"✓ 执行成功")
                        # 打印执行输出（截断）
                        truncated_output = exec_stdout[-2000:] if len(exec_stdout) > 2000 else exec_stdout
                        if truncated_output.strip():
                            print("  " + "\n  ".join(parts), flush=True)
                            print(f"  输出:\n{truncated_output}", flush=True)
                        else:
                            print("  " + " | ".join(parts), flush=True)
                        # 更新 observation
                        self._append_observation(f"迭代{feedback_iterations}: {purpose}", truncated_output)
                    else:
                        parts.append(f"✗ 执行失败: {exec_err[:200]}")
                        print("  " + " | ".join(parts), flush=True)
                        write_text(attempt_dir / "error.txt", exec_err)
                        self._append_observation(f"迭代{feedback_iterations}: 执行失败", exec_err)
                        continue

                if not report_saved:
                    final_dir = step_dir / "feedback_final"
                    final_dir.mkdir(parents=True, exist_ok=True)
                    final_prompt = self._render_feedback_prompt(force_final=True)
                    final_prompt += "\n\n必须输出 ```Feedback_Report``` 块，否则视为未完成。"
                    write_text(final_dir / "prompt.md", final_prompt)
                    llm_text = self._call_llm(final_prompt)
                    write_text(final_dir / "llm_response.md", llm_text)
                    report_match = re.search(r'```Feedback_Report\s*(.*?)\s*```', llm_text, re.DOTALL)
                    feedback_dir = self.output_dir / "feedback"
                    feedback_dir.mkdir(parents=True, exist_ok=True)
                    if report_match:
                        report_content = report_match.group(1).strip()
                        write_text(feedback_dir / "report.md", report_content)
                        print("✓ 反馈报告已保存: feedback/report.md", flush=True)
                    else:
                        fallback = (
                            "# 训练反馈报告\n\n"
                            "## 一、结果复盘\n"
                            "（自动生成：未从 LLM 获取到 Feedback_Report）\n\n"
                            "## 二、发现的问题\n"
                            "（自动生成：参考历史观测与错误日志）\n\n"
                            "## 三、改进建议\n"
                            "请重新运行反馈分析以生成完整报告。\n"
                        )
                        write_text(feedback_dir / "report.md", fallback)
                        write_text(final_dir / "error.txt", "未获取到 Feedback_Report，已写入占位报告。")
                        print("✗ 未获取到 Feedback_Report，已写入占位报告", flush=True)

            # 检查是否达到最大步数
            if step_no >= self.max_steps and decision != "FINISH":
                print(f"\n警告: 已达到最大步数 {self.max_steps}，强制结束", flush=True)

        # 保存完整代码
        full_code_parts = []
        for r in self.step_results:
            if r.success:
                full_code_parts.append(f"# 步骤 {r.step_no}: {r.execution_content}\n{r.code}")

        if full_code_parts:
            full_notebook = "# AutoHealth 完整执行记录\n"
            full_notebook += f"# 创建时间: {datetime.now().isoformat(timespec='seconds')}\n"
            full_notebook += f"# 输出目录: {self.output_dir.absolute()}\n\n"
            full_notebook += "\n\n".join(full_code_parts)
            notebook_path = self.output_dir / "full_notebook.py"
            write_text(notebook_path, full_notebook)
            print(f"\n完整notebook已保存: {notebook_path.absolute()}")

        duration = (datetime.now() - start).total_seconds()
        self._cleanup_resources()
        summary = {
            "output_dir": str(self.output_dir.absolute()),
            "status": run_status,
            "total_steps": len(self.step_results),
            "successful_steps": sum(1 for r in self.step_results if r.success),
            "duration_seconds": f"{duration:.1f}",
            "final_error": final_error if run_status == "failed" else None,
            "notebook_path": str((self.output_dir / "full_notebook.py").absolute()) if full_code_parts else None,
            "token_usage": self.llm.get_usage() if hasattr(self.llm, "get_usage") else {},
            "vision_token_usage": self.vision_llm.get_usage() if self.vision_llm and hasattr(self.vision_llm, "get_usage") else {},
            "memory_cleanup": True,
            "step_durations_path": str((self.output_dir / "step_durations.json").absolute()),
        }

        write_text(self.output_dir / "step_durations.json", json.dumps(step_timings, ensure_ascii=False, indent=2, default=str))
        summary_path = self.output_dir / "run_summary.json"
        write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2, default=str))

        return summary
