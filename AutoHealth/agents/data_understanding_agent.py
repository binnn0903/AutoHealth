from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from AutoHealth.llm import OpenAICompatClient
from AutoHealth.ptompts import load_prompt_template, render_prompt
from AutoHealth.tools import python_code_safety, run_python_file
from AutoHealth.tools.executor import write_text


_PY_BLOCK_RE = re.compile(r"```(?:python|py)[ \t]*\n(.*?)\n```", flags=re.IGNORECASE | re.DOTALL)


def _extract_first_python_block(text: str) -> Tuple[str, int]:
    blocks = _PY_BLOCK_RE.findall(text or "")
    if not blocks:
        return "", 0
    return blocks[0].strip(), len(blocks)


def _extract_fenced_block(text: str, tag: str) -> str:
    pattern = re.compile(rf"```{re.escape(tag)}\s*(.*?)```", flags=re.DOTALL)
    m = pattern.search(text or "")
    return (m.group(1) or "").strip() if m else ""


class DataUnderstandingAgent:
    def __init__(
        self,
        *,
        task_description: str,
        data_paths: Dict[str, str],
        output_dir: str,
        llm_client: OpenAICompatClient,
        llm_model: str,
        llm_temperature: float,
        llm_top_p: float,
        llm_max_tokens: int,
        llm_extra: Optional[Dict[str, Any]] = None,
        conda_env: str = "dl110",
        timeout_seconds: int = 600,
        max_iterations: int = 15,
        max_observation_chars: int = 10000,
        prompt_name: str = "data_understanding",
        previous_report_path: Optional[str] = None,
        additional_requirements: Optional[str] = None,
    ):
        self.task_description = task_description
        self.data_paths = data_paths

        # 迭代式探索支持：读取上一次的报告
        self.previous_report = ""
        self.additional_requirements = additional_requirements or ""
        if previous_report_path and Path(previous_report_path).exists():
            try:
                self.previous_report = Path(previous_report_path).read_text(encoding="utf-8")
            except Exception as e:
                print(f"⚠️  无法读取上一次的报告: {e}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir = self.output_dir / "trace"
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        self.llm = llm_client
        self.model = llm_model
        self.temperature = llm_temperature
        self.top_p = llm_top_p
        self.max_tokens = llm_max_tokens
        self.llm_extra = dict(llm_extra or {})

        self.conda_env = conda_env
        self.timeout_seconds = timeout_seconds
        self.max_iterations = max_iterations
        self.max_observation_chars = max_observation_chars
        self.prompt_templates = load_prompt_template(prompt_name)

        self.iteration = 0
        # 存储探索记录：purpose/output/code_snippet
        self.exploration_history = []

    def _get_history_summary(self) -> str:
        """获取历史探索记录的摘要"""
        if not self.exploration_history:
            return "（暂无历史探索）"

        lines = ["## 历史探索记录\n"]
        for idx, record in enumerate(self.exploration_history, start=1):
            purpose = record.get("purpose") or "（未说明探索目的）"
            output = record.get("output") or "（无终端输出）"
            code_snippet = record.get("code") or ""
            lines.append(f"### 第 {idx} 轮")
            lines.append(f"**探索目的**: {purpose}")
            # 代码片段截断（前1000字符）
            code_preview = code_snippet[:1000] + ("...(截断)" if len(code_snippet) > 1000 else "")
            lines.append("**代码片段（前1000字符）**:")
            lines.append(f"```\n{code_preview or '（未生成代码）'}\n```\n")
            lines.append(f"**终端输出**:")
            # 截断过长的输出（增加到8000字符，确保数据探索输出完整）
            max_output_chars = 8000
            truncated_output = output[:max_output_chars] + ("...(截断)" if len(output) > max_output_chars else "")
            lines.append(f"```\n{truncated_output}\n```\n")

        return "\n".join(lines)

    def _render_user_prompt(self, *, force_final: bool) -> str:
        # 判断是否为迭代式探索（第二次及以后）
        is_iterative = bool(self.previous_report)

        # 选择模板
        if is_iterative:
            tpl_key = "iterative_final_prompt" if force_final else "iterative_step_prompt"
            # 如果没有迭代式模板，降级使用普通模板
            if tpl_key not in self.prompt_templates:
                tpl_key = "final_prompt" if force_final else "step_prompt"
        else:
            tpl_key = "final_prompt" if force_final else "step_prompt"

        template = self.prompt_templates.get(tpl_key, "")

        context = {
            "task_description": self.task_description,
            "history_summary": self._get_history_summary(),
            "current_iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "output_dir": str(self.output_dir.absolute()),
        }

        # 如果是迭代式探索，添加额外的上下文
        if is_iterative:
            context["previous_report"] = self.previous_report
            context["additional_requirements"] = self.additional_requirements

        return render_prompt(template, context)

    def _call_llm(self, user_prompt: str) -> str:
        system_prompt = self.prompt_templates.get("system_prompt", "")
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

    def _usage_summary(self) -> Dict[str, int]:
        usage = self.llm.get_usage() if hasattr(self.llm, "get_usage") else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        cache_hit_tokens = int(usage.get("cache_hit_tokens", 0) or 0)
        return {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "cache_hit_tokens": cache_hit_tokens,
            "total_tokens": total_tokens if total_tokens else prompt_tokens + completion_tokens + cache_hit_tokens,
        }

    def _extract_report_fallback(self, llm_text: str) -> str:
        cleaned = (llm_text or "")
        marker = "程序结束OVER"
        idx = cleaned.find(marker)
        if idx >= 0:
            cleaned = cleaned[idx + len(marker):]
        cleaned = cleaned.strip()
        if not cleaned:
            return ""
        if cleaned.lower().startswith("data_analysis_report"):
            parts = cleaned.splitlines()
            cleaned = "\n".join(parts[1:]).strip() if len(parts) > 1 else ""
        return cleaned.strip()

    def run(self) -> Dict[str, str]:
        start = datetime.now()
        last_llm_text = ""
        print("=" * 80)
        print("[DataUnderstandingAgent] 开始运行")
        print(f"[DataUnderstandingAgent] output_dir = {self.output_dir.absolute()}")
        print(f"[DataUnderstandingAgent] max_iterations = {self.max_iterations}")
        print("=" * 80, flush=True)
        for i in range(1, self.max_iterations + 1):
            self.iteration = i
            step_dir = self.trace_dir / f"step_{i:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "-" * 80)
            print(f"[步骤 {i}] 生成探索代码...")
            print("-" * 80, flush=True)
            force_final = i == self.max_iterations
            prompt = self._render_user_prompt(force_final=force_final)
            write_text(step_dir / "prompt.md", prompt)

            llm_text = self._call_llm(prompt)
            last_llm_text = llm_text
            write_text(step_dir / "llm.md", llm_text)

            if "程序结束OVER" in llm_text:
                code, n_blocks = _extract_first_python_block(llm_text)
                if code:
                    note = (
                        "检测到同时输出 Python 代码块和最终报告（程序结束OVER），该输出格式无效。"
                        "请在同一轮只输出 A) 或 B) 之一。"
                    )
                    write_text(step_dir / "observation.txt", note)
                    self.exploration_history.append(
                        {
                            "purpose": "格式错误",
                            "output": note,
                            "code": code,
                        }
                    )
                    print("✗ 同时输出代码与报告，继续下一轮", flush=True)
                    continue

                print("✓ LLM 输出最终报告信号", flush=True)
                report = _extract_fenced_block(llm_text, "Data_Analysis_Report") or _extract_fenced_block(
                    llm_text, "Data_Analysis_Report".lower()
                )

                report_path = self.output_dir / "report.md"
                if not report:
                    report = self._extract_report_fallback(llm_text)
                if report:
                    write_text(report_path, report)

                duration = (datetime.now() - start).total_seconds()
                summary = {
                    "output_dir": str(self.output_dir.absolute()),
                    "report_path": str(report_path.absolute()),
                    "duration_seconds": f"{duration:.1f}",
                    "token_usage": self._usage_summary(),
                }
                write_text(self.output_dir / "run_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
                print(f"[DataUnderstandingAgent] 完成，耗时 {duration:.1f}s", flush=True)
                return summary

            # 提取探索目的（代码块之前的文字）
            purpose = ""
            code_start = llm_text.find("```")
            if code_start > 0:
                purpose_text = llm_text[:code_start].strip()
                # 取第一行或前100个字符作为目的
                purpose_lines = [line.strip() for line in purpose_text.split("\n") if line.strip()]
                purpose = purpose_lines[0] if purpose_lines else "（未说明探索目的）"
                if len(purpose) > 100:
                    purpose = purpose[:100] + "..."

            code, n_blocks = _extract_first_python_block(llm_text)
            if not code:
                # 调试信息：看看为什么提取失败
                debug_info = f"LLM未返回python代码块\n\n"
                debug_info += f"LLM响应长度: {len(llm_text)}\n"
                debug_info += f"是否包含 ```python: {'```python' in llm_text}\n"
                debug_info += f"是否包含 ```py: {'```py' in llm_text}\n"
                debug_info += f"LLM响应前500字符:\n{repr(llm_text[:500])}\n\n"
                debug_info += f"LLM响应后500字符:\n{repr(llm_text[-500:])}\n"
                write_text(step_dir / "observation.txt", debug_info)
                self.exploration_history.append(
                    {
                        "purpose": purpose or "数据探索",
                        "output": debug_info,
                        "code": "",
                    }
                )
                print("✗ 未检测到 Python 代码块，继续下一轮", flush=True)
                continue

            is_safe, reason = python_code_safety(code)
            if not is_safe:
                obs = f"代码不安全，已拒绝执行：{reason}"
                write_text(step_dir / "action.py", code)
                write_text(step_dir / "observation.txt", obs)
                # 记录失败的探索
                self.exploration_history.append(
                    {
                        "purpose": purpose or "安全检查失败",
                        "output": obs,
                        "code": code,
                    }
                )
                print(f"✗ 代码安全检查失败: {reason}", flush=True)
                continue

            action_path = step_dir / "action.py"
            write_text(action_path, code)

            project_root = Path(__file__).resolve().parents[2]
            existing_pp = os.environ.get("PYTHONPATH", "")
            merged_pp = str(project_root) if not existing_pp else str(project_root) + os.pathsep + existing_pp

            exec_res = run_python_file(
                str(action_path.absolute()),
                cwd=str(step_dir),
                conda_env=self.conda_env,
                timeout_seconds=self.timeout_seconds,
                env={
                    "AUTOCLINE_OUTPUT_DIR": str(self.output_dir.absolute()),
                    "AUTOCLINE_STEP_DIR": str(step_dir.absolute()),
                    "PYTHONPATH": merged_pp,
                },
            )

            stdout = (exec_res.stdout or "").strip()
            stderr = (exec_res.stderr or "").strip()
            write_text(step_dir / "stdout.txt", stdout)
            write_text(step_dir / "stderr.txt", stderr)
            print(f"[步骤 {i}] 执行完成: success={exec_res.success} return_code={exec_res.return_code}", flush=True)

            obs_text = (
                f"success={exec_res.success} return_code={exec_res.return_code}\n\n"
                f"--- stdout ---\n{stdout}\n\n--- stderr ---\n{stderr}\n"
            )
            write_text(step_dir / "observation.txt", obs_text)

            # 更新历史记录：（探索目的，终端输出，代码片段）
            terminal_output = stdout if exec_res.success else f"{stdout}\n[错误] {stderr}"
            self.exploration_history.append(
                {
                    "purpose": purpose or "数据探索",
                    "output": terminal_output,
                    "code": code,
                }
            )

        # 兜底：如果最后一轮未按要求产出最终块，再强制请求一次 final 输出
        fallback_dir = self.trace_dir / "final_fallback"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        final_prompt = self._render_user_prompt(force_final=True)
        write_text(fallback_dir / "prompt.md", final_prompt)
        llm_text = self._call_llm(final_prompt)
        write_text(fallback_dir / "llm.md", llm_text)

        if "程序结束OVER" in llm_text:
            report = _extract_fenced_block(llm_text, "Data_Analysis_Report")
            report_path = self.output_dir / "report.md"
            if not report:
                # Fallback: strip a bare leading label if fenced block is missing.
                cleaned = llm_text.replace("程序结束OVER", "").strip()
                if cleaned.lower().startswith("data_analysis_report"):
                    cleaned = cleaned.split("\n", 1)[1].strip() if "\n" in cleaned else ""
                report = cleaned.strip()
            if report:
                write_text(report_path, report)

            duration = (datetime.now() - start).total_seconds()
            summary = {
                "output_dir": str(self.output_dir.absolute()),
                "report_path": str(report_path.absolute()),
                "duration_seconds": f"{duration:.1f}",
                "note": "final_fallback_used",
                "token_usage": self._usage_summary(),
            }
            write_text(self.output_dir / "run_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"[DataUnderstandingAgent] 完成（fallback），耗时 {duration:.1f}s", flush=True)
            return summary

        duration = (datetime.now() - start).total_seconds()
        print(f"[DataUnderstandingAgent] 失败，耗时 {duration:.1f}s", flush=True)
        return {
            "output_dir": str(self.output_dir.absolute()),
            "duration_seconds": f"{duration:.1f}",
            "error": "未生成最终报告（LLM未输出程序结束OVER）",
            "last_llm_excerpt": (last_llm_text or "")[:1000],
        }
