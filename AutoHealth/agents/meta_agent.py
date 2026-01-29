from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from AutoHealth.llm import OpenAICompatClient
from AutoHealth.ptompts import load_prompt_template, render_prompt
from AutoHealth.tools.executor import write_text


def _extract_fenced_block(text: str, tag: str) -> str:
    pattern = re.compile(rf"```{re.escape(tag)}\s*(.*?)```", flags=re.DOTALL)
    m = pattern.search(text or "")
    return (m.group(1) or "").strip() if m else ""


def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if max_chars <= 0:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + "\n...(截断)"


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidate = raw[start : end + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _read_file_content(path: str, max_chars: int = 5000) -> str:
    """安全读取文件内容，用于提取各 Agent 的输出"""
    if not path:
        return "（未提供）"
    p = Path(path)
    if not p.exists() or not p.is_file():
        return "（文件不存在）"
    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
        return _truncate(content, max_chars)
    except Exception:
        return "（读取失败）"


def _read_file_content_tail(path: str, max_chars: int = 5000) -> str:
    """安全读取文件内容（尾部截断）"""
    if not path:
        return "（未提供）"
    p = Path(path)
    if not p.exists() or not p.is_file():
        return "（文件不存在）"
    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
        if max_chars <= 0:
            return ""
        return content if len(content) <= max_chars else content[-max_chars:]
    except Exception:
        return "（读取失败）"


def _validate_new_decision(decision: Dict[str, Any], *, allowed_next: List[str]) -> Tuple[bool, str]:
    """验证新格式的决策输出"""
    action = str(decision.get("action") or "").strip().lower()
    if action not in {"continue", "stop"}:
        return False, "decision.action 必须为 continue 或 stop"

    if action == "continue":
        next_start = str(decision.get("next_start") or "").strip()
        if next_start not in allowed_next:
            return False, f"decision.next_start 必须为 {allowed_next} 之一"

    # 可选字段校验
    if "stop_reason" in decision and not isinstance(decision["stop_reason"], str):
        return False, "decision.stop_reason 必须为字符串"
    if "decision_reason" in decision and not isinstance(decision["decision_reason"], str):
        return False, "decision.decision_reason 必须为字符串"

    return True, ""


class MetaAgent:
    """
    Meta 决策器：在每轮迭代结束后进行两个判断：
    1. 终止判断：是否达到最大迭代轮次
    2. 起点判断：决定下一轮从哪个智能体开始

    输入：
      - 轮次信息（当前轮次、最大轮次、patience、min_delta、评估指标）
      - 任务描述
      - 数据描述（来自 DataUnderstandingAgent）
      - 本轮计划（来自 PlanningAgent）
      - 本轮执行情况（来自 CodeExecutionAgent）
      - 历史执行情况
      - 反馈分析（来自 FeedbackReflectionAgent）

    输出：
      - decision.json：决策结果（action、next_start、stop_reason、decision_reason）
      - meta_summary.md：人类可读的决策总览
      - trace/：prompt 与 LLM 输出
    """

    ALLOWED_NEXT_START: List[str] = ["data_understanding", "planning", "code_execution"]

    def __init__(
        self,
        *,
        task_description: str,
        output_dir: str,
        llm_client: OpenAICompatClient,
        llm_model: str,
        llm_temperature: float,
        llm_top_p: float,
        llm_max_tokens: int,
        llm_extra: Optional[Dict[str, Any]] = None,
        max_chars_task: int = 3000,
        max_chars_context: int = 8000,
        prompt_name: str = "meta",
    ):
        self.task_description = task_description or ""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir = self.output_dir / "trace"
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        self.llm = llm_client
        self.model = llm_model
        self.temperature = float(llm_temperature)
        self.top_p = float(llm_top_p)
        self.max_tokens = int(llm_max_tokens)
        self.llm_extra = dict(llm_extra or {})

        self.max_chars_task = max(0, int(max_chars_task))
        self.max_chars_context = max(0, int(max_chars_context))
        self.prompt_templates = load_prompt_template(prompt_name)

    def _call_llm(self, *, system_prompt: str, user_prompt: str) -> str:
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

    def _write_trace(self, step_idx: int, name: str, *, prompt: str, llm_text: str) -> Path:
        step_dir = self.trace_dir / f"step_{step_idx:03d}_{name}"
        step_dir.mkdir(parents=True, exist_ok=True)
        write_text(step_dir / "prompt.md", prompt)
        write_text(step_dir / "llm.md", llm_text)
        return step_dir

    def _render(self, key: str, variables: Dict[str, Any]) -> str:
        template = str(self.prompt_templates.get(key, "") or "")
        return render_prompt(template, variables)

    def _extract_data_description(self, current_round_summary: Dict[str, Any]) -> str:
        """从 current_round_summary 提取数据描述"""
        return "（无数据报告）"

    def _extract_current_plan(self, current_round_summary: Dict[str, Any]) -> str:
        """从 current_round_summary 提取本轮计划"""
        plan_text = str(current_round_summary.get("plan_text") or "")
        if plan_text:
            return _truncate(plan_text, self.max_chars_context)
        return "（无计划文件）"

    def _extract_execution_status(self, current_round_summary: Dict[str, Any]) -> str:
        """从 current_round_summary 提取执行情况"""
        execution_result_text = str(current_round_summary.get("execution_result_text") or "")
        if execution_result_text:
            return _truncate(execution_result_text, self.max_chars_context)
        return "（无执行信息）"

    def _extract_history_summary(self, history_summary: Dict[str, Any]) -> str:
        """从 history_summary 提取历史概况"""
        history_text = str(history_summary.get("history_text") or "")
        if history_text:
            return _truncate(history_text, self.max_chars_context)
        best_metric = history_summary.get("best_metric_value")
        best_round = history_summary.get("best_round_idx")
        no_improve = history_summary.get("no_improve_rounds")
        previous_rounds = history_summary.get("previous_rounds", [])

        parts = []
        if best_metric is not None:
            parts.append(f"历史最佳指标: {best_metric} (轮次 {best_round})")
        if no_improve is not None:
            parts.append(f"连续无提升轮数: {no_improve}")

        if previous_rounds:
            parts.append("\n最近轮次:")
            for r in previous_rounds[-5:]:  # 最近5轮
                plan_text = str(r.get("plan_text") or "") or "（无计划）"
                exec_tail = str(r.get("execution_result_text") or "") or "（无执行结果）"
                parts.append(
                    "\n".join(
                        [
                            f"  - Round {r.get('round_idx')}: {r.get('primary_metric_value')} ({r.get('status')})",
                            "    [计划摘要]",
                            _truncate(plan_text, 1200),
                            "    [执行结果]",
                            exec_tail,
                        ]
                    )
                )

        return "\n".join(parts) if parts else "（无历史信息）"

    def _extract_feedback_analysis(self, current_round_summary: Dict[str, Any]) -> str:
        """从 current_round_summary 提取反馈分析"""
        feedback_text = str(current_round_summary.get("feedback_report_text") or "")
        if feedback_text:
            return _truncate(feedback_text, self.max_chars_context)
        return "（无反馈分析）"

    def run(
        self,
        *,
        round_idx: int,
        max_rounds: int,
        patience: int,
        min_delta: float,
        evaluation_metric: str,
        history_summary: Dict[str, Any],
        current_round_summary: Dict[str, Any],
        heuristic_decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """运行 Meta 决策器"""
        start = datetime.now()

        # 提取各部分输入
        data_description = self._extract_data_description(current_round_summary)
        current_plan = self._extract_current_plan(current_round_summary)
        execution_status = self._extract_execution_status(current_round_summary)
        history_text = self._extract_history_summary(history_summary)
        feedback_analysis = self._extract_feedback_analysis(current_round_summary)

        # 构建 prompt
        system_prompt = str(self.prompt_templates.get("system_prompt", "") or "")
        user_prompt = self._render(
            "decision_prompt",
            {
                "task_description": _truncate(self.task_description, self.max_chars_task) or "（无）",
                "round_idx": int(round_idx),
                "max_rounds": int(max_rounds),
                "patience": int(patience),
                "min_delta": float(min_delta),
                "evaluation_metric": str(evaluation_metric or "（未知）"),
                "data_description": data_description,
                "current_plan": current_plan,
                "execution_status": execution_status,
                "history_summary": history_text,
                "feedback_analysis": feedback_analysis,
            },
        )

        # 调用 LLM
        llm_text = self._call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
        self._write_trace(1, "decision", prompt=user_prompt, llm_text=llm_text)

        # 解析输出
        decision_obj = _safe_json_loads(_extract_fenced_block(llm_text, "decision_json")) or _safe_json_loads(llm_text) or {}
        # 验证决策
        ok, reason = _validate_new_decision(decision_obj, allowed_next=self.ALLOWED_NEXT_START)

        if not ok:
            # LLM 输出无效，使用启发式决策
            decision_obj = {
                "action": "stop" if int(round_idx) >= int(max_rounds) else "continue",
                "next_start": str(heuristic_decision.get("next_start", "planning")),
                "stop_reason": "达到最大轮次" if int(round_idx) >= int(max_rounds) else "",
                "decision_reason": f"LLM 决策解析失败: {reason}，已回退到启发式规则",
                "next_start_reason": str(heuristic_decision.get("reason", "使用默认起点")),
            }
        # 写入输出文件
        decision_path = self.output_dir / "decision.json"
        write_text(decision_path, json.dumps(decision_obj, ensure_ascii=False, indent=2))

        duration = (datetime.now() - start).total_seconds()
        run_summary = {
            "output_dir": str(self.output_dir.absolute()),
            "decision_path": str(decision_path.absolute()),
            "action": str(decision_obj.get("action") or ""),
            "next_start": str(decision_obj.get("next_start") or ""),
            "stop_reason": str(decision_obj.get("stop_reason") or ""),
            "next_start_reason": str(decision_obj.get("next_start_reason") or ""),
            "duration_seconds": f"{duration:.1f}",
        }
        write_text(self.output_dir / "run_summary.json", json.dumps(run_summary, ensure_ascii=False, indent=2))
        return run_summary
