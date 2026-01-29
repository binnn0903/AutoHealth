from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agents import (
    CodeExecutionAgent,
    DataUnderstandingAgent,
    MetaAgent,
    PlanningAgent,
    ReportGenerationAgent,
    SimpleReportInput,
)
from .config import get_agent_config, get_execution_config, get_llm_config, load_config
from .llm import OpenAICompatClient


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _truncate_tail(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if max_chars <= 0:
        return ""
    return text if len(text) <= max_chars else text[-max_chars:]


def _truncate_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if max_chars <= 0:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + "\n...(截断)"


def _usage_from_client(client: OpenAICompatClient, *, fallback_dir: Optional[Path] = None) -> Dict[str, int]:
    usage = client.get_usage() if hasattr(client, "get_usage") else {}
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


def _parse_task_file(task_path: Path) -> Tuple[str, str, Dict[str, str]]:
    text = _read_text(task_path)
    task_name = ""
    eval_metric = ""
    data_paths: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("task name:"):
            task_name = line.split(":", 1)[1].strip()
            continue
        if line.lower().startswith("evaluation metric:"):
            eval_metric = line.split(":", 1)[1].strip()
            continue

        m = re.match(r"^-?\s*(train|validation|test)\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if m:
            key = m.group(1).strip().lower()
            path = m.group(2).strip()
            data_paths[key] = path

    task_name = task_name or task_path.stem
    eval_metric = eval_metric or "RMSLE"
    return task_name, eval_metric, data_paths


def _find_metrics_file(output_dir: Path) -> Optional[Path]:
    candidates = [
        "test_results.json",
        "evaluation_results.json",
        "validation_results.json",
        "metrics.json",
        "test_results.txt",
        "evaluation_results.csv",
    ]
    for name in candidates:
        p = output_dir / name
        if p.exists() and p.is_file():
            return p
    for p in output_dir.rglob("test_results.json"):
        return p
    for p in output_dir.rglob("evaluation_results.json"):
        return p
    return None


def _extract_metric_value(metrics_path: Optional[Path], metric_name: str) -> Optional[float]:
    if not metrics_path or not metrics_path.exists():
        return None

    raw = metrics_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return None

    obj: Any
    try:
        obj = json.loads(raw)
    except Exception:
        try:
            obj = json.loads(json.loads(raw))
        except Exception:
            return None

    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return None

    metric_key = metric_name.strip().lower()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                for k2, v2 in value.items():
                    if str(k2).strip().lower() == metric_key:
                        try:
                            return float(v2)
                        except Exception:
                            return None
            if str(key).strip().lower() == metric_key:
                try:
                    return float(value)
                except Exception:
                    return None
    return None


def _build_device_info() -> str:
    # 优先使用外部指定的设备信息，便于在脚本中覆盖
    env_override = os.environ.get("DEVICE_INFO")
    if env_override:
        return env_override.strip()

    cpu_cores_env = os.environ.get("CPU_CORES")
    gpu_name_env = os.environ.get("GPU_NAME")
    gpu_mem_env = os.environ.get("GPU_MEM_GB")
    disable_gpu = os.environ.get("GPU_OFF")

    lines = []
    # CPU 信息
    try:
        cores = int(cpu_cores_env) if cpu_cores_env else os.cpu_count()
    except Exception:
        cores = os.cpu_count()
    if cores:
        lines.append(f"CPU: {cores} cores")

    # GPU 信息
    if disable_gpu:
        lines.append("GPU: disabled by env (GPU_OFF)")
    elif gpu_name_env:
        mem = f", VRAM: {gpu_mem_env} GiB" if gpu_mem_env else ""
        lines.append(f"GPU: {gpu_name_env}{mem}")
    else:
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                total_gb = props.total_memory / (1024 ** 3)
                lines.append(f"GPU: {props.name}, VRAM: {total_gb:.1f} GiB")
            else:
                lines.append("GPU: 不可用")
        except Exception as exc:
            lines.append(f"GPU: 未知（{exc.__class__.__name__}）")

    return "\n".join(lines).strip() or "（无）"


@dataclass
class RoundArtifacts:
    round_idx: int
    data_report_path: str = ""
    plan_path: str = ""
    code_execution_dir: str = ""
    code_execution_status: str = ""
    feedback_report_path: str = ""
    metrics_path: str = ""
    primary_metric_value: Optional[float] = None
    stage_durations: Dict[str, float] = None
    stage_token_usage: Dict[str, Dict[str, int]] = None


def run_pipeline(
    *,
    task_file: str,
    output_root: Optional[str] = None,
    max_rounds: int = 5,
    patience: int = 1,
    min_delta: float = 0.0,
    resume_round_dir: Optional[str] = None,
    resume_from_stage: Optional[str] = None,
) -> Dict[str, Any]:
    """
    运行完整管线：数据理解 -> 计划产生 -> 代码生成 -> 反馈分析 -> Meta 决策
    若 Meta 决策结束，则进入报告生成。
    """
    task_path = Path(task_file).resolve()
    task_text = _read_text(task_path)
    task_name, eval_metric, data_paths = _parse_task_file(task_path)

    project_root = Path(__file__).resolve().parent
    if resume_round_dir:
        pipeline_root = Path(resume_round_dir).resolve().parent
    elif output_root:
        pipeline_root = Path(output_root).resolve()
    else:
        pipeline_root = project_root / "outputs" / "pipeline" / task_name / _now_ts()
    pipeline_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    device_info = _build_device_info()

    def _make_client(agent_name: str) -> Tuple[OpenAICompatClient, Dict[str, Any]]:
        llm_cfg = get_llm_config(cfg, agent_name=agent_name)
        client = OpenAICompatClient(api_key=llm_cfg["api_key"], base_url=llm_cfg["base_url"])
        return client, llm_cfg

    history_summary: Dict[str, Any] = {
        "best_metric_value": None,
        "best_round_idx": None,
        "no_improve_rounds": 0,
        "previous_rounds": [],
    }

    artifacts_by_round: List[RoundArtifacts] = []
    next_start = "data_understanding"
    report_result: Optional[Dict[str, str]] = None
    snapshots_path = pipeline_root / "snapshots.md"

    start_round = 1
    resume_dir = None
    if resume_round_dir:
        cand = Path(resume_round_dir).resolve()
        if cand.exists():
            resume_dir = cand
            match = re.search(r"round_(\d+)", str(resume_round_dir))
            if match:
                start_round = int(match.group(1))

    for round_idx in range(start_round, max_rounds + 1):
        round_dir = resume_dir if resume_dir and round_idx == start_round else (pipeline_root / f"round_{round_idx:02d}")
        round_dir.mkdir(parents=True, exist_ok=True)

        data_report_path = ""
        plan_path = ""
        plan_text = ""
        feedback_report_text = ""
        execution_result_text = ""
        stage_durations: Dict[str, float] = {}
        stage_token_usage: Dict[str, Dict[str, int]] = {}

        if resume_dir:
            data_report_path = str((resume_dir / "data_understanding" / "report.md").resolve())
            plan_path = str((resume_dir / "planning" / "final_plan.md").resolve())
            plan_text = _read_text(Path(plan_path)) if Path(plan_path).exists() else ""
            feedback_report_path = str((resume_dir / "code_execution" / "feedback" / "report.md").resolve())
            feedback_report_text = _read_text(Path(feedback_report_path)) if Path(feedback_report_path).exists() else ""
            execution_log_path = resume_dir / "code_execution" / "run_stdout.log"
            execution_result_text = _truncate_tail(_read_text(execution_log_path), 8000) if execution_log_path.exists() else ""
            ce_dir = resume_dir / "code_execution"
            metrics_path = _find_metrics_file(ce_dir) if ce_dir.exists() else None
            primary_metric_value = _extract_metric_value(metrics_path, eval_metric)
            if not resume_from_stage:
                resume_from_stage = "meta"

        # 数据理解
        if resume_dir and resume_from_stage in {"planning", "code_execution", "meta"}:
            pass
        elif next_start == "data_understanding" or not artifacts_by_round:
            du_dir = round_dir / "data_understanding"
            du_client, du_llm_cfg = _make_client("data_understanding")
            du_exec_cfg = get_execution_config(cfg, agent_name="data_understanding")
            du_agent_cfg = get_agent_config(cfg, agent_name="data_understanding")

            prev_report_path = artifacts_by_round[-1].data_report_path if artifacts_by_round else None
            additional_requirements = ""
            if artifacts_by_round and artifacts_by_round[-1].feedback_report_path:
                additional_requirements = _read_text(Path(artifacts_by_round[-1].feedback_report_path))
            if artifacts_by_round and meta_summary:
                next_start_reason = str(meta_summary.get("next_start_reason") or "")
                if next_start_reason:
                    additional_requirements = next_start_reason

            du_agent = DataUnderstandingAgent(
                task_description=task_text,
                data_paths=data_paths,
                output_dir=str(du_dir),
                llm_client=du_client,
                llm_model=str(du_llm_cfg.get("model", "deepseek-chat")),
                llm_temperature=float(du_llm_cfg.get("temperature", 0.3)),
                llm_top_p=float(du_llm_cfg.get("top_p", 0.3)),
                llm_max_tokens=int(du_llm_cfg.get("max_tokens", 8192)),
                llm_extra=(du_llm_cfg.get("extra") if isinstance(du_llm_cfg.get("extra"), dict) else None),
                conda_env=str(du_exec_cfg.get("conda_env", "dl110")),
                timeout_seconds=int(du_exec_cfg.get("timeout_seconds", 600)),
                max_iterations=int(du_agent_cfg.get("max_iterations", 15)),
                max_observation_chars=int(du_agent_cfg.get("max_observation_chars", 10000)),
                prompt_name="data_understanding",
                previous_report_path=prev_report_path,
                additional_requirements=additional_requirements or None,
            )
            du_start = datetime.now()
            du_result = du_agent.run()
            stage_durations["data_understanding"] = (datetime.now() - du_start).total_seconds()
            stage_token_usage["data_understanding"] = _usage_from_client(du_client, fallback_dir=du_dir)
            du_tokens = stage_token_usage["data_understanding"]
            print(
                f"[DataUnderstandingAgent] tokens in={du_tokens['input_tokens']} out={du_tokens['output_tokens']} "
                f"cache={du_tokens['cache_hit_tokens']} total={du_tokens['total_tokens']} "
                f"time={stage_durations['data_understanding']:.1f}s",
                flush=True,
            )
            data_report_path = du_result.get("report_path", "")
        else:
            data_report_path = artifacts_by_round[-1].data_report_path

        # 计划产生
        if resume_dir and resume_from_stage in {"code_execution", "meta"}:
            pass
        elif next_start in {"data_understanding", "planning"} or not artifacts_by_round:
            if not data_report_path and artifacts_by_round:
                data_report_path = artifacts_by_round[-1].data_report_path

            plan_dir = round_dir / "planning"
            plan_client, plan_llm_cfg = _make_client("planning")
            plan_agent_cfg = get_agent_config(cfg, agent_name="planning")

            previous_feedback = ""
            if snapshots_path.exists():
                previous_feedback = _read_text(snapshots_path)

            plan_agent = PlanningAgent(
                task_description=task_text,
                data_report=_read_text(Path(data_report_path)) if data_report_path else "",
                previous_feedback=previous_feedback,
                device_info=device_info,
                output_dir=str(plan_dir),
                llm_client=plan_client,
                llm_model=str(plan_llm_cfg.get("model", "deepseek-chat")),
                llm_temperature=float(plan_llm_cfg.get("temperature", 0.4)),
                llm_top_p=float(plan_llm_cfg.get("top_p", 0.7)),
                llm_max_tokens=int(plan_llm_cfg.get("max_tokens", 8192)),
                llm_extra=(plan_llm_cfg.get("extra") if isinstance(plan_llm_cfg.get("extra"), dict) else None),
                review_rounds=int(plan_agent_cfg.get("review_rounds", 1)),
                enable_retrieval=bool(plan_agent_cfg.get("enable_retrieval", False)),
                enable_kaggle_retrieval=bool(plan_agent_cfg.get("enable_kaggle_retrieval", False)),
                enable_arxiv_retrieval=bool(plan_agent_cfg.get("enable_arxiv_retrieval", False)),
                enable_web_retrieval=bool(plan_agent_cfg.get("enable_web_retrieval", False)),
                enable_uncertainty=bool(plan_agent_cfg.get("enable_uncertainty", True)),
                kaggle_top_k=int(plan_agent_cfg.get("kaggle_top_k", 10)),
                kaggle_language=str(plan_agent_cfg.get("kaggle_language", "python")),
                kaggle_sort_by=str(plan_agent_cfg.get("kaggle_sort_by", "relevance")),
                arxiv_top_k=int(plan_agent_cfg.get("arxiv_top_k", 5)),
                web_top_k=int(plan_agent_cfg.get("web_top_k", 5)),
                web_search_url=str(plan_agent_cfg.get("web_search_url", "https://duckduckgo.com/html/?q={query}")),
                max_chars_data_report=int(plan_agent_cfg.get("max_chars_data_report", 20000)),
                max_chars_feedback=int(plan_agent_cfg.get("max_chars_feedback", 8000)),
                max_chars_kaggle_context=int(plan_agent_cfg.get("max_chars_kaggle_context", 40000)),
                max_chars_notebook=int(plan_agent_cfg.get("max_chars_notebook", 20000)),
                uncertainty_methods_path=str(plan_agent_cfg.get("uncertainty_methods_path", "")) or None,
                prompt_name="planning",
            )
            plan_start = datetime.now()
            plan_result = plan_agent.run()
            stage_durations["planning"] = (datetime.now() - plan_start).total_seconds()
            stage_token_usage["planning"] = _usage_from_client(plan_client, fallback_dir=plan_dir)
            plan_tokens = stage_token_usage["planning"]
            print(
                f"[PlanningAgent] tokens in={plan_tokens['input_tokens']} out={plan_tokens['output_tokens']} "
                f"cache={plan_tokens['cache_hit_tokens']} total={plan_tokens['total_tokens']} "
                f"time={stage_durations['planning']:.1f}s",
                flush=True,
            )
            plan_path = plan_result.get("plan_path", "")
        else:
            plan_path = artifacts_by_round[-1].plan_path
        if plan_path:
            plan_text = _read_text(Path(plan_path))

        # 代码执行
        if resume_dir and resume_from_stage == "meta":
            ce_dir = resume_dir / "code_execution"
            code_status = "success" if (ce_dir / "run_summary.json").exists() else "unknown"
        else:
            ce_dir = round_dir / "code_execution"
            ce_client, ce_llm_cfg = _make_client("code_execution")
            ce_agent_cfg = get_agent_config(cfg, agent_name="code_execution")
            ce_exec_cfg = get_execution_config(cfg, agent_name="code_execution")

            code_agent = CodeExecutionAgent(
                task_description=task_text,
                data_report=_read_text(Path(data_report_path)) if data_report_path else "",
                plan_markdown=plan_text,
                device_info=device_info,
                output_dir=str(ce_dir),
                llm_client=ce_client,
                llm_model=str(ce_llm_cfg.get("model", "deepseek-chat")),
                llm_temperature=float(ce_llm_cfg.get("temperature", 0.3)),
                llm_top_p=float(ce_llm_cfg.get("top_p", 0.7)),
                llm_max_tokens=int(ce_llm_cfg.get("max_tokens", 8192)),
                llm_extra=(ce_llm_cfg.get("extra") if isinstance(ce_llm_cfg.get("extra"), dict) else None),
                max_steps=int(ce_agent_cfg.get("max_steps", 10)),
                max_retries=int(ce_agent_cfg.get("max_retries", 10)),
                conda_env=str(ce_exec_cfg.get("conda_env", "dl110")),
                enable_dependencies=bool(ce_agent_cfg.get("enable_dependencies", True)),
            )
            ce_start = datetime.now()
            ce_summary = code_agent.run()
            stage_durations["code_execution"] = (datetime.now() - ce_start).total_seconds()
            stage_token_usage["code_execution"] = _usage_from_client(ce_client, fallback_dir=ce_dir)
            ce_tokens = stage_token_usage["code_execution"]
            print(
                f"[CodeExecutionAgent] tokens in={ce_tokens['input_tokens']} out={ce_tokens['output_tokens']} "
                f"cache={ce_tokens['cache_hit_tokens']} total={ce_tokens['total_tokens']} "
                f"time={stage_durations['code_execution']:.1f}s",
                flush=True,
            )
            code_status = ce_summary.get("status", "unknown")

            metrics_path = _find_metrics_file(ce_dir)
            primary_metric_value = _extract_metric_value(metrics_path, eval_metric)

            # 反馈分析已内置在 CodeExecutionAgent 中
            feedback_report_path = str((ce_dir / "feedback" / "report.md").resolve())
            if not Path(feedback_report_path).exists():
                feedback_report_path = ""
            feedback_report_text = _read_text(Path(feedback_report_path)) if feedback_report_path else ""

            execution_log_path = ce_dir / "run_stdout.log"
            execution_result_text = _truncate_tail(_read_text(execution_log_path), 8000) if execution_log_path.exists() else ""

        # Meta 决策
        meta_dir = round_dir / "meta"
        meta_client, meta_llm_cfg = _make_client("meta")

        heuristic_decision = {
            "next_start": "planning" if code_status != "success" else "code_execution",
            "reason": "基于执行状态的兜底决策",
        }

        snapshot_path = round_dir / "round_snapshot.md"
        run_summary_path = ce_dir / "run_summary.json"
        snapshot_lines = [
            f"# Round {round_idx} Snapshot",
            "",
            "## Plan",
            _truncate_text(plan_text, 8000) or "（无）",
            "",
            "## Feedback Analysis",
            _truncate_text(feedback_report_text, 8000) or "（无）",
        ]
        snapshot_path.write_text("\n".join(snapshot_lines) + "\n", encoding="utf-8")
        with snapshots_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(snapshot_lines) + "\n\n")

        current_round_summary = {
            "plan_text": plan_text,
            "execution_result_text": execution_result_text,
            "feedback_report_text": feedback_report_text,
            "history_text": _read_text(snapshots_path) if snapshots_path.exists() else "",
        }

        meta_agent = MetaAgent(
            task_description=task_text,
            output_dir=str(meta_dir),
            llm_client=meta_client,
            llm_model=str(meta_llm_cfg.get("model", "deepseek-chat")),
            llm_temperature=float(meta_llm_cfg.get("temperature", 0.4)),
            llm_top_p=float(meta_llm_cfg.get("top_p", 0.7)),
            llm_max_tokens=int(meta_llm_cfg.get("max_tokens", 8192)),
            llm_extra=(meta_llm_cfg.get("extra") if isinstance(meta_llm_cfg.get("extra"), dict) else None),
            max_chars_task=3000,
            max_chars_context=8000,
            prompt_name="meta",
        )

        meta_start = datetime.now()
        meta_summary = meta_agent.run(
            round_idx=round_idx,
            max_rounds=max_rounds,
            patience=patience,
            min_delta=min_delta,
            evaluation_metric=eval_metric,
            history_summary=history_summary,
            current_round_summary=current_round_summary,
            heuristic_decision=heuristic_decision,
        )
        stage_durations["meta"] = (datetime.now() - meta_start).total_seconds()
        stage_token_usage["meta"] = _usage_from_client(meta_client, fallback_dir=meta_dir)
        meta_tokens = stage_token_usage["meta"]
        print(
            f"[MetaAgent] tokens in={meta_tokens['input_tokens']} out={meta_tokens['output_tokens']} "
            f"cache={meta_tokens['cache_hit_tokens']} total={meta_tokens['total_tokens']} "
            f"time={stage_durations['meta']:.1f}s",
            flush=True,
        )

        artifacts = RoundArtifacts(
            round_idx=round_idx,
            data_report_path=data_report_path,
            plan_path=plan_path,
            code_execution_dir=str(ce_dir),
            code_execution_status=code_status,
            feedback_report_path=feedback_report_path,
            metrics_path=str(metrics_path) if metrics_path else "",
            primary_metric_value=primary_metric_value,
            stage_durations=stage_durations,
            stage_token_usage=stage_token_usage,
        )
        artifacts_by_round.append(artifacts)

        history_summary["previous_rounds"].append(
            {
                "round_idx": round_idx,
                "primary_metric_value": primary_metric_value,
                "status": code_status,
                "plan_text": plan_text,
                "execution_result_text": execution_result_text,
                "feedback_report_text": feedback_report_text,
            }
        )

        if primary_metric_value is not None:
            best = history_summary.get("best_metric_value")
            if best is None or primary_metric_value < best - min_delta:
                history_summary["best_metric_value"] = primary_metric_value
                history_summary["best_round_idx"] = round_idx
                history_summary["no_improve_rounds"] = 0
            else:
                history_summary["no_improve_rounds"] = int(history_summary.get("no_improve_rounds", 0)) + 1

        action = meta_summary.get("action", "")
        next_start = meta_summary.get("next_start", "planning")

        if action == "stop":
            break

        # 报告生成
    if artifacts_by_round:
        last = artifacts_by_round[-1]
        report_dir = pipeline_root / "report_generation"
        report_client, report_llm_cfg = _make_client("report_generation")

        data_report_text = _read_text(Path(last.data_report_path)) if last.data_report_path else ""
        plan_text = _read_text(Path(last.plan_path)) if last.plan_path else ""

        code_exec_dir = Path(last.code_execution_dir)
        run_log_path = code_exec_dir / "run_stdout.log"
        training_output = _truncate_tail(_read_text(run_log_path), 12000) if run_log_path.exists() else ""

        image_descriptions: Dict[str, str] = {}
        cache_path = code_exec_dir / "image_descriptions.json"
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(data, dict):
                    image_descriptions = data
            except Exception:
                image_descriptions = {}

        report_start = datetime.now()
        report_agent = ReportGenerationAgent(
            task_description=task_text,
            output_dir=str(report_dir),
            llm_client=report_client,
            llm_model=str(report_llm_cfg.get("model", "deepseek-chat")),
            llm_temperature=float(report_llm_cfg.get("temperature", 0.7)),
            llm_top_p=float(report_llm_cfg.get("top_p", 0.7)),
            llm_max_tokens=int(report_llm_cfg.get("max_tokens", 8192)),
            llm_extra=(report_llm_cfg.get("extra") if isinstance(report_llm_cfg.get("extra"), dict) else None),
            compile_pdf=True,
        )

        report_inputs = SimpleReportInput(
            task_description=task_text,
            data_report=data_report_text,
            training_plan=plan_text,
            training_output=training_output,
            image_descriptions=image_descriptions,
            image_descriptions_path=str(cache_path.resolve()) if cache_path.exists() else None,
            title=f"AutoML Experiment Report: {task_name}",
            author="",
        )
        report_result = report_agent.run_simple(report_inputs)
        report_duration = (datetime.now() - report_start).total_seconds()
        report_result["duration_seconds"] = f"{report_duration:.1f}"
        report_result["token_usage"] = _usage_from_client(report_client, fallback_dir=report_dir)
        report_tokens = report_result["token_usage"]
        print(
            f"[ReportGenerationAgent] tokens in={report_tokens['input_tokens']} out={report_tokens['output_tokens']} "
            f"cache={report_tokens['cache_hit_tokens']} total={report_tokens['total_tokens']} "
            f"time={report_duration:.1f}s",
            flush=True,
        )

    summary = {
        "task_name": task_name,
        "task_file": str(task_path),
        "output_dir": str(pipeline_root),
        "rounds_completed": len(artifacts_by_round),
        "report_generation": report_result or {},
        "rounds": [artifact.__dict__ for artifact in artifacts_by_round],
    }
    summary_path = pipeline_root / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


__all__ = ["run_pipeline"]
