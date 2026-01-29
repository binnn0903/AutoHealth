#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple


def _find_task_files(task_root: Path) -> List[Path]:
    return sorted(task_root.glob("*/task.txt"), reverse=True)


def _build_env(project_root: Path) -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing}" if existing else str(project_root)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _run_task(
    *,
    task_file: Path,
    python_bin: str,
    max_rounds: int,
    patience: int,
    min_delta: float,
    logs_dir: Path,
    env: dict,
    output_root: Path | None,
) -> Tuple[str, int, str]:
    task_name = task_file.parent.name
    output_root_task = None
    if output_root:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root_task = output_root / task_name / run_ts
    log_path = logs_dir / f"{task_name}.log"
    code = "\n".join(
        [
            "import os",
            "from AutoHealth.run_pipeline import run_pipeline",
            (
                "kwargs = dict(task_file=os.environ['TASK_FILE'], "
                f"max_rounds={max_rounds}, patience={patience}, min_delta={min_delta})"
            ),
            "out_root = os.environ.get('OUTPUT_ROOT')",
            "if out_root:",
            "    kwargs['output_root'] = out_root",
            "run_pipeline(**kwargs)",
        ]
    )
    run_env = env.copy()
    run_env["TASK_FILE"] = str(task_file)
    if output_root_task:
        run_env["OUTPUT_ROOT"] = str(output_root_task)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            [python_bin, "-c", code],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=run_env,
            text=True,
        )
    return task_name, proc.returncode, str(log_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AutoHealth pipeline for all tasks.")
    parser.add_argument("--task-root", default="/root/Dataset", help="Root directory containing tasks.")
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent tasks.")
    parser.add_argument("--max-rounds", type=int, default=5, help="Max optimization rounds per task.")
    parser.add_argument("--patience", type=int, default=1, help="Early stopping patience.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum delta for improvement.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--output-root", default="", help="Output root directory for pipeline runs.")
    args = parser.parse_args()

    task_root = Path(args.task_root).resolve()
    if not task_root.exists():
        raise SystemExit(f"Task root not found: {task_root}")

    task_files = _find_task_files(task_root)
    if not task_files:
        raise SystemExit(f"No task.txt found under: {task_root}")

    project_root = Path(__file__).resolve().parents[2]
    logs_dir = project_root / "outputs" / "batch_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = _build_env(project_root)
    output_root = Path(args.output_root).resolve() if args.output_root else None

    print(f"Found {len(task_files)} tasks under {task_root}")
    print(f"Logs directory: {logs_dir}")
    print(f"Workers: {args.workers} | Max rounds: {args.max_rounds}")
    if output_root:
        print(f"Output root: {output_root}")

    results: List[Tuple[str, int, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                _run_task,
                task_file=task_file,
                python_bin=args.python_bin,
                max_rounds=args.max_rounds,
                patience=args.patience,
                min_delta=args.min_delta,
                logs_dir=logs_dir,
                env=env,
                output_root=output_root,
            )
            for task_file in task_files
        ]
        for fut in as_completed(futures):
            name, code, log_path = fut.result()
            status = "ok" if code == 0 else f"fail({code})"
            print(f"[{status}] {name} -> {log_path}", flush=True)
            results.append((name, code, log_path))

    failed = [r for r in results if r[1] != 0]
    print(f"Completed: {len(results)} | Failed: {len(failed)}")
    if failed:
        print("Failed tasks:")
        for name, code, log_path in failed:
            print(f"- {name}: {log_path} (exit {code})")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
