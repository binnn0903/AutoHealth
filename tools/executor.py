from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
        }


def run_python_file(
    script_path: str,
    *,
    cwd: str,
    conda_env: str = "dl110",
    timeout_seconds: int = 600,
    env: Optional[Dict[str, str]] = None,
) -> ExecutionResult:
    cmd = ["conda", "run", "-n", conda_env, "python", script_path]
    run_env = os.environ.copy()
    if env:
        run_env.update({k: str(v) for k, v in env.items()})
    run_env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=run_env,
        )
        return ExecutionResult(
            success=(proc.returncode == 0),
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            return_code=proc.returncode,
        )
    except subprocess.TimeoutExpired as e:
        return ExecutionResult(
            success=False,
            stdout=(e.stdout or "") if hasattr(e, "stdout") else "",
            stderr=f"执行超时（{timeout_seconds}s）",
            return_code=-1,
        )
    except Exception as e:  # pragma: no cover
        return ExecutionResult(success=False, stdout="", stderr=str(e), return_code=-1)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or "", encoding="utf-8")
