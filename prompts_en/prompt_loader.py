from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_prompt_template(template_name: str) -> Dict[str, Any]:
    prompts_dir = Path(__file__).parent
    template_path = prompts_dir / f"{template_name}.yaml"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}


def render_prompt(template: str, variables: Dict[str, Any]) -> str:
    result = template or ""
    for key, value in variables.items():
        result = result.replace("{" + key + "}", str(value))
    return result
