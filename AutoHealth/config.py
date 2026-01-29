from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_CACHE: Dict[str, Any] | None = None
_CACHE_PATH: str | None = None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并字典：override 中的非 None 值覆盖 base；子字典继续递归。
    """
    merged: Dict[str, Any] = dict(base or {})
    for key, value in (override or {}).items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _get_agent_block(cfg: Dict[str, Any], agent_name: Optional[str]) -> Dict[str, Any]:
    if not agent_name:
        return {}
    agents = cfg.get("agents") or {}
    if not isinstance(agents, dict):
        return {}
    block = agents.get(agent_name) or {}
    return dict(block) if isinstance(block, dict) else {}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    global _CACHE, _CACHE_PATH

    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")

    config_file = Path(config_path).resolve()
    if _CACHE is not None and _CACHE_PATH == str(config_file):
        return _CACHE

    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    _CACHE = cfg
    _CACHE_PATH = str(config_file)
    return cfg


def get_llm_config(cfg: Dict[str, Any], agent_name: Optional[str] = None) -> Dict[str, Any]:
    llm_cfg = dict(cfg.get("llm") or {})
    api_key = (llm_cfg.get("api_key") or "").strip()
    if not api_key:
        api_key = (os.environ.get("GLM_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("缺少 GLM API Key：请在 config.yaml 的 llm.api_key 填写，或设置环境变量 GLM_API_KEY")
    llm_cfg["api_key"] = api_key
    # agent 级别覆盖：只覆盖温度/采样等参数，默认不允许用空值覆盖 api_key
    agent_block = _get_agent_block(cfg, agent_name)
    agent_llm = dict(agent_block.get("llm") or {}) if isinstance(agent_block.get("llm"), dict) else {}
    if "api_key" in agent_llm and not str(agent_llm.get("api_key") or "").strip():
        agent_llm.pop("api_key", None)
    return _deep_merge(llm_cfg, agent_llm)


def get_execution_config(cfg: Dict[str, Any], agent_name: Optional[str] = None) -> Dict[str, Any]:
    base = dict(cfg.get("execution") or {})
    agent_block = _get_agent_block(cfg, agent_name)
    agent_exec = dict(agent_block.get("execution") or {}) if isinstance(agent_block.get("execution"), dict) else {}
    return _deep_merge(base, agent_exec)


def get_agent_config(cfg: Dict[str, Any], agent_name: Optional[str] = None) -> Dict[str, Any]:
    base = dict(cfg.get("agent") or {})
    agent_block = _get_agent_block(cfg, agent_name)
    agent_cfg = dict(agent_block.get("agent") or {}) if isinstance(agent_block.get("agent"), dict) else {}
    return _deep_merge(base, agent_cfg)
