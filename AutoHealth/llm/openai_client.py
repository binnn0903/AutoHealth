from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from openai import OpenAI


def encode_image(image_path: str) -> str:
    """将图像编码为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """获取图片的 MIME 类型"""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/jpeg"


class OpenAICompatClient:
    """
    使用 OpenAI SDK 调用 OpenAI-compat API（如智谱 open.bigmodel.cn）。
    """

    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._last_usage: Dict[str, int] = {}
        self._total_usage: Dict[str, int] = {}

    def _record_usage(self, usage: Optional[Any]) -> None:
        if not usage:
            return
        if isinstance(usage, dict):
            usage_dict = usage
        else:
            usage_dict = usage.__dict__ if hasattr(usage, "__dict__") else {}

        prompt_tokens = int(usage_dict.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage_dict.get("completion_tokens", 0) or 0)
        total_tokens = int(usage_dict.get("total_tokens", 0) or 0)

        cache_hit_tokens = 0
        if "prompt_cache_hit_tokens" in usage_dict:
            cache_hit_tokens = int(usage_dict.get("prompt_cache_hit_tokens", 0) or 0)
        elif "cache_hit_tokens" in usage_dict:
            cache_hit_tokens = int(usage_dict.get("cache_hit_tokens", 0) or 0)
        elif "cached_tokens" in usage_dict:
            cache_hit_tokens = int(usage_dict.get("cached_tokens", 0) or 0)
        else:
            details = usage_dict.get("prompt_tokens_details") or {}
            if not isinstance(details, dict) and hasattr(details, "__dict__"):
                details = details.__dict__
            cache_hit_tokens = int(details.get("cached_tokens", 0) or 0)

        self._last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cache_hit_tokens": cache_hit_tokens,
        }
        for key, value in self._last_usage.items():
            self._total_usage[key] = int(self._total_usage.get(key, 0) or 0) + int(value or 0)

    def get_usage(self) -> Dict[str, int]:
        return dict(self._total_usage)

    def reset_usage(self) -> None:
        self._last_usage = {}
        self._total_usage = {}

    def chat(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        top_p: float = 0.7,
        temperature: float = 0.9,
        max_tokens: int = 8192,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        extra = extra or {}
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        self._record_usage(getattr(completion, "usage", None))
        return (completion.choices[0].message.content or "").strip()

    def vision_chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        top_p: float = 0.7,
        temperature: float = 0.9,
        max_tokens: int = 2048,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        多模态对话（OpenAI-compat）。

        `messages` 需要是 OpenAI Chat Completions 兼容格式，且 content 可为:
          [{"type":"text","text":"..."},{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]
        """
        extra = extra or {}
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        self._record_usage(getattr(completion, "usage", None))
        return (completion.choices[0].message.content or "").strip()

    def analyze_image(
        self,
        *,
        image_path: str,
        prompt: str,
        model: str,
        top_p: float = 0.7,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        便捷方法：分析图片内容

        Args:
            image_path: 图片文件路径
            prompt: 提示词/问题
            model: 模型名称（如 glm-4v, glm-4.6v 等）
            top_p: 采样参数
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            extra: 额外参数

        Returns:
            模型的分析结果
        """
        # 编码图片
        image_base64 = encode_image(image_path)
        mime_type = get_image_mime_type(image_path)

        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                    },
                ],
            }
        ]

        return self.vision_chat(
            messages=messages,
            model=model,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )
