from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from AutoHealth.llm import OpenAICompatClient
from AutoHealth.ptompts import load_prompt_template, render_prompt
from AutoHealth.tools.executor import write_text


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _latex_escape(text: str) -> str:
    if text is None:
        return ""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def _normalize_latex_text(text: str) -> str:
    if not text:
        return ""
    normalized = text
    normalized = re.sub(r"([A-Za-z])²", r"$\1^2$", normalized)
    normalized = re.sub(r"([A-Za-z])³", r"$\1^3$", normalized)
    normalized = re.sub(r"([0-9])²", r"$\1^2$", normalized)
    normalized = re.sub(r"([0-9])³", r"$\1^3$", normalized)
    normalized = normalized.replace("±", r"$\pm$")
    return normalized


def _escape_latex_outside_math(text: str) -> str:
    if not text:
        return ""
    out = []
    in_math = False
    for i, ch in enumerate(text):
        if ch == "$" and (i == 0 or text[i - 1] != "\\"):
            in_math = not in_math
            out.append(ch)
            continue
        if not in_math:
            if ch == "_" and (i == 0 or text[i - 1] != "\\"):
                out.append(r"\_")
                continue
            if ch == "%" and (i == 0 or text[i - 1] != "\\"):
                out.append(r"\%")
                continue
        out.append(ch)
    return "".join(out)


def _unescape_label_like_commands(text: str) -> str:
    commands = ["ref", "label", "eqref", "pageref", "cite"]
    for cmd in commands:
        pattern = r"(\\%s\{)([^}]*)\}" % cmd
        def repl(m):
            inner = m.group(2).replace(r"\_", "_")
            return m.group(1) + inner + "}"
        text = re.sub(pattern, repl, text)
    return text


def _normalize_includegraphics_paths(
    text: str,
    *,
    assets_dir: Path,
    missing_log: Path,
) -> str:
    asset_files = [p for p in assets_dir.glob("*") if p.is_file()]
    asset_set = {str(p.resolve()) for p in asset_files}
    name_map = {p.name: p for p in asset_files}

    def _pick_asset(path_str: str) -> Optional[Path]:
        raw = re.sub(r"^\\detokenize\\{(.+)\\}$", r"\\1", path_str)
        raw = raw.replace(r"\_", "_")
        raw_path = Path(raw)
        if raw_path.exists():
            return raw_path
        if raw in asset_set:
            return Path(raw)
        if raw_path.name in name_map:
            return name_map[raw_path.name]
        base = re.sub(r"_[0-9]+(?=\\.[A-Za-z0-9]+$)", "", raw_path.name)
        if base:
            for p in asset_files:
                if p.name == base or p.stem == Path(base).stem:
                    return p
        return None

    missing = []
    updated_parts = []
    last_idx = 0
    pattern = re.compile(r"\\includegraphics(?:\\[[^\\]]*\\])?\\{")
    for match in pattern.finditer(text):
        start = match.start()
        brace_start = match.end() - 1
        brace_end = text.find("}", brace_start + 1)
        if brace_end == -1:
            continue
        cmd = text[start:brace_start]
        has_options = "[" in cmd
        if not has_options:
            cmd = "\\includegraphics[width=0.9\\linewidth,keepaspectratio]"
        path = text[brace_start + 1:brace_end]
        chosen = _pick_asset(path)
        updated_parts.append(text[last_idx:start])
        if not chosen:
            missing.append(path)
            updated_parts.append("% missing image omitted")
        else:
            updated_parts.append(f"{cmd}{{\\detokenize{{{str(chosen.resolve())}}}}}")
        last_idx = brace_end + 1
    updated_parts.append(text[last_idx:])
    updated = "".join(updated_parts)

    if missing:
        write_text(missing_log, "\n".join(missing))

    return updated


def _copy_into(src: Path, dst_dir: Path) -> Optional[Path]:
    if not src.exists() or not src.is_file():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^0-9A-Za-z._-]+", "_", src.name)
    dst = dst_dir / safe_name
    if dst.exists():
        stem, suffix = dst.stem, dst.suffix
        for i in range(1, 10_000):
            cand = dst_dir / f"{stem}_{i}{suffix}"
            if not cand.exists():
                dst = cand
                break
    shutil.copy2(src, dst)
    return dst


def _find_template_entrypoint(template_dir: Path, preferred: Optional[str] = None) -> Optional[Path]:
    if preferred:
        p = (template_dir / preferred).resolve()
        if p.exists() and p.is_file():
            return p
    for name in ["main.tex", "report.tex", "paper.tex", "tex/main.tex"]:
        p = (template_dir / name).resolve()
        if p.exists() and p.is_file():
            return p
    for p in sorted(template_dir.rglob("*.tex")):
        try:
            s = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "\\begin{document}" in s:
            return p
    tex_files = sorted(template_dir.rglob("*.tex"))
    return tex_files[0] if tex_files else None


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _compile_latex(entry_tex: Path, *, workdir: Path, timeout_seconds: int = 900) -> Tuple[Optional[Path], str]:
    pipelines: List[List[List[str]]] = []
    if _which("latexmk"):
        if _which("xelatex"):
            pipelines.append(
                [
                    [
                        "latexmk",
                        "-xelatex",
                        "-pdf",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "-file-line-error",
                        entry_tex.name,
                    ]
                ]
            )
        else:
            pipelines.append(
                [["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", entry_tex.name]]
            )
    if _which("xelatex"):
        pipelines.append(
            [
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error", entry_tex.name],
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error", entry_tex.name],
            ]
        )
    if _which("pdflatex"):
        pipelines.append(
            [
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", entry_tex.name],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", entry_tex.name],
            ]
        )

    if not pipelines:
        return None, "未检测到 latexmk/xelatex/pdflatex，跳过PDF编译。"

    logs_by_pipeline: List[str] = []
    for pipeline in pipelines:
        log_lines: List[str] = []
        ok = True
        for cmd in pipeline:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(workdir),
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
                if proc.stdout:
                    log_lines.append(proc.stdout)
                if proc.stderr:
                    log_lines.append(proc.stderr)
                if proc.returncode != 0:
                    ok = False
                    break
            except Exception as e:
                log_lines.append(str(e))
                ok = False
                break

        pdf = entry_tex.with_suffix(".pdf")
        if ok and pdf.exists():
            return pdf, "\n".join(log_lines)[-8000:]
        logs_by_pipeline.append("\n".join(log_lines))

    pdf = entry_tex.with_suffix(".pdf")
    if pdf.exists():
        return pdf, "\n".join(logs_by_pipeline)[-8000:]
    return None, "\n".join(logs_by_pipeline)[-8000:]


@dataclass
class SimpleReportInput:
    """
    简化版报告生成输入

    输入：
    - task_description: 任务描述
    - data_report: 数据分析报告（Markdown 文本）
    - training_plan: 训练计划（Markdown 文本）
    - training_output: 训练执行结果/日志（文本）
    - image_descriptions: 图像描述字典 {path: description}
    - image_descriptions_path: 图像描述文件路径（可选）
    - template_dir: 预设模板目录（可选，默认使用内置模板）
    """
    task_description: str
    data_report: str
    training_plan: str
    training_output: str
    image_descriptions: Dict[str, str] = field(default_factory=dict)
    image_descriptions_path: Optional[str] = None
    template_dir: Optional[str] = None
    title: str = "AutoML Experiment Report"
    author: str = ""


class ReportGenerationAgent:
    """
    简化版报告生成智能体（LaTeX）

    功能：
    - 基于 SimpleReportInput 生成完整报告
    - 使用 LLM 一次性生成所有内容
    - 将内容注入 LaTeX 模板并（可选）编译 PDF
    """

    def __init__(
        self,
        *,
        task_description: str,
        output_dir: str,
        llm_client: OpenAICompatClient,
        llm_model: str = "deepseek-chat",
        llm_temperature: float = 0.7,
        llm_top_p: float = 0.7,
        llm_max_tokens: int = 8192,
        llm_extra: Optional[Dict[str, Any]] = None,
        compile_pdf: bool = True,
    ):
        self.task_description = task_description
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
        self.compile_pdf = bool(compile_pdf)

    def run_simple(self, inputs: SimpleReportInput) -> Dict[str, str]:
        """
        简化版报告生成：一次性 LLM 生成完整报告

        输入：SimpleReportInput
        输出：Dict with keys: output_dir, tex_path, pdf_path, duration_seconds
        """
        start = datetime.now()

        if not self.llm:
            raise RuntimeError("LLM client is required for simplified report generation")

        # 准备工作目录
        report_root = self.output_dir / "report"
        report_root.mkdir(parents=True, exist_ok=True)
        assets_dir = report_root / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        # 选择模板
        if inputs.template_dir:
            tpl_dir = Path(inputs.template_dir).resolve()
        else:
            tpl_dir = Path(__file__).resolve().parents[1] / "templates" / "latex_default"

        template_dst = report_root / "template"
        if template_dst.exists():
            shutil.rmtree(template_dst)
        shutil.copytree(tpl_dir, template_dst)

        entry_tex = _find_template_entrypoint(template_dst)
        if not entry_tex:
            raise FileNotFoundError(f"Cannot find .tex file in template: {template_dst}")

        # 准备图像信息
        images_by_category = {"data_analysis": [], "training": [], "uncertainty": []}
        for img_path, desc in inputs.image_descriptions.items():
            src = Path(img_path)
            if src.exists():
                dst = _copy_into(src, assets_dir)
                if dst:
                    abs_path = str(dst.resolve())
                    img_info = f"- **{abs_path}**\n{desc}"
                    if "data_analysis" in img_path:
                        images_by_category["data_analysis"].append(img_info)
                    elif "uncertainty" in img_path:
                        images_by_category["uncertainty"].append(img_info)
                    else:
                        images_by_category["training"].append(img_info)

        # 准备图像描述文本
        images_text = "\n\n".join(
            f"## {cat.upper()} IMAGES:\n" + "\n".join(imgs)
            for cat, imgs in images_by_category.items() if imgs
        )

        # 读取模板内容（传递给 LLM）
        template_content = entry_tex.read_text(encoding="utf-8", errors="ignore")

        # 准备 prompt 变量
        prompt_dict = load_prompt_template("report_generation")

        prompt_vars = {
            "task_description": inputs.task_description,
            "data_report": inputs.data_report[:30000],
            "training_plan": inputs.training_plan[:20000],
            "training_output": inputs.training_output[:20000],
            "images_info": images_text,
            "latex_template": template_content,
            "title": inputs.title,
            "author": inputs.author,
        }

        def _truncate_log(log_text: str, max_chars: int = 8000) -> str:
            if not log_text:
                return ""
            return log_text if len(log_text) <= max_chars else log_text[-max_chars:]

        # 获取具体的模板字符串
        template_str = prompt_dict.get("generate_full_report_simple", prompt_dict.get("system_prompt", ""))

        # 编译失败时最多重试 3 次（包含首次）
        max_attempts = 3
        pdf_path: Optional[Path] = None
        compile_log = ""
        generation_attempts = 0

        for attempt in range(1, max_attempts + 1):
            generation_attempts = attempt
            prompt = render_prompt(template_str, prompt_vars)
            if attempt > 1 and compile_log:
                prompt += (
                    "\n\n## LaTeX compile error log\n"
                    + _truncate_log(compile_log)
                    + "\n\n请根据错误修复 LaTeX，并输出完整文档。"
                )

            prompt_path = (
                self.trace_dir / f"simple_generation_prompt_attempt_{attempt:02d}.md"
                if attempt > 1
                else self.trace_dir / "simple_generation_prompt.md"
            )
            write_text(prompt_path, prompt)

            # 调用 LLM 生成报告
            content = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.7,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra=self.llm_extra,
            )

            response_path = (
                self.trace_dir / f"simple_generation_response_attempt_{attempt:02d}.md"
                if attempt > 1
                else self.trace_dir / "simple_generation_response.md"
            )
            write_text(response_path, content)

            # 提取 LaTeX 内容（LLM 输出完整文档）
            latex_match = re.search(r"```latex\s*\n(.*?)```", content, re.DOTALL | re.IGNORECASE)
            if latex_match:
                full_latex = latex_match.group(1).strip()
            else:
                full_latex = content.strip()

            full_latex = _normalize_latex_text(full_latex)
            full_latex = _escape_latex_outside_math(full_latex)
            full_latex = _unescape_label_like_commands(full_latex)

            # 规范化 \includegraphics 路径并自动修复不存在的文件
            missing_img_log = self.trace_dir / "missing_images.log"
            full_latex = _normalize_includegraphics_paths(
                full_latex,
                assets_dir=assets_dir,
                missing_log=missing_img_log,
            )

            # 保存修复后的 LaTeX 文件
            tex_path = entry_tex
            write_text(tex_path, full_latex)

            # 编译 PDF
            if self.compile_pdf:
                pdf_path, compile_log = _compile_latex(tex_path, workdir=tex_path.parent)
                log_path = (
                    self.trace_dir / f"latex_compile_attempt_{attempt:02d}.log"
                    if attempt > 1
                    else self.trace_dir / "latex_compile.log"
                )
                write_text(log_path, compile_log or "")
                if pdf_path:
                    if attempt > 1:
                        write_text(self.trace_dir / "latex_compile.log", compile_log or "")
                    break
                if "未检测到 latexmk" in (compile_log or ""):
                    break
                continue
            break

        duration = (datetime.now() - start).total_seconds()

        summary: Dict[str, str] = {
            "output_dir": str(self.output_dir.resolve()),
            "tex_path": str(tex_path.resolve()),
            "pdf_path": str(pdf_path.resolve()) if pdf_path else "",
            "duration_seconds": f"{duration:.1f}",
            "images_count": len(inputs.image_descriptions),
            "image_descriptions_path": str(Path(inputs.image_descriptions_path).resolve())
            if inputs.image_descriptions_path else "",
            "latex_compile_log_path": str((self.trace_dir / "latex_compile.log").resolve()),
            "latex_compile_error": _truncate_log(compile_log, 800) if (self.compile_pdf and not pdf_path and compile_log) else "",
            "generation_attempts": str(generation_attempts),
        }
        write_text(self.output_dir / "simple_run_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
