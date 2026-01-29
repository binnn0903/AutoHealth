from __future__ import annotations

import json
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from dataclasses import dataclass
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


@dataclass
class KaggleNotebook:
    ref: str
    title: str
    author: str
    total_votes: int
    content: str


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    authors: List[str]
    published: str
    url: str


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str


class _DuckDuckGoParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._in_title = False
        self._in_snippet = False
        self._current_url = ""
        self._current_title = ""
        self._current_snippet = ""
        self.results: List[WebResult] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        attrs_dict = {k: v for k, v in attrs}
        if tag == "a" and attrs_dict.get("class") == "result__a":
            self._in_title = True
            self._current_url = attrs_dict.get("href") or ""
        if tag == "a" and "result__snippet" in (attrs_dict.get("class") or ""):
            self._in_snippet = True

        if tag == "span" and "result__snippet" in (attrs_dict.get("class") or ""):
            self._in_snippet = True

    def handle_endtag(self, tag: str):
        if tag == "a" and self._in_title:
            self._in_title = False
        if tag in {"a", "span"} and self._in_snippet:
            self._in_snippet = False
            if self._current_title and self._current_url:
                self.results.append(
                    WebResult(
                        title=self._current_title.strip(),
                        url=self._current_url.strip(),
                        snippet=self._current_snippet.strip(),
                    )
                )
                self._current_title = ""
                self._current_url = ""
                self._current_snippet = ""

    def handle_data(self, data: str):
        if self._in_title:
            self._current_title += data
        if self._in_snippet:
            self._current_snippet += data


class PlanningAgent:
    """
    计划产生智能体

    输入：
      - task_description: 任务描述（文本/JSON原文）
      - data_report: 数据分析报告（Markdown文本）
      - previous_feedback: 上一轮反馈（可选）

    输出：
      - final_plan.md（Markdown）
      - trace/（每一步 prompt / llm 输出）
    """

    def __init__(
        self,
        *,
        task_description: str,
        data_report: str,
        previous_feedback: str,
        device_info: str = "",
        output_dir: str,
        llm_client: OpenAICompatClient,
        llm_model: str,
        llm_temperature: float,
        llm_top_p: float,
        llm_max_tokens: int,
        llm_extra: Optional[Dict[str, Any]] = None,
        review_rounds: int = 1,
        enable_retrieval: bool = False,
        enable_kaggle_retrieval: bool = False,
        enable_arxiv_retrieval: bool = False,
        enable_web_retrieval: bool = False,
        enable_uncertainty: bool = True,
        kaggle_top_k: int = 10,
        kaggle_language: str = "python",
        kaggle_sort_by: str = "relevance",
        arxiv_top_k: int = 5,
        web_top_k: int = 5,
        web_search_url: str = "https://duckduckgo.com/html/?q={query}",
        max_chars_data_report: int = 20000,
        max_chars_feedback: int = 8000,
        max_chars_kaggle_context: int = 40000,
        max_chars_notebook: int = 20000,
        uncertainty_methods_path: Optional[str] = None,
        prompt_name: str = "planning",
    ):
        self.task_description = task_description
        self.data_report = data_report or ""
        self.previous_feedback = previous_feedback or ""
        self.device_info = device_info or ""

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

        self.review_rounds = max(0, int(review_rounds))
        self.enable_retrieval = bool(enable_retrieval)
        self.enable_kaggle_retrieval = bool(enable_kaggle_retrieval)
        self.enable_arxiv_retrieval = bool(enable_arxiv_retrieval)
        self.enable_web_retrieval = bool(enable_web_retrieval)
        self.enable_uncertainty = bool(enable_uncertainty)

        self.kaggle_top_k = max(1, int(kaggle_top_k))
        self.kaggle_language = str(kaggle_language or "python")
        self.kaggle_sort_by = str(kaggle_sort_by or "relevance")
        self.arxiv_top_k = max(1, int(arxiv_top_k))
        self.web_top_k = max(1, int(web_top_k))
        self.web_search_url = str(web_search_url or "https://duckduckgo.com/html/?q={query}")
        self.max_chars_data_report = max(0, int(max_chars_data_report))
        self.max_chars_feedback = max(0, int(max_chars_feedback))
        self.max_chars_kaggle_context = max(0, int(max_chars_kaggle_context))
        self.max_chars_notebook = max(0, int(max_chars_notebook))

        self.uncertainty_methods_path = uncertainty_methods_path
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

    def _build_requirement_summary(self) -> str:
        # 不截断：直接使用完整的数据报告和反馈
        parts = [
            "## 任务描述\n" + (self.task_description or "（无）"),
            "\n## 数据分析报告\n" + (self.data_report or "（无）"),
            "\n## 上一轮反馈\n" + (self.previous_feedback or "（无）"),
            "\n## 设备信息\n" + (self.device_info or "（无）") + "\n请根据显存/算力控制模型规模与训练时间。",
        ]
        return "\n".join(parts)

    def _generate_search_query(self, *, requirement_summary: str, step_idx: int) -> str:
        system_prompt = str(self.prompt_templates.get("system_prompt", "") or "")
        key = "search_query_prompt" if "search_query_prompt" in self.prompt_templates else "kaggle_query_prompt"
        prompt = self._render(key, {"requirement_summary": requirement_summary})
        llm_text = self._call_llm(system_prompt=system_prompt, user_prompt=prompt)
        self._write_trace(step_idx, "search_query", prompt=prompt, llm_text=llm_text)

        query = _extract_fenced_block(llm_text, "query")
        if not query:
            query = (llm_text or "").strip().splitlines()[0].strip() if (llm_text or "").strip() else ""
        return query.strip()

    def _retrieve_kaggle_notebooks(self, *, query: str) -> Tuple[List[KaggleNotebook], str]:
        """
        使用 Kaggle API 拉取 notebooks（需要用户配置 kaggle API 凭证）。
        返回：(notebooks, error_message)
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
        except Exception as e:
            return [], f"kaggle 包不可用: {e}"

        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as e:
            return [], f"Kaggle API 鉴权失败: {e}"

        try:
            kernels = api.kernels_list(
                search=query,
                sort_by=self.kaggle_sort_by,
                language=self.kaggle_language,
                page_size=self.kaggle_top_k,
            )
        except Exception as e:
            return [], f"Kaggle kernels_list 失败: {e}"

        notebooks: List[KaggleNotebook] = []
        temp_root = tempfile.mkdtemp(prefix="autocline_kaggle_")
        try:
            for k in kernels or []:
                ref = getattr(k, "ref", None) or getattr(k, "kernelRef", None) or ""
                ref = str(ref)
                if not ref:
                    continue

                nb_dir = Path(temp_root) / ref.replace("/", "__")
                nb_dir.mkdir(parents=True, exist_ok=True)
                try:
                    api.kernels_pull(ref, path=str(nb_dir), metadata=False, quiet=True)
                except Exception:
                    continue

                content = self._load_notebook_content(nb_dir)
                if not content:
                    continue

                title = str(getattr(k, "title", "") or "")
                author = str(getattr(k, "author", "") or "")
                votes = int(getattr(k, "totalVotes", 0) or 0)
                notebooks.append(
                    KaggleNotebook(
                        ref=ref,
                        title=title,
                        author=author,
                        total_votes=votes,
                        content=_truncate(content, self.max_chars_notebook),
                    )
                )
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        return notebooks, ""

    def _retrieve_arxiv_papers(self, *, query: str) -> Tuple[List[ArxivPaper], str]:
        try:
            q = urllib.parse.quote(query)
            url = f"http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={self.arxiv_top_k}"
            req = urllib.request.Request(url, headers={"User-Agent": "AutoHealth/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            return [], f"arXiv 检索失败: {e}"

        try:
            root = ET.fromstring(content)
        except Exception as e:
            return [], f"arXiv 解析失败: {e}"

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers: List[ArxivPaper] = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
            authors = [a.findtext("atom:name", default="", namespaces=ns) for a in entry.findall("atom:author", ns)]
            url = arxiv_id
            papers.append(
                ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    summary=summary,
                    authors=[a for a in authors if a],
                    published=published,
                    url=url,
                )
            )
        return papers, ""

    def _retrieve_web_results(self, *, query: str) -> Tuple[List[WebResult], str]:
        try:
            url = self.web_search_url.format(query=urllib.parse.quote(query))
            req = urllib.request.Request(url, headers={"User-Agent": "AutoHealth/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            return [], f"网页检索失败: {e}"

        parser = _DuckDuckGoParser()
        try:
            parser.feed(content)
        except Exception as e:
            return [], f"网页解析失败: {e}"

        results = parser.results[: self.web_top_k]
        return results, ""

    def _load_notebook_content(self, nb_dir: Path) -> str:
        ipynbs = list(nb_dir.glob("*.ipynb"))
        if ipynbs:
            try:
                obj = json.loads(ipynbs[0].read_text(encoding="utf-8", errors="ignore"))
                cells = obj.get("cells") or []
                parts: List[str] = []
                for cell in cells:
                    src = cell.get("source") or []
                    text = "".join(src) if isinstance(src, list) else str(src)
                    ctype = str(cell.get("cell_type") or "").lower()
                    if ctype == "markdown":
                        parts.append(text)
                    elif ctype == "code":
                        parts.append(f"\n```python\n{text}\n```")
                return "\n".join(parts).strip()
            except Exception:
                pass

        pys = list(nb_dir.glob("*.py"))
        if pys:
            try:
                return pys[0].read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                pass
        return ""

    def _summarize_kaggle(
        self,
        *,
        requirement_summary: str,
        query: str,
        notebooks: List[KaggleNotebook],
        step_idx: int,
    ) -> str:
        system_prompt = str(self.prompt_templates.get("system_prompt", "") or "")

        hits_md = "\n".join(
            [
                f"- {n.ref} | votes={n.total_votes} | title={n.title} | author={n.author}"
                for n in notebooks
            ]
        )

        context_parts: List[str] = []
        for n in notebooks:
            header = f"\n\n## Notebook: {n.ref}\n- votes: {n.total_votes}\n- title: {n.title}\n- author: {n.author}\n"
            context_parts.append(header + "\n" + (n.content or ""))
        context = _truncate("\n".join(context_parts).strip(), self.max_chars_kaggle_context)

        prompt = self._render(
            "kaggle_summarize_prompt",
            {
                "requirement_summary": requirement_summary,
                "kaggle_query": query or "（空）",
                "kaggle_hits": hits_md or "（无命中）",
                "kaggle_context": context or "（无）",
            },
        )
        llm_text = self._call_llm(system_prompt=system_prompt, user_prompt=prompt)
        self._write_trace(step_idx, "kaggle_summarize", prompt=prompt, llm_text=llm_text)

        info = _extract_fenced_block(llm_text, "info_augment")
        if not info:
            info = (llm_text or "").strip()
        return info.strip()

    def _summarize_arxiv(
        self,
        *,
        requirement_summary: str,
        query: str,
        papers: List[ArxivPaper],
        step_idx: int,
    ) -> str:
        system_prompt = str(self.prompt_templates.get("system_prompt", "") or "")
        hits_md = "\n".join([f"- {p.title} | {p.url}" for p in papers])
        context_parts = []
        for p in papers:
            header = f"\n\n## {p.title}\n- id: {p.arxiv_id}\n- published: {p.published}\n- authors: {', '.join(p.authors)}\n"
            context_parts.append(header + "\n" + (p.summary or ""))
        context = _truncate("\n".join(context_parts).strip(), self.max_chars_kaggle_context)
        prompt = self._render(
            "arxiv_summarize_prompt",
            {
                "requirement_summary": requirement_summary,
                "arxiv_query": query or "（空）",
                "arxiv_hits": hits_md or "（无命中）",
                "arxiv_context": context or "（无）",
            },
        )
        llm_text = self._call_llm(system_prompt=system_prompt, user_prompt=prompt)
        self._write_trace(step_idx, "arxiv_summarize", prompt=prompt, llm_text=llm_text)
        info = _extract_fenced_block(llm_text, "info_augment")
        if not info:
            info = (llm_text or "").strip()
        return info.strip()

    def _summarize_web(
        self,
        *,
        requirement_summary: str,
        query: str,
        results: List[WebResult],
        step_idx: int,
    ) -> str:
        system_prompt = str(self.prompt_templates.get("system_prompt", "") or "")
        hits_md = "\n".join([f"- {r.title} | {r.url}" for r in results])
        context_parts = []
        for r in results:
            header = f"\n\n## {r.title}\n- url: {r.url}\n"
            context_parts.append(header + "\n" + (r.snippet or ""))
        context = _truncate("\n".join(context_parts).strip(), self.max_chars_kaggle_context)
        prompt = self._render(
            "web_summarize_prompt",
            {
                "requirement_summary": requirement_summary,
                "web_query": query or "（空）",
                "web_hits": hits_md or "（无命中）",
                "web_context": context or "（无）",
            },
        )
        llm_text = self._call_llm(system_prompt=system_prompt, user_prompt=prompt)
        self._write_trace(step_idx, "web_summarize", prompt=prompt, llm_text=llm_text)
        info = _extract_fenced_block(llm_text, "info_augment")
        if not info:
            info = (llm_text or "").strip()
        return info.strip()

    def _planner(
        self,
        *,
        requirement_summary: str,
        info_augment: str,
        uncertainty_methods: str,
        previous_plan: str,
        review: str,
        step_idx: int,
    ) -> str:
        system_prompt = str(self.prompt_templates.get("system_prompt", "") or "")
        prompt_key = "planner_prompt" if not previous_plan else "planner_refine_prompt"
        prompt = self._render(
            prompt_key,
            {
                "requirement_summary": requirement_summary,
                "info_augment": info_augment or "（无）",
                "uncertainty_methods": uncertainty_methods or "（未找到不确定性方法清单）",
                "previous_plan": previous_plan or "（无）",
                "review": review or "（无）",
            },
        )
        llm_text = self._call_llm(system_prompt=system_prompt, user_prompt=prompt)
        self._write_trace(step_idx, prompt_key, prompt=prompt, llm_text=llm_text)

        plan = _extract_fenced_block(llm_text, "plan_md")
        if not plan:
            plan = (llm_text or "").strip()
        return plan.strip()

    def _reviewer(self, *, requirement_summary: str, info_augment: str, plan: str, step_idx: int) -> str:
        system_prompt = str(self.prompt_templates.get("system_prompt", "") or "")
        prompt = self._render(
            "reviewer_prompt",
            {
                "requirement_summary": requirement_summary,
                "info_augment": info_augment or "（无）",
                "plan": plan or "（无）",
            },
        )
        llm_text = self._call_llm(system_prompt=system_prompt, user_prompt=prompt)
        self._write_trace(step_idx, "reviewer_prompt", prompt=prompt, llm_text=llm_text)

        review = _extract_fenced_block(llm_text, "review")
        if not review:
            review = (llm_text or "").strip()
        return review.strip()

    def run(self) -> Dict[str, str]:
        start = datetime.now()
        step_idx = 1

        requirement_summary = self._build_requirement_summary()
        info_augment = ""

        # 加载不确定性方法清单（从一开始就融入计划）
        uncertainty_methods = ""
        if self.enable_uncertainty:
            if self.uncertainty_methods_path:
                p = Path(self.uncertainty_methods_path)
                if p.exists():
                    uncertainty_methods = p.read_text(encoding="utf-8", errors="ignore")

            if not uncertainty_methods:
                # 兜底：默认按项目根目录的文件名查找
                # 这里直接指向仓库根目录的“不确定性方法.md”
                guess = Path(__file__).resolve().parents[1] / "不确定性方法.md"
                if guess.exists():
                    uncertainty_methods = guess.read_text(encoding="utf-8", errors="ignore")

            uncertainty_methods = _truncate(uncertainty_methods, 20000)

        if self.enable_retrieval:
            query = self._generate_search_query(requirement_summary=requirement_summary, step_idx=step_idx)
            step_idx += 1

            info_parts: List[str] = []

            if self.enable_kaggle_retrieval:
                notebooks, err = self._retrieve_kaggle_notebooks(query=query)
                hits_path = self.output_dir / "kaggle_hits.md"
                if err:
                    write_text(hits_path, f"# Kaggle 检索失败\n\n- query: {query}\n- error: {err}\n")
                    info_parts.append(f"# 信息增强（Kaggle）\n- 检索失败：{err}")
                else:
                    hits_md = "\n".join(
                        [f"- {n.ref} | votes={n.total_votes} | title={n.title} | author={n.author}" for n in notebooks]
                    )
                    write_text(hits_path, f"# Kaggle 命中列表\n\n- query: {query}\n\n{hits_md}\n")
                    info = self._summarize_kaggle(
                        requirement_summary=requirement_summary,
                        query=query,
                        notebooks=notebooks,
                        step_idx=step_idx,
                    )
                    info_parts.append(info)
                    step_idx += 1

            if self.enable_arxiv_retrieval:
                papers, err = self._retrieve_arxiv_papers(query=query)
                hits_path = self.output_dir / "arxiv_hits.md"
                if err:
                    write_text(hits_path, f"# arXiv 检索失败\n\n- query: {query}\n- error: {err}\n")
                    info_parts.append(f"# 信息增强（arXiv）\n- 检索失败：{err}")
                else:
                    hits_md = "\n".join([f"- {p.title} | {p.url}" for p in papers])
                    write_text(hits_path, f"# arXiv 命中列表\n\n- query: {query}\n\n{hits_md}\n")
                    info = self._summarize_arxiv(
                        requirement_summary=requirement_summary,
                        query=query,
                        papers=papers,
                        step_idx=step_idx,
                    )
                    info_parts.append(info)
                    step_idx += 1

            if self.enable_web_retrieval:
                results, err = self._retrieve_web_results(query=query)
                hits_path = self.output_dir / "web_hits.md"
                if err:
                    write_text(hits_path, f"# 网页检索失败\n\n- query: {query}\n- error: {err}\n")
                    info_parts.append(f"# 信息增强（网页）\n- 检索失败：{err}")
                else:
                    hits_md = "\n".join([f"- {r.title} | {r.url}" for r in results])
                    write_text(hits_path, f"# 网页命中列表\n\n- query: {query}\n\n{hits_md}\n")
                    info = self._summarize_web(
                        requirement_summary=requirement_summary,
                        query=query,
                        results=results,
                        step_idx=step_idx,
                    )
                    info_parts.append(info)
                    step_idx += 1

            info_augment = "\n\n".join([p for p in info_parts if p.strip()]) if info_parts else ""
            write_text(self.output_dir / "info_augment.md", info_augment or "（无）")

        # 计划者初稿（已融入不确定性）
        plan = self._planner(
            requirement_summary=requirement_summary,
            info_augment=info_augment,
            uncertainty_methods=uncertainty_methods,
            previous_plan="",
            review="",
            step_idx=step_idx,
        )
        write_text(self.output_dir / "plan_round_00.md", plan)
        step_idx += 1

        # 评价者反问 + 计划者改写（多轮迭代）
        for r in range(1, self.review_rounds + 1):
            review = self._reviewer(
                requirement_summary=requirement_summary,
                info_augment=info_augment,
                plan=plan,
                step_idx=step_idx,
            )
            write_text(self.output_dir / f"review_round_{r:02d}.md", review)
            step_idx += 1

            plan = self._planner(
                requirement_summary=requirement_summary,
                info_augment=info_augment,
                uncertainty_methods=uncertainty_methods,
                previous_plan=plan,
                review=review,
                step_idx=step_idx,
            )
            write_text(self.output_dir / f"plan_round_{r:02d}.md", plan)
            step_idx += 1

        # 最终计划即为最后一轮迭代的结果
        final_plan_path = self.output_dir / "final_plan.md"
        write_text(final_plan_path, plan)

        duration = (datetime.now() - start).total_seconds()
        summary_md = (
            "# PlanningAgent Run Summary\n\n"
            f"- output_dir: {self.output_dir.absolute()}\n"
            f"- final_plan: {final_plan_path.absolute()}\n"
            f"- enable_retrieval: {self.enable_retrieval}\n"
            f"- enable_uncertainty: {self.enable_uncertainty}\n"
            f"- review_rounds: {self.review_rounds}\n"
            f"- duration_seconds: {duration:.1f}\n"
        )
        write_text(self.output_dir / "run_summary.md", summary_md)

        return {
            "output_dir": str(self.output_dir.absolute()),
            "plan_path": str(final_plan_path.absolute()),
            "duration_seconds": f"{duration:.1f}",
        }
