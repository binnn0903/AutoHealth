# AutoHealth

English | [简体中文](#简体中文)

AutoHealth is an uncertainty-aware, multi-agent AutoML pipeline for health data. It coordinates data understanding, task-conditioned modeling, training/optimization, and report generation around an OpenAI-compatible LLM endpoint (supports DeepSeek/GLM), jointly focusing on predictive performance and uncertainty estimation. The system produces ready-to-use models plus comprehensive reports (metrics, plots, PDF) to support trustworthy, risk-aware decision-making.

## Features
- Uncertainty-aware multi-round pipeline with specialized agents: DataUnderstanding → Planning → CodeExecution → ReportGeneration.
- Health-data oriented: handles heterogeneous tabular/vision tasks with task-conditioned modeling rather than fixed templates.
- Joint focus on performance + uncertainty quantification to aid reliable decisions.
- OpenAI-compatible client (tested with DeepSeek / GLM endpoints); vision analysis support.
- Prompt templates and report templates customizable via YAML / LaTeX.
- Batch runner for multiple tasks and environment bootstrap script.

## Project layout
- `run_pipeline.py`: main entry for a single task run.
- `agents/`: agent implementations.
- `llm/`: OpenAI-compatible client wrapper.
- `ptompts/`: prompt templates and loader.
- `templates/`: LaTeX/report templates.
- `tools/`: safety/quicklook helpers for executing generated code.
- `scripts/run_all_tasks.py`: optional batch runner; `scripts/fix_env.sh`: environment bootstrapper.
- `config.yaml`: runtime configuration (LLM/execution/agent settings).

## Quickstart
1) **Environment**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# allow Python to import the package (repo name == package name)
export PYTHONPATH="$(pwd)/.."
```
Alternatively, run `scripts/fix_env.sh` to install common ML dependencies into the `automl` conda env.

2) **Configure**
- Fill LLM credentials in `config.yaml` (`llm.api_key`, `llm.base_url`, and optionally vision keys).
- Ensure your tasks live under a dataset root with `task.txt` per task (see `examples/report_spec.example.json` for report input structure).

3) **Run a single task**
```bash
python - <<'PY'
from AutoHealth.run_pipeline import run_pipeline

result = run_pipeline(
    task_file="/root/Dataset/your_task/task.txt",
    max_rounds=3,
    patience=1,
    min_delta=0.0,
)
print(result.get("output_dir"))
PY
```

4) **Batch run (optional)**
```bash
python scripts/run_all_tasks.py --task-root /root/Dataset --workers 4 --max-rounds 5
```

## Configuration notes
- All API keys are empty by default; set them before running.
- Execution defaults: conda env name `automl`, per-agent sampling/iteration limits configurable in `config.yaml`.
- Vision analysis uses `llm.vision_*` fields; leave blank to disable.

## Outputs
Each run writes under `outputs/pipeline/<task>/<timestamp>/round_x/`:
- Data understanding summaries, plans, executed code/logs, metrics (`test_results*.json`), plots, and generated PDF report (if report generation is enabled).

## Development
- Code is pure Python; no compiled extensions in repo.
- Run linters/tests as needed in your environment; none are bundled.

---

## 简体中文

AutoHealth 是面向健康数据的、不确定性感知的多智能体 AutoML 管线。系统围绕可配置的 OpenAI 兼容接口（支持 DeepSeek/GLM），协同完成数据理解、任务自适应建模、训练/优化和报告生成，兼顾性能与不确定性评估，输出可直接使用的模型与支持风控决策的可视化/报告（指标、图表、PDF）。

### 功能
- 不确定性感知的多轮迭代：数据理解 → 规划 → 代码执行 → 报告生成。
- 面向健康数据，支持多模态（表格/视觉），任务自适应建模而非固定模板。
- 同时关注性能和不确定性量化，支撑可信、风险感知的决策。
- 兼容 OpenAI 接口（含视觉分析）；可切换到自定义 endpoint。
- 提示词模板（YAML）和报告模板（LaTeX）可自定义。
- 提供批量跑任务脚本与环境初始化脚本。

### 目录结构
- `run_pipeline.py`：单任务入口。
- `agents/`：各智能体实现。
- `llm/`：OpenAI 兼容客户端封装。
- `ptompts/`：提示词模板与加载器。
- `templates/`：LaTeX/报告模板。
- `tools/`：代码执行安全/快览工具。
- `scripts/run_all_tasks.py`：批量运行；`scripts/fix_env.sh`：环境安装脚本。
- `config.yaml`：配置 LLM、执行与智能体参数。

### 快速开始
1) **环境**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 由于仓库名与包名相同，需让 Python 能看到上一级目录
export PYTHONPATH="$(pwd)/.."
# 或执行 scripts/fix_env.sh 在 automl conda 环境中安装依赖
```

2) **配置**
- 在 `config.yaml` 填写 LLM 访问密钥与 base_url（视觉密钥可选）。
- 准备包含 `task.txt` 的任务目录（示例可参考 `examples/report_spec.example.json` 的输入格式）。

3) **运行单任务**
```bash
python - <<'PY'
from AutoHealth.run_pipeline import run_pipeline

result = run_pipeline(
    task_file="/root/Dataset/your_task/task.txt",
    max_rounds=3,
    patience=1,
    min_delta=0.0,
)
print(result.get("output_dir"))
PY
```

4) **批量运行（可选）**
```bash
python scripts/run_all_tasks.py --task-root /root/Dataset --workers 4 --max-rounds 5
```

### 配置提示
- API 密钥默认留空，运行前请填写。
- 执行默认使用 conda 环境 `automl`；各智能体的采样/迭代参数可在 `config.yaml` 调整。
- 视觉分析依赖 `llm.vision_*` 配置，不需要可留空。

### 输出
运行结果写入 `outputs/pipeline/<task>/<timestamp>/round_x/`，包含：
- 数据理解报告、计划、执行代码与日志、指标文件（如 `test_results*.json`）、可视化图片、PDF 报告（若开启报告生成）。

### 开发
- 纯 Python 代码，无二进制扩展。
- 仓库未自带测试/格式化流程，可按需在本地运行。
