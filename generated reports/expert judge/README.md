# Expert Report Evaluation

这个目录保存 rebuttal 阶段新增的人类专家评分材料。三位独立 healthcare professionals 使用与 LLM judge 相同的 rubric，对 `generated reports/report/` 里的 17 个自动生成报告逐项评分，并和 `generated reports/Report Judgment Score/scores/` 里的 LLM-as-a-judge 评分直接对比。

专家身份与 rebuttal 说明保持一致：

- `expert 1/`: biomedical engineering postdoctoral researcher
- `expert 2/`: cardiologist
- `expert 3/`: senior emergency physician

## 功能

- 严格复刻 `../Report Judgment Score/ChatGPT.txt` rubric 的 17 个评分项和 `summary` 字段
- 每个评分项均支持 `1-5` 分和 `justification`
- 实时计算与 `../Report Judgment Score/get_score.py` 一致的大项均值和总分
- 自动载入现有 LLM 评分 JSON
- 支持把 LLM 评分填入表单，或和人类评分并排比较
- 支持浏览器本地自动保存
- 支持导出“人类评分 JSON”和“人类 vs LLM 对比 JSON”

## 启动

建议从当前工作区根目录启动一个静态服务器，否则浏览器直接打开 `index.html` 时，`fetch` 本地 JSON 可能会被拦截。

```bash
cd /path/to/AutoHealth/generated\ reports
python -m http.server 8000
```

然后访问：

[http://localhost:8000/expert%20judge/](http://localhost:8000/expert%20judge/)

## 数据路径约定

- 报告 PDF：`../report/T{n}.pdf`
- LLM 评分：`../Report Judgment Score/scores/Score{n}.json`

如果后续新增任务，只需要同时补充：

1. `report/T18.pdf`
2. `Report Judgment Score/scores/Score18.json`
3. 在 `app.js` 里把任务总数从 `17` 改成对应数量

## 导出的 JSON

“导出人类评分 JSON”会输出和现有 LLM 评分结构兼容的 JSON。只要保存成例如 `Score1_human.json` 这一类文件名，就可以直接被你现有的聚合脚本或其轻微改版复用。
