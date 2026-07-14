const REPORT_TASKS = Array.from({ length: 17 }, (_, index) => {
  const taskNumber = index + 1;
  return {
    id: `T${taskNumber}`,
    label: `Task ${taskNumber}`,
    reportPath: `../report/T${taskNumber}.pdf`,
    llmScorePath: `../Report Judgment Score/scores/Score${taskNumber}.json`,
  };
});

const RUBRIC = [
  {
    section: "Clinical Readability",
    description: "Readability, completeness, actionability, and the cognitive load placed on clinicians.",
    metrics: [
      {
        code: "A1",
        name: "Clinical Readability",
        prompt: "Can clinicians understand the report without advanced ML expertise? Consider clarity of language, jargon, sentence complexity, logical organization, and readability of conclusions.",
      },
      {
        code: "A2",
        name: "Explanation Completeness",
        prompt: "Does the report sufficiently explain task definition, dataset, preprocessing, model choice, metrics, uncertainty, limitations, and deployment implications?",
      },
      {
        code: "A3",
        name: "Actionability",
        prompt: "Does the report provide clinically actionable insights such as threshold implications, uncertainty triage, referral recommendations, or workflow suggestions?",
      },
      {
        code: "A4",
        name: "Cognitive Burden",
        prompt: "Is the report concise and cognitively manageable for clinicians? Penalize unnecessary length, repetition, technical derivations, or unclear structure.",
      },
    ],
  },
  {
    section: "Clinical Reliability",
    description: "Calibration, uncertainty, safety awareness, and fairness or subgroup concerns.",
    metrics: [
      {
        code: "B1",
        name: "Calibration Awareness",
        prompt: "Does the report appropriately discuss probability calibration, including ECE, Brier score, NLL, calibration curves, pre/post calibration discussion, or interpretation of calibrated probabilities?",
      },
      {
        code: "B2",
        name: "Uncertainty Quality",
        prompt: "Does the report meaningfully quantify and interpret uncertainty, including uncertainty-error relationships, selective prediction, abstention logic, triage, or subgroup uncertainty analysis?",
      },
      {
        code: "B3",
        name: "Clinical Safety Awareness",
        prompt: "Does the report acknowledge risks and limitations such as non-autonomous use, need for oversight, deployment cautions, uncertainty limitations, or subgroup risks?",
      },
      {
        code: "B4",
        name: "Fairness and Bias Awareness",
        prompt: "Does the report discuss subgroup behavior, demographic bias, shortcut learning, fairness limitations, or subgroup calibration?",
      },
    ],
  },
  {
    section: "Deployment Readiness",
    description: "Whether the report could credibly support real deployment discussions in a clinical workflow.",
    metrics: [
      {
        code: "C1",
        name: "Deployment Readiness",
        prompt: "Could this report realistically support clinical deployment discussions? Consider workflow integration, threshold selection, triage strategy, risk stratification, and use-case clarity.",
      },
      {
        code: "C2",
        name: "Human-in-the-Loop Design",
        prompt: "Does the report appropriately integrate clinician oversight such as manual review for uncertain cases, selective automation, escalation logic, or clinician confirmation?",
      },
      {
        code: "C3",
        name: "External Validity Awareness",
        prompt: "Does the report discuss external validation, domain shift, demographic shift, hospital or site variation, or generalizability limitations?",
      },
    ],
  },
  {
    section: "Scientific Rigor",
    description: "Transparency, evidence-grounded reasoning, and seriousness of limitation discussion.",
    metrics: [
      {
        code: "D1",
        name: "Methodological Transparency",
        prompt: "Does the report clearly explain preprocessing, feature engineering, model selection, validation strategy, and leakage prevention?",
      },
      {
        code: "D2",
        name: "Evidence-grounded Reasoning",
        prompt: "Are claims appropriately supported by evidence and metrics? Penalize exaggerated deployment claims, unsupported statements, or hallucinated conclusions.",
      },
      {
        code: "D3",
        name: "Limitation Quality",
        prompt: "Does the report critically discuss limitations such as calibration limits, threshold dependence, subgroup weakness, uncertainty limits, or lack of external validation?",
      },
    ],
  },
  {
    section: "Overall Assessment",
    description: "High-level judgement of usefulness, trustworthiness, and publication quality.",
    metrics: [
      {
        code: "E1",
        name: "Overall Clinical Usefulness",
        prompt: "Overall, how useful would this report be to a clinician?",
      },
      {
        code: "E2",
        name: "Overall Trustworthiness",
        prompt: "How trustworthy and responsible is the report?",
      },
      {
        code: "E3",
        name: "Publication-level Quality",
        prompt: "Could this style of report appear in a serious clinical AI paper?",
      },
    ],
  },
];

const BIG_METRIC_MAP = {
  "Clinical Readability": [
    "Clinical Readability",
    "Explanation Completeness",
    "Actionability",
    "Cognitive Burden",
  ],
  "Clinical Reliability": [
    "Calibration Awareness",
    "Uncertainty Quality",
    "Clinical Safety Awareness",
    "Fairness and Bias Awareness",
  ],
  "Deployment Readiness": [
    "Deployment Readiness",
    "Human-in-the-Loop Design",
    "External Validity Awareness",
  ],
  "Scientific Rigor": [
    "Methodological Transparency",
    "Evidence-grounded Reasoning",
    "Limitation Quality",
  ],
  "Overall Assessment": [
    "Overall Clinical Usefulness",
    "Overall Trustworthiness",
    "Publication-level Quality",
  ],
};

const SCORE_LABELS = {
  1: "Very poor",
  2: "Weak",
  3: "Acceptable",
  4: "Strong",
  5: "Excellent",
};

const state = {
  currentTaskId: REPORT_TASKS[0].id,
  llmScores: new Map(),
};

const elements = {
  taskSelect: document.querySelector("#task-select"),
  taskPath: document.querySelector("#task-path"),
  reportTitle: document.querySelector("#report-title"),
  reportLink: document.querySelector("#report-link"),
  reportFrame: document.querySelector("#report-frame"),
  rubricSections: document.querySelector("#rubric-sections"),
  progressSummary: document.querySelector("#progress-summary"),
  bigMetricSummary: document.querySelector("#big-metric-summary"),
  comparisonWrapper: document.querySelector("#comparison-table-wrapper"),
  dashboard: document.querySelector("#dashboard"),
  autosaveIndicator: document.querySelector("#autosave-indicator"),
  llmIndicator: document.querySelector("#llm-indicator"),
  majorStrengths: document.querySelector("#major-strengths"),
  majorWeaknesses: document.querySelector("#major-weaknesses"),
  deploymentReadiness: document.querySelector("#deployment-readiness"),
  overallRecommendation: document.querySelector("#overall-recommendation"),
  metricTemplate: document.querySelector("#metric-card-template"),
};

document.addEventListener("DOMContentLoaded", async () => {
  buildTaskOptions();
  buildRubricForm();
  bindEvents();
  await preloadAllLlmScores();
  hydrateTask();
});

function buildTaskOptions() {
  for (const task of REPORT_TASKS) {
    const option = document.createElement("option");
    option.value = task.id;
    option.textContent = `${task.id} - ${task.label}`;
    elements.taskSelect.append(option);
  }
}

function buildRubricForm() {
  for (const sectionConfig of RUBRIC) {
    const section = document.createElement("section");
    section.className = "rubric-section";
    section.dataset.section = sectionConfig.section;

    const panel = document.createElement("div");
    panel.className = "panel";

    const header = document.createElement("div");
    header.className = "rubric-header";
    header.innerHTML = `
      <div>
        <p class="eyebrow">Section</p>
        <h2>${sectionConfig.section}</h2>
      </div>
      <p class="muted">${sectionConfig.description}</p>
    `;
    panel.append(header);
    section.append(panel);

    for (const metric of sectionConfig.metrics) {
      const fragment = elements.metricTemplate.content.cloneNode(true);
      const card = fragment.querySelector(".metric-card");
      const codeEl = fragment.querySelector(".metric-code");
      const titleEl = fragment.querySelector("h3");
      const descEl = fragment.querySelector(".metric-desc");
      const scoreRow = fragment.querySelector(".score-row");
      const textarea = fragment.querySelector("textarea");
      const llmHint = fragment.querySelector(".llm-hint");

      card.dataset.section = sectionConfig.section;
      card.dataset.metric = metric.name;
      codeEl.textContent = metric.code;
      titleEl.textContent = metric.name;
      descEl.textContent = metric.prompt;
      textarea.name = `${sectionConfig.section}::${metric.name}::justification`;
      llmHint.id = hintId(sectionConfig.section, metric.name);

      for (let score = 1; score <= 5; score += 1) {
        const option = document.createElement("label");
        option.className = "score-option";

        const input = document.createElement("input");
        input.type = "radio";
        input.name = `${sectionConfig.section}::${metric.name}::score`;
        input.value = String(score);

        const span = document.createElement("span");
        span.innerHTML = `<strong>${score}</strong><small>${SCORE_LABELS[score]}</small>`;

        option.append(input, span);
        scoreRow.append(option);
      }

      section.append(fragment);
    }

    elements.rubricSections.append(section);
  }
}

function bindEvents() {
  elements.taskSelect.addEventListener("change", () => {
    state.currentTaskId = elements.taskSelect.value;
    hydrateTask();
  });

  document.querySelector("#open-report-btn").addEventListener("click", () => {
    window.open(currentTask().reportPath, "_blank", "noopener");
  });

  document.querySelector("#load-llm-btn").addEventListener("click", () => {
    const llmData = state.llmScores.get(state.currentTaskId);
    if (!llmData) {
      window.alert("当前任务未找到 LLM 评分 JSON。");
      return;
    }
    applyEvaluationToForm(llmData);
    persistCurrentTask();
    renderEverything();
  });

  document.querySelector("#save-local-btn").addEventListener("click", () => {
    persistCurrentTask();
    renderEverything();
  });

  document.querySelector("#export-json-btn").addEventListener("click", () => {
    downloadJson(buildEvaluationFromForm(), `${state.currentTaskId}_human_score.json`);
  });

  document.querySelector("#export-comparison-btn").addEventListener("click", () => {
    const human = buildEvaluationFromForm();
    const llm = state.llmScores.get(state.currentTaskId) || null;
    downloadJson(
      {
        task: state.currentTaskId,
        human,
        llm,
        human_big_metrics: computeBigMetrics(human),
        llm_big_metrics: llm ? computeBigMetrics(llm) : null,
      },
      `${state.currentTaskId}_human_vs_llm.json`,
    );
  });

  document.querySelector("#clear-local-btn").addEventListener("click", () => {
    const shouldClear = window.confirm(`确认清空 ${state.currentTaskId} 的本地评分吗？`);
    if (!shouldClear) {
      return;
    }
    localStorage.removeItem(storageKey(state.currentTaskId));
    hydrateTask();
  });

  document.querySelector("#import-json-input").addEventListener("change", async (event) => {
    const [file] = event.target.files;
    if (!file) {
      return;
    }

    try {
      const parsed = JSON.parse(await file.text());
      applyEvaluationToForm(parsed);
      persistCurrentTask();
      renderEverything();
    } catch (error) {
      window.alert(`导入失败：${error.message}`);
    } finally {
      event.target.value = "";
    }
  });

  document.querySelector("#evaluation-form").addEventListener("input", () => {
    markDirty();
    persistCurrentTask(true);
    renderEverything();
  });
}

async function preloadAllLlmScores() {
  await Promise.all(REPORT_TASKS.map(async (task) => {
    try {
      const response = await fetch(task.llmScorePath);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      state.llmScores.set(task.id, await response.json());
    } catch (error) {
      console.warn(`Failed to load ${task.llmScorePath}`, error);
    }
  }));
}

function hydrateTask() {
  const task = currentTask();
  elements.taskSelect.value = task.id;
  elements.taskPath.textContent = task.reportPath;
  elements.reportTitle.textContent = `${task.id} 评分问卷`;
  elements.reportLink.href = task.reportPath;
  elements.reportFrame.src = task.reportPath;

  const saved = localStorage.getItem(storageKey(task.id));
  if (saved) {
    applyEvaluationToForm(JSON.parse(saved));
    elements.autosaveIndicator.textContent = "已从浏览器恢复";
    elements.autosaveIndicator.classList.remove("subtle");
  } else {
    clearForm();
    elements.autosaveIndicator.textContent = "当前任务尚未保存";
  }

  const llmExists = state.llmScores.has(task.id);
  elements.llmIndicator.textContent = llmExists ? "LLM 评分已载入" : "LLM 评分缺失";
  renderEverything();
}

function clearForm() {
  for (const input of document.querySelectorAll('input[type="radio"]')) {
    input.checked = false;
  }
  for (const textarea of document.querySelectorAll(".metric-card textarea")) {
    textarea.value = "";
  }
  elements.majorStrengths.value = "";
  elements.majorWeaknesses.value = "";
  elements.deploymentReadiness.value = "";
  elements.overallRecommendation.value = "";
}

function applyEvaluationToForm(data) {
  clearForm();

  for (const sectionConfig of RUBRIC) {
    const sectionData = data[sectionConfig.section] || {};
    for (const metric of sectionConfig.metrics) {
      const metricData = sectionData[metric.name] || {};
      const scoreValue = metricData.score != null ? String(metricData.score) : null;
      const justification = metricData.justification || "";

      if (scoreValue) {
        const radio = document.querySelector(
          `input[name="${cssName(sectionConfig.section, metric.name, "score")}"][value="${scoreValue}"]`,
        );
        if (radio) {
          radio.checked = true;
        }
      }

      const textarea = document.querySelector(
        `textarea[name="${cssName(sectionConfig.section, metric.name, "justification")}"]`,
      );
      if (textarea) {
        textarea.value = justification;
      }
    }
  }

  const summary = data.summary || {};
  elements.majorStrengths.value = (summary.major_strengths || []).join("\n");
  elements.majorWeaknesses.value = (summary.major_weaknesses || []).join("\n");
  elements.deploymentReadiness.value = summary.deployment_readiness || "";
  elements.overallRecommendation.value = summary.overall_recommendation || "";
}

function buildEvaluationFromForm() {
  const result = {};

  for (const sectionConfig of RUBRIC) {
    result[sectionConfig.section] = {};

    for (const metric of sectionConfig.metrics) {
      const scoreInput = document.querySelector(
        `input[name="${cssName(sectionConfig.section, metric.name, "score")}"]:checked`,
      );
      const justificationInput = document.querySelector(
        `textarea[name="${cssName(sectionConfig.section, metric.name, "justification")}"]`,
      );

      result[sectionConfig.section][metric.name] = {
        score: scoreInput ? Number(scoreInput.value) : null,
        justification: justificationInput.value.trim(),
      };
    }
  }

  result.summary = {
    major_strengths: splitLines(elements.majorStrengths.value),
    major_weaknesses: splitLines(elements.majorWeaknesses.value),
    deployment_readiness: elements.deploymentReadiness.value || "",
    overall_recommendation: elements.overallRecommendation.value.trim(),
  };

  return result;
}

function computeBigMetrics(data) {
  const result = {};

  for (const [section, metrics] of Object.entries(BIG_METRIC_MAP)) {
    const scores = metrics
      .map((metric) => {
        const score = data?.[section]?.[metric]?.score;
        if (score === null || score === undefined || score === "") {
          return Number.NaN;
        }
        return typeof score === "number" ? score : Number(score);
      })
      .filter((score) => Number.isFinite(score));

    result[section] = scores.length ? average(scores) : null;
  }

  const validScores = Object.values(result).filter((score) => Number.isFinite(score));
  result["Overall Score"] = validScores.length ? average(validScores) : null;
  return result;
}

function renderEverything() {
  const human = buildEvaluationFromForm();
  const llm = state.llmScores.get(state.currentTaskId) || null;
  renderProgress(human);
  renderBigMetricSummary(human, llm);
  renderComparisonTable(human, llm);
  renderDashboard();
  renderLlmHints(llm);
}

function renderProgress(data) {
  const metricCount = RUBRIC.reduce((sum, section) => sum + section.metrics.length, 0);
  let scored = 0;
  let withJustification = 0;

  for (const section of RUBRIC) {
    for (const metric of section.metrics) {
      const metricData = data?.[section.section]?.[metric.name];
      if (Number.isFinite(metricData?.score)) {
        scored += 1;
      }
      if (metricData?.justification?.trim()) {
        withJustification += 1;
      }
    }
  }

  const stats = [
    { label: "已评分条目", value: `${scored}/${metricCount}` },
    { label: "已写 justification", value: `${withJustification}/${metricCount}` },
    { label: "完成率", value: `${Math.round((scored / metricCount) * 100)}%` },
  ];

  elements.progressSummary.innerHTML = stats
    .map((item) => `<div class="stat-card"><strong>${item.label}</strong><span>${item.value}</span></div>`)
    .join("");
}

function renderBigMetricSummary(human, llm) {
  const humanBig = computeBigMetrics(human);
  const llmBig = llm ? computeBigMetrics(llm) : null;
  const metrics = [...Object.keys(BIG_METRIC_MAP), "Overall Score"];

  elements.bigMetricSummary.innerHTML = metrics
    .map((metric) => {
      const humanScore = formatScore(humanBig[metric]);
      const llmScore = llmBig ? formatScore(llmBig[metric]) : "N/A";
      const delta = llmBig && humanBig[metric] != null && llmBig[metric] != null
        ? formatSignedDelta(humanBig[metric] - llmBig[metric])
        : "N/A";

      return `
        <div class="summary-item">
          <strong>${metric}</strong>
          <div>Human: ${humanScore}</div>
          <div>LLM: ${llmScore}</div>
          <div>Delta: ${delta}</div>
        </div>
      `;
    })
    .join("");
}

function renderComparisonTable(human, llm) {
  const rows = [];

  for (const section of RUBRIC) {
    rows.push(`
      <tr>
        <th colspan="4">${section.section}</th>
      </tr>
    `);

    for (const metric of section.metrics) {
      const humanMetric = human?.[section.section]?.[metric.name] || {};
      const llmMetric = llm?.[section.section]?.[metric.name] || {};
      const delta = Number.isFinite(humanMetric.score) && Number.isFinite(llmMetric.score)
        ? humanMetric.score - llmMetric.score
        : null;

      rows.push(`
        <tr>
          <td>${metric.code} ${metric.name}</td>
          <td>${formatScore(humanMetric.score)}</td>
          <td>${formatScore(llmMetric.score)}</td>
          <td class="${deltaClass(delta)}">${formatSignedDelta(delta)}</td>
        </tr>
      `);
    }
  }

  elements.comparisonWrapper.innerHTML = `
    <table class="comparison-table">
      <thead>
        <tr>
          <th>Metric</th>
          <th>Human</th>
          <th>LLM</th>
          <th>Delta</th>
        </tr>
      </thead>
      <tbody>${rows.join("")}</tbody>
    </table>
  `;
}

function renderDashboard() {
  const rows = REPORT_TASKS.map((task) => {
    const saved = localStorage.getItem(storageKey(task.id));
    const human = saved ? JSON.parse(saved) : null;
    const llm = state.llmScores.get(task.id) || null;
    const humanOverall = human ? computeBigMetrics(human)["Overall Score"] : null;
    const llmOverall = llm ? computeBigMetrics(llm)["Overall Score"] : null;
    const delta = Number.isFinite(humanOverall) && Number.isFinite(llmOverall)
      ? humanOverall - llmOverall
      : null;

    return `
      <div class="dashboard-item">
        <strong>${task.id}</strong>
        <div>Human: ${formatScore(humanOverall)}</div>
        <div>LLM: ${formatScore(llmOverall)}</div>
        <div>Delta: <span class="${deltaClass(delta)}">${formatSignedDelta(delta)}</span></div>
      </div>
    `;
  });

  elements.dashboard.innerHTML = rows.join("");
}

function renderLlmHints(llm) {
  for (const section of RUBRIC) {
    for (const metric of section.metrics) {
      const hint = document.querySelector(`#${hintId(section.section, metric.name)}`);
      const llmMetric = llm?.[section.section]?.[metric.name];

      if (!llmMetric) {
        hint.classList.add("hidden");
        hint.textContent = "";
        continue;
      }

      hint.classList.remove("hidden");
      hint.innerHTML = `<strong>LLM score: ${formatScore(llmMetric.score)}</strong><br>${escapeHtml(llmMetric.justification || "")}`;
    }
  }
}

function persistCurrentTask(isAutosave = false) {
  localStorage.setItem(storageKey(state.currentTaskId), JSON.stringify(buildEvaluationFromForm(), null, 2));
  elements.autosaveIndicator.textContent = isAutosave ? "自动保存中" : "已保存到浏览器";
}

function markDirty() {
  elements.autosaveIndicator.textContent = "正在编辑";
}

function currentTask() {
  return REPORT_TASKS.find((task) => task.id === state.currentTaskId) || REPORT_TASKS[0];
}

function storageKey(taskId) {
  return `human-eval::${taskId}`;
}

function hintId(section, metric) {
  return `hint-${slugify(`${section}-${metric}`)}`;
}

function cssName(section, metric, field) {
  return `${section}::${metric}::${field}`.replace(/"/g, '\\"');
}

function splitLines(value) {
  return value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function formatScore(score) {
  return Number.isFinite(score) ? score.toFixed(2) : "N/A";
}

function formatSignedDelta(delta) {
  if (!Number.isFinite(delta)) {
    return "N/A";
  }
  return `${delta > 0 ? "+" : ""}${delta.toFixed(2)}`;
}

function deltaClass(delta) {
  if (!Number.isFinite(delta) || delta === 0) {
    return "";
  }
  return delta > 0 ? "delta-positive" : "delta-negative";
}

function slugify(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function downloadJson(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
