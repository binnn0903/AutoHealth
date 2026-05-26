import os
import json
import pandas as pd

BIG_METRIC_MAP = {
    "Clinical Readability": [
        "Clinical Readability",
        "Explanation Completeness",
        "Actionability",
        "Cognitive Burden"
    ],
    "Clinical Reliability": [
        "Calibration Awareness",
        "Uncertainty Quality",
        "Clinical Safety Awareness",
        "Fairness and Bias Awareness"
    ],
    "Deployment Readiness": [
        "Deployment Readiness",
        "Human-in-the-Loop Design",
        "External Validity Awareness"
    ],
    "Scientific Rigor": [
        "Methodological Transparency",
        "Evidence-grounded Reasoning",
        "Limitation Quality"
    ],
    "Overall Assessment": [
        "Overall Clinical Usefulness",
        "Overall Trustworthiness",
        "Publication-level Quality"
    ]
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_score(data, section, metric):
    try:
        return float(data[section][metric]["score"])
    except Exception:
        return None


def compute_big_metrics(data):
    result = {}

    for section, metrics in BIG_METRIC_MAP.items():
        scores = [
            get_score(data, section, metric)
            for metric in metrics
        ]
        scores = [s for s in scores if s is not None]

        result[section] = sum(scores) / len(scores) if scores else None

    valid_scores = [v for v in result.values() if v is not None]
    result["Overall Score"] = sum(valid_scores) / len(valid_scores) if valid_scores else None

    return result


def summarize_all_json(json_dir):
    rows = []

    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(json_dir, filename)

        try:
            data = load_json(path)
            scores = compute_big_metrics(data)

            row = {
                "task": os.path.splitext(filename)[0],
                **scores
            }
            rows.append(row)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    df = pd.DataFrame(rows)
    return df


def metric_distribution(df):
    metric_cols = [c for c in df.columns if c != "task"]

    summary = df[metric_cols].agg(
        [ "mean", "std", "min","median", "max"]
    ).T.reset_index()

    summary = summary.rename(columns={"index": "metric"})
    return summary


# =========================
# Example usage
# =========================

json_dir = "scores"

df_scores = summarize_all_json(json_dir)
df_dist = metric_distribution(df_scores)

print("Task-level scores:")
print(df_scores)

print("\nMetric distribution:")
print(df_dist)

df_scores.to_csv("all_task_big_metric_scores.csv", index=False)
df_dist.to_csv("big_metric_distribution.csv", index=False)