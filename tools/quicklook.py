from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def print_json(obj: Any, *, title: str | None = None) -> None:
    if title:
        print(f"\n=== {title} ===")
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def path_exists(path: str) -> Dict[str, Any]:
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
        "size_bytes": p.stat().st_size if p.exists() and p.is_file() else None,
    }


def count_lines(path: str, *, encoding: str = "utf-8") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {"path": str(p), "error": "file_not_found"}
    n = 0
    try:
        with p.open("r", encoding=encoding, errors="ignore") as f:
            for _ in f:
                n += 1
    except Exception as e:
        return {"path": str(p), "error": str(e)}
    return {"path": str(p), "lines": n}


def safe_read_csv(path: str, *, nrows: Optional[int] = 5000, sep: Optional[str] = None) -> "Any":
    import pandas as pd

    kwargs: Dict[str, Any] = {}
    if nrows is not None:
        kwargs["nrows"] = int(nrows)
    if sep:
        kwargs["sep"] = sep
    return pd.read_csv(path, **kwargs)


def quick_profile_tabular_csv(
    path: str,
    *,
    target_col: str | None = None,
    id_col: str | None = "id",
    sample_rows: int = 5000,
) -> Dict[str, Any]:
    """
    快速表格画像（无绘图）：列名、dtype、缺失、重复、目标分布（若提供）。
    默认只读取前 sample_rows 行以加速。
    """
    import pandas as pd

    df = safe_read_csv(path, nrows=sample_rows)
    cols = [str(c) for c in df.columns.tolist()]
    dtypes = {str(k): str(v) for k, v in df.dtypes.items()}
    missing = {str(k): int(v) for k, v in df.isna().sum().items()}
    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])
    dup_rows = int(df.duplicated().sum())

    target_info: Dict[str, Any] = {}
    if target_col and target_col in df.columns:
        vc = df[target_col].value_counts(dropna=False)
        target_info = {
            "target_col": target_col,
            "value_counts": {str(k): int(v) for k, v in vc.items()},
        }

    id_info: Dict[str, Any] = {}
    if id_col and id_col in df.columns:
        nunique = int(df[id_col].nunique(dropna=False))
        id_info = {"id_col": id_col, "nunique": nunique, "is_unique_in_sample": bool(nunique == n_rows)}

    # 数值列的粗略统计（前 sample_rows）
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_summary: Dict[str, Any] = {}
    if numeric_cols:
        desc = df[numeric_cols].describe(include="all").to_dict()
        # 只保留常用统计以避免输出过大
        keep_stats = {"count", "mean", "std", "min", "25%", "50%", "75%", "max"}
        numeric_summary = {
            "numeric_cols": [str(c) for c in numeric_cols],
            "describe": {str(col): {k: desc[col][k] for k in desc[col] if k in keep_stats} for col in desc},
        }

    return {
        "path": path,
        "sample_rows": sample_rows,
        "shape_in_sample": [n_rows, n_cols],
        "columns": cols,
        "dtypes": dtypes,
        "missing_count_in_sample": missing,
        "duplicate_rows_in_sample": dup_rows,
        "id_info": id_info,
        "target_info": target_info,
        "numeric_summary": numeric_summary,
    }


def detect_waveform_columns(columns: List[str]) -> List[str]:
    """
    识别波形列：默认匹配纯数字列名（如 "0","1",...）。
    返回按数值排序后的列名列表。
    """
    digit_cols = [c for c in columns if re.fullmatch(r"\d+", str(c))]
    if not digit_cols:
        return []
    return sorted(digit_cols, key=lambda x: int(x))


def quick_profile_ecg_wide_csv(
    path: str,
    *,
    target_col: str | None = None,
    id_col: str | None = "id",
    sample_rows: int = 2000,
) -> Dict[str, Any]:
    """
    针对“1D ECG 波形值以宽表列存储(0..N)”的快速画像（无绘图）。
    """
    import numpy as np

    df = safe_read_csv(path, nrows=sample_rows)
    cols = [str(c) for c in df.columns.tolist()]
    waveform_cols = detect_waveform_columns(cols)

    base = quick_profile_tabular_csv(path, target_col=target_col, id_col=id_col, sample_rows=sample_rows)
    if not waveform_cols:
        base["ecg_wide"] = {"waveform_cols_detected": 0, "note": "未检测到纯数字波形列名"}
        return base

    X = df[waveform_cols].to_numpy(dtype=float, copy=False)
    # 行级统计（每条波形的均值/方差/极值）
    row_mean = np.nanmean(X, axis=1)
    row_std = np.nanstd(X, axis=1)
    row_min = np.nanmin(X, axis=1)
    row_max = np.nanmax(X, axis=1)
    ecg_stats = {
        "waveform_length": int(len(waveform_cols)),
        "waveform_cols_first_last": [waveform_cols[0], waveform_cols[-1]],
        "row_mean": {
            "min": float(np.nanmin(row_mean)),
            "p50": float(np.nanmedian(row_mean)),
            "max": float(np.nanmax(row_mean)),
        },
        "row_std": {
            "min": float(np.nanmin(row_std)),
            "p50": float(np.nanmedian(row_std)),
            "max": float(np.nanmax(row_std)),
        },
        "row_min": float(np.nanmin(row_min)),
        "row_max": float(np.nanmax(row_max)),
    }

    # 若提供目标列，给出每类的行均值/行方差概览（仅样本）
    per_class: Dict[str, Any] = {}
    if target_col and target_col in df.columns:
        for k, sub in df.groupby(target_col):
            subX = sub[waveform_cols].to_numpy(dtype=float, copy=False)
            m = np.nanmean(subX, axis=1)
            s = np.nanstd(subX, axis=1)
            per_class[str(k)] = {
                "n": int(sub.shape[0]),
                "row_mean_p50": float(np.nanmedian(m)),
                "row_std_p50": float(np.nanmedian(s)),
            }

    base["ecg_wide"] = {
        "waveform_cols_detected": int(len(waveform_cols)),
        "stats_in_sample": ecg_stats,
        "per_class_in_sample": per_class,
    }
    return base


def _guess_text_column(df: "Any") -> Tuple[Optional[str], Dict[str, Any]]:
    import pandas as pd

    candidates: List[str] = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_string_dtype(s) or s.dtype == object:
            candidates.append(str(c))

    if not candidates:
        return None, {"candidates": []}

    name_hits = {"text", "sentence", "note", "report", "desc", "description", "content", "comment", "remark"}
    scores: Dict[str, float] = {}
    for c in candidates:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        avg_len = float(s.str.len().mean())
        name = str(c).lower()
        bonus = 20.0 if any(k in name for k in name_hits) else 0.0
        scores[c] = avg_len + bonus

    if not scores:
        return None, {"candidates": candidates, "scores": {}}

    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return best, {"candidates": candidates, "scores": scores, "selected": best}


def quick_profile_text_csv(
    path: str,
    *,
    text_col: str | None = None,
    target_col: str | None = None,
    id_col: str | None = "id",
    sample_rows: int = 5000,
) -> Dict[str, Any]:
    """
    文本类 CSV 的快速画像（无绘图、只读）。
    - 若未指定 text_col，会基于列名与平均文本长度尝试自动识别。
    """
    import numpy as np
    import pandas as pd

    df = safe_read_csv(path, nrows=sample_rows)
    base = quick_profile_tabular_csv(path, target_col=target_col, id_col=id_col, sample_rows=sample_rows)

    guess_info: Dict[str, Any] = {}
    if not text_col or text_col not in df.columns:
        text_col, guess_info = _guess_text_column(df)

    if not text_col or text_col not in df.columns:
        base["text"] = {"note": "未检测到文本列（请显式指定 text_col）", "guess": guess_info}
        return base

    s = df[text_col]
    non_null = s.dropna().astype(str)
    stripped = non_null.str.strip()
    lengths = stripped.str.len().to_numpy(dtype=float)
    token_counts = stripped.str.split().str.len().to_numpy(dtype=float)

    text_stats = {
        "text_col": str(text_col),
        "non_null_in_sample": int(non_null.shape[0]),
        "empty_string_in_sample": int((stripped == "").sum()),
        "nunique_in_sample": int(non_null.nunique(dropna=False)),
        "duplicate_text_rows_in_sample": int(non_null.duplicated().sum()),
        "char_len": {
            "min": float(np.nanmin(lengths)) if lengths.size else None,
            "p50": float(np.nanmedian(lengths)) if lengths.size else None,
            "mean": float(np.nanmean(lengths)) if lengths.size else None,
            "max": float(np.nanmax(lengths)) if lengths.size else None,
        },
        "token_len_whitespace": {
            "min": float(np.nanmin(token_counts)) if token_counts.size else None,
            "p50": float(np.nanmedian(token_counts)) if token_counts.size else None,
            "mean": float(np.nanmean(token_counts)) if token_counts.size else None,
            "max": float(np.nanmax(token_counts)) if token_counts.size else None,
        },
        "guess": guess_info,
    }

    # 目标列（若提供）补充：按类的文本长度中位数概览（样本内）
    per_class: Dict[str, Any] = {}
    if target_col and target_col in df.columns:
        for k, sub in df[[text_col, target_col]].groupby(target_col):
            sub_s = sub[text_col].dropna().astype(str).str.strip()
            sub_l = sub_s.str.len().to_numpy(dtype=float)
            per_class[str(k)] = {"n": int(sub.shape[0]), "char_len_p50": float(np.nanmedian(sub_l)) if sub_l.size else None}
    text_stats["per_class_in_sample"] = per_class

    base["text"] = text_stats
    return base


def _guess_path_column(df: "Any", *, exts: Tuple[str, ...]) -> Tuple[Optional[str], Dict[str, Any]]:
    import pandas as pd

    candidates: List[str] = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_string_dtype(s) or s.dtype == object:
            candidates.append(str(c))

    if not candidates:
        return None, {"candidates": []}

    ext_re = re.compile(rf"(?i)\\.({'|'.join([re.escape(x.lstrip('.')) for x in exts])})$")
    scores: Dict[str, float] = {}
    for c in candidates:
        series = df[c].dropna().astype(str)
        if series.empty:
            continue
        matched = series.str.contains(ext_re, na=False).mean()
        name = str(c).lower()
        bonus = 0.2 if any(k in name for k in ["path", "file", "image", "img", "audio", "wav", "mp3"]) else 0.0
        scores[c] = float(matched) + bonus

    if not scores:
        return None, {"candidates": candidates, "scores": {}}

    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return best, {"candidates": candidates, "scores": scores, "selected": best}


def quick_profile_image_path_csv(
    path: str,
    *,
    image_col: str | None = None,
    target_col: str | None = None,
    id_col: str | None = "id",
    sample_rows: int = 3000,
    root_dir: str | None = None,
    max_images_to_open: int = 16,
) -> Dict[str, Any]:
    """
    图像路径型 CSV 快速画像（无绘图、只读）。
    - 统计路径存在性、扩展名分布
    - 可选：用 PIL 读取少量图片以获得尺寸/通道信息（若环境无 PIL，会自动跳过）
    """
    import pandas as pd

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")
    df = safe_read_csv(path, nrows=sample_rows)
    base = quick_profile_tabular_csv(path, target_col=target_col, id_col=id_col, sample_rows=sample_rows)

    guess_info: Dict[str, Any] = {}
    if not image_col or image_col not in df.columns:
        image_col, guess_info = _guess_path_column(df, exts=exts)

    if not image_col or image_col not in df.columns:
        base["image"] = {"note": "未检测到图像路径列（请显式指定 image_col）", "guess": guess_info}
        return base

    base_dir = Path(root_dir).resolve() if root_dir else Path(path).resolve().parent
    series = df[image_col].dropna().astype(str).str.strip()
    resolved: List[Path] = []
    for v in series.tolist():
        p = Path(v)
        resolved.append(p if p.is_absolute() else (base_dir / p))

    exists_flags = [p.exists() for p in resolved]
    ext_counts: Dict[str, int] = {}
    for p in resolved:
        ext = p.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    image_info: Dict[str, Any] = {
        "image_col": str(image_col),
        "base_dir": str(base_dir),
        "paths_non_null_in_sample": int(len(resolved)),
        "exists_in_sample": int(sum(1 for x in exists_flags if x)),
        "missing_in_sample": int(sum(1 for x in exists_flags if not x)),
        "ext_counts_in_sample": ext_counts,
        "guess": guess_info,
    }

    # 可选打开图片，获取尺寸/模式分布
    opened = 0
    sizes: List[Tuple[int, int]] = []
    modes: Dict[str, int] = {}
    try:
        from PIL import Image  # type: ignore

        for p, ok in zip(resolved, exists_flags):
            if not ok:
                continue
            try:
                with Image.open(p) as im:
                    sizes.append(tuple(map(int, im.size)))
                    modes[im.mode] = modes.get(im.mode, 0) + 1
                    opened += 1
            except Exception:
                continue
            if opened >= int(max_images_to_open):
                break
    except Exception:
        image_info["pil"] = {"available": False, "note": "PIL 未安装或不可用，已跳过读取图片"}

    if sizes:
        widths = [w for w, _ in sizes]
        heights = [h for _, h in sizes]
        image_info["opened_images"] = opened
        image_info["size_stats_opened"] = {
            "n": int(len(sizes)),
            "width_min_max": [int(min(widths)), int(max(widths))],
            "height_min_max": [int(min(heights)), int(max(heights))],
            "unique_sizes": int(len(set(sizes))),
            "modes": modes,
        }

    base["image"] = image_info
    return base


def quick_profile_timeseries_long_csv(
    path: str,
    *,
    id_col: str | None = "id",
    time_col: str | None = None,
    value_col: str | None = None,
    target_col: str | None = None,
    sample_rows: int = 50000,
) -> Dict[str, Any]:
    """
    长表时序数据（id, time, value...）的快速画像（无绘图、只读）。
    适用于：每一行是一个时间点/观测值；一个样本由多行构成。
    """
    import numpy as np
    import pandas as pd

    df = safe_read_csv(path, nrows=sample_rows)
    base = quick_profile_tabular_csv(path, target_col=target_col, id_col=id_col, sample_rows=sample_rows)

    cols = [str(c) for c in df.columns]
    if not id_col or id_col not in cols:
        id_candidates = [c for c in cols if "id" in c.lower()]
        id_col = id_candidates[0] if id_candidates else None

    if time_col and time_col not in cols:
        time_col = None
    if value_col and value_col not in cols:
        value_col = None

    if not time_col:
        for k in ["time", "timestamp", "t"]:
            hit = [c for c in cols if k == c.lower() or k in c.lower()]
            if hit:
                time_col = hit[0]
                break

    if not value_col:
        numeric_cols = []
        for c in df.columns:
            if str(c) in {id_col, time_col, target_col}:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_cols.append(str(c))
        if numeric_cols:
            # 常见 value/signal 优先，否则取第一个数值列
            priority = [c for c in numeric_cols if c.lower() in {"value", "signal", "x", "y"}]
            value_col = priority[0] if priority else numeric_cols[0]

    ts_info: Dict[str, Any] = {
        "id_col": id_col,
        "time_col": time_col,
        "value_col": value_col,
        "note": "统计基于前 sample_rows 行（可能不是全量）",
    }

    if id_col and id_col in df.columns:
        counts = df[id_col].value_counts(dropna=False)
        arr = counts.to_numpy(dtype=float)
        ts_info["unique_ids_in_sample"] = int(counts.shape[0])
        ts_info["points_per_id_in_sample"] = {
            "min": float(np.nanmin(arr)) if arr.size else None,
            "p50": float(np.nanmedian(arr)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
        }

    if time_col and time_col in df.columns:
        t = df[time_col]
        t_info: Dict[str, Any] = {"dtype": str(t.dtype)}
        if pd.api.types.is_numeric_dtype(t):
            t_info["min_max_in_sample"] = [float(np.nanmin(t.to_numpy(dtype=float))), float(np.nanmax(t.to_numpy(dtype=float)))]
        else:
            dt = pd.to_datetime(t, errors="coerce")
            ok = dt.notna().mean()
            t_info["parseable_ratio_in_sample"] = float(ok)
            if ok > 0:
                t_info["min_max_in_sample"] = [str(dt.min()), str(dt.max())]
        ts_info["time_stats"] = t_info

    if value_col and value_col in df.columns and pd.api.types.is_numeric_dtype(df[value_col]):
        v = df[value_col].to_numpy(dtype=float)
        ts_info["value_stats_in_sample"] = {
            "min": float(np.nanmin(v)) if v.size else None,
            "p50": float(np.nanmedian(v)) if v.size else None,
            "max": float(np.nanmax(v)) if v.size else None,
        }

    base["timeseries_long"] = ts_info
    return base


def list_datasets(root: str = "./Dataset") -> Dict[str, Any]:
    root_path = Path(root).resolve()
    datasets: List[Dict[str, Any]] = []
    if not root_path.exists() or not root_path.is_dir():
        return {"root": str(root_path), "error": "root_not_found", "datasets": []}

    for entry in sorted(root_path.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        data_dir = entry / "data"
        task_txt = entry / "task.txt"
        splits = {}
        for split in ["train", "val", "test"]:
            split_path = data_dir / split
            splits[split] = {
                "path": str(split_path),
                "exists": split_path.exists(),
                "is_dir": split_path.is_dir(),
            }
        datasets.append(
            {
                "name": entry.name,
                "path": str(entry),
                "has_task_txt": task_txt.exists(),
                "has_data_dir": data_dir.exists(),
                "splits": splits,
            }
        )
    return {"root": str(root_path), "datasets": datasets}


def discover_splits(dataset_dir: str) -> Dict[str, Any]:
    ds = Path(dataset_dir).resolve()
    data_dir = ds / "data"
    splits: Dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        split_path = data_dir / split
        if split_path.exists():
            splits[split] = {"path": str(split_path), "exists": True, "is_dir": split_path.is_dir()}
        else:
            csv_path = data_dir / f"{split}.csv"
            splits[split] = {"path": str(csv_path), "exists": csv_path.exists(), "is_dir": False}
    return {"dataset_dir": str(ds), "data_dir": str(data_dir), "splits": splits}


def sample_files(path: str, *, exts: Optional[Tuple[str, ...]] = None, k: int = 20) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    if p.is_file():
        return [str(p)]
    results: List[str] = []
    exts_norm = tuple(x.lower() for x in exts) if exts else None
    for fp in p.rglob("*"):
        if fp.is_dir():
            continue
        if fp.name.startswith("."):
            continue
        if exts_norm and fp.suffix.lower() not in exts_norm:
            continue
        results.append(str(fp))
        if len(results) >= int(k):
            break
    return results


def _count_exts(paths: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in paths:
        ext = Path(p).suffix.lower()
        counts[ext] = counts.get(ext, 0) + 1
    return counts


def infer_modality(dataset_dir: str) -> Dict[str, Any]:
    info = discover_splits(dataset_dir)
    splits = info.get("splits", {})
    sample_paths: List[str] = []
    for split in ["train", "val", "test"]:
        sp = splits.get(split, {}).get("path")
        if not sp:
            continue
        sample_paths.extend(sample_files(sp, k=20))
    ext_counts = _count_exts(sample_paths)

    modality = "unknown"
    has_tabular = any(ext in ext_counts for ext in [".csv", ".tsv", ".parquet", ".json"])
    has_text = any(ext in ext_counts for ext in [".txt"])
    has_image = any(ext in ext_counts for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"])
    has_audio = any(ext in ext_counts for ext in [".wav", ".mp3", ".flac", ".ogg", ".npy"])
    has_graph = any(ext in ext_counts for ext in [".pt"])

    train_dir = Path(dataset_dir) / "data" / "train"
    seg_pairs = [
        ("image", "mask"),
        ("images", "mask"),
        ("image", "masks"),
        ("images", "masks"),
    ]
    is_seg = False
    for img_name, mask_name in seg_pairs:
        if (train_dir / img_name).exists() and (train_dir / mask_name).exists():
            is_seg = True
            break

    if is_seg:
        modality = "segmentation"
    elif has_graph:
        modality = "graph"
    elif has_audio:
        modality = "audio"
    elif has_image:
        modality = "image"
    elif has_text and has_tabular:
        modality = "text"
    elif has_text:
        modality = "text"
    elif has_tabular:
        modality = "tabular"

    return {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "modality": modality,
        "ext_counts_sample": ext_counts,
        "sample_paths": sample_paths,
    }


def quick_profile_image_dir(
    split_dir: str,
    *,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"),
    sample_n: int = 32,
    max_classes: int = 50,
) -> Dict[str, Any]:
    base = {"split_dir": str(Path(split_dir).resolve())}
    p = Path(split_dir)
    if not p.exists() or not p.is_dir():
        base["error"] = "split_dir_not_found"
        return base

    class_dirs = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")]
    class_dirs = class_dirs[: int(max_classes)]
    class_counts: Dict[str, int] = {}
    for d in class_dirs:
        count = 0
        for fp in d.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in exts:
                count += 1
        class_counts[d.name] = count

    samples = sample_files(split_dir, exts=exts, k=sample_n)
    base["class_counts"] = class_counts
    base["sample_paths"] = samples

    opened = 0
    sizes: List[Tuple[int, int]] = []
    modes: Dict[str, int] = {}
    try:
        from PIL import Image  # type: ignore

        for s in samples:
            try:
                with Image.open(s) as im:
                    sizes.append(tuple(map(int, im.size)))
                    modes[im.mode] = modes.get(im.mode, 0) + 1
                    opened += 1
            except Exception:
                continue
    except Exception:
        base["pil"] = {"available": False, "note": "PIL not available"}

    if sizes:
        widths = [w for w, _ in sizes]
        heights = [h for _, h in sizes]
        base["image_stats_sample"] = {
            "opened": opened,
            "width_min_max": [int(min(widths)), int(max(widths))],
            "height_min_max": [int(min(heights)), int(max(heights))],
            "unique_sizes": int(len(set(sizes))),
            "modes": modes,
        }
    return base


def quick_profile_seg_dir(
    images_dir: str,
    masks_dir: str,
    *,
    sample_n: int = 32,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "images_dir": str(Path(images_dir).resolve()),
        "masks_dir": str(Path(masks_dir).resolve()),
    }
    img_dir = Path(images_dir)
    msk_dir = Path(masks_dir)
    if not img_dir.exists() or not msk_dir.exists():
        out["error"] = "images_or_masks_not_found"
        return out

    img_files = sample_files(str(img_dir), exts=exts, k=sample_n)
    mask_files = sample_files(str(msk_dir), exts=exts, k=sample_n)
    out["sample_images"] = img_files
    out["sample_masks"] = mask_files

    img_stems = {Path(p).stem for p in img_files}
    mask_stems = {Path(p).stem for p in mask_files}
    out["sample_match_ratio"] = float(len(img_stems & mask_stems)) / max(len(img_stems), 1)

    sizes_ok = 0
    sizes_total = 0
    mask_values: List[int] = []
    try:
        from PIL import Image  # type: ignore

        for p in img_files:
            stem = Path(p).stem
            candidate = msk_dir / f"{stem}{Path(p).suffix}"
            if not candidate.exists():
                continue
            try:
                with Image.open(p) as im, Image.open(candidate) as mm:
                    sizes_total += 1
                    if im.size == mm.size:
                        sizes_ok += 1
                    vals = list(mm.getdata())
                    if vals:
                        mask_values.extend(vals[: min(2000, len(vals))])
            except Exception:
                continue
    except Exception:
        out["pil"] = {"available": False, "note": "PIL not available"}

    out["size_match_ratio_sample"] = float(sizes_ok) / max(sizes_total, 1)
    if mask_values:
        out["mask_value_range_sample"] = [int(min(mask_values)), int(max(mask_values))]
    return out


def _audio_info(path: str) -> Dict[str, Any]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".npy":
        import numpy as np

        arr = np.load(p)
        length = int(arr.shape[0]) if arr.ndim > 0 else 0
        return {"ext": ext, "length": length, "sr": None, "channels": 1 if arr.ndim == 1 else int(arr.shape[1])}

    try:
        import soundfile as sf  # type: ignore

        info = sf.info(p)
        return {"ext": ext, "frames": int(info.frames), "sr": int(info.samplerate), "channels": int(info.channels)}
    except Exception:
        pass

    if ext == ".wav":
        import wave

        with wave.open(str(p), "rb") as wf:
            return {
                "ext": ext,
                "frames": int(wf.getnframes()),
                "sr": int(wf.getframerate()),
                "channels": int(wf.getnchannels()),
            }

    try:
        import librosa  # type: ignore

        y, sr = librosa.load(str(p), sr=None, mono=False)
        length = int(y.shape[-1])
        channels = 1 if y.ndim == 1 else int(y.shape[0])
        return {"ext": ext, "length": length, "sr": int(sr), "channels": channels}
    except Exception:
        return {"ext": ext, "error": "unsupported_audio_format"}


def quick_profile_audio_dir(
    split_dir: str,
    *,
    exts: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".npy"),
    sample_n: int = 64,
) -> Dict[str, Any]:
    base = {"split_dir": str(Path(split_dir).resolve())}
    p = Path(split_dir)
    if not p.exists() or not p.is_dir():
        base["error"] = "split_dir_not_found"
        return base

    samples = sample_files(str(p), exts=exts, k=sample_n)
    base["sample_paths"] = samples
    infos: List[Dict[str, Any]] = []
    errors = 0
    for s in samples:
        info = _audio_info(s)
        if "error" in info:
            errors += 1
        infos.append(info)
    base["audio_infos_sample"] = infos
    base["errors_in_sample"] = errors
    return base


def quick_profile_audio_path_csv(
    path: str,
    *,
    audio_col: str | None = None,
    target_col: str | None = None,
    id_col: str | None = "id",
    sample_rows: int = 3000,
    root_dir: str | None = None,
) -> Dict[str, Any]:
    import pandas as pd

    df = safe_read_csv(path, nrows=sample_rows)
    base = quick_profile_tabular_csv(path, target_col=target_col, id_col=id_col, sample_rows=sample_rows)

    guess_info: Dict[str, Any] = {}
    if not audio_col or audio_col not in df.columns:
        audio_col, guess_info = _guess_path_column(df, exts=(".wav", ".mp3", ".flac", ".ogg", ".npy"))

    if not audio_col or audio_col not in df.columns:
        base["audio"] = {"note": "audio path col not found", "guess": guess_info}
        return base

    base_dir = Path(root_dir).resolve() if root_dir else Path(path).resolve().parent
    series = df[audio_col].dropna().astype(str).str.strip()
    resolved = [Path(v) if Path(v).is_absolute() else (base_dir / v) for v in series.tolist()]
    exists_flags = [p.exists() for p in resolved]
    base["audio"] = {
        "audio_col": str(audio_col),
        "base_dir": str(base_dir),
        "paths_non_null_in_sample": int(len(resolved)),
        "exists_in_sample": int(sum(1 for x in exists_flags if x)),
        "missing_in_sample": int(sum(1 for x in exists_flags if not x)),
        "guess": guess_info,
    }
    return base


def quick_profile_graph_pt(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {"path": str(p), "error": "file_not_found"}
    try:
        import torch  # type: ignore

        obj = None
        try:
            obj = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            obj = torch.load(p, map_location="cpu", weights_only=False)
        out: Dict[str, Any] = {"path": str(p), "type": str(type(obj))}
        if isinstance(obj, dict):
            out["keys"] = list(obj.keys())
            edge_index = obj.get("edge_index")
            if edge_index is not None and hasattr(edge_index, "shape"):
                out["edge_index_shape"] = list(edge_index.shape)
            num_nodes = obj.get("num_nodes")
            if num_nodes is not None:
                out["num_nodes"] = int(num_nodes)
        elif hasattr(obj, "edge_index"):
            out["edge_index_shape"] = list(getattr(obj, "edge_index").shape)
            if hasattr(obj, "num_nodes"):
                out["num_nodes"] = int(getattr(obj, "num_nodes"))
        return out
    except Exception as e:
        return {"path": str(p), "error": str(e)}


def quick_profile_graph_dir(split_dir: str, *, sample_n: int = 5) -> Dict[str, Any]:
    p = Path(split_dir)
    if not p.exists():
        return {"split_dir": str(p), "error": "split_dir_not_found"}
    pt_files = sample_files(str(p), exts=(".pt",), k=sample_n)
    return {
        "split_dir": str(p.resolve()),
        "pt_files_sample": pt_files,
        "pt_summaries_sample": [quick_profile_graph_pt(fp) for fp in pt_files],
    }


def quick_profile_dataset(dataset_dir: str) -> Dict[str, Any]:
    ds = Path(dataset_dir).resolve()
    modality_info = infer_modality(str(ds))
    splits_info = discover_splits(str(ds))
    splits = splits_info.get("splits", {})

    out: Dict[str, Any] = {"dataset_dir": str(ds), "modality": modality_info.get("modality"), "splits": {}}

    for split in ["train", "val", "test"]:
        sp = splits.get(split, {})
        sp_path = sp.get("path")
        if not sp_path or not Path(sp_path).exists():
            out["splits"][split] = {"error": "split_not_found", "path": sp_path}
            continue

        modality = modality_info.get("modality")
        if modality in {"image"} and Path(sp_path).is_dir():
            out["splits"][split] = quick_profile_image_dir(sp_path)
        elif modality in {"segmentation"}:
            images_dir = Path(sp_path) / "images"
            masks_dir = Path(sp_path) / "mask"
            if not masks_dir.exists():
                masks_dir = Path(sp_path) / "masks"
            out["splits"][split] = quick_profile_seg_dir(str(images_dir), str(masks_dir))
        elif modality in {"audio"} and Path(sp_path).is_dir():
            out["splits"][split] = quick_profile_audio_dir(sp_path)
        elif modality in {"graph"}:
            out["splits"][split] = quick_profile_graph_dir(sp_path)
        else:
            # default to tabular for csv-like splits
            if Path(sp_path).is_file() and Path(sp_path).suffix.lower() in {".csv", ".tsv", ".parquet", ".json"}:
                out["splits"][split] = quick_profile_tabular_csv(sp_path)
            else:
                # try common train/val/test csv inside split dir
                candidate = Path(sp_path) / f"{split}.csv"
                if candidate.exists():
                    out["splits"][split] = quick_profile_tabular_csv(str(candidate))
                else:
                    out["splits"][split] = {"path": sp_path, "note": "no handler for split"}
    return out


def quick_profile_all_datasets(root: str = "./Dataset") -> Dict[str, Any]:
    listing = list_datasets(root)
    results: List[Dict[str, Any]] = []
    for ds in listing.get("datasets", []):
        results.append({"name": ds.get("name"), "summary": infer_modality(ds.get("path", ""))})
    return {"root": listing.get("root"), "datasets": results}
