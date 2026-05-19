#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from lenta_shelf_ai.schema import LEGACY_COLUMN_ALIASES, OUTPUT_COLUMNS
from lenta_shelf_ai.utils import mkdir, smart_float, crop_xyxy
from lenta_shelf_ai.video import read_frame_at_ms


def _describe(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "min": round(float(np.min(arr)), 3),
        "p10": round(float(np.percentile(arr, 10)), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "max": round(float(np.max(arr)), 3),
        "mean": round(float(np.mean(arr)), 3),
    }


def _non_empty_rate(series: pd.Series) -> float:
    vals = series.astype(str).str.strip()
    return round(float(((vals != "") & (vals.str.lower() != "nan") & (vals != "нет")).mean()), 4)


def _resize_tile(image: np.ndarray, size: int = 180) -> np.ndarray:
    if image is None or image.size == 0:
        return np.full((size, size, 3), 245, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def analyze(data_dir: Path, output_dir: Path, max_crops: int = 12) -> dict[str, Any]:
    mkdir(output_dir)
    rows = []
    for csv_path in sorted(data_dir.glob("*/*.csv")):
        video_path = csv_path.with_suffix(".mp4")
        if not video_path.exists():
            video_path = csv_path.parent / f"{csv_path.parent.name}.mp4"
        if not video_path.exists():
            continue
        df = pd.read_csv(csv_path).rename(columns=LEGACY_COLUMN_ALIASES)
        df["__video"] = video_path.name
        df["__video_path"] = str(video_path)
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No labeled CSV/video pairs under {data_dir}")
    all_df = pd.concat(rows, ignore_index=True)
    widths = [smart_float(x2) - smart_float(x1) for x1, x2 in zip(all_df["x_min"], all_df["x_max"])]
    heights = [smart_float(y2) - smart_float(y1) for y1, y2 in zip(all_df["y_min"], all_df["y_max"])]
    areas = [w * h for w, h in zip(widths, heights)]
    aspect = [w / max(1e-6, h) for w, h in zip(widths, heights)]
    by_video = all_df.groupby("__video").size().to_dict()
    by_timestamp = all_df.groupby(["__video", "frame_timestamp"]).size()
    field_coverage = {col: _non_empty_rate(all_df[col]) for col in OUTPUT_COLUMNS if col in all_df.columns}
    metrics = {
        "rows": int(len(all_df)),
        "videos": by_video,
        "unique_timestamps": int(by_timestamp.shape[0]),
        "tags_per_labeled_frame": _describe([float(x) for x in by_timestamp.values]),
        "bbox_width": _describe(widths),
        "bbox_height": _describe(heights),
        "bbox_area": _describe(areas),
        "bbox_aspect_w_over_h": _describe(aspect),
        "field_coverage": field_coverage,
    }
    (output_dir / "public_data_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    crops = []
    sample = all_df.copy()
    sample["__area"] = areas
    sample = sample.sort_values("__area")
    if max_crops > 0 and len(sample) > 0:
        sample = sample.iloc[np.linspace(0, len(sample) - 1, min(max_crops, len(sample)), dtype=int)]
    else:
        sample = sample.iloc[[]]
    for _, row in sample.iterrows():
        frame = read_frame_at_ms(Path(str(row["__video_path"])), smart_float(row["frame_timestamp"]))
        if frame is None:
            continue
        box = [smart_float(row[c]) for c in ["x_min", "y_min", "x_max", "y_max"]]
        crop = crop_xyxy(frame, box, pad=8)
        crops.append(_resize_tile(crop))
    if crops:
        cols = 4
        rows_n = int(np.ceil(len(crops) / cols))
        blank = np.full_like(crops[0], 245)
        while len(crops) < rows_n * cols:
            crops.append(blank.copy())
        sheet = np.vstack([np.hstack(crops[i * cols : (i + 1) * cols]) for i in range(rows_n)])
        cv2.imwrite(str(output_dir / "label_crop_contact_sheet.jpg"), sheet)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze public Lenta labels and generate crop contact sheet")
    parser.add_argument("--data-dir", default="data/Данные")
    parser.add_argument("--output-dir", default="outputs/data_analysis")
    parser.add_argument("--max-crops", type=int, default=12)
    args = parser.parse_args()
    metrics = analyze(Path(args.data_dir), Path(args.output_dir), max_crops=args.max_crops)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
