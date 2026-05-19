#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import pandas as pd

from lenta_shelf_ai.detectors import HybridDetector
from lenta_shelf_ai.schema import LEGACY_COLUMN_ALIASES
from lenta_shelf_ai.utils import iou_xyxy, smart_float
from lenta_shelf_ai.video import read_frame_at_ms


def evaluate(data_dir: Path, weights: str = "", imgsz: int = 1600, conf: float = 0.18, iou_threshold: float = 0.35) -> dict[str, object]:
    detector = HybridDetector(yolo_weights=weights, yolo_conf=conf, imgsz=imgsz)
    total = 0
    matched = 0
    per_video = {}
    for csv_path in sorted(data_dir.glob("*/*.csv")):
        video_path = csv_path.with_suffix(".mp4")
        if not video_path.exists():
            video_path = csv_path.parent / f"{csv_path.parent.name}.mp4"
        if not video_path.exists():
            continue
        df = pd.read_csv(csv_path).rename(columns=LEGACY_COLUMN_ALIASES)
        video_total = 0
        video_matched = 0
        detections_per_frame = []
        best_ious = []
        for ts, group in df.groupby("frame_timestamp"):
            frame = read_frame_at_ms(video_path, smart_float(ts))
            if frame is None:
                continue
            detections = detector.predict(frame)
            detections_per_frame.append(len(detections))
            for _, row in group.iterrows():
                box = [smart_float(row[c]) for c in ["x_min", "y_min", "x_max", "y_max"]]
                best_iou = max([iou_xyxy(box, det.xyxy) for det in detections] or [0.0])
                best_ious.append(best_iou)
                video_total += 1
                if best_iou >= iou_threshold:
                    video_matched += 1
        total += video_total
        matched += video_matched
        per_video[video_path.name] = {
            "matched": video_matched,
            "total": video_total,
            "recall": video_matched / max(1, video_total),
            "avg_detections_per_frame": sum(detections_per_frame) / max(1, len(detections_per_frame)),
        }
    return {
        "weights": weights or "fallback_only",
        "imgsz": imgsz,
        "conf": conf,
        "iou_threshold": iou_threshold,
        "matched": matched,
        "total": total,
        "recall": matched / max(1, total),
        "per_video": per_video,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate detector recall at public labeled timestamps")
    parser.add_argument("--data-dir", default="data/Данные")
    parser.add_argument("--weights", default="")
    parser.add_argument("--imgsz", type=int, default=1600)
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou-threshold", type=float, default=0.35)
    parser.add_argument("--output", default="outputs/detector_recall.json")
    args = parser.parse_args()
    metrics = evaluate(Path(args.data_dir), args.weights, args.imgsz, args.conf, args.iou_threshold)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
