#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import pandas as pd

from lenta_shelf_ai.schema import LEGACY_COLUMN_ALIASES
from lenta_shelf_ai.utils import mkdir, smart_float, clip_xyxy
from lenta_shelf_ai.video import read_frame_at_ms


def yolo_line(box, width: int, height: int, cls: int = 0) -> str:
    x1, y1, x2, y2 = clip_xyxy(box, width, height)
    xc = ((x1 + x2) / 2) / width
    yc = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def template_track(prev_frame, next_frame, box, search_pad=80):
    x1, y1, x2, y2 = map(int, box)
    h, w = prev_frame.shape[:2]
    x1, y1, x2, y2 = map(int, clip_xyxy([x1, y1, x2, y2], w, h))
    tpl = cv2.cvtColor(prev_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if tpl.size == 0 or tpl.shape[0] < 8 or tpl.shape[1] < 8:
        return box, 0.0
    nx1 = max(0, x1 - search_pad); ny1 = max(0, y1 - search_pad)
    nx2 = min(w, x2 + search_pad); ny2 = min(h, y2 + search_pad)
    search = cv2.cvtColor(next_frame[ny1:ny2, nx1:nx2], cv2.COLOR_BGR2GRAY)
    if search.shape[0] < tpl.shape[0] or search.shape[1] < tpl.shape[1]:
        return box, 0.0
    res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    dx, dy = max_loc
    new_box = [nx1 + dx, ny1 + dy, nx1 + dx + (x2 - x1), ny1 + dy + (y2 - y1)]
    return new_box, float(max_val)


def build_dataset(data_dir: Path, out_dir: Path, propagate: int = 0, val_ratio: float = 0.2) -> None:
    images_train = mkdir(out_dir / "images/train")
    images_val = mkdir(out_dir / "images/val")
    labels_train = mkdir(out_dir / "labels/train")
    labels_val = mkdir(out_dir / "labels/val")
    csvs = sorted(data_dir.glob("*/*.csv"))
    frame_id = 0
    for csv_path in csvs:
        df = pd.read_csv(csv_path).rename(columns=LEGACY_COLUMN_ALIASES)
        video_path = csv_path.with_suffix(".mp4")
        if not video_path.exists():
            video_path = csv_path.parent / f"{csv_path.parent.name}.mp4"
        if not video_path.exists():
            print(f"[WARN] no video for {csv_path}")
            continue
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        for ts, group in df.groupby("frame_timestamp"):
            tsf = smart_float(ts)
            base_frame = read_frame_at_ms(video_path, tsf)
            if base_frame is None:
                continue
            boxes = []
            for _, row in group.iterrows():
                boxes.append([smart_float(row.x_min), smart_float(row.y_min), smart_float(row.x_max), smart_float(row.y_max)])
            frame_variants = [(base_frame, boxes, 0, 1.0)]
            if propagate > 0:
                # Automatically add neighboring frames by local template matching.
                for direction in [-1, 1]:
                    prev_frame = base_frame
                    prev_boxes = boxes
                    for step in range(1, propagate + 1):
                        t = max(0.0, tsf + direction * step * 1000.0 / fps)
                        frame = read_frame_at_ms(video_path, t)
                        if frame is None:
                            break
                        tracked = []
                        scores = []
                        for b in prev_boxes:
                            nb, score = template_track(prev_frame, frame, b)
                            tracked.append(nb); scores.append(score)
                        if scores and min(scores) >= 0.42:
                            frame_variants.append((frame, tracked, direction * step, min(scores)))
                            prev_frame, prev_boxes = frame, tracked
                        else:
                            break
            for frame, bxs, offset, q in frame_variants:
                h, w = frame.shape[:2]
                split_val = (frame_id % max(2, int(round(1 / max(1e-6, val_ratio))))) == 0
                img_dir = images_val if split_val else images_train
                lab_dir = labels_val if split_val else labels_train
                name = f"{video_path.stem}_{int(round(tsf)):08d}_{offset:+03d}_{frame_id:06d}.jpg"
                cv2.imwrite(str(img_dir / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                with open(lab_dir / name.replace(".jpg", ".txt"), "w", encoding="utf-8") as f:
                    for b in bxs:
                        f.write(yolo_line(b, w, h) + "\n")
                frame_id += 1
        cap.release()
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: price_tag\n",
        encoding="utf-8",
    )
    print(f"[DONE] wrote {frame_id} frames to {out_dir}")
    print(f"[DONE] data yaml: {data_yaml}")


def main():
    ap = argparse.ArgumentParser(description="Build YOLO dataset from provided Lenta CSV labels")
    ap.add_argument("--data-dir", default="data/Данные", help="Directory containing labeled video folders")
    ap.add_argument("--out-dir", default="datasets/lenta_yolo", help="Output YOLO dataset directory")
    ap.add_argument("--propagate", type=int, default=8, help="Automatic neighbor-frame propagation steps per labeled frame")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()
    build_dataset(Path(args.data_dir), Path(args.out_dir), propagate=args.propagate, val_ratio=args.val_ratio)

if __name__ == "__main__":
    main()
