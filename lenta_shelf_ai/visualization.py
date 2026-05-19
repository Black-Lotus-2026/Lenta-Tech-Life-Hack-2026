from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import cv2
import pandas as pd

from .utils import smart_float


def draw_rows_on_frame(frame_bgr, rows: Iterable[Mapping[str, object]]):
    out = frame_bgr.copy()
    for row in rows:
        x1 = smart_float(row.get("x_min")); y1 = smart_float(row.get("y_min")); x2 = smart_float(row.get("x_max")); y2 = smart_float(row.get("y_max"))
        if any(v != v for v in [x1, y1, x2, y2]):
            continue
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        label = str(row.get("product_name") or row.get("barcode") or row.get("price_card") or "tag")[:40]
        cv2.putText(out, label, (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def annotate_video_preview(video_path: str, csv_path: str, out_path: str, max_seconds: float = 20.0) -> str:
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    max_frames = int(max_seconds * fps)
    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        near = df[(df["frame_timestamp"].astype(float) - ts).abs() < 550]
        if len(near):
            frame = draw_rows_on_frame(frame, near.to_dict("records"))
        writer.write(frame)
        idx += 1
    cap.release(); writer.release()
    return out_path
