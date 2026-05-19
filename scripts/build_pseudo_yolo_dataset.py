#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import cv2

from lenta_shelf_ai.detectors import YOLODetector
from lenta_shelf_ai.utils import mkdir
from scripts.build_yolo_dataset import yolo_line


def iter_unlabeled_videos(data_dir: Path) -> Iterable[Path]:
    unlabeled = data_dir / "Unlabeled"
    if unlabeled.exists():
        yield from sorted(unlabeled.glob("*.mp4"))


def copy_base_dataset(base_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for rel in ["images/train", "images/val", "labels/train", "labels/val"]:
        src = base_dir / rel
        dst = out_dir / rel
        if src.exists():
            shutil.copytree(src, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)


def build_pseudo_dataset(
    data_dir: Path,
    base_dataset: Path,
    out_dir: Path,
    weights: Path,
    sample_fps: float = 1.0,
    conf: float = 0.65,
    imgsz: int = 1600,
    max_frames_per_video: int = 0,
) -> dict[str, object]:
    copy_base_dataset(base_dataset, out_dir)
    images_train = mkdir(out_dir / "images/train")
    labels_train = mkdir(out_dir / "labels/train")
    detector = YOLODetector(str(weights), conf=conf, imgsz=imgsz)
    summary: dict[str, object] = {
        "weights": str(weights),
        "sample_fps": sample_fps,
        "conf": conf,
        "imgsz": imgsz,
        "videos": {},
    }
    pseudo_frames = 0
    pseudo_boxes = 0
    for video_path in iter_unlabeled_videos(data_dir):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        step = max(1, int(round(fps / max(0.1, sample_fps))))
        frame_idx = 0
        kept_frames = 0
        kept_boxes = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            if max_frames_per_video and kept_frames >= max_frames_per_video:
                break
            detections = detector.predict(frame)
            if detections:
                h, w = frame.shape[:2]
                stem = f"pseudo_{video_path.stem}_{frame_idx:08d}"
                image_name = f"{stem}.jpg"
                label_name = f"{stem}.txt"
                cv2.imwrite(str(images_train / image_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                with open(labels_train / label_name, "w", encoding="utf-8") as f:
                    for det in detections:
                        f.write(yolo_line(det.xyxy, w, h) + "\n")
                kept_frames += 1
                kept_boxes += len(detections)
            frame_idx += 1
        cap.release()
        pseudo_frames += kept_frames
        pseudo_boxes += kept_boxes
        summary["videos"][video_path.name] = {
            "frames": kept_frames,
            "boxes": kept_boxes,
            "source_fps": fps,
            "step": step,
        }
    summary["pseudo_frames"] = pseudo_frames
    summary["pseudo_boxes"] = pseudo_boxes
    (out_dir / "pseudo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: price_tag\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build self-training YOLO dataset from unlabeled videos")
    parser.add_argument("--data-dir", default="data/Данные")
    parser.add_argument("--base-dataset", default="datasets/lenta_yolo")
    parser.add_argument("--out-dir", default="datasets/lenta_yolo_self")
    parser.add_argument("--weights", default="models/price_tag_yolo.pt")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--conf", type=float, default=0.65)
    parser.add_argument("--imgsz", type=int, default=1600)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    args = parser.parse_args()
    build_pseudo_dataset(
        data_dir=Path(args.data_dir),
        base_dataset=Path(args.base_dataset),
        out_dir=Path(args.out_dir),
        weights=Path(args.weights),
        sample_fps=args.sample_fps,
        conf=args.conf,
        imgsz=args.imgsz,
        max_frames_per_video=args.max_frames_per_video,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
