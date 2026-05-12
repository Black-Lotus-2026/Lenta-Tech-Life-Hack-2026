#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Train compact YOLO detector for Lenta price tags")
    ap.add_argument("--data", default="datasets/lenta_yolo/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt/yolov8n.pt or local .pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--project", default="runs/lenta")
    ap.add_argument("--name", default="price_tag_yolo")
    args = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=30,
        cos_lr=True,
        close_mosaic=15,
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.30,
        degrees=8.0,
        translate=0.08,
        scale=0.45,
        shear=2.0,
        perspective=0.0008,
        fliplr=0.0,
        mosaic=0.55,
        mixup=0.05,
        copy_paste=0.0,
        workers=2,
        seed=42,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"[DONE] best weights expected at: {best}")

if __name__ == "__main__":
    main()
