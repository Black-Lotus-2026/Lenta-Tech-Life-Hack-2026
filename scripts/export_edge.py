#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse


def main():
    ap = argparse.ArgumentParser(description="Export detector to ONNX/OpenVINO/RKNN where supported")
    ap.add_argument("--weights", default="runs/lenta/price_tag_yolo/weights/best.pt")
    ap.add_argument("--format", default="onnx", choices=["onnx", "openvino", "torchscript", "rknn"])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--name", default="rk3588", help="RKNN target name, e.g. rk3588/rk3566")
    args = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    kwargs = {"format": args.format, "imgsz": args.imgsz}
    if args.format == "openvino":
        kwargs["int8"] = args.int8
    if args.format == "rknn":
        kwargs["name"] = args.name
    path = model.export(**kwargs)
    print(f"[DONE] exported to {path}")

if __name__ == "__main__":
    main()
