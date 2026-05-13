#!/usr/bin/env python3
"""Smoke-test QR decode + Lenta field parsing on a still image or one video frame."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from lenta_shelf_ai.qr import decode_qr_payloads, parse_qr_payloads


def _load_bgr(path: Path, frame_index: int | None) -> tuple[np.ndarray, str]:
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video: {path}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = frame_index
        if idx is None:
            idx = max(0, n // 2) if n > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, bgr = cap.read()
        cap.release()
        if not ok or bgr is None:
            raise SystemExit(f"Cannot read frame {idx} from {path}")
        return bgr, f"video_frame={idx}"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise SystemExit(f"Cannot read image: {path}")
    return bgr, "image"


def main() -> int:
    p = argparse.ArgumentParser(description="Decode QR from image or video frame, print raw + parsed fields.")
    p.add_argument("path", type=Path, help="Image (.jpg/.png/…) or video path")
    p.add_argument(
        "--frame",
        type=int,
        default=None,
        metavar="N",
        help="For video: 0-based frame index (default: middle of stream)",
    )
    p.add_argument("--json", action="store_true", help="Print one JSON object to stdout")
    args = p.parse_args()

    path = args.path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Not a file: {path}")

    bgr, source_tag = _load_bgr(path, args.frame)
    h, w = bgr.shape[:2]
    payloads = decode_qr_payloads(bgr)
    parsed = parse_qr_payloads(payloads)

    out = {
        "path": str(path),
        "source": source_tag,
        "shape_hw": [int(h), int(w)],
        "raw_payloads": payloads,
        "parsed_fields": parsed,
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print(f"{path} ({source_tag})  shape={h}x{w}")
    print("--- raw payloads ---")
    if not payloads:
        print("(none — try pyzbar / zxing-cpp if OpenCV alone fails)")
    else:
        for i, s in enumerate(payloads):
            print(f"  [{i}] {s!r}")
    print("--- parsed (Lenta aliases) ---")
    if not parsed:
        print("(empty)")
    else:
        for k in sorted(parsed):
            print(f"  {k}: {parsed[k]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
