#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from lenta_shelf_ai.qr import decode_qr_payloads_with_debug


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark QR/barcode decoding on extracted price-tag crops")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-regions", type=int, default=3)
    parser.add_argument("--max-variants", type=int, default=6)
    parser.add_argument("--opencv-variants", type=int, default=3)
    parser.add_argument("--native-variants", type=int, default=1)
    parser.add_argument("--native-timeout-sec", type=float, default=2.5)
    parser.add_argument("--native-always", action="store_true")
    parser.add_argument("--enable-point-regions", action="store_true")
    parser.add_argument("--disable-zxing", action="store_true")
    parser.add_argument("--disable-pyzbar", action="store_true")
    args = parser.parse_args()

    os.environ["LENTA_QR_MAX_REGIONS"] = str(args.max_regions)
    os.environ["LENTA_QR_MAX_VARIANTS"] = str(args.max_variants)
    os.environ["LENTA_QR_OPENCV_MAX_VARIANTS"] = str(args.opencv_variants)
    os.environ["LENTA_QR_NATIVE_SUBPROCESS"] = "1"
    os.environ["LENTA_QR_NATIVE_BACKEND"] = "processpool"
    os.environ["LENTA_QR_NATIVE_MAX_VARIANTS"] = str(args.native_variants)
    os.environ["LENTA_QR_NATIVE_TIMEOUT_SEC"] = str(args.native_timeout_sec)
    os.environ["LENTA_QR_NATIVE_ALWAYS"] = "1" if args.native_always else "0"
    os.environ["LENTA_QR_ENABLE_POINT_REGIONS"] = "1" if args.enable_point_regions else "0"
    if args.disable_zxing:
        os.environ["LENTA_QR_ENABLE_ZXING"] = "0"
    if args.disable_pyzbar:
        os.environ["LENTA_QR_ENABLE_PYZBAR"] = "0"

    paths = sorted([p for p in args.image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    if args.limit > 0:
        paths = paths[: args.limit]

    rows = []
    hits = 0
    native_calls = 0
    opencv_points = 0
    start = time.time()
    for path in paths:
        image = cv2.imread(str(path))
        if image is None or image.size == 0:
            rows.append({"file": str(path), "error": "read_failed"})
            continue
        t0 = time.time()
        payloads, stats = decode_qr_payloads_with_debug(image)
        elapsed = round(time.time() - t0, 4)
        hits += int(bool(payloads))
        native_calls += int(bool(stats.get("native_processpool") or stats.get("native_subprocess")))
        opencv_points += int(stats.get("opencv_points_detected", 0) or 0)
        rows.append(
            {
                "file": str(path),
                "shape": list(image.shape[:2]),
                "elapsed_sec": elapsed,
                "payloads": payloads,
                "stats": stats,
            }
        )
    report = {
        "image_dir": str(args.image_dir),
        "images_tested": len(paths),
        "qr_hits": hits,
        "hit_rate": hits / max(1, len(paths)),
        "native_calls": native_calls,
        "opencv_points_detected": opencv_points,
        "total_elapsed_sec": round(time.time() - start, 3),
        "env": {k: v for k, v in os.environ.items() if k.startswith("LENTA_QR_")},
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["images_tested", "qr_hits", "hit_rate", "native_calls", "opencv_points_detected", "total_elapsed_sec"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
