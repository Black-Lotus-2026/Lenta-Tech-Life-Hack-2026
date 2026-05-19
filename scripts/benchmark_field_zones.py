#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lenta_shelf_ai.zones import build_field_zone_detector, crop_zone, zones_to_debug
from lenta_shelf_ai.qr import decode_qr_payloads_with_debug
from lenta_shelf_ai.parsers import parse_observation


def iter_images(root: Path) -> Iterable[Path]:
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        yield from sorted(root.rglob(ext))


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark field-zone detector + QR/barcode cascade on price-tag crops/images")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--weights", default="models/field_zone_yolo.pt")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--save-debug-dir", default="")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-heuristic", action="store_true")
    ap.add_argument("--skip-decode", action="store_true", help="Only run field-zone detection; do not call QR/barcode decoders")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.save_debug_dir) if args.save_debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    detector = build_field_zone_detector(
        args.weights,
        enabled=True,
        conf=args.conf,
        imgsz=args.imgsz,
        use_heuristic_fallback=not args.no_heuristic,
    )
    if detector is None:
        raise SystemExit("No field-zone detector is available")

    files = list(iter_images(image_dir))
    if args.max_images > 0:
        files = files[: args.max_images]

    rows = []
    for idx, path in enumerate(files):
        image = cv2.imread(str(path))
        if image is None:
            continue
        zones = detector.predict(image)
        machine_zones = [z for z in zones if z.label in {"barcode", "qr_code_barcode"}]
        payloads = []
        attempts = []
        for zi, zone in enumerate(machine_zones[:8]):
            zcrop = crop_zone(image, zone, pad_ratio=0.12, min_pad=6)
            if zcrop.size == 0:
                continue
            if args.skip_decode:
                stats = {"zone": zones_to_debug([zone])[0], "skipped_decode": True}
                found = []
            else:
                found, stats = decode_qr_payloads_with_debug(zcrop, force_native=True, debug_label=zone.label)
                stats["zone"] = zones_to_debug([zone])[0]
            attempts.append(stats)
            for payload in found:
                if payload not in payloads:
                    payloads.append(payload)
            if debug_dir:
                cv2.imwrite(str(debug_dir / f"{idx:04d}_{path.stem}_{zi}_{zone.label}.jpg"), zcrop)
        parsed = parse_observation([], payloads)
        rows.append(
            {
                "image": str(path),
                "zones": zones_to_debug(zones),
                "machine_zone_count": len(machine_zones),
                "payloads": payloads,
                "parsed": parsed,
                "attempts": attempts,
            }
        )

    summary = {
        "images": len(rows),
        "images_with_machine_zones": sum(1 for r in rows if r["machine_zone_count"] > 0),
        "images_with_payloads": sum(1 for r in rows if r["payloads"]),
        "rows": rows,
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["images", "images_with_machine_zones", "images_with_payloads"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
