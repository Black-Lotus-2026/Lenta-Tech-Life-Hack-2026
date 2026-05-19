from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def _dedupe(values):
    out = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def decode_paths(paths: list[str], enable_zxing: bool = True, enable_pyzbar: bool = True) -> dict:
    payloads: list[str] = []
    stats = {
        "images": 0,
        "zxing_attempts": 0,
        "pyzbar_attempts": 0,
        "errors": [],
    }
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None or image.size == 0:
            continue
        images.append(image)
    stats["images"] = len(images)

    if enable_zxing:
        try:
            import zxingcpp  # type: ignore

            fmt = getattr(zxingcpp.BarcodeFormat, "All", None)
            binarizer = getattr(zxingcpp.Binarizer, "LocalAverage", None)
            for image in images:
                stats["zxing_attempts"] += 1
                kwargs = dict(try_rotate=True, try_downscale=True, try_invert=True)
                if fmt is not None:
                    kwargs["formats"] = fmt
                if binarizer is not None:
                    kwargs["binarizer"] = binarizer
                try:
                    results = zxingcpp.read_barcodes(image, **kwargs)
                except TypeError:
                    kwargs.pop("binarizer", None)
                    results = zxingcpp.read_barcodes(image, **kwargs)
                for item in results:
                    text = getattr(item, "text", "") or ""
                    if text:
                        payloads.append(text)
        except Exception as exc:  # pragma: no cover - depends on optional native libs
            stats["errors"].append(f"zxing:{type(exc).__name__}:{exc}")

    if enable_pyzbar:
        try:
            from pyzbar import pyzbar  # type: ignore

            for image in images:
                stats["pyzbar_attempts"] += 1
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                for obj in pyzbar.decode(rgb):
                    try:
                        text = obj.data.decode("utf-8", errors="ignore")
                    except Exception:
                        text = str(obj.data)
                    if text:
                        payloads.append(text)
        except Exception as exc:  # pragma: no cover - depends on optional native libs
            stats["errors"].append(f"pyzbar:{type(exc).__name__}:{exc}")

    payloads = _dedupe(payloads)
    stats["payloads"] = len(payloads)
    return {"payloads": payloads, "stats": stats}


def main() -> int:
    parser = argparse.ArgumentParser(description="Crash-isolated QR/barcode native decoder worker")
    parser.add_argument("images", nargs="*")
    parser.add_argument("--no-zxing", action="store_true")
    parser.add_argument("--no-pyzbar", action="store_true")
    args = parser.parse_args()
    payload = decode_paths(args.images, enable_zxing=not args.no_zxing, enable_pyzbar=not args.no_pyzbar)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
