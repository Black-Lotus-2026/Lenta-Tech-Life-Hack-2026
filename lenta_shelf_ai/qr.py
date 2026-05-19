from __future__ import annotations

import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .schema import QR_FIELD_ALIASES
from .utils import price_to_str


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _dedupe_payloads(payloads: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for payload in payloads:
        text = str(payload or "").strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


_NATIVE_EXECUTOR: concurrent.futures.ProcessPoolExecutor | None = None


def _native_decode_arrays_worker(images: List[np.ndarray], enable_zxing: bool, enable_pyzbar: bool) -> Dict[str, object]:
    """Process-pool worker. A native SIGABRT kills only this process."""
    payloads: List[str] = []
    stats: Dict[str, object] = {"images": len(images), "zxing_attempts": 0, "pyzbar_attempts": 0, "errors": []}
    if enable_zxing:
        try:
            import zxingcpp  # type: ignore

            fmt = getattr(zxingcpp.BarcodeFormat, "All", None)
            binarizer = getattr(zxingcpp.Binarizer, "LocalAverage", None)
            for image in images:
                stats["zxing_attempts"] = int(stats.get("zxing_attempts", 0)) + 1
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
        except Exception as exc:  # pragma: no cover - optional native dep
            stats.setdefault("errors", []).append(f"zxing:{type(exc).__name__}:{exc}")
    if enable_pyzbar:
        try:
            from pyzbar import pyzbar  # type: ignore

            for image in images:
                stats["pyzbar_attempts"] = int(stats.get("pyzbar_attempts", 0)) + 1
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                for obj in pyzbar.decode(rgb):
                    try:
                        text = obj.data.decode("utf-8", errors="ignore")
                    except Exception:
                        text = str(obj.data)
                    if text:
                        payloads.append(text)
        except Exception as exc:  # pragma: no cover - optional native dep
            stats.setdefault("errors", []).append(f"pyzbar:{type(exc).__name__}:{exc}")
    payloads = _dedupe_payloads(payloads)
    stats["payloads"] = len(payloads)
    return {"payloads": payloads, "stats": stats}


def _get_native_executor() -> concurrent.futures.ProcessPoolExecutor:
    global _NATIVE_EXECUTOR
    if _NATIVE_EXECUTOR is None:
        _NATIVE_EXECUTOR = concurrent.futures.ProcessPoolExecutor(max_workers=1)
    return _NATIVE_EXECUTOR


def _reset_native_executor() -> None:
    global _NATIVE_EXECUTOR
    if _NATIVE_EXECUTOR is not None:
        try:
            _NATIVE_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    _NATIVE_EXECUTOR = None


def _resize_max_side(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0 or max_side <= 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return np.ascontiguousarray(image_bgr)
    scale = float(max_side) / float(side)
    return cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _add_quiet_zone(image_bgr: np.ndarray, ratio: float = 0.18, min_px: int = 8) -> np.ndarray:
    """Add white QR quiet-zone padding.

    Crops from detector/tracker are often too tight around the QR modules. QR
    decoders expect a quiet zone; adding a white border is cheap and does not
    change the encoded modules.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    pad = max(int(min_px), int(round(ratio * max(h, w))))
    return cv2.copyMakeBorder(image_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]       # top-left
    ordered[2] = pts[np.argmax(s)]       # bottom-right
    ordered[1] = pts[np.argmin(diff)]    # top-right
    ordered[3] = pts[np.argmax(diff)]    # bottom-left
    return ordered


def _warp_quad(image_bgr: np.ndarray, points: np.ndarray, pad_ratio: float = 0.25) -> np.ndarray | None:
    if image_bgr is None or image_bgr.size == 0:
        return None
    try:
        ordered = _order_quad_points(points)
    except Exception:
        return None
    w1 = np.linalg.norm(ordered[1] - ordered[0])
    w2 = np.linalg.norm(ordered[2] - ordered[3])
    h1 = np.linalg.norm(ordered[3] - ordered[0])
    h2 = np.linalg.norm(ordered[2] - ordered[1])
    side = int(max(32.0, w1, w2, h1, h2))
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32)
    try:
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image_bgr, matrix, (side, side), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    except Exception:
        return None
    return _add_quiet_zone(warped, ratio=pad_ratio)


def _opencv_point_regions(image_bgr: np.ndarray, max_regions: int = 4) -> List[np.ndarray]:
    """Localize QR with OpenCV and return rectified/bbox crops even if decode fails."""
    if image_bgr is None or image_bgr.size == 0 or max_regions <= 0:
        return []
    regions: List[np.ndarray] = []
    h, w = image_bgr.shape[:2]
    detector = cv2.QRCodeDetector()
    point_sets: list[np.ndarray] = []
    try:
        ok, points = detector.detectMulti(image_bgr)
        if ok and points is not None:
            for pts in np.asarray(points).reshape(-1, 4, 2):
                point_sets.append(pts)
    except Exception:
        pass
    if not point_sets:
        try:
            ok, points = detector.detect(image_bgr)
            if ok and points is not None:
                point_sets.append(np.asarray(points).reshape(4, 2))
        except Exception:
            pass
    for pts in point_sets[:max_regions]:
        warped = _warp_quad(image_bgr, pts)
        if warped is not None and warped.size:
            regions.append(np.ascontiguousarray(warped))
        x1 = int(np.floor(np.min(pts[:, 0]))); y1 = int(np.floor(np.min(pts[:, 1])))
        x2 = int(np.ceil(np.max(pts[:, 0]))); y2 = int(np.ceil(np.max(pts[:, 1])))
        side = max(x2 - x1, y2 - y1)
        pad = max(8, int(0.40 * side))
        x1, y1, x2, y2 = _clip_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, w=w, h=h)
        if x2 > x1 and y2 > y1:
            regions.append(np.ascontiguousarray(image_bgr[y1:y2, x1:x2]))
    return regions[:max_regions]


def _rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _variant_signature(image: np.ndarray) -> tuple[int, int, int, int, int, int]:
    """Cheap content-aware signature for variant dedupe."""
    h, w = image.shape[:2]
    step_y = max(1, h // 16)
    step_x = max(1, w // 16)
    sample = image[::step_y, ::step_x]
    flat = sample.reshape(-1)
    return (
        h,
        w,
        int(flat[0]) if flat.size else 0,
        int(flat[-1]) if flat.size else 0,
        int(np.mean(flat)) if flat.size else 0,
        int(np.std(flat)) if flat.size else 0,
    )


def _suppress_specular_glare(image_bgr: np.ndarray) -> np.ndarray:
    """Cheap glare/haze suppression used only as decoder variant.

    Shelf videos often contain glass reflections: QR modules are localized but
    decode fails because a few white blobs erase finder/module contrast.  This
    inpaints only low-saturation high-value pixels and then applies CLAHE, so
    the original crop remains first in the decoder cascade.
    """
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if h < 24 or w < 24:
        return image_bgr
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((val > 238) & (sat < 45)).astype(np.uint8) * 255
    # Ignore tiny isolated white text pixels; keep broad reflection blobs.
    k = max(3, int(round(min(h, w) * 0.025)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    ratio = float((mask > 0).mean())
    if ratio < 0.006 or ratio > 0.42:
        return image_bgr
    try:
        repaired = cv2.inpaint(image_bgr, mask, max(2, k // 3), cv2.INPAINT_TELEA)
        lab = cv2.cvtColor(repaired, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        return image_bgr


def _barcode_image_variants(image_bgr: np.ndarray) -> List[np.ndarray]:
    """Barcode-specific preprocessing for wide 1D strips."""
    if image_bgr is None or image_bgr.size == 0:
        return []
    max_output_side = _env_int("LENTA_QR_MAX_SIDE", 1400)
    base = _resize_max_side(np.ascontiguousarray(image_bgr), max_output_side)
    h, w = base.shape[:2]
    if h < 12 or w < 12:
        return []
    oriented = [base]
    if h > w:
        oriented.insert(0, cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE))

    variants: List[np.ndarray] = []
    for strip in oriented[:2]:
        sh, sw = strip.shape[:2]
        if sw / max(1, sh) < 1.35:
            continue
        variants.append(strip)
        for sx, sy in [(1.6, 2.0), (2.2, 2.6)]:
            target_w = min(max_output_side, max(sw, int(sw * sx)))
            target_h = min(max_output_side, max(sh, int(sh * sy)))
            up = cv2.resize(strip, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY) if len(up.shape) == 3 else up
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 4)).apply(gray)
            blur = cv2.GaussianBlur(clahe, (0, 0), 0.8)
            sharp = cv2.addWeighted(clahe, 1.7, blur, -0.7, 0)
            _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive = cv2.adaptiveThreshold(
                sharp,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                31,
                7,
            )
            kernel_w = max(1, int(round(target_w / 180)))
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 2))
            closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, close_kernel)
            for single in [gray, clahe, sharp, otsu, adaptive, closed, cv2.bitwise_not(otsu)]:
                variants.append(_add_quiet_zone(cv2.cvtColor(single, cv2.COLOR_GRAY2BGR), ratio=0.06, min_px=10))

    unique: List[np.ndarray] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    for im in variants:
        if im is None or im.size == 0:
            continue
        key = _variant_signature(im)
        if key in seen:
            continue
        unique.append(np.ascontiguousarray(im))
        seen.add(key)
    return unique


def _qr_image_variants(image_bgr: np.ndarray) -> List[np.ndarray]:
    if image_bgr is None or image_bgr.size == 0:
        return []
    max_output_side = _env_int("LENTA_QR_MAX_SIDE", 1400)
    image_bgr = _resize_max_side(np.ascontiguousarray(image_bgr), max_output_side)
    variants: List[np.ndarray] = [image_bgr, _add_quiet_zone(image_bgr)]
    if os.environ.get("LENTA_QR_ENABLE_GLARE_SUPPRESSION", "1") != "0":
        glare_fixed = _suppress_specular_glare(image_bgr)
        if glare_fixed is not image_bgr and getattr(glare_fixed, "size", 0):
            variants.append(glare_fixed)
            variants.append(_add_quiet_zone(glare_fixed, ratio=0.12))
    h, w = image_bgr.shape[:2]
    max_side = max(h, w)

    # QR modules in 4K retail crops are frequently below decoder sweet spot.
    # Try bounded upscales of local regions before global threshold variants.
    scale_factors: List[float] = []
    if max_side < 180:
        scale_factors = [2.0, 3.0, 4.0]
    elif max_side < 450:
        scale_factors = [1.7, 2.5, 3.2]
    elif max_side < 1200:
        scale_factors = [1.5, 2.0]
    for scale in scale_factors:
        up = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append(_resize_max_side(up, max_output_side))
        variants.append(_add_quiet_zone(_resize_max_side(up, max_output_side), ratio=0.12))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    denoise = cv2.bilateralFilter(clahe, 5, 30, 30)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    # Small-block and large-block adaptive thresholding handle glare and
    # non-uniform shelf lighting differently.
    adaptive_small = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    adaptive_large = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        10,
    )
    adaptive_inv = cv2.bitwise_not(adaptive)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph_variants = [
        cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, morph_kernel, iterations=1),
        cv2.morphologyEx(otsu, cv2.MORPH_OPEN, morph_kernel, iterations=1),
        cv2.morphologyEx(adaptive_small, cv2.MORPH_CLOSE, morph_kernel, iterations=1),
        cv2.bitwise_not(cv2.morphologyEx(adaptive_large, cv2.MORPH_OPEN, morph_kernel, iterations=1)),
    ]
    for single in [gray, clahe, denoise, sharp, otsu, cv2.bitwise_not(otsu), adaptive, adaptive_small, adaptive_large, adaptive_inv, *morph_variants]:
        bgr = cv2.cvtColor(single, cv2.COLOR_GRAY2BGR)
        variants.append(bgr)
        variants.append(_add_quiet_zone(bgr, ratio=0.12))

    # Rotation variants cover phone/robot roll and vertical shelf labels.
    for rotated in [
        cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image_bgr, cv2.ROTATE_180),
        cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]:
        variants.append(rotated)
    for angle in (-15.0, -8.0, 8.0, 15.0):
        variants.append(_resize_max_side(_rotate_bound(image_bgr, angle), max_output_side))

    # Keep only content-distinct variants to avoid hammering native decoders.
    unique: List[np.ndarray] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    for im in variants:
        if im is None or im.size == 0:
            continue
        key = _variant_signature(im)
        if key in seen:
            continue
        unique.append(np.ascontiguousarray(im))
        seen.add(key)
    return unique


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _add_region(
    image_bgr: np.ndarray,
    regions: List[np.ndarray],
    boxes: List[Tuple[int, int, int, int]],
    box: Tuple[int, int, int, int],
    min_side: int = 28,
) -> None:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_box(*box, w=w, h=h)
    if x2 - x1 < min_side or y2 - y1 < min_side:
        return
    candidate = (x1, y1, x2, y2)
    for other in boxes:
        ox1, oy1, ox2, oy2 = other
        if other == (0, 0, w, h) and candidate != other:
            continue
        ix1, iy1, ix2, iy2 = max(x1, ox1), max(y1, oy1), min(x2, ox2), min(y2, oy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = (x2 - x1) * (y2 - y1)
        other_area = (ox2 - ox1) * (oy2 - oy1)
        if inter / max(1, min(area, other_area)) > 0.92:
            return
    boxes.append(candidate)
    regions.append(np.ascontiguousarray(image_bgr[y1:y2, x1:x2]))


def _qr_candidate_regions(image_bgr: np.ndarray, max_regions: int = 12) -> List[np.ndarray]:
    """Return full tag plus likely QR/barcode sub-crops.

    Lenta templates place QR/barcode payloads in either lower/right white
    blocks or next to price panels. Returning only 2-3 regions was too lossy:
    contour-discovered QR areas were silently truncated before decoding.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    image_bgr = np.ascontiguousarray(image_bgr)
    h, w = image_bgr.shape[:2]
    regions: List[np.ndarray] = [image_bgr]
    boxes: List[Tuple[int, int, int, int]] = [(0, 0, w, h)]

    # Point-aware regions have the highest value: OpenCV often localizes QR
    # corners but returns empty text on blurred retail frames. Rectified point
    # crops are cheap and should not be truncated by layout priors.
    if os.environ.get("LENTA_QR_ENABLE_POINT_REGIONS", "1") != "0":
        for region in _opencv_point_regions(image_bgr, max_regions=4):
            regions.append(np.ascontiguousarray(region))
            boxes.append((-1000 - len(boxes), -1000, -999 - len(boxes), -999))
            if len(regions) >= max_regions:
                return regions[:max_regions]

    # Layout priors cover 1D barcode strips and QR blocks.  Jury feedback says
    # barcode+bbox+timestamp is an early scoring gate, so wide lower strips must
    # be near the front of the native-decoder budget; otherwise max_native=2
    # spends all zxing/pyzbar attempts on full-crop variants and never sees the
    # barcode.
    priors = [
        (int(0.52 * w), 0, w, int(0.72 * h)),             # top-right QR block on many tags
        (int(0.58 * w), 0, w, min(h, int(0.52 * w))),     # top-right square QR prior
        (int(0.48 * w), int(0.05 * h), w, int(0.55 * h)), # upper-right/right-middle QR block
        (0, int(0.50 * h), w, h),                         # lower barcode/SKU strip
        (0, int(0.62 * h), w, h),                         # very lower 1D barcode strip
        (int(0.30 * w), int(0.42 * h), w, h),             # lower-right QR/barcode block
        (int(0.45 * w), 0, w, h),
        (int(0.35 * w), int(0.10 * h), w, h),
        (int(0.50 * w), int(0.35 * h), w, h),
        (0, int(0.35 * h), int(0.70 * w), h),
        (0, 0, int(0.70 * w), int(0.70 * h)),
        (0, 0, w, int(0.50 * h)),
        (0, 0, int(0.50 * w), h),
        (int(0.25 * w), int(0.25 * h), int(0.85 * w), int(0.90 * h)),
    ]
    for box in priors:
        _add_region(image_bgr, regions, boxes, box)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(clahe, (3, 3), 0), 35, 140)
    dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 16 or bh < 16 or bw > 0.90 * w or bh > 0.90 * h:
            continue
        aspect = bw / max(1, bh)
        if not 0.35 <= aspect <= 2.8:
            continue
        roi_gray = gray[y : y + bh, x : x + bw]
        if roi_gray.size == 0:
            continue
        edge_density = float((edges[y : y + bh, x : x + bw] > 0).mean())
        dark_density = float((roi_gray < 130).mean())
        if edge_density < 0.030 or dark_density < 0.025:
            continue
        square_score = 1.0 - min(1.0, abs(aspect - 1.0))
        pad = int(max(8, 0.55 * max(bw, bh)))
        area = (bw + 2 * pad) * (bh + 2 * pad)
        score = (0.55 * edge_density + 0.35 * dark_density + 0.10 * square_score) * area
        scored.append((score, (x - pad, y - pad, x + bw + pad, y + bh + pad)))
    for _, box in sorted(scored, reverse=True)[: max(0, max_regions - len(regions)) + 4]:
        _add_region(image_bgr, regions, boxes, box)

    return regions[:max_regions]


def _ean13_is_valid(code: str) -> bool:
    digits = re.sub(r"\D", "", str(code or ""))
    if len(digits) != 13:
        return False
    nums = [int(ch) for ch in digits]
    checksum = (10 - ((sum(nums[:-1:2]) + 3 * sum(nums[1:-1:2])) % 10)) % 10
    return checksum == nums[-1]


def _ean8_is_valid(code: str) -> bool:
    digits = re.sub(r"\D", "", str(code or ""))
    if len(digits) != 8:
        return False
    nums = [int(ch) for ch in digits]
    checksum = (10 - ((3 * sum(nums[0:7:2]) + sum(nums[1:7:2])) % 10)) % 10
    return checksum == nums[-1]


def _gtin14_is_valid(code: str) -> bool:
    digits = re.sub(r"\D", "", str(code or ""))
    if len(digits) != 14:
        return False
    nums = [int(ch) for ch in digits]
    total = 0
    # GS1 check digit: from the right, excluding check digit, weights 3/1.
    for idx, digit in enumerate(reversed(nums[:-1])):
        total += digit * (3 if idx % 2 == 0 else 1)
    checksum = (10 - (total % 10)) % 10
    return checksum == nums[-1]


def _reliable_numeric_payload(raw: str) -> str:
    digits = re.sub(r"\D", "", str(raw or ""))
    if len(digits) == 13 and _ean13_is_valid(digits):
        return digits
    if len(digits) == 14 and _gtin14_is_valid(digits):
        return digits[1:] if digits.startswith("0") and _ean13_is_valid(digits[1:]) else digits
    if len(digits) == 8 and _ean8_is_valid(digits) and os.environ.get("LENTA_QR_ACCEPT_EAN8", "0") != "0":
        return digits
    return ""


def _looks_like_keyed_qr_payload(payload: str) -> bool:
    raw = str(payload or "").strip()
    if not raw:
        return False
    if raw.startswith("{") and raw.endswith("}"):
        return True
    if re.search(r"(?:^|[?;&|,\s])(?:barcode|b|p1|p2|p3|p4|price1|price2|price3|price4|aP|aC|action|wL1|wL2)\s*[:=]", raw, re.I):
        return True
    if re.search(r"\(01\)\s*\d{14}", raw):
        return True
    return False


def _has_structured_payload(payloads: List[str]) -> bool:
    # Fast-exit only on payloads that are likely to survive CSV scoring.  The
    # previous version treated any 8-14 digit blob as structured; debug output
    # showed short noise such as 11111108/24210522 then blocked full/context
    # QR fallbacks.  Keep decoding until we get a valid EAN/GTIN or keyed QR.
    for payload in payloads:
        raw = str(payload or "").strip()
        if _looks_like_keyed_qr_payload(raw):
            return True
        if raw == re.sub(r"\D", "", raw) and _reliable_numeric_payload(raw):
            return True
    return False


def _region_has_machine_code_texture(image_bgr: np.ndarray) -> bool:
    """Cheap gate for 1D/2D code-like texture before native zbar/zxing calls."""
    if image_bgr is None or image_bgr.size == 0:
        return False
    h, w = image_bgr.shape[:2]
    if h < 20 or w < 20:
        return False
    aspect = w / max(1, h)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark_density = float((gray < 135).mean())
    edges = cv2.Canny(gray, 45, 160)
    edge_density = float((edges > 0).mean())
    # 1D barcode: very wide/tall and high-frequency; QR: square-ish dense module
    # texture.  Keep thresholds loose because video blur reduces edge density.
    barcode_like = (aspect >= 1.65 or aspect <= 0.60) and dark_density >= 0.035 and edge_density >= 0.015
    qr_like = 0.45 <= aspect <= 2.2 and dark_density >= 0.045 and edge_density >= 0.020
    return barcode_like or qr_like


def _native_priority_variants(regions: List[np.ndarray], fallback_variants: List[np.ndarray]) -> List[np.ndarray]:
    """Prioritize variants for zxing/pyzbar.

    Native decoders are the only local path that reads 1D barcodes.  They are
    bounded/crash-isolated, so choose a small, high-value set: barcode-like
    wide strips first, then QR-like local crops, then full-crop fallback.
    """
    textured: List[np.ndarray] = []
    plain: List[np.ndarray] = []
    for region in regions:
        if region is None or region.size == 0:
            continue
        (textured if _region_has_machine_code_texture(region) else plain).append(region)
    ordered = textured + plain
    out: List[np.ndarray] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()
    for region in ordered[: max(1, _env_int("LENTA_QR_NATIVE_REGION_BUDGET", 6))]:
        h, w = region.shape[:2]
        aspect = max(w / max(1, h), h / max(1, w))
        region_variants: List[np.ndarray] = []
        if aspect >= 1.35:
            region_variants.extend(_barcode_image_variants(region))
        region_variants.extend(_qr_image_variants(region))
        for im in region_variants[: max(1, _env_int("LENTA_QR_NATIVE_VARIANTS_PER_REGION", 2))]:
            if im is None or im.size == 0:
                continue
            key = _variant_signature(im)
            if key in seen:
                continue
            out.append(np.ascontiguousarray(im))
            seen.add(key)
    if not out:
        out = list(fallback_variants)
    return out


def _decode_native_variants_processpool(variants: List[np.ndarray], stats: Dict[str, object]) -> List[str]:
    if os.environ.get("LENTA_QR_NATIVE_SUBPROCESS", "1") == "0":
        return []
    max_native = max(0, _env_int("LENTA_QR_NATIVE_MAX_VARIANTS", _env_int("LENTA_QR_ZXING_MAX_VARIANTS", 2)))
    if max_native <= 0 or not variants:
        return []
    enable_zxing = os.environ.get("LENTA_QR_ENABLE_ZXING", "1") != "0"
    enable_pyzbar = os.environ.get("LENTA_QR_ENABLE_PYZBAR", "1") != "0"
    if not enable_zxing and not enable_pyzbar:
        return []
    timeout_sec = max(0.2, _env_float("LENTA_QR_NATIVE_TIMEOUT_SEC", 2.5))
    max_side = _env_int("LENTA_QR_NATIVE_MAX_SIDE", 900)
    images = [_resize_max_side(np.ascontiguousarray(im), max_side) for im in variants[:max_native] if im is not None and im.size]
    if not images:
        return []
    stats["native_processpool"] = True
    stats["native_images"] = len(images)
    try:
        future = _get_native_executor().submit(_native_decode_arrays_worker, images, enable_zxing, enable_pyzbar)
        result = future.result(timeout=timeout_sec)
        worker_stats = result.get("stats") if isinstance(result, dict) else None
        if isinstance(worker_stats, dict):
            stats["native_worker_stats"] = worker_stats
        return _dedupe_payloads([str(x) for x in result.get("payloads", [])]) if isinstance(result, dict) else []
    except concurrent.futures.TimeoutError:
        stats["native_timeout"] = True
        _reset_native_executor()
    except Exception as exc:
        stats.setdefault("errors", []).append(f"native-processpool:{type(exc).__name__}:{exc}")
        _reset_native_executor()
    return []


def _decode_native_variants_subprocess(variants: List[np.ndarray], stats: Dict[str, object]) -> List[str]:
    """Run zxing/pyzbar out of process so native crashes do not kill Kaggle."""
    if os.environ.get("LENTA_QR_NATIVE_SUBPROCESS", "1") == "0":
        return []
    backend = os.environ.get("LENTA_QR_NATIVE_BACKEND", "subprocess").lower()
    if backend == "processpool":
        return _decode_native_variants_processpool(variants, stats)
    if not variants:
        return []
    max_native = max(0, _env_int("LENTA_QR_NATIVE_MAX_VARIANTS", _env_int("LENTA_QR_ZXING_MAX_VARIANTS", 12)))
    if max_native <= 0:
        return []
    timeout_sec = max(1.0, _env_float("LENTA_QR_NATIVE_TIMEOUT_SEC", 8.0))
    enable_zxing = os.environ.get("LENTA_QR_ENABLE_ZXING", "1") != "0"
    enable_pyzbar = os.environ.get("LENTA_QR_ENABLE_PYZBAR", "1") != "0"
    if not enable_zxing and not enable_pyzbar:
        return []

    payloads: List[str] = []
    with tempfile.TemporaryDirectory(prefix="lenta_qr_") as tmp:
        tmpdir = Path(tmp)
        paths: List[str] = []
        for idx, im in enumerate(variants[:max_native]):
            if im is None or im.size == 0:
                continue
            out = tmpdir / f"v{idx:03d}.png"
            try:
                cv2.imwrite(str(out), im)
                paths.append(str(out))
            except Exception:
                continue
        if not paths:
            return []
        cmd = [sys.executable, "-m", "lenta_shelf_ai.qr_worker"]
        if not enable_zxing:
            cmd.append("--no-zxing")
        if not enable_pyzbar:
            cmd.append("--no-pyzbar")
        cmd.extend(paths)
        stats["native_subprocess"] = True
        stats["native_images"] = len(paths)
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec)
            stats["native_returncode"] = int(proc.returncode)
            if proc.stderr:
                stats.setdefault("errors", []).append(str(proc.stderr)[-500:])
            if proc.returncode != 0:
                return []
            data = json.loads(proc.stdout or "{}")
            worker_stats = data.get("stats") if isinstance(data, dict) else None
            if isinstance(worker_stats, dict):
                stats["native_worker_stats"] = worker_stats
            payloads = [str(x) for x in data.get("payloads", [])] if isinstance(data, dict) else []
        except subprocess.TimeoutExpired:
            stats["native_timeout"] = True
        except Exception as exc:
            stats.setdefault("errors", []).append(f"native-subprocess:{type(exc).__name__}:{exc}")
    return _dedupe_payloads(payloads)


def _decode_native_variants_inprocess(variants: List[np.ndarray], stats: Dict[str, object]) -> List[str]:
    """Fast path for trusted environments. Disabled by default due native crashes."""
    payloads: List[str] = []
    if os.environ.get("LENTA_QR_NATIVE_SUBPROCESS", "1") != "0":
        return payloads
    max_zxing = max(0, _env_int("LENTA_QR_ZXING_MAX_VARIANTS", len(variants)))
    max_pyzbar = max(0, _env_int("LENTA_QR_PYZBAR_MAX_VARIANTS", len(variants)))

    if os.environ.get("LENTA_QR_ENABLE_ZXING", "1") != "0" and max_zxing:
        try:  # pragma: no cover - optional native package
            import zxingcpp

            fmt = getattr(zxingcpp.BarcodeFormat, "All", None)
            binarizer = getattr(zxingcpp.Binarizer, "LocalAverage", None)
            for im in variants[:max_zxing]:
                stats["zxing_attempts"] = int(stats.get("zxing_attempts", 0)) + 1
                kwargs = dict(try_rotate=True, try_downscale=True, try_invert=True)
                if fmt is not None:
                    kwargs["formats"] = fmt
                if binarizer is not None:
                    kwargs["binarizer"] = binarizer
                try:
                    results = zxingcpp.read_barcodes(im, **kwargs)
                except TypeError:
                    kwargs.pop("binarizer", None)
                    results = zxingcpp.read_barcodes(im, **kwargs)
                for r in results:
                    text = getattr(r, "text", "") or ""
                    if text:
                        payloads.append(text)
                if _has_structured_payload(payloads):
                    break
        except Exception as exc:
            stats.setdefault("errors", []).append(f"zxing:{type(exc).__name__}:{exc}")

    if os.environ.get("LENTA_QR_ENABLE_PYZBAR", "1") != "0" and max_pyzbar:
        try:
            from pyzbar import pyzbar

            for im in variants[:max_pyzbar]:
                stats["pyzbar_attempts"] = int(stats.get("pyzbar_attempts", 0)) + 1
                rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                for obj in pyzbar.decode(rgb):
                    try:
                        text = obj.data.decode("utf-8", errors="ignore")
                    except Exception:
                        text = str(obj.data)
                    if text:
                        payloads.append(text)
        except Exception as exc:
            stats.setdefault("errors", []).append(f"pyzbar:{type(exc).__name__}:{exc}")
    return _dedupe_payloads(payloads)



def _decode_opencv_barcode_variants(variants: List[np.ndarray], stats: Dict[str, object]) -> List[str]:
    """Optional OpenCV 1D barcode decoder when opencv-contrib is available."""
    payloads: List[str] = []
    if os.environ.get("LENTA_QR_ENABLE_OPENCV_BARCODE", "1") == "0":
        return payloads
    barcode_mod = getattr(cv2, "barcode", None)
    detector_cls = getattr(barcode_mod, "BarcodeDetector", None) if barcode_mod is not None else None
    if detector_cls is None:
        stats["opencv_barcode_available"] = False
        return payloads
    max_barcode = max(0, _env_int("LENTA_QR_OPENCV_BARCODE_MAX_VARIANTS", 6))
    if max_barcode <= 0:
        return payloads
    try:
        detector = detector_cls()
    except Exception as exc:
        stats.setdefault("errors", []).append(f"opencv-barcode-init:{type(exc).__name__}:{exc}")
        return payloads
    stats["opencv_barcode_available"] = True
    for im in variants[:max_barcode]:
        stats["opencv_barcode_attempts"] = int(stats.get("opencv_barcode_attempts", 0)) + 1
        try:
            # OpenCV Python signatures differ across builds. Try the richer API
            # first, then fall back to detectAndDecode.
            if hasattr(detector, "detectAndDecodeWithType"):
                out = detector.detectAndDecodeWithType(im)
                decoded_info = out[1] if isinstance(out, tuple) and len(out) >= 2 else []
            else:
                out = detector.detectAndDecode(im)
                decoded_info = out[0] if isinstance(out, tuple) else out
            if isinstance(decoded_info, str):
                if decoded_info:
                    payloads.append(decoded_info)
            else:
                for text in decoded_info or []:
                    if text:
                        payloads.append(str(text))
        except Exception as exc:
            stats.setdefault("errors", []).append(f"opencv-barcode:{type(exc).__name__}:{exc}")
    return _dedupe_payloads(payloads)


def _decode_opencv_warped_variants(detector: cv2.QRCodeDetector, image_bgr: np.ndarray, points: np.ndarray, stats: Dict[str, object]) -> List[str]:
    """Decode a perspective-normalized QR when OpenCV sees corners but returns empty text."""
    payloads: List[str] = []
    if os.environ.get("LENTA_QR_ENABLE_WARP", "1") == "0":
        return payloads
    warped = _warp_quad(image_bgr, points, pad_ratio=_env_float("LENTA_QR_WARP_QUIET_ZONE", 0.32))
    if warped is None or getattr(warped, "size", 0) == 0:
        return payloads
    max_warp = max(0, _env_int("LENTA_QR_WARP_MAX_VARIANTS", 8))
    if max_warp <= 0:
        return payloads
    stats["opencv_warp_regions"] = int(stats.get("opencv_warp_regions", 0)) + 1
    enable_curved = os.environ.get("LENTA_QR_ENABLE_OPENCV_CURVED", "0") != "0"
    methods = ("detectAndDecode", "detectAndDecodeCurved") if enable_curved else ("detectAndDecode",)
    for variant in _qr_image_variants(warped)[:max_warp]:
        stats["opencv_warp_attempts"] = int(stats.get("opencv_warp_attempts", 0)) + 1
        for method in methods:
            try:
                text, _, straight = getattr(detector, method)(variant)
            except Exception:
                continue
            if text:
                payloads.append(str(text))
            if straight is not None and getattr(straight, "size", 0):
                try:
                    straight_bgr = cv2.cvtColor(straight, cv2.COLOR_GRAY2BGR) if straight.ndim == 2 else straight
                    text2, _, _ = detector.detectAndDecode(straight_bgr)
                    if text2:
                        payloads.append(str(text2))
                except Exception:
                    pass
        if payloads and os.environ.get("LENTA_QR_FAST_EXIT", "1") != "0" and _has_structured_payload(payloads):
            break
    if payloads:
        stats["opencv_warp_payloads"] = int(stats.get("opencv_warp_payloads", 0)) + len(payloads)
    return _dedupe_payloads(payloads)

def _decode_opencv_variants(variants: List[np.ndarray], stats: Dict[str, object]) -> List[str]:
    payloads: List[str] = []
    if os.environ.get("LENTA_QR_ENABLE_OPENCV", "1") == "0":
        return payloads
    enable_curved = os.environ.get("LENTA_QR_ENABLE_OPENCV_CURVED", "0") != "0"
    max_opencv = max(0, _env_int("LENTA_QR_OPENCV_MAX_VARIANTS", len(variants)))
    detector = cv2.QRCodeDetector()
    for setter in ("setEpsX", "setEpsY"):
        try:
            getattr(detector, setter)(0.2)
        except Exception:
            pass
    for im in variants[:max_opencv]:
        stats["opencv_attempts"] = int(stats.get("opencv_attempts", 0)) + 1
        try:
            ok, decoded, points, _ = detector.detectAndDecodeMulti(im)
            if ok and decoded:
                payloads.extend(str(text) for text in decoded if text)
            try:
                ok_points, points_only = detector.detectMulti(im)
                if ok_points and points_only is not None:
                    for pts in np.asarray(points_only).reshape(-1, 4, 2)[:3]:
                        try:
                            stats["opencv_points_detected"] = int(stats.get("opencv_points_detected", 0)) + 1
                            text, straight = detector.decode(im, pts.astype(np.float32))
                            if text:
                                payloads.append(text)
                            else:
                                payloads.extend(_decode_opencv_warped_variants(detector, im, pts.astype(np.float32), stats))
                            if straight is not None and getattr(straight, "size", 0):
                                text2, _, _ = detector.detectAndDecode(cv2.cvtColor(straight, cv2.COLOR_GRAY2BGR) if straight.ndim == 2 else straight)
                                if text2:
                                    payloads.append(text2)
                        except Exception:
                            pass
            except Exception:
                pass
            methods = ("detectAndDecode", "detectAndDecodeCurved") if enable_curved else ("detectAndDecode",)
            for method in methods:
                try:
                    text, points, straight = getattr(detector, method)(im)
                except Exception:
                    continue
                if text:
                    payloads.append(text)
                elif points is not None and getattr(points, "size", 0):
                    try:
                        payloads.extend(_decode_opencv_warped_variants(detector, im, np.asarray(points).reshape(-1, 2)[:4], stats))
                    except Exception:
                        pass
                if straight is not None and getattr(straight, "size", 0):
                    try:
                        text2, _, _ = detector.detectAndDecode(cv2.cvtColor(straight, cv2.COLOR_GRAY2BGR) if straight.ndim == 2 else straight)
                        if text2:
                            payloads.append(text2)
                    except Exception:
                        pass
        except Exception as exc:
            stats.setdefault("errors", []).append(f"opencv:{type(exc).__name__}:{exc}")
            continue
        if os.environ.get("LENTA_QR_FAST_EXIT", "1") != "0" and _has_structured_payload(payloads):
            break
    return _dedupe_payloads(payloads)



def _default_wechat_model_paths() -> List[str]:
    """Find bundled OpenCV WeChatQRCode model files when env vars are empty."""
    env_paths = [
        os.environ.get("LENTA_WECHAT_QR_DETECT_PROTOTXT", ""),
        os.environ.get("LENTA_WECHAT_QR_DETECT_CAFFEMODEL", ""),
        os.environ.get("LENTA_WECHAT_QR_SR_PROTOTXT", ""),
        os.environ.get("LENTA_WECHAT_QR_SR_CAFFEMODEL", ""),
    ]
    if all(env_paths) and all(Path(item).exists() for item in env_paths):
        return [str(item) for item in env_paths]

    roots = [
        Path.cwd() / "models/wechat_qr",
        Path(__file__).resolve().parents[1] / "models/wechat_qr",
    ]
    names = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
    for root in roots:
        paths = [root / name for name in names]
        if all(path.exists() for path in paths):
            return [str(path) for path in paths]
    return []

def _decode_wechat_qr_variants(variants: List[np.ndarray], stats: Dict[str, object]) -> List[str]:
    """Optional OpenCV WeChat QR fallback for tiny/blurred QR crops.

    The detector is available only in opencv-contrib builds. Model paths are
    optional and can be provided through LENTA_WECHAT_QR_* env vars. Missing
    contrib/model support is treated as a normal skip, not as a pipeline error.
    """
    payloads: List[str] = []
    if os.environ.get("LENTA_QR_ENABLE_WECHAT", "1") == "0":
        return payloads
    module = getattr(cv2, "wechat_qrcode", None)
    detector_cls = getattr(module, "WeChatQRCode", None) if module is not None else None
    if detector_cls is None:
        stats["wechat_available"] = False
        return payloads
    max_wechat = max(0, _env_int("LENTA_QR_WECHAT_MAX_VARIANTS", 8))
    if max_wechat <= 0:
        return payloads

    args: list[str] = _default_wechat_model_paths()
    stats["wechat_model_paths"] = bool(args)
    try:
        detector = detector_cls(*args)
    except Exception as exc:
        stats.setdefault("errors", []).append(f"wechat-init:{type(exc).__name__}:{exc}")
        return payloads

    stats["wechat_available"] = True
    for im in variants[:max_wechat]:
        stats["wechat_attempts"] = int(stats.get("wechat_attempts", 0)) + 1
        try:
            out = detector.detectAndDecode(im)
        except Exception as exc:
            stats.setdefault("errors", []).append(f"wechat:{type(exc).__name__}:{exc}")
            continue
        decoded = out[0] if isinstance(out, tuple) and len(out) >= 1 else out
        if isinstance(decoded, str):
            if decoded:
                payloads.append(decoded)
        else:
            for text in decoded or []:
                if text:
                    payloads.append(str(text))
        if hasattr(detector, "detectAndDecodeMulti"):
            try:
                multi = detector.detectAndDecodeMulti(im)
                decoded_multi = multi[0] if isinstance(multi, tuple) and len(multi) >= 1 else multi
                if isinstance(decoded_multi, str):
                    if decoded_multi:
                        payloads.append(decoded_multi)
                else:
                    for text in decoded_multi or []:
                        if text:
                            payloads.append(str(text))
            except Exception:
                pass
        if os.environ.get("LENTA_QR_FAST_EXIT", "1") != "0" and _has_structured_payload(payloads):
            break
    return _dedupe_payloads(payloads)


def decode_qr_payloads_with_debug(image_bgr: np.ndarray, force_native: bool = False, debug_label: str = "") -> tuple[List[str], Dict[str, object]]:
    """Decode QR/barcodes with bounded local cascade and debug counters."""
    start = time.time()
    stats: Dict[str, object] = {
        "regions": 0,
        "variants": 0,
        "payloads": 0,
        "native_subprocess": False,
        "zxing_attempts": 0,
        "pyzbar_attempts": 0,
        "opencv_attempts": 0,
        "errors": [],
        "force_native": bool(force_native),
        "debug_label": debug_label,
    }
    if image_bgr is None or image_bgr.size == 0:
        stats["elapsed_ms"] = 0
        return [], stats
    max_regions = max(1, _env_int("LENTA_QR_MAX_REGIONS", 8))
    max_variants = _env_int("LENTA_QR_MAX_VARIANTS", 24)
    regions = _qr_candidate_regions(image_bgr, max_regions=max_regions)
    stats["regions"] = len(regions)
    fast_variants: List[np.ndarray] = []
    slow_variants: List[np.ndarray] = []
    for region_idx, region in enumerate(regions):
        if max_variants > 0 and len(fast_variants) + len(slow_variants) >= max_variants:
            break
        variants_for_region = _qr_image_variants(region)
        for im in variants_for_region[:4]:
            if max_variants > 0 and len(fast_variants) + len(slow_variants) >= max_variants:
                break
            fast_variants.append(im)
        if region_idx < 2:
            for im in variants_for_region[4:10]:
                if max_variants > 0 and len(fast_variants) + len(slow_variants) >= max_variants:
                    break
                slow_variants.append(im)
    variants = fast_variants + slow_variants
    if max_variants > 0:
        variants = variants[:max_variants]
    stats["variants"] = len(variants)

    payloads: List[str] = []
    # OpenCV first: pure Python binding and usually safe; native zbar/zxing is
    # crash-isolated below.
    payloads.extend(_decode_opencv_variants(variants, stats))
    if not (_has_structured_payload(payloads) and os.environ.get("LENTA_QR_FAST_EXIT", "1") != "0"):
        payloads.extend(_decode_wechat_qr_variants(variants, stats))
    if not (_has_structured_payload(payloads) and os.environ.get("LENTA_QR_FAST_EXIT", "1") != "0"):
        payloads.extend(_decode_opencv_barcode_variants(_native_priority_variants(regions, variants), stats))
    if not (_has_structured_payload(payloads) and os.environ.get("LENTA_QR_FAST_EXIT", "1") != "0"):
        native_variants = _native_priority_variants(regions, variants)
        stats["native_priority_variants"] = len(native_variants)
        has_code_texture = any(_region_has_machine_code_texture(region) for region in regions[:8])
        native_reason = ""
        if force_native:
            native_reason = "force_native"
        elif os.environ.get("LENTA_QR_NATIVE_ALWAYS", "0") != "0":
            native_reason = "always"
        elif int(stats.get("opencv_points_detected", 0) or 0) > 0:
            native_reason = "opencv_points"
        elif has_code_texture:
            native_reason = "machine_code_texture"
        elif os.environ.get("LENTA_QR_ENABLE_OPENCV", "1") == "0":
            native_reason = "opencv_disabled"
        native_allowed = bool(native_reason)
        stats["native_allowed"] = bool(native_allowed)
        stats["native_allowed_reason"] = native_reason
        if native_allowed:
            payloads.extend(_decode_native_variants_subprocess(native_variants, stats))
            payloads.extend(_decode_native_variants_inprocess(native_variants, stats))
    payloads = _dedupe_payloads(payloads)
    stats["payloads"] = len(payloads)
    stats["elapsed_ms"] = int(round((time.time() - start) * 1000.0))
    return payloads, stats


def decode_qr_payloads(image_bgr: np.ndarray) -> List[str]:
    payloads, _ = decode_qr_payloads_with_debug(image_bgr)
    return payloads

def _flatten_qs(qs: Dict[str, List[str]]) -> Dict[str, str]:
    return {k: (v[-1] if isinstance(v, list) and v else str(v)) for k, v in qs.items()}


def parse_qr_payload(payload: str) -> Dict[str, str]:
    """Parse known Lenta QR fields from JSON/query/semicolon payloads."""
    if not payload:
        return {}
    raw = payload.strip()
    data: Dict[str, str] = {}
    raw_digits = re.sub(r"\D", "", raw)
    if raw == raw_digits and 8 <= len(raw_digits) <= 14:
        reliable = _reliable_numeric_payload(raw_digits)
        return {"qr_code_barcode": reliable} if reliable else {}
    gs1_gtin = re.search(r"\(01\)\s*(\d{14})", raw)
    if gs1_gtin:
        gtin = _reliable_numeric_payload(gs1_gtin.group(1))
        return {"qr_code_barcode": gtin} if gtin else {}

    # JSON QR. Some public/team prototypes encode payloads as
    # {"ean": "...", "prices": ["p1", "p2", ...]} instead of flat p1/p2 keys.
    if raw.startswith("{") and raw.endswith("}"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if v is None or isinstance(v, (list, tuple, dict)):
                        continue
                    data[str(k)] = str(v)
                prices = obj.get("prices")
                if isinstance(prices, (list, tuple)):
                    for idx, value in enumerate(prices[:4], start=1):
                        if value is not None:
                            data.setdefault(f"price{idx}", str(value))
        except Exception:
            pass

    # URL or query string.
    if not data:
        parsed = urlparse(raw)
        query = parsed.query or (raw if "=" in raw else "")
        if query:
            query = query.replace(";", "&").replace("|", "&")
            try:
                data.update(_flatten_qs(parse_qs(query, keep_blank_values=True)))
            except Exception:
                pass

    # key:value/key=value pairs.
    if not data:
        for m in re.finditer(r"([A-Za-z0-9_]+)\s*[:=]\s*([^;&|,\n\r]+)", raw):
            data[m.group(1)] = m.group(2).strip()

    # Delimited raw payloads like "460...|129.99|99.99" appear in team
    # prototypes. Accept only checksum-valid numeric IDs to avoid the old
    # 8-digit OCR-noise failure mode.
    if not data:
        parts = [part.strip() for part in re.split(r"[|;,\n\r]+", raw) if part.strip()]
        prices: List[str] = []
        for part in parts:
            digits = re.sub(r"\D", "", part)
            reliable = _reliable_numeric_payload(digits) if 8 <= len(digits) <= 14 else ""
            if reliable and "barcode" not in data:
                data["barcode"] = reliable
                continue
            if re.fullmatch(r"\d{1,5}\s*[.,]\s*\d{2}", part):
                prices.append(part)
        for idx, value in enumerate(prices[:4], start=1):
            data.setdefault(f"price{idx}", value)

    # Compact pairs like b467..., p1123.45 are rare but cheap to support.
    if not data:
        for key in ["barcode", "b", "p1", "p2", "p3", "p4", "aP", "aC", "wL1C", "wL1P", "wL2C", "wL2P"]:
            m = re.search(rf"(?:^|\W){re.escape(key)}\W*([0-9A-Za-z.,_-]+)", raw)
            if m:
                data[key] = m.group(1)

    # Decode aliases case-insensitively; payloads from QR generators are not
    # consistent about camelCase/lowercase keys.
    lower_data = {str(k).lower(): v for k, v in data.items()}
    normalized: Dict[str, str] = {}
    for target, aliases in QR_FIELD_ALIASES.items():
        for alias in aliases:
            source_val = data.get(alias)
            if source_val is None:
                source_val = lower_data.get(alias.lower())
            if source_val is not None and str(source_val).strip() != "":
                val = str(source_val).strip()
                if "price" in target or target.endswith("_price") or target.startswith("price") or target == "action_price_qr":
                    val = price_to_str(val)
                normalized[target] = val
                break
    return normalized


def parse_qr_payloads(payloads: Iterable[str]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for payload in payloads:
        parsed = parse_qr_payload(payload)
        for k, v in parsed.items():
            if v and k not in merged:
                merged[k] = v
    return merged
