from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .schema import QR_FIELD_ALIASES
from .utils import price_to_str


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


def _qr_image_variants(image_bgr: np.ndarray) -> List[np.ndarray]:
    if image_bgr is None or image_bgr.size == 0:
        return []
    image_bgr = np.ascontiguousarray(image_bgr)
    variants: List[np.ndarray] = [image_bgr]
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
        variants.append(cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
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
    adaptive_inv = cv2.bitwise_not(adaptive)
    for single in [gray, clahe, sharp, otsu, adaptive, adaptive_inv]:
        variants.append(cv2.cvtColor(single, cv2.COLOR_GRAY2BGR))

    # Rotation variants cover phone/robot roll and vertical shelf labels.
    for rotated in [
        cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image_bgr, cv2.ROTATE_180),
        cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]:
        variants.append(rotated)
    for angle in (-15.0, -8.0, 8.0, 15.0):
        variants.append(_rotate_bound(image_bgr, angle))
    return variants


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

    # Layout priors cover right/left QR placement, lower barcode strips, and
    # rotated tags where a "lower" strip appears as a side strip.
    priors = [
        (int(0.45 * w), 0, w, h),
        (int(0.35 * w), int(0.10 * h), w, h),
        (int(0.50 * w), int(0.35 * h), w, h),
        (0, int(0.35 * h), int(0.70 * w), h),
        (0, 0, int(0.70 * w), int(0.70 * h)),
        (0, int(0.55 * h), w, h),
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


def _has_structured_payload(payloads: List[str]) -> bool:
    for payload in payloads:
        if "=" in payload or re.fullmatch(r"\d{8,14}", payload.strip()):
            return True
    return False


def decode_qr_payloads(image_bgr: np.ndarray) -> List[str]:
    """Decode QR/barcodes with local libraries only.

    Order: zxing-cpp (best for distorted codes), pyzbar, OpenCV QR. All imports are
    optional, so the app runs in minimal CPU environments too.
    """
    payloads: List[str] = []
    if image_bgr is None or image_bgr.size == 0:
        return payloads
    regions = _qr_candidate_regions(image_bgr)
    fast_variants: List[np.ndarray] = []
    slow_variants: List[np.ndarray] = []
    for region_idx, region in enumerate(regions):
        variants = _qr_image_variants(region)
        fast_variants.extend(variants[:3])
        if region_idx == 0:
            slow_variants.extend(variants[3:6])
    variants = fast_variants + slow_variants

    # Try zxing-cpp if installed.
    try:  # pragma: no cover - optional native package
        import zxingcpp

        scan_configs = [
            (getattr(zxingcpp.BarcodeFormat, "All", None), getattr(zxingcpp.Binarizer, "LocalAverage", None)),
        ]
        scan_configs = [(fmt, binarizer) for fmt, binarizer in scan_configs if fmt is not None and binarizer is not None]
        for im in variants:
            for fmt, binarizer in scan_configs:
                results = zxingcpp.read_barcodes(
                    im,
                    formats=fmt,
                    try_rotate=True,
                    try_downscale=True,
                    try_invert=True,
                    binarizer=binarizer,
                )
                for r in results:
                    text = getattr(r, "text", "") or ""
                    if text and text not in payloads:
                        payloads.append(text)
                if _has_structured_payload(payloads):
                    break
            if _has_structured_payload(payloads):
                break
    except Exception:
        pass

    # Try pyzbar/zbar.
    try:
        from pyzbar import pyzbar

        for im in variants:
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            for obj in pyzbar.decode(rgb):
                try:
                    text = obj.data.decode("utf-8", errors="ignore")
                except Exception:
                    text = str(obj.data)
                if text and text not in payloads:
                    payloads.append(text)
    except Exception:
        pass

    # OpenCV QR detector. Use single, multi and curved variants because retail
    # video crops can be skewed/warped by the robot perspective.
    detector = cv2.QRCodeDetector()
    for setter in ("setEpsX", "setEpsY"):
        try:
            getattr(detector, setter)(0.2)
        except Exception:
            pass
    for im in variants:
        try:
            ok, decoded, points, _ = detector.detectAndDecodeMulti(im)
            if ok and decoded:
                for text in decoded:
                    if text and text not in payloads:
                        payloads.append(text)
            for method in ("detectAndDecode", "detectAndDecodeCurved"):
                try:
                    text, _, straight = getattr(detector, method)(im)
                except Exception:
                    continue
                if text and text not in payloads:
                    payloads.append(text)
                if straight is not None and getattr(straight, "size", 0):
                    try:
                        text2, _, _ = detector.detectAndDecode(cv2.cvtColor(straight, cv2.COLOR_GRAY2BGR) if straight.ndim == 2 else straight)
                        if text2 and text2 not in payloads:
                            payloads.append(text2)
                    except Exception:
                        pass
        except Exception:
            continue
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
        return {"qr_code_barcode": raw_digits}
    gs1_gtin = re.search(r"\(01\)\s*(\d{14})", raw)
    if gs1_gtin:
        gtin = gs1_gtin.group(1)
        if gtin.startswith("0"):
            gtin = gtin[1:]
        return {"qr_code_barcode": gtin}

    # JSON QR.
    if raw.startswith("{") and raw.endswith("}"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                data.update({str(k): str(v) for k, v in obj.items() if v is not None})
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
