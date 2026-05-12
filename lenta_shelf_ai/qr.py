from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .schema import QR_FIELD_ALIASES
from .utils import price_to_str


def decode_qr_payloads(image_bgr: np.ndarray) -> List[str]:
    """Decode QR/barcodes with local libraries only.

    Order: zxing-cpp (best for distorted codes), pyzbar, OpenCV QR. All imports are
    optional, so the app runs in minimal CPU environments too.
    """
    payloads: List[str] = []
    if image_bgr is None or image_bgr.size == 0:
        return payloads

    # Try zxing-cpp if installed.
    try:  # pragma: no cover - optional native package
        import zxingcpp

        results = zxingcpp.read_barcodes(image_bgr)
        for r in results:
            text = getattr(r, "text", "") or ""
            if text and text not in payloads:
                payloads.append(text)
    except Exception:
        pass

    # Try pyzbar/zbar.
    try:
        from pyzbar import pyzbar

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        for obj in pyzbar.decode(rgb):
            try:
                text = obj.data.decode("utf-8", errors="ignore")
            except Exception:
                text = str(obj.data)
            if text and text not in payloads:
                payloads.append(text)
    except Exception:
        pass

    # OpenCV QR detector. Upscale improves tiny QR codes.
    detector = cv2.QRCodeDetector()
    variants = [image_bgr]
    h, w = image_bgr.shape[:2]
    if max(h, w) < 900:
        variants.append(cv2.resize(image_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variants.extend([
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR),
    ])
    for im in variants:
        try:
            ok, decoded, points, _ = detector.detectAndDecodeMulti(im)
            if ok and decoded:
                for text in decoded:
                    if text and text not in payloads:
                        payloads.append(text)
            else:
                text, _, _ = detector.detectAndDecode(im)
                if text and text not in payloads:
                    payloads.append(text)
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

    normalized: Dict[str, str] = {}
    for target, aliases in QR_FIELD_ALIASES.items():
        for alias in aliases:
            if alias in data and str(data[alias]).strip() != "":
                val = str(data[alias]).strip()
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
