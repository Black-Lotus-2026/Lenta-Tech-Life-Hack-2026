from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def smart_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(" ", "").replace("\u00a0", "").replace(",", ".")
    if text.lower() in {"", "nan", "none", "нет"}:
        return default
    try:
        return float(text)
    except ValueError:
        m = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
        if not m:
            return default
        return float(m.group(0).replace(",", "."))


def smart_int(value: Any, default: int = 0) -> int:
    f = smart_float(value)
    if math.isnan(f):
        return default
    return int(round(f))


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a[:4])
    bx1, by1, bx2, by2 = map(float, b[:4])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def nms_xyxy(boxes: List[Sequence[float]], iou_threshold: float = 0.45) -> List[List[float]]:
    if not boxes:
        return []
    boxes_sorted = sorted([list(map(float, b[:5])) for b in boxes], key=lambda x: x[4], reverse=True)
    keep: List[List[float]] = []
    for box in boxes_sorted:
        if all(iou_xyxy(box, k) < iou_threshold for k in keep):
            keep.append(box)
    return keep


def clip_xyxy(xyxy: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = map(float, xyxy[:4])
    x1 = max(0.0, min(width - 1.0, x1))
    x2 = max(0.0, min(width - 1.0, x2))
    y1 = max(0.0, min(height - 1.0, y1))
    y2 = max(0.0, min(height - 1.0, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def crop_xyxy(image: np.ndarray, xyxy: Sequence[float], pad: int = 0) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clip_xyxy([xyxy[0] - pad, xyxy[1] - pad, xyxy[2] + pad, xyxy[3] + pad], w, h)
    return image[int(y1) : int(y2), int(x1) : int(x2)].copy()


def sharpness_laplacian(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def price_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        s = value.strip().replace(" ", "").replace("₽", "")
        if not s or s.lower() in {"nan", "none"}:
            return ""
        s = s.replace(",", ".")
        m = re.search(r"\d{1,5}(?:\.\d{1,2})?", s)
        if m:
            try:
                return f"{float(m.group(0)):.2f}"
            except ValueError:
                return m.group(0)
        return value.strip()
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def read_yaml_or_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except Exception:
        # tiny fallback for flat key: value configs
        out: Dict[str, Any] = {}
        for line in text.splitlines():
            if ":" in line and not line.strip().startswith("#"):
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip().strip('"\'')
        return out


def mkdir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
