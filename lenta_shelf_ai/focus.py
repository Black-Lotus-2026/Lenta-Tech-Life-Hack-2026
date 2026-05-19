from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import cv2
import numpy as np


def _safe_gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if image_bgr.ndim == 2:
        return image_bgr.astype(np.uint8, copy=False)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _crop_rel(image: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(round(box[0] * w))))
    y1 = max(0, min(h - 1, int(round(box[1] * h))))
    x2 = max(0, min(w, int(round(box[2] * w))))
    y2 = max(0, min(h, int(round(box[3] * h))))
    if x2 <= x1 or y2 <= y1:
        return image[:0, :0]
    return image[y1:y2, x1:x2]


def _entropy(gray: np.ndarray) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).reshape(-1)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / 6.0)  # normalize 64-bin entropy to ~[0, 1]


def _focus_stats(gray: np.ndarray) -> Dict[str, float]:
    if gray is None or gray.size == 0:
        return {
            "laplacian": 0.0,
            "tenengrad": 0.0,
            "brenner": 0.0,
            "spatial_frequency": 0.0,
            "entropy": 0.0,
            "contrast": 0.0,
            "dark_ratio": 0.0,
            "overexposed_ratio": 1.0,
            "underexposed_ratio": 1.0,
        }
    gray_f = gray.astype(np.float32)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    ten = float(np.sqrt(np.mean(gx * gx + gy * gy)))
    if gray_f.shape[1] > 2:
        brenner = float(np.mean((gray_f[:, 2:] - gray_f[:, :-2]) ** 2))
    else:
        brenner = 0.0
    rf = float(np.sqrt(np.mean(np.diff(gray_f, axis=0) ** 2))) if gray_f.shape[0] > 1 else 0.0
    cf = float(np.sqrt(np.mean(np.diff(gray_f, axis=1) ** 2))) if gray_f.shape[1] > 1 else 0.0
    spatial = float(np.sqrt(rf * rf + cf * cf))
    contrast = float(np.std(gray_f))
    return {
        "laplacian": lap,
        "tenengrad": ten,
        "brenner": brenner,
        "spatial_frequency": spatial,
        "entropy": _entropy(gray),
        "contrast": contrast,
        "dark_ratio": float(np.mean(gray < 75)),
        "overexposed_ratio": float(np.mean(gray > 248)),
        "underexposed_ratio": float(np.mean(gray < 8)),
    }


def _score_from_stats(stats: Dict[str, float]) -> float:
    # Saturating terms make the score robust across video resolutions.  Penalize
    # crushed black/white crops because they often OCR well on edges but decode
    # QR/barcodes poorly. The weights are dependency-free and bounded for
    # regression safety.
    lap = min(1.0, stats.get("laplacian", 0.0) / 900.0)
    ten = min(1.0, stats.get("tenengrad", 0.0) / 95.0)
    brenner = min(1.0, stats.get("brenner", 0.0) / 1500.0)
    spatial = min(1.0, stats.get("spatial_frequency", 0.0) / 55.0)
    entropy = min(1.0, max(0.0, stats.get("entropy", 0.0)))
    contrast = min(1.0, stats.get("contrast", 0.0) / 70.0)
    exposure_penalty = min(0.35, 0.45 * stats.get("overexposed_ratio", 0.0) + 0.45 * stats.get("underexposed_ratio", 0.0))
    score = 0.22 * lap + 0.22 * ten + 0.16 * brenner + 0.16 * spatial + 0.12 * entropy + 0.12 * contrast - exposure_penalty
    return float(max(0.0, min(1.0, score)))


def compute_focus_quality(image_bgr: np.ndarray) -> Dict[str, float]:
    """Return crop quality scores for best-frame selection.

    The F run showed that QR-only low-FPS settings picked too few usable frames.
    This score separates frame/crop quality from bbox geometry: it is used for
    semantic best-crop selection, not as proof that a row is correct.
    """
    gray = _safe_gray(image_bgr)
    global_stats = _focus_stats(gray)
    h, w = gray.shape[:2]
    if h <= 2 or w <= 2:
        score = _score_from_stats(global_stats)
        out = {"score": score, "global_score": score, "price_score": score, "header_score": score, "qr_score": score}
        out.update({f"global_{k}": float(v) for k, v in global_stats.items()})
        return out

    # Generic Lenta tag priors: name/header at top, prices in lower/left-middle,
    # QR/barcodes often right/lower-right.  Use broad overlapping regions so it
    # still works across red/yellow/white templates.
    price_roi = _crop_rel(gray, (0.00, 0.20, 0.72, 0.95))
    header_roi = _crop_rel(gray, (0.00, 0.00, 1.00, 0.45))
    qr_roi = _crop_rel(gray, (0.45, 0.00, 1.00, 1.00))
    price_stats = _focus_stats(price_roi)
    header_stats = _focus_stats(header_roi)
    qr_stats = _focus_stats(qr_roi)
    global_score = _score_from_stats(global_stats)
    price_score = _score_from_stats(price_stats)
    header_score = _score_from_stats(header_stats)
    qr_score = _score_from_stats(qr_stats)
    score = float(max(0.0, min(1.0, 0.30 * global_score + 0.34 * price_score + 0.18 * header_score + 0.18 * qr_score)))
    return {
        "score": score,
        "global_score": global_score,
        "price_score": price_score,
        "header_score": header_score,
        "qr_score": qr_score,
        "global_laplacian": float(global_stats["laplacian"]),
        "global_contrast": float(global_stats["contrast"]),
        "global_entropy": float(global_stats["entropy"]),
        "overexposed_ratio": float(global_stats["overexposed_ratio"]),
        "underexposed_ratio": float(global_stats["underexposed_ratio"]),
    }
