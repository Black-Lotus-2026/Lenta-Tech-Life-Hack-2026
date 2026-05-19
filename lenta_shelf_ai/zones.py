from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

from .utils import clip_xyxy, nms_xyxy


MACHINE_ZONE_LABELS = {"qr_code_barcode", "barcode"}
OCR_ZONE_LABELS = {
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "special_symbols",
}
PRICE_ZONE_LABELS = {"price_default", "price_card", "price_discount"}
FIELD_CLASS_ALIASES = {
    "product": "product_name",
    "name": "product_name",
    "prod_name": "product_name",
    "qr": "qr_code_barcode",
    "qrcode": "qr_code_barcode",
    "qr_code": "qr_code_barcode",
    "qr-code": "qr_code_barcode",
    "barcode_1d": "barcode",
    "bar_code": "barcode",
    "default_price": "price_default",
    "price": "price_default",
    "regular_price": "price_default",
    "card_price": "price_card",
    "discount": "discount_amount",
    "discount_price": "price_discount",
    "price_action": "price_discount",
    "sku": "id_sku",
    "datetime": "print_datetime",
    "date": "print_datetime",
    "zone_code": "code",
}


@dataclass(frozen=True)
class FieldZone:
    label: str
    score: float
    xyxy: tuple[float, float, float, float]
    source: str = "unknown"

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def aspect(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(1.0, x2 - x1) / max(1.0, y2 - y1)

    def clamp(self, width: int, height: int) -> "FieldZone":
        x1, y1, x2, y2 = clip_xyxy(self.xyxy, width, height)
        return FieldZone(self.label, self.score, (x1, y1, x2, y2), self.source)

    def expanded(self, width: int, height: int, pad_ratio: float = 0.08, min_pad: int = 3) -> "FieldZone":
        x1, y1, x2, y2 = self.xyxy
        bw = x2 - x1
        bh = y2 - y1
        pad = max(float(min_pad), float(max(bw, bh)) * float(pad_ratio))
        return FieldZone(
            self.label,
            self.score,
            clip_xyxy((x1 - pad, y1 - pad, x2 + pad, y2 + pad), width, height),
            self.source,
        )


def canonical_zone_label(label: object) -> str:
    text = str(label or "").strip().lower().replace(" ", "_")
    return FIELD_CLASS_ALIASES.get(text, text)


def crop_zone(image_bgr: np.ndarray, zone: FieldZone, pad_ratio: float = 0.08, min_pad: int = 3) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    h, w = image_bgr.shape[:2]
    z = zone.expanded(w, h, pad_ratio=pad_ratio, min_pad=min_pad)
    x1, y1, x2, y2 = [int(round(v)) for v in z.xyxy]
    x1, y1, x2, y2 = _clip_int_box(x1, y1, x2, y2, w, h)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return np.ascontiguousarray(image_bgr[y1:y2, x1:x2])


def _clip_int_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _dedupe_zones(zones: Iterable[FieldZone], iou_threshold: float = 0.78) -> List[FieldZone]:
    boxes_by_label: dict[str, list[list[float]]] = {}
    zones_by_label: dict[str, list[FieldZone]] = {}
    for zone in zones:
        zones_by_label.setdefault(zone.label, []).append(zone)
        x1, y1, x2, y2 = zone.xyxy
        boxes_by_label.setdefault(zone.label, []).append([x1, y1, x2, y2, float(zone.score)])
    out: List[FieldZone] = []
    for label, boxes in boxes_by_label.items():
        kept = nms_xyxy(boxes, iou_threshold=iou_threshold)
        src_zones = zones_by_label[label]
        for box in kept:
            best = max(
                src_zones,
                key=lambda z: z.score if abs(z.xyxy[0] - box[0]) + abs(z.xyxy[1] - box[1]) + abs(z.xyxy[2] - box[2]) + abs(z.xyxy[3] - box[3]) < 6 else -1,
            )
            out.append(FieldZone(label, float(box[4]), (float(box[0]), float(box[1]), float(box[2]), float(box[3])), best.source))
    out.sort(key=lambda z: (0 if z.label in MACHINE_ZONE_LABELS else 1, -float(z.score), z.xyxy[1], z.xyxy[0]))
    return out


class YOLOFieldZoneDetector:
    def __init__(self, weights: str | Path, conf: float = 0.12, iou: float = 0.45, imgsz: int = 640, device: str = ""):
        self.weights = str(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install ultralytics to enable field-zone YOLO") from exc
        self.model = YOLO(self.weights)
        raw_names = getattr(self.model, "names", {}) or {}
        self.names = self._normalize_names(raw_names)

    @staticmethod
    def _normalize_names(raw_names: object) -> dict[int, str]:
        if isinstance(raw_names, dict):
            return {int(k): canonical_zone_label(v) for k, v in raw_names.items()}
        if isinstance(raw_names, (list, tuple)):
            return {i: canonical_zone_label(v) for i, v in enumerate(raw_names)}
        return {}

    def predict(self, image_bgr: np.ndarray) -> List[FieldZone]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        h, w = image_bgr.shape[:2]
        results = self.model.predict(
            source=image_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device or None,
            verbose=False,
        )
        zones: List[FieldZone] = []
        for res in results:
            result_names = self._normalize_names(getattr(res, "names", {}) or self.names)
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
            for box, score, cls_id in zip(xyxy, confs, clss):
                label = result_names.get(int(cls_id), str(int(cls_id)))
                label = canonical_zone_label(label)
                if label not in MACHINE_ZONE_LABELS and label not in OCR_ZONE_LABELS:
                    continue
                x1, y1, x2, y2 = clip_xyxy(box, w, h)
                if x2 - x1 >= 6 and y2 - y1 >= 6:
                    zones.append(FieldZone(label, float(score), (x1, y1, x2, y2), source="zone_yolo"))
        return _dedupe_zones(zones)


class HeuristicFieldZoneDetector:
    """Deterministic zone detector for Lenta tags when zone YOLO is unavailable.

    It is deliberately conservative: it does not try to replace a trained model,
    but it exposes high-value machine-readable zones and broad OCR zones so the
    rest of the pipeline can route QR/barcode/OCR locally instead of hammering the
    whole crop.
    """

    def __init__(self, enable_priors: bool = True):
        self.enable_priors = bool(enable_priors)

    @staticmethod
    def _mask_price_panels(image_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        ranges = [
            ((0, 35, 60), (18, 255, 255)),
            ((160, 35, 60), (179, 255, 255)),
            ((12, 45, 85), (42, 255, 255)),
            ((42, 35, 60), (88, 255, 255)),
        ]
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
        return mask

    @staticmethod
    def _texture_boxes(image_bgr: np.ndarray) -> List[FieldZone]:
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(gray)
        dark = cv2.inRange(clahe, 0, 135)
        edges = cv2.Canny(clahe, 45, 160)
        texture = cv2.bitwise_or(dark, edges)
        texture = cv2.morphologyEx(texture, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        contours, _ = cv2.findContours(texture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        zones: List[FieldZone] = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 12 or bh < 10:
                continue
            area_ratio = (bw * bh) / max(1.0, float(w * h))
            if area_ratio < 0.002 or area_ratio > 0.45:
                continue
            aspect = bw / max(1, bh)
            roi_dark = dark[y : y + bh, x : x + bw]
            roi_edges = edges[y : y + bh, x : x + bw]
            dark_density = float((roi_dark > 0).mean()) if roi_dark.size else 0.0
            edge_density = float((roi_edges > 0).mean()) if roi_edges.size else 0.0
            lower_or_right = y > 0.36 * h or x > 0.42 * w
            if not lower_or_right:
                continue
            qr_like = 0.55 <= aspect <= 1.85 and dark_density >= 0.12 and edge_density >= 0.035
            barcode_like = aspect >= 1.8 and dark_density >= 0.06 and edge_density >= 0.025 and y > 0.42 * h
            if not (qr_like or barcode_like):
                continue
            pad = max(4, int(0.24 * max(bw, bh)))
            x1, y1, x2, y2 = _clip_int_box(x - pad, y - pad, x + bw + pad, y + bh + pad, w, h)
            label = "barcode" if barcode_like and aspect >= 1.8 else "qr_code_barcode"
            score = 0.35 + min(0.30, edge_density * 2.5) + min(0.25, dark_density * 0.8) + min(0.10, area_ratio * 4.0)
            zones.append(FieldZone(label, float(score), (x1, y1, x2, y2), source="zone_heuristic_texture"))
        return zones

    @staticmethod
    def _price_zones(image_bgr: np.ndarray) -> List[FieldZone]:
        h, w = image_bgr.shape[:2]
        mask = HeuristicFieldZoneDetector._mask_price_panels(image_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        zones: List[FieldZone] = []
        candidates: List[tuple[float, tuple[int, int, int, int]]] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < max(80.0, 0.004 * w * h):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 18 or bh < 14:
                continue
            aspect = bw / max(1, bh)
            if aspect < 0.35 or aspect > 3.8:
                continue
            fill = area / max(1.0, bw * bh)
            if fill < 0.12:
                continue
            pad_x = int(max(4, 0.18 * bw))
            pad_y = int(max(4, 0.18 * bh))
            box = _clip_int_box(x - pad_x, y - pad_y, x + bw + pad_x, y + bh + pad_y, w, h)
            score = area * (0.5 + fill)
            candidates.append((score, box))
        for rank, (_, box) in enumerate(sorted(candidates, reverse=True)[:3]):
            label = "price_default" if rank == 0 else "price_discount"
            zones.append(FieldZone(label, 0.42 - rank * 0.03, tuple(float(v) for v in box), source="zone_heuristic_color"))
        return zones

    @staticmethod
    def _prior_zones(image_bgr: np.ndarray) -> List[FieldZone]:
        h, w = image_bgr.shape[:2]
        if h < 24 or w < 24:
            return []
        priors = [
            FieldZone("product_name", 0.20, (0.02 * w, 0.02 * h, 0.92 * w, 0.42 * h), "zone_prior"),
            FieldZone("price_default", 0.18, (0.38 * w, 0.20 * h, 0.98 * w, 0.72 * h), "zone_prior"),
            FieldZone("barcode", 0.18, (0.04 * w, 0.70 * h, 0.96 * w, 0.98 * h), "zone_prior"),
            FieldZone("qr_code_barcode", 0.18, (0.52 * w, 0.06 * h, 0.98 * w, 0.58 * h), "zone_prior"),
            FieldZone("id_sku", 0.16, (0.02 * w, 0.58 * h, 0.98 * w, 0.98 * h), "zone_prior"),
        ]
        return [z.clamp(w, h) for z in priors]

    def predict(self, image_bgr: np.ndarray) -> List[FieldZone]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        h, w = image_bgr.shape[:2]
        zones: List[FieldZone] = []
        zones.extend(self._texture_boxes(image_bgr))
        zones.extend(self._price_zones(image_bgr))
        if self.enable_priors:
            zones.extend(self._prior_zones(image_bgr))
        zones = [z.clamp(w, h) for z in zones if z.area >= 25]
        return _dedupe_zones(zones)


class CompositeFieldZoneDetector:
    def __init__(self, yolo: Optional[YOLOFieldZoneDetector], fallback: Optional[HeuristicFieldZoneDetector] = None):
        self.yolo = yolo
        self.fallback = fallback
        self.names = getattr(yolo, "names", {}) if yolo is not None else {}

    def predict(self, image_bgr: np.ndarray) -> List[FieldZone]:
        yolo_zones: List[FieldZone] = []
        if self.yolo is not None:
            try:
                yolo_zones = self.yolo.predict(image_bgr)
            except Exception as exc:  # pragma: no cover - optional dep/runtime
                print(f"[WARN] field-zone YOLO failed: {exc}")
                yolo_zones = []
        fallback_zones = self.fallback.predict(image_bgr) if self.fallback is not None else []
        if not yolo_zones:
            return fallback_zones
        # Keep heuristic machine-code zones even when YOLO predicted the same
        # label. A wrong/tight YOLO barcode crop is worse than an additional
        # low-cost heuristic strip because the decoder can try both and the row
        # fusion later rejects non-checksummed noise. Limit per label to avoid
        # exploding native decode budget.
        supplemental: List[FieldZone] = []
        per_label: dict[str, int] = {label: 0 for label in MACHINE_ZONE_LABELS}
        for z in sorted(fallback_zones, key=lambda item: (-float(item.score), -float(item.area))):
            if z.label not in MACHINE_ZONE_LABELS:
                continue
            if per_label.get(z.label, 0) >= 2:
                continue
            supplemental.append(z)
            per_label[z.label] = per_label.get(z.label, 0) + 1
        return _dedupe_zones(yolo_zones + supplemental)


def build_field_zone_detector(
    weights: str | Path | None,
    enabled: bool = True,
    conf: float = 0.12,
    imgsz: int = 640,
    use_heuristic_fallback: bool = True,
    device: str = "",
) -> Optional[CompositeFieldZoneDetector]:
    if not enabled:
        return None
    yolo: Optional[YOLOFieldZoneDetector] = None
    if weights:
        path = Path(weights)
        if path.exists():
            try:
                yolo = YOLOFieldZoneDetector(path, conf=conf, imgsz=imgsz, device=device)
                print(f"[INFO] field-zone YOLO loaded: {path} names={yolo.names}")
            except Exception as exc:
                print(f"[WARN] field-zone YOLO disabled: {exc}")
    fallback = HeuristicFieldZoneDetector() if use_heuristic_fallback else None
    if yolo is None and fallback is None:
        return None
    return CompositeFieldZoneDetector(yolo, fallback)


def zones_to_debug(zones: Sequence[FieldZone]) -> list[dict[str, object]]:
    return [
        {
            "label": z.label,
            "score": round(float(z.score), 4),
            "source": z.source,
            "xyxy": [round(float(v), 1) for v in z.xyxy],
        }
        for z in zones
    ]
