from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

from .schema import Detection
from .utils import nms_xyxy, clip_xyxy


def _split_weight_paths(value: object) -> List[str]:
    """Parse one-or-many YOLO weight paths from config/env.

    Accepts a string with comma/semicolon/os.pathsep/newline separators or a
    Python sequence. Keeps order and removes duplicates.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = [str(item) for item in value if str(item).strip()]
    else:
        text = str(value).strip()
        if not text:
            return []
        # Split on common list separators while preserving Windows drive
        # prefixes such as C:\models\a.pt. Kaggle uses POSIX paths, but local
        # handoff/testing often passes Windows paths through the same parser.
        parts = [text]
        for sep in ("\n", ",", ";"):
            next_parts: List[str] = []
            for part in parts:
                next_parts.extend(part.split(sep))
            parts = next_parts
        colon_parts: List[str] = []
        for part in parts:
            start = 0
            token_start = 0
            for idx, ch in enumerate(part):
                if ch != ":":
                    continue
                is_windows_drive = (
                    idx == token_start + 1
                    and part[token_start : token_start + 1].isalpha()
                    and idx + 1 < len(part)
                    and part[idx + 1] in "\\/"
                )
                if is_windows_drive:
                    continue
                colon_parts.append(part[start:idx])
                start = idx + 1
                token_start = start
            colon_parts.append(part[start:])
        parts = colon_parts
        raw_items = parts
    out: List[str] = []
    seen: set[str] = set()
    for item in raw_items:
        value = str(item).strip().strip('"').strip("'")
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def _auto_ensemble_yolo_weights() -> List[str]:
    """Return optional ensemble detector weights in a conservative order."""
    root = _repo_root_from_here()
    candidates = [
        root / "models/ensemble/shelf_detector_a.pt",
        root / "models/ensemble/shelf_detector_b.pt",
        root / "models/ensemble/shelf_detector_c.pt",
        root / "models/ensemble/shelf_detector_d.pt",
        root / "models/ensemble/shelf_detector_e.pt",
        root / "models/ensemble/shelf_detector_f.pt",
    ]
    return [str(path) for path in candidates if path.exists()]


def _auto_yolo_world_weights() -> List[str]:
    root = _repo_root_from_here()
    candidates = [
        root / "models/ensemble/open_vocab_detector.pt",
        root / "yolov8s-worldv2.pt",
    ]
    return [str(path) for path in candidates if path.exists()]


class BaseDetector:
    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        raise NotImplementedError

class YOLODetector(BaseDetector):
    """Ultralytics YOLO wrapper. Loaded lazily; falls back is handled by HybridDetector."""

    def __init__(self, weights: str | Path, conf: float = 0.25, iou: float = 0.5, imgsz: int = 1280, device: str = "", source_label: str = "yolo"):
        self.weights = str(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.source_label = source_label or "yolo"
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - depends on optional dep
            raise ImportError("Install ultralytics or use the heuristic detector") from exc
        self.model = YOLO(self.weights)

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device or None,
            verbose=False,
        )
        detections: List[Detection] = []
        h, w = frame_bgr.shape[:2]
        for res in results:
            if getattr(res, "boxes", None) is None:
                continue
            boxes = res.boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            for box, score in zip(xyxy, confs):
                x1, y1, x2, y2 = clip_xyxy(box, w, h)
                if x2 - x1 > 8 and y2 - y1 > 8:
                    detections.append(Detection(x1, y1, x2, y2, float(score), source=self.source_label))
        return detections


class YOLOWorldPromptDetector(BaseDetector):
    """Optional open-vocabulary YOLO-World detector for pseudo-labeling / ablation.

    This is intentionally disabled by default. Open-vocabulary detectors can
    emit broad product boxes; keep them as extra proposal generators gated by
    size/aspect priors and downstream semantic evidence.
    """

    def __init__(
        self,
        weights: str | Path,
        prompts: Optional[Sequence[str]] = None,
        conf: float = 0.012,
        imgsz: int = 1280,
        device: str = "",
        source_label: str = "yolo_world",
        min_box_frac: float = 0.0003,
        max_box_frac: float = 0.060,
        max_aspect: float = 8.0,
        max_boxes: int = 60,
    ):
        self.weights = str(weights)
        self.prompts = [str(p).strip() for p in (prompts or ["small rectangular price sticker on shelf"]) if str(p).strip()]
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.source_label = source_label or "yolo_world"
        self.min_box_frac = float(min_box_frac)
        self.max_box_frac = float(max_box_frac)
        self.max_aspect = float(max_aspect)
        self.max_boxes = int(max_boxes)
        try:
            from ultralytics import YOLOWorld
        except Exception as exc:  # pragma: no cover - depends on optional dep
            raise ImportError("Install ultralytics with YOLOWorld support") from exc
        self.model = YOLOWorld(self.weights)
        if self.prompts:
            self.model.set_classes(self.prompts)

    def _passes_gate(self, xyxy: Sequence[float], frame_area: float) -> bool:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if bw <= 8 or bh <= 8:
            return False
        frac = (bw * bh) / max(1.0, frame_area)
        if frac < self.min_box_frac or frac > self.max_box_frac:
            return False
        aspect = max(bw / max(1.0, bh), bh / max(1.0, bw))
        return aspect <= self.max_aspect

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device or None,
            verbose=False,
        )
        detections: List[Detection] = []
        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w)
        for res in results:
            if getattr(res, "boxes", None) is None:
                continue
            boxes = res.boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            order = np.argsort(confs)[::-1][: self.max_boxes]
            for i in order:
                box = xyxy[i]
                if not self._passes_gate(box, frame_area):
                    continue
                x1, y1, x2, y2 = clip_xyxy(box, w, h)
                if x2 - x1 > 8 and y2 - y1 > 8:
                    detections.append(Detection(x1, y1, x2, y2, float(confs[i]), source=self.source_label))
        return detections

class ColorGeometryDetector(BaseDetector):
    """Fast offline detector for Lenta price tags.

    It is not intended to outperform the trained model. It provides deterministic
    operation before fine-tuning and helps generate pseudo-labels around QR/color cues.
    """

    def __init__(
        self,
        max_width: int = 1920,
        min_area: int = 90,
        max_area_ratio: float = 0.015,
        nms_iou: float = 0.30,
        enable_white_tags: bool = True,
    ):
        self.max_width = max_width
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.nms_iou = nms_iou
        self.enable_white_tags = enable_white_tags

    def _resize(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame_bgr.shape[:2]
        scale = 1.0
        if w > self.max_width:
            scale = self.max_width / float(w)
            frame_bgr = cv2.resize(frame_bgr, (self.max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        return frame_bgr, scale

    def _color_mask(self, im: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        ranges = [
            ((0, 35, 55), (15, 255, 255)),     # red/orange
            ((160, 30, 55), (179, 255, 255)),  # red/pink
            ((12, 45, 85), (38, 255, 255)),    # orange/yellow promo tags
            ((40, 35, 55), (90, 255, 255)),    # green shelf labels
            ((95, 35, 55), (135, 255, 255)),   # blue parts
        ]
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _white_mask(self, im: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        # White tags are low saturation and high value; edge-density filter later removes bottles/labels.
        mask = cv2.inRange(hsv, np.array((0, 0, 160), np.uint8), np.array((179, 75, 255), np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def _boxes_from_mask(self, im: np.ndarray, mask: np.ndarray, source: str) -> List[List[float]]:
        ih, iw = im.shape[:2]
        max_area = iw * ih * self.max_area_ratio
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        out: List[List[float]] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 8 or bh < 8:
                continue
            ar = bw / max(1, bh)
            if ar < 0.18 or ar > 6.0:
                continue
            fill = area / max(1.0, bw * bh)
            if fill < 0.10:
                continue
            # Tiny price color area is only part of the label; expand to include QR/name/barcode.
            pad_x = int(max(14, min(110, 0.70 * bw)))
            pad_y = int(max(12, min(115, 0.85 * bh)))
            x1, y1, x2, y2 = clip_xyxy([x - pad_x, y - pad_y, x + bw + pad_x, y + bh + pad_y], iw, ih)
            roi_edges = edges[int(y1) : int(y2), int(x1) : int(x2)]
            edge_density = float((roi_edges > 0).mean()) if roi_edges.size else 0.0
            # Product packs have huge saturated areas with high edges; tags are compact and rectangular.
            if source == "white" and edge_density < 0.02:
                continue
            ew, eh = x2 - x1, y2 - y1
            if ew * eh > iw * ih * 0.020 or ew > iw * 0.34 or eh > ih * 0.45:
                continue
            score = 0.35 + min(0.45, fill * 0.45) + min(0.20, edge_density)
            out.append([x1, y1, x2, y2, score])
        return out

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        im, scale = self._resize(frame_bgr)
        boxes = self._boxes_from_mask(im, self._color_mask(im), "color")
        if self.enable_white_tags:
            # White detector is intentionally conservative: only lower half near shelf rail
            white_boxes = self._boxes_from_mask(im, self._white_mask(im), "white")
            boxes.extend([b for b in white_boxes if (b[2]-b[0]) < 500 and (b[3]-b[1]) < 350])
        boxes = nms_xyxy(boxes, self.nms_iou)
        detections: List[Detection] = []
        h, w = frame_bgr.shape[:2]
        for b in boxes:
            x1, y1, x2, y2, score = b
            x1, y1, x2, y2 = [v / scale for v in (x1, y1, x2, y2)]
            x1, y1, x2, y2 = clip_xyxy([x1, y1, x2, y2], w, h)
            if x2 - x1 >= 20 and y2 - y1 >= 20:
                detections.append(Detection(x1, y1, x2, y2, float(score), source="heuristic"))
        return detections

class RedWhiteTagDetector(BaseDetector):
    """Detect common Lenta tags from a red price panel plus adjacent white text/QR area."""

    def __init__(
        self,
        max_width: int = 1600,
        min_red_area: int = 80,
        max_candidate_area_ratio: float = 0.035,
        nms_iou: float = 0.25,
    ):
        self.max_width = max_width
        self.min_red_area = min_red_area
        self.max_candidate_area_ratio = max_candidate_area_ratio
        self.nms_iou = nms_iou

    def _resize(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame_bgr.shape[:2]
        scale = 1.0
        if w > self.max_width:
            scale = self.max_width / float(w)
            frame_bgr = cv2.resize(frame_bgr, (self.max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        return frame_bgr, scale

    @staticmethod
    def _red_mask(im: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        ranges = [
            ((0, 45, 70), (14, 255, 255)),
            ((165, 35, 70), (179, 255, 255)),
            ((4, 35, 85), (28, 235, 255)),  # faded orange/red price panels
        ]
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    @staticmethod
    def _white_mask(im: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array((0, 0, 135), np.uint8), np.array((179, 90, 255), np.uint8))

    @staticmethod
    def _candidate_boxes(x: int, y: int, bw: int, bh: int) -> List[List[float]]:
        boxes: List[List[float]] = []
        # Price panel is most often on the left or right side of the tag.
        for width_mult in (2.1, 2.5, 2.85, 3.2):
            boxes.append([x - 0.10 * bw, y - 0.06 * bh, x + width_mult * bw, y + 1.06 * bh])
            boxes.append([x - (width_mult - 1.0) * bw, y - 0.06 * bh, x + 1.10 * bw, y + 1.06 * bh])
        # Some shelf labels are rotated relative to the image and expose a horizontal red strip.
        for height_mult in (2.0, 2.4, 2.8):
            boxes.append([x - 0.06 * bw, y - 0.10 * bh, x + 1.06 * bw, y + height_mult * bh])
            boxes.append([x - 0.06 * bw, y - (height_mult - 1.0) * bh, x + 1.06 * bw, y + 1.10 * bh])
        return boxes

    def _score_candidate(
        self,
        box: Sequence[float],
        im: np.ndarray,
        red_mask: np.ndarray,
        white_mask: np.ndarray,
        edges: np.ndarray,
    ) -> Optional[List[float]]:
        ih, iw = im.shape[:2]
        x1, y1, x2, y2 = clip_xyxy(box, iw, ih)
        cw, ch = x2 - x1, y2 - y1
        if cw < 24 or ch < 24:
            return None
        ar = cw / max(1.0, ch)
        area = cw * ch
        area_ratio = area / max(1.0, iw * ih)
        if ar < 0.24 or ar > 1.85:
            return None
        if cw > max(iw * 0.18, 260.0) or ch > max(ih * 0.32, 260.0):
            return None
        if area_ratio < 0.0008:
            return None
        max_candidate_area = max(iw * ih * self.max_candidate_area_ratio, 60000.0)
        if area > max_candidate_area:
            return None
        roi = (slice(int(y1), int(y2)), slice(int(x1), int(x2)))
        red_ratio = float((red_mask[roi] > 0).mean())
        white_ratio = float((white_mask[roi] > 0).mean())
        edge_density = float((edges[roi] > 0).mean())
        if red_ratio < 0.045 or white_ratio < 0.16 or edge_density < 0.012:
            return None
        shape_score = max(0.0, 1.0 - abs(ar - 0.82) / 0.82)
        area_score = max(0.0, 1.0 - abs(area_ratio - 0.0075) / 0.016)
        score = (
            0.45
            + 0.18 * min(1.0, red_ratio / 0.30)
            + 0.18 * min(1.0, white_ratio / 0.45)
            + 0.14 * min(1.0, edge_density / 0.08)
            + 0.12 * shape_score
            + 0.08 * area_score
        )
        return [x1, y1, x2, y2, float(score)]

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        im, scale = self._resize(frame_bgr)
        ih, iw = im.shape[:2]
        red_mask = self._red_mask(im)
        white_mask = self._white_mask(im)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[List[float]] = []
        max_red_area = max(iw * ih * 0.020, 18000.0)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_red_area or area > max_red_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 8 or bh < 8:
                continue
            fill = area / max(1.0, bw * bh)
            if fill < 0.18:
                continue
            for candidate in self._candidate_boxes(x, y, bw, bh):
                scored = self._score_candidate(candidate, im, red_mask, white_mask, edges)
                if scored is not None:
                    boxes.append(scored)

        boxes = nms_xyxy(boxes, self.nms_iou)
        detections: List[Detection] = []
        h, w = frame_bgr.shape[:2]
        for x1, y1, x2, y2, score in boxes:
            x1, y1, x2, y2 = [v / scale for v in (x1, y1, x2, y2)]
            x1, y1, x2, y2 = clip_xyxy([x1, y1, x2, y2], w, h)
            detections.append(Detection(x1, y1, x2, y2, float(score), source="red_white_tag"))
        return detections

class QRSeedDetector(BaseDetector):
    """Finds QR codes and expands them into approximate tag boxes."""

    def __init__(self, max_width: int = 1920, expand_x: float = 2.6, expand_y: float = 2.2):
        self.max_width = max_width
        self.expand_x = expand_x
        self.expand_y = expand_y
        self.detector = cv2.QRCodeDetector()

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        scale = 1.0
        im = frame_bgr
        if w > self.max_width:
            scale = self.max_width / w
            im = cv2.resize(frame_bgr, (self.max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        detections: List[Detection] = []
        try:
            ok, decoded, points, _ = self.detector.detectAndDecodeMulti(im)
        except Exception:
            ok, points = False, None
        if ok and points is not None:
            for pts in points:
                pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
                x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
                cx, cy = x + bw / 2, y + bh / 2
                tw, th = bw * self.expand_x, bh * self.expand_y
                # QR is usually on left or right side of label; expand to both sides.
                x1, y1, x2, y2 = cx - tw, cy - th * 0.55, cx + tw, cy + th * 0.55
                x1, y1, x2, y2 = [v / scale for v in (x1, y1, x2, y2)]
                x1, y1, x2, y2 = clip_xyxy([x1, y1, x2, y2], w, h)
                detections.append(Detection(x1, y1, x2, y2, 0.60, source="qr_seed"))
        return detections

class HybridDetector(BaseDetector):
    def __init__(
        self,
        yolo_weights: Optional[str | Sequence[str]] = None,
        yolo_conf: float = 0.25,
        imgsz: int = 1280,
        enable_fallbacks: bool = True,
    ):
        self.detectors: List[BaseDetector] = []
        loaded_yolo = False

        weight_paths = _split_weight_paths(yolo_weights)
        weight_paths.extend(_split_weight_paths(os.environ.get("LENTA_EXTRA_YOLO_WEIGHTS", "")))
        if os.environ.get("LENTA_ENABLE_YOLO_ENSEMBLE", "0") != "0":
            weight_paths.extend(_auto_ensemble_yolo_weights())

        resolved: List[Path] = []
        seen: set[str] = set()
        for raw in weight_paths:
            path = Path(raw)
            if not path.exists():
                alt = Path.cwd() / raw
                if alt.exists():
                    path = alt
            if not path.exists():
                print(f"[WARN] YOLO weights missing: {raw}")
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            resolved.append(path)

        max_models = int(os.environ.get("LENTA_MAX_YOLO_MODELS", "0") or "0")
        if max_models > 0:
            resolved = resolved[:max_models]

        for idx, path in enumerate(resolved):
            label = "yolo" if len(resolved) == 1 and idx == 0 else f"yolo:{path.stem}"
            try:
                self.detectors.append(YOLODetector(path, conf=yolo_conf, imgsz=imgsz, source_label=label))
                loaded_yolo = True
            except Exception as exc:
                print(f"[WARN] YOLO disabled for {path}: {exc}")

        if os.environ.get("LENTA_ENABLE_YOLO_WORLD_FALLBACK", "0") != "0":
            world_weights = _split_weight_paths(os.environ.get("LENTA_YOLO_WORLD_WEIGHTS", ""))
            if not world_weights:
                world_weights = _auto_yolo_world_weights()
            prompts = _split_weight_paths(os.environ.get("LENTA_YOLO_WORLD_PROMPTS", "small rectangular price sticker on shelf"))
            world_conf = float(os.environ.get("LENTA_YOLO_WORLD_CONF", "0.012") or "0.012")
            world_min_frac = float(os.environ.get("LENTA_YOLO_WORLD_MIN_BOX_FRAC", "0.0003") or "0.0003")
            world_max_frac = float(os.environ.get("LENTA_YOLO_WORLD_MAX_BOX_FRAC", "0.060") or "0.060")
            world_max_aspect = float(os.environ.get("LENTA_YOLO_WORLD_MAX_ASPECT", "8.0") or "8.0")
            for raw in world_weights[:1]:
                path = Path(raw)
                if not path.exists():
                    alt = Path.cwd() / raw
                    if alt.exists():
                        path = alt
                if not path.exists():
                    print(f"[WARN] YOLO-World weights missing: {raw}")
                    continue
                try:
                    self.detectors.append(
                        YOLOWorldPromptDetector(
                            path,
                            prompts=prompts,
                            conf=world_conf,
                            imgsz=imgsz,
                            source_label=f"yolo_world:{path.stem}",
                            min_box_frac=world_min_frac,
                            max_box_frac=world_max_frac,
                            max_aspect=world_max_aspect,
                        )
                    )
                except Exception as exc:
                    print(f"[WARN] YOLO-World disabled for {path}: {exc}")

        if enable_fallbacks or not loaded_yolo:
            self.detectors.append(QRSeedDetector(max_width=imgsz))
            self.detectors.append(RedWhiteTagDetector(max_width=imgsz))
            self.detectors.append(ColorGeometryDetector(max_width=imgsz))

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        boxes: List[List[float]] = []
        dets_by_box: List[Detection] = []
        for detector in self.detectors:
            try:
                dets = detector.predict(frame_bgr)
            except Exception as exc:
                print(f"[WARN] detector {detector.__class__.__name__} failed: {exc}")
                continue
            for d in dets:
                dets_by_box.append(d)
                boxes.append([d.x_min, d.y_min, d.x_max, d.y_max, d.score])
        keep_boxes = nms_xyxy(boxes, iou_threshold=0.38)
        out: List[Detection] = []
        for b in keep_boxes:
            # preserve source of closest original
            best = max(dets_by_box, key=lambda d: d.score if abs(d.x_min-b[0])+abs(d.y_min-b[1])+abs(d.x_max-b[2])+abs(d.y_max-b[3]) < 4 else -1)
            out.append(Detection(b[0], b[1], b[2], b[3], b[4], source=best.source))
        return out
