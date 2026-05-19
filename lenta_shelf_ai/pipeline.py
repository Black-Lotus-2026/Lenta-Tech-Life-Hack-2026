from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import pandas as pd

from .catalog import ProductCatalog, should_replace_product_name
from .focus import compute_focus_quality
from .detectors import HybridDetector
from .ocr import EnsembleOCREngine
from .parsers import ean13_is_valid, merge_field_values, parse_observation
from .qr import decode_qr_payloads_with_debug, parse_qr_payloads
from .zones import MACHINE_ZONE_LABELS, OCR_ZONE_LABELS, PRICE_ZONE_LABELS, FieldZone, build_field_zone_detector, crop_zone, zones_to_debug
from .schema import ABSENT_VALUE, OUTPUT_COLUMNS, QR_FIELD_ALIASES, Detection, TagObservation, ensure_columns
from .tracker import SimpleTracker, Track
from .utils import crop_xyxy, mkdir, sharpness_laplacian, read_yaml_or_json, normalize_text, price_to_str, iou_xyxy, perceptual_hash, hamming_distance_hex, text_similarity
from .video import iter_video_frames, get_video_meta, read_frame_at_ms

@dataclass
class PipelineConfig:
    sample_fps: float = 4.0
    yolo_weights: str = "models/price_tag_yolo.pt"
    yolo_conf: float = 0.08
    detector_imgsz: int = 1600
    enable_fallback_detectors: bool = False
    min_sharpness: float = 18.0
    max_frames: int = 0
    max_detections_per_frame: int = 80
    detection_edge_margin_ratio: float = 0.0
    enable_focus_quality: bool = True
    enable_ocr: bool = True
    enable_qr: bool = True
    defer_ocr: bool = False
    prefer_paddle: bool = False
    ocr_lang: str = "ru"
    use_gpu: bool = False
    crop_pad_px: int = 8
    tracker_iou: float = 0.12
    tracker_center_threshold: float = 250.0
    max_lost: int = 5
    min_track_observations: int = 1
    dedupe_iou: float = 0.30
    dedupe_center_threshold: float = 90.0
    dedupe_time_window_ms: int = 1600
    representative_temporal_weight: float = 0.0
    save_crops: bool = False
    save_debug_json: bool = True
    enable_zonal_ocr: bool = True
    qr_expansion_x: float = 0.55
    qr_expansion_y: float = 0.45
    dedupe_visual_hash_threshold: int = 14
    dedupe_text_similarity: float = 0.86
    dedupe_extended_time_window_ms: int = 12000
    dedupe_row_y_threshold_ratio: float = 0.55
    fallback_min_observations: int = 3
    fallback_require_evidence: bool = True
    deferred_qr_top_k: int = 5
    deferred_ocr_top_k: int = 1
    deferred_recognition_top_k: int = 5
    enable_field_zone_detector: bool = True
    field_zone_weights: str = "models/field_zone_yolo.pt"
    field_zone_conf: float = 0.10
    field_zone_imgsz: int = 640
    field_zone_use_heuristic_fallback: bool = True
    field_zone_padding_ratio: float = 0.10
    field_zone_qr_top_k: int = 4
    field_zone_barcode_top_k: int = 4
    field_zone_ocr_top_k: int = 12
    field_zone_ocr_per_label_top_k: int = 1
    field_zone_full_crop_fallback: bool = True
    field_zone_qr_full_crop_fallback: bool = True
    field_zone_qr_context_fallback: bool = True
    field_zone_ocr_full_crop_fallback: bool = True
    field_zone_save_crops: bool = False
    field_zone_max_failure_crops: int = 120
    field_zone_crop_rotations: tuple[int, ...] = (270, 0, 90, 180)
    temporal_qr_reconstruction: bool = False
    temporal_qr_top_k: int = 6
    temporal_qr_min_crops: int = 2
    temporal_qr_max_side: int = 384
    temporal_qr_max_composites: int = 8
    enable_crop_rectification: bool = False
    crop_rectification_min_area_ratio: float = 0.16
    crop_rectification_max_area_ratio: float = 0.98

    @classmethod
    def from_file(cls, path: Optional[str]) -> "PipelineConfig":
        data = read_yaml_or_json(path)
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

class PriceTagPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        weights = self.config.yolo_weights
        if weights and not Path(weights).exists():
            # Also check relative to CWD and package parent.
            alt = Path.cwd() / weights
            weights = str(alt) if alt.exists() else ""
        self.detector = HybridDetector(
            yolo_weights=weights,
            yolo_conf=self.config.yolo_conf,
            imgsz=int(self.config.detector_imgsz),
            enable_fallbacks=bool(self.config.enable_fallback_detectors),
        )
        zone_weights = str(getattr(self.config, "field_zone_weights", "") or "")
        if zone_weights and not Path(zone_weights).exists():
            alt = Path.cwd() / zone_weights
            zone_weights = str(alt) if alt.exists() else zone_weights
        self.zone_detector = build_field_zone_detector(
            zone_weights,
            enabled=bool(getattr(self.config, "enable_field_zone_detector", True)),
            conf=float(getattr(self.config, "field_zone_conf", 0.10)),
            imgsz=int(getattr(self.config, "field_zone_imgsz", 640)),
            use_heuristic_fallback=bool(getattr(self.config, "field_zone_use_heuristic_fallback", True)),
            device="cuda:0" if bool(self.config.use_gpu) else "",
        )
        self.product_catalog = ProductCatalog.from_env_or_default(Path.cwd())
        self.ocr = None
        self._qr_failure_crops_saved = 0
        if self.config.enable_ocr:
            self.ocr = EnsembleOCREngine(prefer_paddle=bool(self.config.prefer_paddle), lang=str(self.config.ocr_lang), use_gpu=bool(self.config.use_gpu))

    @staticmethod
    def _rotate_crop_right_angle(crop: np.ndarray, degrees: int) -> np.ndarray:
        deg = int(degrees) % 360
        if deg == 90:
            return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        if deg == 180:
            return cv2.rotate(crop, cv2.ROTATE_180)
        if deg == 270:
            return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return crop

    @staticmethod
    def _rotation_candidates(value: object) -> List[int]:
        if value is None:
            return [0]
        if isinstance(value, str):
            raw_items = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            raw_items = [value]
        out: List[int] = []
        for item in raw_items:
            try:
                deg = int(item) % 360
            except Exception:
                continue
            if deg in {0, 90, 180, 270} and deg not in out:
                out.append(deg)
        return out or [0]

    @staticmethod
    def _zone_orientation_score(zones: List[FieldZone]) -> float:
        if not zones:
            return -1.0
        score = 0.0
        for zone in zones:
            source_bonus = 3.0 if zone.source == "zone_yolo" else 0.45
            label_bonus = 2.0 if zone.label in MACHINE_ZONE_LABELS else (1.5 if zone.label in PRICE_ZONE_LABELS else 1.0)
            score += float(zone.score) * label_bonus * source_bonus
        return score + min(2.0, len(zones) * 0.08)

    def _detect_zones_with_best_orientation(self, crop: np.ndarray) -> tuple[np.ndarray, int, List[FieldZone], Dict[str, object]]:
        if self.zone_detector is None or crop is None or crop.size == 0:
            return crop, 0, [], {"enabled": False, "zones": [], "semantic_rotation": 0}

        best_crop = crop
        best_rotation = 0
        best_zones: List[FieldZone] = []
        best_score = -1.0
        attempts: List[Dict[str, object]] = []

        for rotation in self._rotation_candidates(getattr(self.config, "field_zone_crop_rotations", (0,))):
            oriented = self._rotate_crop_right_angle(crop, rotation)
            try:
                zones = self.zone_detector.predict(oriented)
                error = ""
            except Exception as exc:
                zones = []
                error = f"zone-detector:{type(exc).__name__}:{exc}"
            score = self._zone_orientation_score(zones)
            counts = {label: sum(1 for z in zones if z.label == label) for label in sorted({z.label for z in zones})}
            item: Dict[str, object] = {
                "rotation": int(rotation),
                "score": round(float(score), 4),
                "zone_count": len(zones),
                "counts": counts,
            }
            if error:
                item["error"] = error
            attempts.append(item)
            if score > best_score:
                best_score = score
                best_crop = oriented
                best_rotation = int(rotation)
                best_zones = zones

        return best_crop, best_rotation, best_zones, {
            "enabled": True,
            "semantic_rotation": best_rotation,
            "orientation_attempts": attempts,
            "zones": zones_to_debug(best_zones),
            "counts": {label: sum(1 for z in best_zones if z.label == label) for label in sorted({z.label for z in best_zones})},
        }

    @staticmethod
    def _order_quad_points(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
        sums = pts.sum(axis=1)
        diffs = np.diff(pts, axis=1).reshape(-1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[int(np.argmin(sums))]
        ordered[2] = pts[int(np.argmax(sums))]
        ordered[1] = pts[int(np.argmin(diffs))]
        ordered[3] = pts[int(np.argmax(diffs))]
        return ordered

    @staticmethod
    def _rectify_tag_crop_static(crop: np.ndarray, min_area_ratio: float = 0.16, max_area_ratio: float = 0.98) -> tuple[np.ndarray, Dict[str, object]]:
        if crop is None or crop.size == 0:
            return crop, {"applied": False, "reason": "empty"}
        h, w = crop.shape[:2]
        if h < 40 or w < 40:
            return crop, {"applied": False, "reason": "too_small"}
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 140)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crop_area = float(h * w)
        best = None
        best_area = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < crop_area * float(min_area_ratio) or area > crop_area * float(max_area_ratio):
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                rect = cv2.minAreaRect(contour)
                approx = cv2.boxPoints(rect).reshape(-1, 1, 2)
            pts = np.asarray(approx, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] != 4:
                continue
            x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
            if bw < 24 or bh < 24:
                continue
            aspect = bw / max(1.0, float(bh))
            if not 0.18 <= aspect <= 6.5:
                continue
            if area > best_area:
                best_area = area
                best = pts
        if best is None:
            return crop, {"applied": False, "reason": "no_quad"}
        ordered = PriceTagPipeline._order_quad_points(best)
        tl, tr, br, bl = ordered
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        out_w = int(max(width_a, width_b))
        out_h = int(max(height_a, height_b))
        if out_w < 24 or out_h < 24:
            return crop, {"applied": False, "reason": "degenerate"}
        out_w = min(max(out_w, 24), max(24, int(w * 1.15)))
        out_h = min(max(out_h, 24), max(24, int(h * 1.15)))
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(crop, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return warped, {"applied": True, "area_ratio": round(best_area / crop_area, 4), "output_shape": [out_h, out_w]}

    def _maybe_rectify_crop(self, crop: np.ndarray) -> tuple[np.ndarray, Dict[str, object]]:
        if not bool(getattr(self.config, "enable_crop_rectification", False)):
            return crop, {"applied": False, "enabled": False}
        rectified, debug = self._rectify_tag_crop_static(
            crop,
            min_area_ratio=float(getattr(self.config, "crop_rectification_min_area_ratio", 0.16)),
            max_area_ratio=float(getattr(self.config, "crop_rectification_max_area_ratio", 0.98)),
        )
        debug["enabled"] = True
        return rectified, debug

    @staticmethod
    def _payloads_have_machine_fields(payloads: Iterable[str]) -> bool:
        fields = parse_qr_payloads(payloads)
        useful = {
            "qr_code_barcode",
            "price1_qr",
            "price2_qr",
            "price3_qr",
            "price4_qr",
            "action_price_qr",
            "action_code_qr",
        }
        return any(str(fields.get(key, "")).strip() not in {"", ABSENT_VALUE} for key in useful)

    @staticmethod
    def _select_ocr_zones(zones: List[FieldZone], budget: int, per_label: int = 1) -> List[FieldZone]:
        if budget <= 0:
            return []
        priority = [
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
        ]
        buckets: Dict[str, List[FieldZone]] = {}
        for zone in zones:
            buckets.setdefault(zone.label, []).append(zone)
        for label in buckets:
            buckets[label].sort(key=lambda z: (-float(z.score), z.xyxy[1], z.xyxy[0], -float(z.area)))

        selected: List[FieldZone] = []
        used_ids: set[int] = set()
        # Guarantee one crop per semantic label before spending extra budget on
        # more price/name crops. The previous global top-4 selection never OCRed
        # id_sku/print_datetime/code in the Kaggle debug runs.
        for label in priority:
            for zone in buckets.get(label, [])[:1]:
                selected.append(zone)
                used_ids.add(id(zone))
                break
            if len(selected) >= budget:
                return selected[:budget]

        extras: List[FieldZone] = []
        label_counts: Dict[str, int] = {z.label: 1 for z in selected}
        for label in priority:
            for zone in buckets.get(label, [])[1: max(1, per_label)]:
                if id(zone) in used_ids:
                    continue
                if label_counts.get(label, 0) >= per_label:
                    continue
                extras.append(zone)
                label_counts[label] = label_counts.get(label, 0) + 1
        extras.extend(z for z in zones if id(z) not in used_ids and z.label not in priority)
        extras.sort(key=lambda z: (-float(z.score), z.xyxy[1], z.xyxy[0]))
        for zone in extras:
            if id(zone) in used_ids:
                continue
            selected.append(zone)
            used_ids.add(id(zone))
            if len(selected) >= budget:
                break
        return selected[:budget]

    @staticmethod
    def _is_present_value(value: object) -> bool:
        text = normalize_text(str(value or ""))
        return text not in {"", ABSENT_VALUE}

    @staticmethod
    def _clear_on_empty_fusion(col: str) -> bool:
        return col in {
            "barcode",
            "qr_code_barcode",
            "id_sku",
            "product_name",
            "price_default",
            "price_card",
            "price_discount",
            "price1_qr",
            "price2_qr",
            "price3_qr",
            "price4_qr",
            "action_price_qr",
            "wholesale_level_1_price",
            "wholesale_level_2_price",
        }

    @staticmethod
    def _normalize_price_candidate(value: object) -> str:
        text = normalize_text(str(value or ""))
        if text in {"", ABSENT_VALUE}:
            return ""
        try:
            normalized = price_to_str(text)
            amount = float(normalized)
        except Exception:
            return ""
        max_price = float(os.environ.get("LENTA_PRICE_MAX", "49999"))
        if amount < 0.5 or amount > max_price:
            return ""
        return normalized

    @staticmethod
    def _merge_column_values(col: str, values: Iterable[object]) -> str:
        raw = [normalize_text(str(v or "")) for v in values if v is not None]
        vals = [v for v in raw if v]
        if not vals:
            return ""
        present = [v for v in vals if v != ABSENT_VALUE]
        if not present:
            return ABSENT_VALUE

        if col in {"barcode", "qr_code_barcode"}:
            digits = [re.sub(r"\D", "", v) for v in present]
            valid = [d for d in digits if ean13_is_valid(d)]
            if not valid:
                return ""
            counts = Counter(valid)
            return max(counts, key=lambda v: (counts[v], len(v)))

        if col in {
            "price_default",
            "price_card",
            "price_discount",
            "price1_qr",
            "price2_qr",
            "price3_qr",
            "price4_qr",
            "action_price_qr",
            "wholesale_level_1_price",
            "wholesale_level_2_price",
        }:
            prices = [PriceTagPipeline._normalize_price_candidate(v) for v in present]
            prices = [p for p in prices if p]
            if not prices:
                return ABSENT_VALUE if ABSENT_VALUE in vals else ""
            counts = Counter(prices)
            amounts = sorted(float(p) for p in prices)
            median = amounts[len(amounts) // 2]
            return max(prices, key=lambda p: (counts[p], -abs(float(p) - median), -len(p)))

        if col == "id_sku":
            skus = [re.sub(r"\D", "", v) for v in present]
            skus = [s for s in skus if 9 <= len(s) <= 13 and not ean13_is_valid(s)]
            if not skus:
                return ""
            counts = Counter(skus)
            return max(skus, key=lambda s: (counts[s], len(s) >= 10 and s.startswith(("2", "3")), len(s)))

        if col == "product_name":
            good = []
            bad = re.compile(r"(товар\s+закончился|привез[её]м|распродано|нет\s+в\s+наличии)", re.I)
            for v in present:
                if len(v) < 5 or bad.search(v):
                    continue
                if not re.search(r"[А-Яа-яЁё]", v):
                    continue
                good.append(v)
            if good:
                counts = Counter(good)
                return max(good, key=lambda v: (counts[v], sum(ch.isalpha() for ch in v), -sum(ch.isdigit() for ch in v)))
            return ""

        counts = Counter(present)
        return max(present, key=lambda v: (counts[v], len(v)))

    def _recognize_detection(
        self,
        frame: np.ndarray,
        det: Detection,
        output_dir: Optional[Path],
        crop_name: str,
        run_qr: bool = True,
        run_ocr: bool = True,
        known_qr_payloads: Optional[List[str]] = None,
    ) -> tuple[str, list, list[str], Dict[str, str], float, Dict[str, object]]:
        h, w = frame.shape[:2]
        raw_crop = crop_xyxy(frame, det.xyxy, pad=int(self.config.crop_pad_px))
        semantic_input_crop, rectification_debug = self._maybe_rectify_crop(raw_crop)
        crop, semantic_rotation, zones, zone_debug = self._detect_zones_with_best_orientation(semantic_input_crop)
        if bool(getattr(self.config, "enable_crop_rectification", False)):
            zone_debug["rectification"] = rectification_debug
        crop_h, crop_w = crop.shape[:2]

        if output_dir is not None and (self.config.save_crops or bool(getattr(self.config, "field_zone_save_crops", False))):
            crop_dir = mkdir(output_dir / "crops")
            if self.config.save_crops:
                cv2.imwrite(str(crop_dir / f"{crop_name}.jpg"), crop)
                if semantic_rotation or bool(getattr(self.config, "enable_crop_rectification", False)):
                    cv2.imwrite(str(crop_dir / f"{crop_name}_raw.jpg"), raw_crop)
            if bool(getattr(self.config, "field_zone_save_crops", False)) and zones:
                zone_dir = mkdir(output_dir / "zone_crops")
                for zi, zone in enumerate(zones[:24]):
                    zcrop = crop_zone(crop, zone, pad_ratio=float(getattr(self.config, "field_zone_padding_ratio", 0.10)))
                    if zcrop.size:
                        cv2.imwrite(str(zone_dir / f"{crop_name}_{zi:02d}_{zone.label}.jpg"), zcrop)

        qr_payloads: List[str] = list(known_qr_payloads or [])
        qr_debug: Dict[str, object] = {"zones": zone_debug, "zone_attempts": []}
        if self.config.enable_qr and run_qr:
            # Machine-readable fields first. The field-zone detector provides
            # tight qr_code_barcode/barcode crops that should be tried before
            # whole-crop fallbacks.
            code_zones = [z for z in zones if z.label in MACHINE_ZONE_LABELS]
            label_priority = {"qr_code_barcode": 0, "barcode": 1}
            code_zones.sort(key=lambda z: (label_priority.get(z.label, 5), -float(z.score), -float(z.area)))
            per_label_budget = {
                "qr_code_barcode": int(getattr(self.config, "field_zone_qr_top_k", 4)),
                "barcode": int(getattr(self.config, "field_zone_barcode_top_k", 4)),
            }
            used_per_label: Dict[str, int] = {"qr_code_barcode": 0, "barcode": 0}
            for zone in code_zones:
                budget = per_label_budget.get(zone.label, 0)
                if budget <= 0 or used_per_label.get(zone.label, 0) >= budget:
                    continue
                used_per_label[zone.label] = used_per_label.get(zone.label, 0) + 1
                zcrop = crop_zone(crop, zone, pad_ratio=float(getattr(self.config, "field_zone_padding_ratio", 0.10)), min_pad=5)
                if zcrop.size == 0:
                    continue
                try:
                    payloads, debug = decode_qr_payloads_with_debug(zcrop, force_native=True, debug_label=zone.label)
                except Exception as exc:
                    payloads, debug = [], {"errors": [f"zone-decode:{type(exc).__name__}:{exc}"], "debug_label": zone.label}
                debug["zone_label"] = zone.label
                debug["zone_source"] = zone.source
                debug["zone_score"] = round(float(zone.score), 4)
                debug["zone_xyxy"] = [round(float(v), 1) for v in zone.xyxy]
                qr_debug["zone_attempts"].append(debug)
                if (not payloads) and output_dir is not None and bool(getattr(self.config, "field_zone_save_crops", False)):
                    limit = int(getattr(self.config, "field_zone_max_failure_crops", 120) or 0)
                    if limit > 0 and int(getattr(self, "_qr_failure_crops_saved", 0)) < limit:
                        try:
                            fail_dir = mkdir(output_dir / "qr_failure_crops")
                            cv2.imwrite(str(fail_dir / f"{crop_name}_{len(qr_debug['zone_attempts']) - 1:02d}_{zone.label}.jpg"), zcrop)
                            self._qr_failure_crops_saved = int(getattr(self, "_qr_failure_crops_saved", 0)) + 1
                        except Exception:
                            pass
                for payload in payloads:
                    if payload not in qr_payloads:
                        qr_payloads.append(payload)
                if qr_payloads and os.environ.get("LENTA_ZONE_QR_FAST_EXIT", "0") != "0":
                    break

            # Whole-crop fallback remains necessary when the zone detector is not
            # installed, misses a field, or the tight crop has lost quiet zone.
            qr_full_fallback = bool(
                getattr(
                    self.config,
                    "field_zone_qr_full_crop_fallback",
                    getattr(self.config, "field_zone_full_crop_fallback", True),
                )
            )
            if qr_full_fallback and not self._payloads_have_machine_fields(qr_payloads):
                direct_payloads, direct_debug = decode_qr_payloads_with_debug(crop, debug_label="full_crop")
                qr_debug["direct"] = direct_debug
                for payload in direct_payloads:
                    if payload not in qr_payloads:
                        qr_payloads.append(payload)
            qr_context_fallback = bool(
                getattr(
                    self.config,
                    "field_zone_qr_context_fallback",
                    getattr(self.config, "field_zone_full_crop_fallback", True),
                )
            )
            if (not self._payloads_have_machine_fields(qr_payloads)) and qr_context_fallback:
                bigger = det.expanded(
                    w,
                    h,
                    px=float(self.config.qr_expansion_x),
                    py=float(self.config.qr_expansion_y),
                )
                context_crop = crop_xyxy(frame, bigger.xyxy, pad=4)
                context_crop = self._rotate_crop_right_angle(context_crop, semantic_rotation)
                context_payloads, context_debug = decode_qr_payloads_with_debug(context_crop, debug_label="expanded_context")
                qr_debug["context"] = context_debug
                for payload in context_payloads:
                    if payload not in qr_payloads:
                        qr_payloads.append(payload)

        ocr_lines = []
        if run_ocr and self.ocr is not None:
            seen_texts: set[str] = set()

            def add_lines(lines, zone_label: str) -> None:
                for line in lines:
                    text_key = normalize_text(str(getattr(line, "text", ""))).lower()
                    if not text_key or text_key in seen_texts:
                        continue
                    try:
                        line.engine = f"{line.engine}|zone:{zone_label}"
                    except Exception:
                        pass
                    ocr_lines.append(line)
                    seen_texts.add(text_key)

            ocr_zones = [z for z in zones if z.label in OCR_ZONE_LABELS and z.label not in MACHINE_ZONE_LABELS]
            ocr_priority = {
                "product_name": 0,
                "price_default": 1,
                "price_card": 2,
                "price_discount": 3,
                "discount_amount": 4,
                "id_sku": 5,
                "print_datetime": 6,
                "code": 7,
            }
            ocr_zones.sort(key=lambda z: (ocr_priority.get(z.label, 20), -float(z.score), z.xyxy[1], z.xyxy[0]))
            selected_ocr_zones = self._select_ocr_zones(
                ocr_zones,
                max(0, int(getattr(self.config, "field_zone_ocr_top_k", 12))),
                max(1, int(getattr(self.config, "field_zone_ocr_per_label_top_k", 1))),
            )
            qr_debug["ocr_zone_selected"] = [z.label for z in selected_ocr_zones]
            for zone in selected_ocr_zones:
                zcrop = crop_zone(crop, zone, pad_ratio=float(getattr(self.config, "field_zone_padding_ratio", 0.10)), min_pad=4)
                if zcrop.size == 0:
                    continue
                try:
                    add_lines(self.ocr.recognize(zcrop), zone.label)
                except Exception as exc:
                    qr_debug.setdefault("ocr_zone_errors", []).append(f"{zone.label}:{type(exc).__name__}:{exc}")

            ocr_full_fallback = bool(
                getattr(
                    self.config,
                    "field_zone_ocr_full_crop_fallback",
                    getattr(self.config, "field_zone_full_crop_fallback", True),
                )
            )
            need_full_fallback = (not ocr_lines) or ocr_full_fallback
            if need_full_fallback:
                try:
                    if bool(self.config.enable_zonal_ocr) and hasattr(self.ocr, "recognize_zoned"):
                        add_lines(self.ocr.recognize_zoned(crop), "heuristic_full")
                    else:
                        add_lines(self.ocr.recognize(crop), "full")
                except Exception as exc:
                    qr_debug.setdefault("ocr_errors", []).append(f"full:{type(exc).__name__}:{exc}")

        text = "\n".join(line.text for line in ocr_lines)
        parsed = parse_observation(ocr_lines, qr_payloads, crop_bgr=crop)
        if self._payloads_have_machine_fields(qr_payloads):
            for qr_col in QR_FIELD_ALIASES:
                parsed.setdefault(qr_col, ABSENT_VALUE)
        qr_debug["payloads"] = len(qr_payloads)
        qr_debug["zone_count"] = len(zones)
        return text, ocr_lines, qr_payloads, parsed, sharpness_laplacian(crop), qr_debug

    def _process_detection(
        self,
        filename: str,
        timestamp_ms: int,
        frame: np.ndarray,
        det: Detection,
        output_dir: Optional[Path],
        crop_idx: int,
        recognize: bool = True,
    ) -> TagObservation:
        h, w = frame.shape[:2]
        det = det.expanded(w, h, px=0.05, py=0.06).clamp(w, h)
        text = ""
        ocr_lines = []
        qr_payloads: List[str] = []
        parsed: Dict[str, str] = {}
        base_crop = crop_xyxy(frame, det.xyxy, pad=int(self.config.crop_pad_px))
        image_quality = sharpness_laplacian(base_crop)
        focus_quality = compute_focus_quality(base_crop) if bool(getattr(self.config, "enable_focus_quality", True)) else {}
        visual_hash = perceptual_hash(base_crop)
        qr_debug: Dict[str, object] = {}
        if recognize:
            text, ocr_lines, qr_payloads, parsed, image_quality, qr_debug = self._recognize_detection(
                frame,
                det,
                output_dir,
                f"{Path(filename).stem}_{timestamp_ms}_{crop_idx:03d}",
            )
        obs = TagObservation(
            filename=filename,
            timestamp_ms=int(timestamp_ms),
            detection=det,
            text=text,
            ocr_lines=ocr_lines,
            qr_payloads=qr_payloads,
            parsed=parsed,
            image_quality=image_quality,
            focus_quality=focus_quality,
            visual_hash=visual_hash,
            qr_debug=qr_debug,
        )
        return obs


    def _detection_passes_edge_margin(self, det: Detection, frame_shape: tuple[int, ...]) -> bool:
        margin = float(getattr(self.config, "detection_edge_margin_ratio", 0.0) or 0.0)
        if margin <= 0:
            return True
        h, w = frame_shape[:2]
        mx = float(w) * margin
        my = float(h) * margin
        return det.x_min >= mx and det.y_min >= my and det.x_max <= (float(w) - mx) and det.y_max <= (float(h) - my)

    def run_video(self, video_path: str | Path, output_dir: str | Path = "outputs", output_csv: Optional[str | Path] = None) -> pd.DataFrame:
        video_path = Path(video_path)
        out_dir = mkdir(output_dir)
        filename = video_path.name
        meta = get_video_meta(video_path)
        tracker = SimpleTracker(
            iou_threshold=float(self.config.tracker_iou),
            center_threshold=float(self.config.tracker_center_threshold),
            max_lost=int(self.config.max_lost),
        )
        frame_limit = int(self.config.max_frames) if int(self.config.max_frames or 0) > 0 else None
        debug: Dict[str, object] = {"video": str(video_path), "meta": asdict(meta), "config": asdict(self.config), "frames": []}
        total_obs = 0
        for packet in iter_video_frames(video_path, sample_fps=float(self.config.sample_fps), max_frames=frame_limit, min_sharpness=float(self.config.min_sharpness)):
            detections = self.detector.predict(packet.frame_bgr)
            detections = [d for d in detections if self._detection_passes_edge_margin(d, packet.frame_bgr.shape)]
            detections = sorted(detections, key=lambda d: d.score, reverse=True)[: int(self.config.max_detections_per_frame)]
            observations: List[TagObservation] = []
            for i, det in enumerate(detections):
                try:
                    recognize_now = not bool(self.config.defer_ocr)
                    obs = self._process_detection(filename, packet.timestamp_ms, packet.frame_bgr, det, out_dir, i, recognize=recognize_now)
                    observations.append(obs)
                except Exception as exc:
                    print(f"[WARN] detection processing failed at {packet.timestamp_ms} ms: {exc}")
            tracker.update(observations)
            total_obs += len(observations)
            debug["frames"].append({
                "index": packet.index,
                "timestamp_ms": packet.timestamp_ms,
                "sharpness": packet.sharpness,
                "detections": len(detections),
                "observations": len(observations),
            })
            print(f"[INFO] {filename} {packet.timestamp_ms} ms: det={len(detections)} obs={len(observations)}")
        tracks = tracker.active_and_finished_tracks()
        if bool(self.config.defer_ocr) and (self.config.enable_ocr or self.config.enable_qr):
            self._enrich_representatives(video_path, tracks, out_dir)
        rows = self._tracks_to_rows(tracks)
        df = pd.DataFrame(ensure_columns(rows), columns=OUTPUT_COLUMNS)
        if output_csv is None:
            output_csv = out_dir / f"{video_path.stem}_recognized.csv"
        else:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        if self.config.save_debug_json:
            debug["total_observations"] = total_obs
            debug["rows"] = len(df)
            debug["tracks"] = self._tracks_debug(tracks)
            with open(out_dir / f"{video_path.stem}_debug.json", "w", encoding="utf-8") as f:
                json.dump(debug, f, ensure_ascii=False, indent=2)
        return df

    def _select_best_observation(self, track: Track) -> Optional[TagObservation]:
        return track.select_best_observation(float(self.config.representative_temporal_weight))

    @staticmethod
    def _enhance_code_crop(crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return crop
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        enhanced_l = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(cv2.merge([enhanced_l, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        return cv2.addWeighted(enhanced, 1.55, blur, -0.55, 0)

    @staticmethod
    def _align_code_crops_ecc(crops: List[np.ndarray]) -> List[np.ndarray]:
        if len(crops) < 2 or os.environ.get("LENTA_TEMPORAL_QR_ECC", "1") == "0":
            return crops
        ref = crops[0]
        if ref is None or ref.size == 0:
            return crops
        h, w = ref.shape[:2]
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) if ref.ndim == 3 else ref
        ref_gray = cv2.GaussianBlur(ref_gray, (3, 3), 0)
        aligned: List[np.ndarray] = [ref]
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 1e-4)
        for crop in crops[1:]:
            if crop is None or crop.size == 0 or crop.shape[:2] != (h, w):
                aligned.append(crop)
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                cv2.findTransformECC(ref_gray, gray, warp, cv2.MOTION_TRANSLATION, criteria, None, 5)
                fixed = cv2.warpAffine(
                    crop,
                    warp,
                    (w, h),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                aligned.append(fixed)
            except Exception:
                aligned.append(crop)
        return aligned

    @staticmethod
    def _temporal_code_composites(crops: List[np.ndarray], max_side: int, max_composites: int) -> List[np.ndarray]:
        usable = [crop for crop in crops if crop is not None and crop.size and crop.shape[0] >= 8 and crop.shape[1] >= 8]
        if not usable:
            return []
        aspects = sorted(float(crop.shape[1]) / max(1.0, float(crop.shape[0])) for crop in usable)
        aspect = aspects[len(aspects) // 2]
        side = max(64, int(max_side))
        if aspect >= 1.0:
            target_w = side
            target_h = max(64, min(side, int(round(side / max(0.25, aspect)))))
        else:
            target_h = side
            target_w = max(64, min(side, int(round(side * max(0.25, aspect)))))
        resized = [cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC) for crop in usable]
        resized.sort(key=sharpness_laplacian, reverse=True)

        composites: List[np.ndarray] = []
        for crop in resized[: min(3, len(resized))]:
            composites.append(crop)
            composites.append(PriceTagPipeline._enhance_code_crop(crop))
        if len(resized) >= 2:
            aligned = PriceTagPipeline._align_code_crops_ecc(resized[: min(6, len(resized))])
            stack = np.stack(aligned).astype(np.float32)
            median = np.median(stack, axis=0).clip(0, 255).astype(np.uint8)
            mean = np.mean(stack, axis=0).clip(0, 255).astype(np.uint8)
            composites.extend([median, PriceTagPipeline._enhance_code_crop(median), mean])

        unique: List[np.ndarray] = []
        seen: set[str] = set()
        for crop in composites:
            key = perceptual_hash(crop)
            if key in seen:
                continue
            seen.add(key)
            unique.append(crop)
            if len(unique) >= max(1, int(max_composites)):
                break
        return unique

    def _decode_temporal_qr_reconstruction(
        self,
        video_path: Path,
        track: Track,
        candidates: List[TagObservation],
        frame_cache: Dict[int, Optional[np.ndarray]],
        output_dir: Optional[Path],
    ) -> tuple[List[str], Dict[str, object]]:
        if not bool(getattr(self.config, "temporal_qr_reconstruction", False)):
            return [], {"enabled": False}
        if self.zone_detector is None:
            return [], {"enabled": True, "skipped": "no_zone_detector"}

        top_k = max(1, int(getattr(self.config, "temporal_qr_top_k", 6)))
        min_crops = max(2, int(getattr(self.config, "temporal_qr_min_crops", 2)))
        max_side = max(64, int(getattr(self.config, "temporal_qr_max_side", 384)))
        max_composites = max(1, int(getattr(self.config, "temporal_qr_max_composites", 8)))
        crops_by_label: Dict[str, List[np.ndarray]] = {"barcode": [], "qr_code_barcode": []}
        crop_debug: List[Dict[str, object]] = []

        for obs in candidates[:top_k]:
            ts = int(obs.timestamp_ms)
            if ts not in frame_cache:
                frame_cache[ts] = read_frame_at_ms(video_path, ts)
            frame = frame_cache.get(ts)
            if frame is None:
                continue
            tag_crop = crop_xyxy(frame, obs.detection.xyxy, pad=int(self.config.crop_pad_px))
            if tag_crop.size == 0:
                continue
            tag_crop, rectification_debug = self._maybe_rectify_crop(tag_crop)
            tag_crop, semantic_rotation, zones, orientation_debug = self._detect_zones_with_best_orientation(tag_crop)
            if bool(getattr(self.config, "enable_crop_rectification", False)):
                orientation_debug["rectification"] = rectification_debug
            if not zones:
                crop_debug.append(
                    {
                        "timestamp_ms": ts,
                        "semantic_rotation": semantic_rotation,
                        "skipped": "no_machine_zones",
                        "orientation_attempts": orientation_debug.get("orientation_attempts", []),
                    }
                )
                continue
            code_zones = [z for z in zones if z.label in MACHINE_ZONE_LABELS]
            code_zones.sort(key=lambda z: (0 if z.label == "barcode" else 1, -float(z.score), -float(z.area)))
            used_labels: set[str] = set()
            for zone in code_zones:
                if zone.label in used_labels:
                    continue
                used_labels.add(zone.label)
                zcrop = crop_zone(
                    tag_crop,
                    zone,
                    pad_ratio=float(getattr(self.config, "field_zone_padding_ratio", 0.10)),
                    min_pad=8,
                )
                if zcrop.size == 0:
                    continue
                crops_by_label.setdefault(zone.label, []).append(zcrop)
                crop_debug.append(
                    {
                        "timestamp_ms": ts,
                        "semantic_rotation": semantic_rotation,
                        "label": zone.label,
                        "source": zone.source,
                        "score": round(float(zone.score), 4),
                        "shape": list(zcrop.shape[:2]),
                        "sharpness": round(float(sharpness_laplacian(zcrop)), 3),
                    }
                )

        payloads: List[str] = []
        attempts: List[Dict[str, object]] = []
        for label, crops in crops_by_label.items():
            if len(crops) < min_crops:
                attempts.append({"label": label, "skipped": "not_enough_crops", "crops": len(crops)})
                continue
            composites = self._temporal_code_composites(crops, max_side=max_side, max_composites=max_composites)
            if output_dir is not None and bool(getattr(self.config, "field_zone_save_crops", False)):
                temporal_dir = mkdir(output_dir / "temporal_qr")
                for idx, composite in enumerate(composites[:max_composites]):
                    cv2.imwrite(str(temporal_dir / f"track{track.track_id:04d}_{label}_{idx:02d}.jpg"), composite)
            for idx, composite in enumerate(composites):
                try:
                    found, debug = decode_qr_payloads_with_debug(
                        composite,
                        force_native=True,
                        debug_label=f"temporal_{label}_{idx}",
                    )
                except Exception as exc:
                    found, debug = [], {"errors": [f"temporal-decode:{type(exc).__name__}:{exc}"]}
                debug["label"] = label
                debug["input_crops"] = len(crops)
                debug["composite_index"] = idx
                attempts.append(debug)
                for payload in found:
                    if payload not in payloads:
                        payloads.append(payload)
                if payloads and os.environ.get("LENTA_TEMPORAL_QR_FAST_EXIT", "1") != "0":
                    break
            if payloads and os.environ.get("LENTA_TEMPORAL_QR_FAST_EXIT", "1") != "0":
                break
        return payloads, {
            "enabled": True,
            "crop_count": sum(len(v) for v in crops_by_label.values()),
            "crops": crop_debug[:32],
            "attempts": attempts,
        }

    @staticmethod
    def _observation_quality_score(obs: TagObservation) -> float:
        # Best-frame selection for recognition: QR/barcode needs sharp modules and
        # enough pixels, while final bbox needs strong detector confidence.
        area_bonus = min(25.0, float(obs.detection.area) / 18000.0)
        sharp_bonus = min(35.0, max(0.0, float(obs.image_quality)) / 35.0)
        focus = getattr(obs, "focus_quality", {}) or {}
        focus_bonus = 26.0 * float(focus.get("score", 0.0) or 0.0)
        qr_focus_bonus = 12.0 * float(focus.get("qr_score", 0.0) or 0.0)
        det_bonus = float(obs.detection.score) * 20.0
        evidence_bonus = 60.0 if obs.qr_payloads else (20.0 if obs.text else 0.0)
        return evidence_bonus + sharp_bonus + focus_bonus + qr_focus_bonus + area_bonus + det_bonus

    def _select_top_observations(self, track: Track, k: int) -> List[TagObservation]:
        if k <= 0 or not track.observations:
            return []
        selected: List[TagObservation] = []
        seen_ts: set[int] = set()
        for obs in sorted(track.observations, key=self._observation_quality_score, reverse=True):
            ts = int(obs.timestamp_ms)
            # Avoid spending top-K budget on near-identical duplicate detections in
            # the same sampled frame; keep one best bbox per timestamp.
            if ts in seen_ts:
                continue
            selected.append(obs)
            seen_ts.add(ts)
            if len(selected) >= k:
                break
        return selected

    @staticmethod
    def _row_representative_score(obs: TagObservation, median_ts: float, span_ms: float) -> float:
        """Score observation for final CSV bbox/timestamp, not semantic fields.

        Hidden/public matching uses the row geometry/time, so a QR-hit frame must
        not automatically become the representative if its bbox is weaker.  Use
        detector confidence, crop sharpness and temporal centrality; semantic
        fields are fused separately from all observations.
        """
        source_bonus = 12.0 if str(obs.detection.source).startswith("yolo") else 0.0
        det_bonus = float(obs.detection.score) * 35.0
        sharp_bonus = min(18.0, max(0.0, float(obs.image_quality)) / 55.0)
        focus_bonus = 5.0 * float((getattr(obs, "focus_quality", {}) or {}).get("score", 0.0) or 0.0)
        area_bonus = min(8.0, float(obs.detection.area) / 35000.0)
        temporal_penalty = abs(float(obs.timestamp_ms) - median_ts) / max(1.0, span_ms) * 8.0
        return source_bonus + det_bonus + sharp_bonus + focus_bonus + area_bonus - temporal_penalty

    def _select_row_observation(self, track: Track) -> Optional[TagObservation]:
        if not track.observations:
            return None
        timestamps = sorted(int(obs.timestamp_ms) for obs in track.observations)
        mid = len(timestamps) // 2
        median_ts = float(timestamps[mid] if len(timestamps) % 2 else (timestamps[mid - 1] + timestamps[mid]) / 2.0)
        span_ms = max(1000.0, float(timestamps[-1] - timestamps[0]))
        return max(track.observations, key=lambda obs: self._row_representative_score(obs, median_ts, span_ms))

    def _enrich_representatives(self, video_path: Path, tracks: Iterable[Track], output_dir: Optional[Path]) -> None:
        frame_cache: Dict[int, Optional[np.ndarray]] = {}
        qr_top_k = max(0, int(self.config.deferred_qr_top_k or 0)) if self.config.enable_qr else 0
        ocr_top_k = max(0, int(self.config.deferred_ocr_top_k or 0)) if self.config.enable_ocr else 0
        recognition_top_k = max(qr_top_k, ocr_top_k, max(0, int(self.config.deferred_recognition_top_k or 0)))
        for tr in tracks:
            if not tr.observations:
                continue
            candidates = self._select_top_observations(tr, recognition_top_k or 1)
            if not candidates:
                continue
            merged_qr_payloads: List[str] = []

            # QR-first: try several top crops from the same physical track. A crop
            # that is best for bbox/sharpness is not always best for QR modules.
            for rank, obs in enumerate(candidates[:qr_top_k]):
                if obs.qr_payloads:
                    for payload in obs.qr_payloads:
                        if payload not in merged_qr_payloads:
                            merged_qr_payloads.append(payload)
                    continue
                ts = int(obs.timestamp_ms)
                if ts not in frame_cache:
                    frame_cache[ts] = read_frame_at_ms(video_path, ts)
                frame = frame_cache.get(ts)
                if frame is None:
                    continue
                try:
                    _, _, qr_payloads, parsed, image_quality, qr_debug = self._recognize_detection(
                        frame,
                        obs.detection,
                        output_dir,
                        f"{Path(obs.filename).stem}_{ts}_track{tr.track_id:04d}_qr{rank}",
                        run_qr=True,
                        run_ocr=False,
                    )
                except Exception as exc:
                    print(f"[WARN] deferred QR failed at {ts} ms track={tr.track_id}: {exc}")
                    continue
                obs.qr_payloads = qr_payloads
                obs.qr_debug = qr_debug
                obs.parsed.update(parsed)
                obs.image_quality = max(float(obs.image_quality), float(image_quality))
                refreshed_crop = crop_xyxy(frame, obs.detection.xyxy, pad=int(self.config.crop_pad_px))
                obs.focus_quality = compute_focus_quality(refreshed_crop) if bool(getattr(self.config, "enable_focus_quality", True)) else obs.focus_quality
                obs.visual_hash = perceptual_hash(refreshed_crop)
                for payload in qr_payloads:
                    if payload not in merged_qr_payloads:
                        merged_qr_payloads.append(payload)
                if merged_qr_payloads and os.environ.get("LENTA_TRACK_QR_FAST_EXIT", "0") != "0":
                    break

            if not merged_qr_payloads and bool(getattr(self.config, "temporal_qr_reconstruction", False)):
                temporal_payloads, temporal_debug = self._decode_temporal_qr_reconstruction(
                    video_path,
                    tr,
                    candidates,
                    frame_cache,
                    output_dir,
                )
                if candidates:
                    candidates[0].qr_debug = {
                        **(candidates[0].qr_debug or {}),
                        "temporal_reconstruction": temporal_debug,
                    }
                for payload in temporal_payloads:
                    if payload not in merged_qr_payloads:
                        merged_qr_payloads.append(payload)
                if temporal_payloads and candidates:
                    qr_fields = parse_observation([], temporal_payloads)
                    candidates[0].qr_payloads = list(dict.fromkeys([*candidates[0].qr_payloads, *temporal_payloads]))
                    candidates[0].parsed.update({k: v for k, v in qr_fields.items() if v})

            # OCR-second: run only on a very small number of best crops and parse it
            # together with any QR payload found on another frame of the track.
            for rank, obs in enumerate(candidates[:ocr_top_k]):
                if obs.text and obs.ocr_lines:
                    continue
                ts = int(obs.timestamp_ms)
                if ts not in frame_cache:
                    frame_cache[ts] = read_frame_at_ms(video_path, ts)
                frame = frame_cache.get(ts)
                if frame is None:
                    continue
                known_qr = obs.qr_payloads or merged_qr_payloads
                try:
                    text, ocr_lines, qr_payloads, parsed, image_quality, qr_debug = self._recognize_detection(
                        frame,
                        obs.detection,
                        output_dir,
                        f"{Path(obs.filename).stem}_{ts}_track{tr.track_id:04d}_ocr{rank}",
                        run_qr=not bool(known_qr),
                        run_ocr=True,
                        known_qr_payloads=list(known_qr),
                    )
                except Exception as exc:
                    print(f"[WARN] deferred OCR failed at {ts} ms track={tr.track_id}: {exc}")
                    continue
                obs.text = text
                obs.ocr_lines = ocr_lines
                obs.qr_payloads = qr_payloads
                obs.qr_debug = qr_debug
                obs.parsed.update(parsed)
                obs.image_quality = max(float(obs.image_quality), float(image_quality))
                refreshed_crop = crop_xyxy(frame, obs.detection.xyxy, pad=int(self.config.crop_pad_px))
                obs.focus_quality = compute_focus_quality(refreshed_crop) if bool(getattr(self.config, "enable_focus_quality", True)) else obs.focus_quality
                obs.visual_hash = perceptual_hash(refreshed_crop)
                for payload in qr_payloads:
                    if payload not in merged_qr_payloads:
                        merged_qr_payloads.append(payload)

            # Propagate machine-readable QR fields to every observation in the track
            # so merge_field_values can recover them even if the representative bbox
            # is not the QR-hit crop.
            if merged_qr_payloads:
                qr_fields = parse_observation([], merged_qr_payloads)
                for obs in tr.observations:
                    for payload in merged_qr_payloads:
                        if payload not in obs.qr_payloads:
                            obs.qr_payloads.append(payload)
                    obs.parsed.update({k: v for k, v in qr_fields.items() if v})

    def _tracks_debug(self, tracks: Iterable[Track]) -> List[Dict[str, object]]:
        items: List[Dict[str, object]] = []
        for tr in tracks:
            best = self._select_best_observation(tr)
            if best is None:
                continue
            items.append(
                {
                    "track_id": tr.track_id,
                    "observations": len(tr.observations),
                    "trajectory": [
                        {
                            "timestamp_ms": int(obs.timestamp_ms),
                            "bbox": [round(float(v), 1) for v in obs.detection.xyxy],
                            "score": round(float(obs.detection.score), 4),
                            "source": obs.detection.source,
                        }
                        for obs in sorted(tr.observations, key=lambda item: int(item.timestamp_ms))
                    ],
                    "best_timestamp_ms": int(best.timestamp_ms),
                    "best_bbox": [round(float(v), 1) for v in best.detection.xyxy],
                    "best_score": round(float(best.detection.score), 4),
                    "best_source": best.detection.source,
                    "best_quality": round(float(best.image_quality), 2),
                    "best_focus_quality": {k: round(float(v), 4) for k, v in (best.focus_quality or {}).items() if isinstance(v, (int, float))},
                    "visual_hash": best.visual_hash,
                    "ocr_engines": sorted({line.engine for line in best.ocr_lines}),
                    "text": best.text[:600],
                    "qr_payloads": best.qr_payloads[:5],
                    "qr_debug": best.qr_debug,
                    "parsed": {col: best.parsed.get(col, "") for col in OUTPUT_COLUMNS if best.parsed.get(col, "")},
                }
            )
        return items

    @staticmethod
    def _catalog_lookup_code(row: Dict[str, object]) -> str:
        for col in ("qr_code_barcode", "barcode"):
            value = str(row.get(col, "") or "")
            digits = re.sub(r"\D", "", value)
            if len(digits) >= 8:
                return digits
        return ""

    def _enrich_row_from_catalog(self, row: Dict[str, object]) -> Dict[str, object]:
        catalog = getattr(self, "product_catalog", None)
        if catalog is None:
            return row
        code = self._catalog_lookup_code(row)
        if code:
            name = catalog.name_for_barcode(code)
            if name:
                if should_replace_product_name(row.get("product_name", "")):
                    row["product_name"] = name
                    row["_catalog_match_source"] = "barcode"
                    row["_catalog_match_score"] = 1.0
                if not self._non_absent(row.get("barcode", "")) and len(code) in {8, 12, 13, 14}:
                    row["barcode"] = code
                return row

        # Riskier catalog text/price enrichment: use only with an explicit
        # env flag and a high-confidence text+price match.  This can rescue
        # noisy product_name rows when QR/barcode is absent, but it is not a
        # replacement for machine-readable evidence.
        if os.environ.get("LENTA_CATALOG_TEXT_MATCH", "0").strip().lower() not in {"1", "true", "yes", "y", "on"}:
            return row
        query = self._non_absent(row.get("product_name", "")) or normalize_text(str(row.get("_best_text", "") or ""))
        if not query:
            return row
        prices = [row.get(col, "") for col in ("price_card", "price_default", "price_discount", "price1_qr", "price4_qr")]
        try:
            match = catalog.best_text_price_match(query, prices=prices)
        except Exception:
            match = {}
        if not match:
            return row
        name = str(match.get("name", "") or "")
        score = float(match.get("score", 0.0) or 0.0)
        if name and (should_replace_product_name(row.get("product_name", "")) or score >= float(os.environ.get("LENTA_CATALOG_TEXT_MATCH_OVERWRITE_SCORE", "0.92"))):
            row["product_name"] = name
            row["_catalog_match_source"] = str(match.get("source", "text_price"))
            row["_catalog_match_score"] = score
        return row

    def _infer_qr_fields_from_visual_evidence(self, row: Dict[str, object]) -> Dict[str, object]:
        """Optional risky fill of QR CSV columns from visual/barcode fields.

        QR optical decode can be zero even when visual price/barcode OCR is
        good. This knob is OFF by default because it
        can overfit public labels; experiments can enable it explicitly.
        """
        if os.environ.get("LENTA_INFER_QR_FROM_VISUAL", "0") == "0":
            return row
        require_barcode = os.environ.get("LENTA_INFER_QR_REQUIRE_BARCODE", "1") != "0"
        barcode = self._non_absent(row.get("qr_code_barcode", "")) or self._non_absent(row.get("barcode", ""))
        if require_barcode and not barcode:
            return row
        if barcode and not self._non_absent(row.get("qr_code_barcode", "")):
            row["qr_code_barcode"] = barcode
        default_price = self._non_absent(row.get("price_default", ""))
        card_price = self._non_absent(row.get("price_card", ""))
        if default_price and not self._non_absent(row.get("price1_qr", "")):
            row["price1_qr"] = default_price
        if card_price and not self._non_absent(row.get("price4_qr", "")):
            row["price4_qr"] = card_price
        ratio = os.environ.get("LENTA_INFER_QR_PRICE2_RATIO", "").strip()
        if ratio and default_price and not self._non_absent(row.get("price2_qr", "")):
            try:
                row["price2_qr"] = price_to_str(float(default_price) * float(ratio))
            except Exception:
                pass
        if os.environ.get("LENTA_INFER_QR_ABSENT_CONSTANTS", "0") != "0":
            for col in ["price3_qr", "action_price_qr", "action_code_qr", "wholesale_level_1_price", "wholesale_level_2_price"]:
                if not self._non_absent(row.get(col, "")):
                    row[col] = ABSENT_VALUE
        return row

    def _tracks_to_rows(self, tracks: Iterable[Track]) -> List[Dict[str, object]]:
        # First fuse per tracker track while keeping private metadata for
        # post-track dedupe. Private keys are stripped by ensure_columns later.
        candidate_rows: List[Dict[str, object]] = []
        for tr in tracks:
            if len(tr.observations) < int(self.config.min_track_observations):
                continue
            best = self._select_best_observation(tr)
            rep_obs = self._select_row_observation(tr)
            if best is None or rep_obs is None:
                continue

            row = rep_obs.to_row()
            # Keep coordinates/timestamp from geometry representative; semantic
            # fields are fused below from all observations including QR-hit crops.
            row.update(rep_obs.to_row())
            for col in OUTPUT_COLUMNS:
                if col not in {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}:
                    fused = self._merge_column_values(col, (obs.parsed.get(col, "") for obs in tr.observations))
                    if fused or self._clear_on_empty_fusion(col):
                        row[col] = fused

            timestamps = [int(obs.timestamp_ms) for obs in tr.observations]
            sources = sorted({str(obs.detection.source) for obs in tr.observations})
            row["_track_id"] = tr.track_id
            row["_observations"] = len(tr.observations)
            row["_track_first_ts"] = min(timestamps) if timestamps else int(best.timestamp_ms)
            row["_track_last_ts"] = max(timestamps) if timestamps else int(best.timestamp_ms)
            row["_best_source"] = rep_obs.detection.source
            row["_sources"] = "|".join(sources)
            row["_visual_hash"] = rep_obs.visual_hash
            row["_best_text"] = best.text
            row["_has_qr"] = any(bool(obs.qr_payloads) for obs in tr.observations)
            row["_det_score"] = float(rep_obs.detection.score)

            self._enrich_row_from_catalog(row)
            self._infer_qr_fields_from_visual_evidence(row)
            if not self._track_row_passes_source_gate(row):
                continue
            candidate_rows.append(row)

        # Collapse duplicate tracks by true stable IDs first. OCR-only
        # name+price is not stable globally: identical products can appear on
        # different shelves, so it is handled only by gated duplicate logic.
        groups: Dict[str, List[Dict[str, object]]] = {}
        ordered_rows = sorted(candidate_rows, key=lambda r: 0 if self._stable_group_key(r) else 1)
        for i, row in enumerate(ordered_rows):
            key = self._stable_group_key(row)
            if not key:
                for existing_key, existing_rows in groups.items():
                    if any(self._rows_duplicate(row, other) for other in existing_rows):
                        key = existing_key
                        break
                if not key:
                    key = f"track:{i}"
            groups.setdefault(key, []).append(row)

        fused_rows: List[Dict[str, object]] = []
        for _, rows in groups.items():
            # Choose representative with most non-empty fields and strongest evidence.
            rep = max(rows, key=self._row_quality)
            out = dict(rep)
            for col in OUTPUT_COLUMNS:
                if col in {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}:
                    continue
                fused = self._merge_column_values(col, (r.get(col, "") for r in rows))
                if fused or self._clear_on_empty_fusion(col):
                    out[col] = fused
            self._enrich_row_from_catalog(out)
            self._infer_qr_fields_from_visual_evidence(out)
            fused_rows.append({col: out.get(col, "") for col in OUTPUT_COLUMNS})
        fused_rows.sort(key=lambda r: (str(r.get("filename", "")), int(float(r.get("frame_timestamp") or 0)), float(r.get("y_min") or 0), float(r.get("x_min") or 0)))
        return fused_rows

    def _track_row_passes_source_gate(self, row: Dict[str, object]) -> bool:
        sources = set(str(row.get("_sources", "")).split("|"))
        # Trained closed-set YOLO weights are trusted geometry evidence. YOLO-World
        # open-vocabulary proposals are treated like heuristic fallbacks because
        # prompt detections can over-fire on product labels.
        if any(source == "yolo" or source.startswith("yolo:") for source in sources):
            return True
        fallback_sources = {"heuristic", "red_white_tag", "qr_seed", "color_geometry", "yolo_world"} | {s for s in sources if s.startswith("yolo_world:")}
        if not (sources & fallback_sources):
            return True
        if not bool(self.config.fallback_require_evidence):
            return True
        # Fallback detectors are proposal generators, not final evidence. A
        # single fallback hit is acceptable only with machine-readable evidence
        # or a product+price pair; weak OCR text still needs repeated support.
        observations = int(row.get("_observations") or 0)
        if self._row_has_strong_fallback_evidence(row):
            return True
        if observations >= int(self.config.fallback_min_observations) and self._row_has_semantic_evidence(row):
            return True
        return False

    @staticmethod
    def _non_absent(value: object) -> str:
        text = str(value or "").strip()
        return text if text and text != ABSENT_VALUE else ""

    def _stable_group_key(self, row: Dict[str, object]) -> str:
        # Only machine-readable identifiers are globally stable. Do not use
        # product_name+price as a key: two real shelf tags can share both.
        for col in ["qr_code_barcode", "barcode", "id_sku", "code"]:
            val = self._non_absent(row.get(col, ""))
            if val:
                return f"{col}:{val}"
        return ""

    def _row_has_semantic_evidence(self, row: Dict[str, object]) -> bool:
        for col in [
            "qr_code_barcode",
            "barcode",
            "id_sku",
            "price_default",
            "price_card",
            "price_discount",
            "product_name",
            "print_datetime",
        ]:
            if self._non_absent(row.get(col, "")):
                return True
        return bool(normalize_text(str(row.get("_best_text", ""))))

    def _row_has_strong_fallback_evidence(self, row: Dict[str, object]) -> bool:
        for col in [
            "qr_code_barcode",
            "barcode",
            "id_sku",
            "code",
            "price1_qr",
            "price2_qr",
            "price3_qr",
            "price4_qr",
            "action_price_qr",
            "action_code_qr",
        ]:
            if self._non_absent(row.get(col, "")):
                return True
        has_product = bool(self._non_absent(row.get("product_name", "")))
        has_price = any(
            self._non_absent(row.get(col, ""))
            for col in ["price_default", "price_card", "price_discount"]
        )
        return has_product and has_price

    def _row_quality(self, row: Dict[str, object]) -> float:
        field_score = sum(1 for c in OUTPUT_COLUMNS if self._non_absent(row.get(c, "")))
        id_bonus = 35 if self._stable_group_key(row) else 0
        qr_bonus = 25 if bool(row.get("_has_qr")) else 0
        text_bonus = min(15.0, len(normalize_text(str(row.get("_best_text", "")))) * 0.03)
        obs_bonus = min(20.0, float(row.get("_observations") or 0))
        try:
            area = max(0.0, float(row.get("x_max", 0)) - float(row.get("x_min", 0))) * max(0.0, float(row.get("y_max", 0)) - float(row.get("y_min", 0)))
        except Exception:
            area = 0.0
        return id_bonus + qr_bonus + text_bonus + obs_bonus + field_score * 8.0 + area * 0.0001

    def _rows_have_conflicting_ids(self, a: Dict[str, object], b: Dict[str, object]) -> bool:
        for col in ["qr_code_barcode", "barcode", "id_sku", "code"]:
            av = self._non_absent(a.get(col, ""))
            bv = self._non_absent(b.get(col, ""))
            if av and bv and av != bv:
                return True
        return False

    @staticmethod
    def _row_box(row: Dict[str, object]) -> Optional[List[float]]:
        try:
            return [float(row[c]) for c in ["x_min", "y_min", "x_max", "y_max"]]
        except Exception:
            return None

    @staticmethod
    def _row_text_for_dedupe(row: Dict[str, object]) -> str:
        parts = [
            str(row.get("product_name", "")),
            str(row.get("price_default", "")),
            str(row.get("price_card", "")),
            str(row.get("_best_text", "")),
        ]
        return " ".join(p for p in parts if p and p != ABSENT_VALUE)

    def _rows_shelf_row_continuous(self, a: Dict[str, object], b: Dict[str, object]) -> bool:
        box_a = self._row_box(a)
        box_b = self._row_box(b)
        if box_a is None or box_b is None:
            return False
        wa, ha = max(1.0, box_a[2] - box_a[0]), max(1.0, box_a[3] - box_a[1])
        wb, hb = max(1.0, box_b[2] - box_b[0]), max(1.0, box_b[3] - box_b[1])
        y_overlap = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])) / max(1.0, min(ha, hb))
        ay = (box_a[1] + box_a[3]) / 2.0
        by = (box_b[1] + box_b[3]) / 2.0
        height_ratio = min(ha, hb) / max(ha, hb)
        return (
            height_ratio >= 0.45
            and (
                y_overlap >= 0.45
                or abs(ay - by) <= float(self.config.dedupe_row_y_threshold_ratio) * max(ha, hb)
            )
            and min(wa, wb) / max(wa, wb) >= 0.40
        )

    def _rows_duplicate(self, a: Dict[str, object], b: Dict[str, object]) -> bool:
        if str(a.get("filename", "")) != str(b.get("filename", "")):
            return False
        if self._rows_have_conflicting_ids(a, b):
            return False
        if self._rows_spatial_duplicate(a, b):
            return True

        try:
            ta = int(float(a.get("frame_timestamp") or 0))
            tb = int(float(b.get("frame_timestamp") or 0))
        except Exception:
            return False
        if abs(ta - tb) > int(self.config.dedupe_extended_time_window_ms):
            return False
        if not self._rows_shelf_row_continuous(a, b):
            return False

        ha = str(a.get("_visual_hash", ""))
        hb = str(b.get("_visual_hash", ""))
        hash_close = hamming_distance_hex(ha, hb) <= int(self.config.dedupe_visual_hash_threshold)

        text_a = self._row_text_for_dedupe(a)
        text_b = self._row_text_for_dedupe(b)
        sim = text_similarity(text_a, text_b)
        text_close = sim >= float(self.config.dedupe_text_similarity)

        # Visual hash alone can merge different adjacent tags with similar colors;
        # require either text agreement or tight spatial continuity. Text alone
        # is allowed when product+price is highly similar on the same shelf row.
        if hash_close and (text_close or self._rows_spatial_near(a, b)):
            return True
        if text_close and self._rows_spatial_near(a, b, loose=True):
            return True
        return False

    def _rows_spatial_duplicate(self, a: Dict[str, object], b: Dict[str, object]) -> bool:
        if str(a.get("filename", "")) != str(b.get("filename", "")):
            return False
        if self._rows_have_conflicting_ids(a, b):
            return False
        try:
            ta = int(float(a.get("frame_timestamp") or 0))
            tb = int(float(b.get("frame_timestamp") or 0))
        except Exception:
            return False
        if abs(ta - tb) > int(self.config.dedupe_time_window_ms):
            return False
        return self._rows_spatial_near(a, b)

    def _rows_spatial_near(self, a: Dict[str, object], b: Dict[str, object], loose: bool = False) -> bool:
        box_a = self._row_box(a)
        box_b = self._row_box(b)
        if box_a is None or box_b is None:
            return False
        if iou_xyxy(box_a, box_b) >= float(self.config.dedupe_iou):
            return True

        wa, ha = max(1.0, box_a[2] - box_a[0]), max(1.0, box_a[3] - box_a[1])
        wb, hb = max(1.0, box_b[2] - box_b[0]), max(1.0, box_b[3] - box_b[1])
        y_overlap = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])) / max(1.0, min(ha, hb))
        height_ratio = min(ha, hb) / max(ha, hb)
        ax, ay = (box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0
        bx, by = (box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0
        x_gate = float(self.config.dedupe_center_threshold) * (2.0 if loose else 1.0)
        y_gate = (0.55 if loose else 0.35) * max(ha, hb)
        return (
            y_overlap >= (0.55 if loose else 0.70)
            and height_ratio >= (0.48 if loose else 0.60)
            and abs(ax - bx) <= x_gate
            and abs(ay - by) <= y_gate
            and min(wa, wb) / max(wa, wb) >= (0.42 if loose else 0.55)
        )



def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Lenta Shelf AI video-to-CSV pipeline")
    parser.add_argument("video", help="Path to input .mp4")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML/JSON config")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--output-csv", default=None, help="Output CSV path")
    parser.add_argument("--sample-fps", type=float, default=None, help="Override frame sampling FPS")
    parser.add_argument("--weights", default=None, help="YOLO weights path")
    parser.add_argument("--fast", action="store_true", help="Disable OCR and use QR/color detector only")
    parser.add_argument("--defer-ocr", action="store_true", help="Run OCR/QR only on the best crop per final track")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug/quick run: process at most N sampled frames")
    args = parser.parse_args(argv)

    cfg = PipelineConfig.from_file(args.config)
    if args.sample_fps is not None:
        cfg.sample_fps = args.sample_fps
    if args.weights is not None:
        cfg.yolo_weights = args.weights
    if args.defer_ocr:
        cfg.defer_ocr = True
    if args.fast:
        cfg.enable_ocr = False
        cfg.sample_fps = min(cfg.sample_fps, 1.0)
        cfg.max_detections_per_frame = min(int(cfg.max_detections_per_frame), 30)
    if args.max_frames is not None:
        cfg.max_frames = args.max_frames
    pipe = PriceTagPipeline(cfg)
    df = pipe.run_video(args.video, output_dir=args.output_dir, output_csv=args.output_csv)
    print(f"[DONE] rows={len(df)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
