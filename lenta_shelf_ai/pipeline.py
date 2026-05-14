from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import pandas as pd

from .detectors import HybridDetector
from .ocr import EnsembleOCREngine
from .parsers import merge_field_values, parse_observation
from .qr import decode_qr_payloads
from .schema import ABSENT_VALUE, OUTPUT_COLUMNS, QR_FIELD_ALIASES, Detection, TagObservation, ensure_columns
from .tracker import SimpleTracker, Track
from .utils import crop_xyxy, mkdir, sharpness_laplacian, read_yaml_or_json, normalize_text, iou_xyxy, perceptual_hash, hamming_distance_hex, text_similarity
from .video import iter_video_frames, get_video_meta, read_frame_at_ms

@dataclass
class PipelineConfig:
    sample_fps: float = 2.0
    yolo_weights: str = "models/price_tag_yolo.pt"
    yolo_conf: float = 0.23
    detector_imgsz: int = 1600
    enable_fallback_detectors: bool = False
    min_sharpness: float = 18.0
    max_frames: int = 0
    max_detections_per_frame: int = 80
    enable_ocr: bool = True
    enable_qr: bool = True
    defer_ocr: bool = False
    prefer_paddle: bool = True
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
        self.ocr = None
        if self.config.enable_ocr:
            self.ocr = EnsembleOCREngine(prefer_paddle=bool(self.config.prefer_paddle), lang=str(self.config.ocr_lang), use_gpu=bool(self.config.use_gpu))

    def _recognize_detection(self, frame: np.ndarray, det: Detection, output_dir: Optional[Path], crop_name: str) -> tuple[str, list, list[str], Dict[str, str], float]:
        h, w = frame.shape[:2]
        crop = crop_xyxy(frame, det.xyxy, pad=int(self.config.crop_pad_px))
        qr_payloads: List[str] = []
        if self.config.enable_qr:
            # Decode both direct crop and enlarged context. QR may sit at crop
            # edge when YOLO bbox is centered on price/text panel only.
            qr_payloads = decode_qr_payloads(crop)
            if not qr_payloads:
                bigger = det.expanded(
                    w,
                    h,
                    px=float(self.config.qr_expansion_x),
                    py=float(self.config.qr_expansion_y),
                )
                qr_payloads = decode_qr_payloads(crop_xyxy(frame, bigger.xyxy, pad=4))
        ocr_lines = []
        if self.ocr is not None:
            if bool(self.config.enable_zonal_ocr) and hasattr(self.ocr, "recognize_zoned"):
                ocr_lines = self.ocr.recognize_zoned(crop)
            else:
                ocr_lines = self.ocr.recognize(crop)
        text = "\n".join(line.text for line in ocr_lines)
        parsed = parse_observation(ocr_lines, qr_payloads, crop_bgr=crop)
        if qr_payloads:
            for qr_col in QR_FIELD_ALIASES:
                parsed.setdefault(qr_col, ABSENT_VALUE)
        if output_dir is not None and self.config.save_crops:
            crop_dir = mkdir(output_dir / "crops")
            cv2.imwrite(str(crop_dir / f"{crop_name}.jpg"), crop)
        return text, ocr_lines, qr_payloads, parsed, sharpness_laplacian(crop)

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
        visual_hash = perceptual_hash(base_crop)
        if recognize:
            text, ocr_lines, qr_payloads, parsed, image_quality = self._recognize_detection(
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
            visual_hash=visual_hash,
        )
        return obs

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

    def _enrich_representatives(self, video_path: Path, tracks: Iterable[Track], output_dir: Optional[Path]) -> None:
        frame_cache: Dict[int, Optional[np.ndarray]] = {}
        for tr in tracks:
            obs = self._select_best_observation(tr)
            if obs is None:
                continue
            if obs.text or obs.qr_payloads:
                continue
            ts = int(obs.timestamp_ms)
            if ts not in frame_cache:
                frame_cache[ts] = read_frame_at_ms(video_path, ts)
            frame = frame_cache.get(ts)
            if frame is None:
                continue
            try:
                text, ocr_lines, qr_payloads, parsed, image_quality = self._recognize_detection(
                    frame,
                    obs.detection,
                    output_dir,
                    f"{Path(obs.filename).stem}_{ts}_track{tr.track_id:04d}",
                )
            except Exception as exc:
                print(f"[WARN] deferred OCR/QR failed at {ts} ms track={tr.track_id}: {exc}")
                continue
            obs.text = text
            obs.ocr_lines = ocr_lines
            obs.qr_payloads = qr_payloads
            obs.parsed = parsed
            obs.image_quality = image_quality
            obs.visual_hash = perceptual_hash(crop_xyxy(frame, obs.detection.xyxy, pad=int(self.config.crop_pad_px)))

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
                    "best_timestamp_ms": int(best.timestamp_ms),
                    "best_bbox": [round(float(v), 1) for v in best.detection.xyxy],
                    "best_score": round(float(best.detection.score), 4),
                    "best_source": best.detection.source,
                    "best_quality": round(float(best.image_quality), 2),
                    "visual_hash": best.visual_hash,
                    "ocr_engines": sorted({line.engine for line in best.ocr_lines}),
                    "text": best.text[:600],
                    "qr_payloads": best.qr_payloads[:5],
                    "parsed": {col: best.parsed.get(col, "") for col in OUTPUT_COLUMNS if best.parsed.get(col, "")},
                }
            )
        return items

    def _tracks_to_rows(self, tracks: Iterable[Track]) -> List[Dict[str, object]]:
        # First fuse per tracker track while keeping private metadata for
        # post-track dedupe. Private keys are stripped by ensure_columns later.
        candidate_rows: List[Dict[str, object]] = []
        for tr in tracks:
            if len(tr.observations) < int(self.config.min_track_observations):
                continue
            best = self._select_best_observation(tr)
            if best is None:
                continue

            row = best.to_row()
            # Merge recognized fields across observations.
            for col in OUTPUT_COLUMNS:
                if col in {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}:
                    continue
                fused = merge_field_values(obs.parsed.get(col, "") for obs in tr.observations)
                if fused:
                    row[col] = fused
            # Keep coordinates/timestamp from best high-quality observation.
            row.update(best.to_row())
            for col in OUTPUT_COLUMNS:
                if col not in {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}:
                    fused = merge_field_values(obs.parsed.get(col, "") for obs in tr.observations)
                    if fused:
                        row[col] = fused

            timestamps = [int(obs.timestamp_ms) for obs in tr.observations]
            sources = sorted({str(obs.detection.source) for obs in tr.observations})
            row["_track_id"] = tr.track_id
            row["_observations"] = len(tr.observations)
            row["_track_first_ts"] = min(timestamps) if timestamps else int(best.timestamp_ms)
            row["_track_last_ts"] = max(timestamps) if timestamps else int(best.timestamp_ms)
            row["_best_source"] = best.detection.source
            row["_sources"] = "|".join(sources)
            row["_visual_hash"] = best.visual_hash
            row["_best_text"] = best.text
            row["_has_qr"] = bool(best.qr_payloads)
            row["_det_score"] = float(best.detection.score)

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
                fused = merge_field_values(r.get(col, "") for r in rows)
                if fused:
                    out[col] = fused
            fused_rows.append({col: out.get(col, "") for col in OUTPUT_COLUMNS})
        fused_rows.sort(key=lambda r: (str(r.get("filename", "")), int(float(r.get("frame_timestamp") or 0)), float(r.get("y_min") or 0), float(r.get("x_min") or 0)))
        return fused_rows

    def _track_row_passes_source_gate(self, row: Dict[str, object]) -> bool:
        sources = set(str(row.get("_sources", "")).split("|"))
        if "yolo" in sources:
            return True
        fallback_sources = {"heuristic", "red_white_tag", "qr_seed", "color_geometry"}
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
