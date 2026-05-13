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
from .utils import crop_xyxy, mkdir, sharpness_laplacian, read_yaml_or_json, normalize_text, iou_xyxy
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
            qr_payloads = decode_qr_payloads(crop)
            # Try whole-frame local area larger if crop failed.
            if not qr_payloads:
                bigger = det.expanded(w, h, px=0.35, py=0.30)
                qr_payloads = decode_qr_payloads(crop_xyxy(frame, bigger.xyxy, pad=4))
        ocr_lines = []
        if self.ocr is not None:
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
        image_quality = sharpness_laplacian(crop_xyxy(frame, det.xyxy, pad=int(self.config.crop_pad_px)))
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
                    "ocr_engines": sorted({line.engine for line in best.ocr_lines}),
                    "text": best.text[:600],
                    "qr_payloads": best.qr_payloads[:5],
                    "parsed": {col: best.parsed.get(col, "") for col in OUTPUT_COLUMNS if best.parsed.get(col, "")},
                }
            )
        return items

    def _tracks_to_rows(self, tracks: Iterable[Track]) -> List[Dict[str, object]]:
        # First fuse per tracker track.
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
                row[col] = merge_field_values(obs.parsed.get(col, "") for obs in tr.observations)
            # Keep coordinates/timestamp from best high-quality observation.
            row.update(best.to_row())
            for col in OUTPUT_COLUMNS:
                if col not in {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}:
                    fused = merge_field_values(obs.parsed.get(col, "") for obs in tr.observations)
                    if fused:
                        row[col] = fused
            candidate_rows.append(row)

        # Collapse duplicate tracks by stable IDs, especially barcode/QR.
        # If a detector fragment has no readable ID, merge only very close
        # spatial/temporal duplicates to avoid creating duplicate empty rows.
        groups: Dict[str, List[Dict[str, object]]] = {}
        ordered_rows = sorted(candidate_rows, key=lambda r: 0 if self._stable_group_key(r) else 1)
        for i, row in enumerate(ordered_rows):
            key = self._stable_group_key(row)
            if not key:
                for existing_key, existing_rows in groups.items():
                    if any(self._rows_spatial_duplicate(row, other) for other in existing_rows):
                        key = existing_key
                        break
                if not key:
                    key = f"track:{i}"
            groups.setdefault(key, []).append(row)

        fused_rows: List[Dict[str, object]] = []
        for _, rows in groups.items():
            # Choose representative with most non-empty fields.
            rep = max(rows, key=self._row_quality)
            out = dict(rep)
            for col in OUTPUT_COLUMNS:
                if col in {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}:
                    continue
                fused = merge_field_values(r.get(col, "") for r in rows)
                if fused:
                    out[col] = fused
            fused_rows.append(out)
        fused_rows.sort(key=lambda r: (str(r.get("filename", "")), int(float(r.get("frame_timestamp") or 0)), float(r.get("y_min") or 0), float(r.get("x_min") or 0)))
        return fused_rows

    @staticmethod
    def _non_absent(value: object) -> str:
        text = str(value or "").strip()
        return text if text and text != ABSENT_VALUE else ""

    def _stable_group_key(self, row: Dict[str, object]) -> str:
        for col in ["qr_code_barcode", "barcode"]:
            val = self._non_absent(row.get(col, ""))
            if val:
                return f"{col}:{val}"
        pn = normalize_text(str(row.get("product_name", ""))).lower()
        pr = str(row.get("price_card") or row.get("price_default") or "").strip()
        if pn and pr:
            return f"name_price:{pn[:80]}:{pr}"
        return ""

    def _row_quality(self, row: Dict[str, object]) -> float:
        field_score = sum(1 for c in OUTPUT_COLUMNS if self._non_absent(row.get(c, "")))
        id_bonus = 20 if self._stable_group_key(row) else 0
        try:
            area = max(0.0, float(row.get("x_max", 0)) - float(row.get("x_min", 0))) * max(0.0, float(row.get("y_max", 0)) - float(row.get("y_min", 0)))
        except Exception:
            area = 0.0
        return id_bonus + field_score * 10.0 + area * 0.0001

    def _rows_have_conflicting_ids(self, a: Dict[str, object], b: Dict[str, object]) -> bool:
        for col in ["qr_code_barcode", "barcode"]:
            av = self._non_absent(a.get(col, ""))
            bv = self._non_absent(b.get(col, ""))
            if av and bv and av != bv:
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
        try:
            box_a = [float(a[c]) for c in ["x_min", "y_min", "x_max", "y_max"]]
            box_b = [float(b[c]) for c in ["x_min", "y_min", "x_max", "y_max"]]
        except Exception:
            return False
        if iou_xyxy(box_a, box_b) >= float(self.config.dedupe_iou):
            return True

        wa, ha = max(1.0, box_a[2] - box_a[0]), max(1.0, box_a[3] - box_a[1])
        wb, hb = max(1.0, box_b[2] - box_b[0]), max(1.0, box_b[3] - box_b[1])
        y_overlap = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])) / max(1.0, min(ha, hb))
        height_ratio = min(ha, hb) / max(ha, hb)
        ax, ay = (box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0
        bx, by = (box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0
        return (
            y_overlap >= 0.70
            and height_ratio >= 0.60
            and abs(ax - bx) <= float(self.config.dedupe_center_threshold)
            and abs(ay - by) <= 0.35 * max(ha, hb)
            and min(wa, wb) / max(wa, wb) >= 0.55
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
