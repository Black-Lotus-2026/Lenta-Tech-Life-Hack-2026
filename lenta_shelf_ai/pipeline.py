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
from .utils import crop_xyxy, mkdir, sharpness_laplacian, read_yaml_or_json, normalize_text
from .video import iter_video_frames, get_video_meta

@dataclass
class PipelineConfig:
    sample_fps: float = 2.0
    yolo_weights: str = "models/price_tag_yolo.pt"
    yolo_conf: float = 0.23
    detector_imgsz: int = 1600
    min_sharpness: float = 18.0
    max_frames: int = 0
    max_detections_per_frame: int = 80
    enable_ocr: bool = True
    enable_qr: bool = True
    prefer_paddle: bool = True
    ocr_lang: str = "ru"
    use_gpu: bool = False
    crop_pad_px: int = 8
    tracker_iou: float = 0.12
    tracker_center_threshold: float = 250.0
    max_lost: int = 5
    min_track_observations: int = 1
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
        self.detector = HybridDetector(yolo_weights=weights, yolo_conf=self.config.yolo_conf, imgsz=int(self.config.detector_imgsz))
        self.ocr = None
        if self.config.enable_ocr:
            self.ocr = EnsembleOCREngine(prefer_paddle=bool(self.config.prefer_paddle), lang=str(self.config.ocr_lang), use_gpu=bool(self.config.use_gpu))

    def _process_detection(self, filename: str, timestamp_ms: int, frame: np.ndarray, det: Detection, output_dir: Optional[Path], crop_idx: int) -> TagObservation:
        h, w = frame.shape[:2]
        det = det.expanded(w, h, px=0.05, py=0.06).clamp(w, h)
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
        obs = TagObservation(
            filename=filename,
            timestamp_ms=int(timestamp_ms),
            detection=det,
            text=text,
            ocr_lines=ocr_lines,
            qr_payloads=qr_payloads,
            parsed=parsed,
            image_quality=sharpness_laplacian(crop),
        )
        if output_dir is not None and self.config.save_crops:
            crop_dir = mkdir(output_dir / "crops")
            cv2.imwrite(str(crop_dir / f"{Path(filename).stem}_{timestamp_ms}_{crop_idx:03d}.jpg"), crop)
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
                    obs = self._process_detection(filename, packet.timestamp_ms, packet.frame_bgr, det, out_dir, i)
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
        rows = self._tracks_to_rows(tracker.active_and_finished_tracks())
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
            with open(out_dir / f"{video_path.stem}_debug.json", "w", encoding="utf-8") as f:
                json.dump(debug, f, ensure_ascii=False, indent=2)
        return df

    def _tracks_to_rows(self, tracks: Iterable[Track]) -> List[Dict[str, object]]:
        # First fuse per tracker track.
        candidate_rows: List[Dict[str, object]] = []
        for tr in tracks:
            if len(tr.observations) < int(self.config.min_track_observations):
                continue
            best = tr.best_observation
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
        groups: Dict[str, List[Dict[str, object]]] = {}
        for i, row in enumerate(candidate_rows):
            key = ""
            for col in ["qr_code_barcode", "barcode"]:
                val = str(row.get(col, "")).strip()
                if val and val != ABSENT_VALUE:
                    key = f"{col}:{val}"
                    break
            if not key:
                pn = normalize_text(str(row.get("product_name", ""))).lower()
                pr = str(row.get("price_card") or row.get("price_default") or "").strip()
                if pn and pr:
                    key = f"name_price:{pn[:80]}:{pr}"
                else:
                    key = f"track:{i}"
            groups.setdefault(key, []).append(row)

        fused_rows: List[Dict[str, object]] = []
        for _, rows in groups.items():
            # Choose representative with most non-empty fields.
            rep = max(rows, key=lambda r: sum(1 for c in OUTPUT_COLUMNS if str(r.get(c, "")).strip()))
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Lenta Shelf AI video-to-CSV pipeline")
    parser.add_argument("video", help="Path to input .mp4")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML/JSON config")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--output-csv", default=None, help="Output CSV path")
    parser.add_argument("--sample-fps", type=float, default=None, help="Override frame sampling FPS")
    parser.add_argument("--weights", default=None, help="YOLO weights path")
    parser.add_argument("--fast", action="store_true", help="Disable OCR and use QR/color detector only")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug/quick run: process at most N sampled frames")
    args = parser.parse_args(argv)

    cfg = PipelineConfig.from_file(args.config)
    if args.sample_fps is not None:
        cfg.sample_fps = args.sample_fps
    if args.weights is not None:
        cfg.yolo_weights = args.weights
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
