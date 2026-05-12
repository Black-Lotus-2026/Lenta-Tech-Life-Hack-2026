#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from lenta_shelf_ai.evaluation import evaluate_csv
from lenta_shelf_ai.pipeline import PipelineConfig, PriceTagPipeline


def main():
    ap = argparse.ArgumentParser(description="Run and evaluate on public labeled videos")
    ap.add_argument("--data-dir", default="data/Данные")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output-dir", default="outputs/eval_public")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig.from_file(args.config)
    pipe = PriceTagPipeline(cfg)
    metrics = {}
    for csv_path in sorted(data_dir.glob("*/*.csv")):
        video_path = csv_path.with_suffix(".mp4")
        if not video_path.exists():
            video_path = csv_path.parent / f"{csv_path.parent.name}.mp4"
        if not video_path.exists():
            continue
        pred_csv = out_dir / f"{video_path.stem}_recognized.csv"
        pipe.run_video(video_path, output_dir=out_dir / video_path.stem, output_csv=pred_csv)
        metrics[str(video_path.name)] = evaluate_csv(csv_path, pred_csv)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
