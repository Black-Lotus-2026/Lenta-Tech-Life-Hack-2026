#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle" / "gpu_experiment.py"
OUT_ROOT = ROOT / "kaggle_work"
DATASET = "whitenigger/lenta-shelf-ai-bundle"
ACCELERATOR = "NvidiaTeslaT4"


EXPERIMENTS = [
    {
        "dir": "kernel_exp_a",
        "id": "whitenigger/lenta-shelf-ai-exp-a",
        "title": "Lenta Shelf AI Exp A",
        "defaults": {
            "EXP_NAME": "exp-a-yolov8n-fallback-gated-4fps-conf008",
            "EXP_REQUIRE_GPU_NAME": "T4",
            "EXP_REQUIRE_GPU_COUNT": "2",
            "EXP_DEVICE": "0",
            "EXP_MODEL": "yolov8n.pt",
            "EXP_EPOCHS": "0",
            "EXP_IMGSZ": "1280",
            "EXP_BATCH": "4",
            "EXP_PROPAGATE": "14",
            "EXP_REQUIREMENTS": "requirements-train.txt",
            "EXP_SKIP_TORCH_PIN": "1",
            "EXP_SKIP_TRAIN": "1",
            "EXP_SKIP_PUBLIC_EVAL": "0",
            "EXP_RECALL_CONF": "0.08",
            "EXP_PIPELINE_ENABLE_OCR": "0",
            "EXP_PIPELINE_ENABLE_QR": "0",
            "EXP_PIPELINE_PREFER_PADDLE": "0",
            "EXP_PIPELINE_USE_GPU": "0",
            "EXP_PIPELINE_ENABLE_FALLBACKS": "1",
            "EXP_PIPELINE_YOLO_CONF": "0.08",
            "EXP_PIPELINE_SAMPLE_FPS": "4.0",
            "EXP_PIPELINE_MAX_DETECTIONS": "120",
            "EXP_PIPELINE_ENABLE_ZONAL_OCR": "0",
            "EXP_PIPELINE_FALLBACK_MIN_OBSERVATIONS": "3",
            "EXP_PIPELINE_FALLBACK_REQUIRE_EVIDENCE": "1",
            "EXP_PIPELINE_DEDUPE_EXTENDED_WINDOW_MS": "15000",
            "EXP_PIPELINE_DEDUPE_VISUAL_HASH": "12",
            "EXP_PIPELINE_DEDUPE_TEXT_SIMILARITY": "0.82",
            "EXP_REPRESENTATIVE_TEMPORAL_WEIGHT": "0.0",
        },
    },
    {
        "dir": "kernel_exp_b",
        "id": "whitenigger/lenta-shelf-ai-exp-b",
        "title": "Lenta Shelf AI Exp B",
        "defaults": {
            "EXP_NAME": "exp-b-yolov8n-zonalqr-dedupe-4fps",
            "EXP_REQUIRE_GPU_NAME": "T4",
            "EXP_REQUIRE_GPU_COUNT": "2",
            "EXP_DEVICE": "0",
            "EXP_MODEL": "models/price_tag_yolo.pt",
            "EXP_EPOCHS": "0",
            "EXP_IMGSZ": "1280",
            "EXP_BATCH": "4",
            "EXP_PROPAGATE": "14",
            "EXP_REQUIREMENTS": "requirements-full.txt",
            "EXP_SKIP_TORCH_PIN": "1",
            "EXP_SKIP_TRAIN": "1",
            "EXP_SKIP_PUBLIC_EVAL": "0",
            "EXP_RECALL_CONF": "0.12",
            "EXP_REQUIRE_PADDLE": "1",
            "EXP_PIPELINE_ENABLE_OCR": "1",
            "EXP_PIPELINE_ENABLE_QR": "1",
            "EXP_PIPELINE_DEFER_OCR": "1",
            "EXP_PIPELINE_PREFER_PADDLE": "1",
            "EXP_PIPELINE_USE_GPU": "0",
            "EXP_PIPELINE_YOLO_CONF": "0.12",
            "EXP_PIPELINE_SAMPLE_FPS": "4.0",
            "EXP_PIPELINE_MAX_DETECTIONS": "60",
            "EXP_PIPELINE_SAVE_CROPS": "0",
            "EXP_PIPELINE_ENABLE_ZONAL_OCR": "1",
            "EXP_PIPELINE_QR_EXPANSION_X": "0.55",
            "EXP_PIPELINE_QR_EXPANSION_Y": "0.45",
            "EXP_PIPELINE_FALLBACK_MIN_OBSERVATIONS": "3",
            "EXP_PIPELINE_FALLBACK_REQUIRE_EVIDENCE": "1",
            "EXP_PIPELINE_DEDUPE_EXTENDED_WINDOW_MS": "12000",
            "EXP_PIPELINE_DEDUPE_VISUAL_HASH": "14",
            "EXP_PIPELINE_DEDUPE_TEXT_SIMILARITY": "0.86",
            "EXP_REPRESENTATIVE_TEMPORAL_WEIGHT": "0.0",
        },
    },
]


def inject_defaults(source: str, defaults: dict[str, str]) -> str:
    marker = "from pathlib import Path\n\n\n"
    if marker not in source:
        raise RuntimeError("Cannot find import marker for default injection")
    block = (
        "from pathlib import Path\n\n\n"
        f"EXPERIMENT_DEFAULTS = {json.dumps(defaults, ensure_ascii=False, indent=4)}\n\n\n"
        "for _key, _value in EXPERIMENT_DEFAULTS.items():\n"
        "    os.environ.setdefault(_key, str(_value))\n"
        "print('[EXP_DEFAULTS]', json.dumps(EXPERIMENT_DEFAULTS, ensure_ascii=False), flush=True)\n\n\n"
    )
    return source.replace(marker, block, 1)


def write_kernel(exp: dict[str, object]) -> Path:
    out_dir = OUT_ROOT / str(exp["dir"])
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    source = SOURCE.read_text(encoding="utf-8")
    (out_dir / "gpu_experiment.py").write_text(
        inject_defaults(source, exp["defaults"]), encoding="utf-8"
    )
    metadata = {
        "id": exp["id"],
        "title": exp["title"],
        "code_file": "gpu_experiment.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [],
        "machine_shape": ACCELERATOR,
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_dir


def main() -> None:
    for exp in EXPERIMENTS:
        out_dir = write_kernel(exp)
        print(out_dir)


if __name__ == "__main__":
    main()
