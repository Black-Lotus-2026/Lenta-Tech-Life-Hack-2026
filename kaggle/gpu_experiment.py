from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd), env=env)


def validate_gpu_runtime() -> dict[str, object]:
    required_name = os.environ.get("EXP_REQUIRE_GPU_NAME", "T4")
    required_count = int(os.environ.get("EXP_REQUIRE_GPU_COUNT", "2"))
    if os.environ.get("EXP_SKIP_GPU_CHECK", "0") == "1":
        return {"skipped": True}

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        )
    except Exception as exc:
        raise RuntimeError("Cannot verify Kaggle GPU runtime with nvidia-smi") from exc

    names = [line.strip() for line in output.splitlines() if line.strip()]
    print("[GPU]", json.dumps(names), flush=True)
    matching = [name for name in names if required_name.lower() in name.lower()]
    if len(matching) < required_count:
        raise RuntimeError(
            f"Refusing to run: expected at least {required_count} GPU(s) containing "
            f"{required_name!r}, got {names}"
        )
    return {"names": names, "required_name": required_name, "required_count": required_count}


def newest_file(root: Path, pattern: str) -> Path:
    candidates = [p for p in root.glob(pattern) if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern!r} under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_bundle() -> Path:
    print("[INPUT]", [str(p) for p in sorted(Path("/kaggle/input").glob("*"))], flush=True)
    candidates = sorted(Path("/kaggle/input").glob("**/repo"))
    if not candidates:
        for dataset_dir in sorted(Path("/kaggle/input").glob("**")):
            repo_zip = dataset_dir / "repo.zip"
            data_zip = dataset_dir / "data.zip"
            if repo_zip.exists() and data_zip.exists():
                extracted = Path("/kaggle/working/input_bundle")
                if extracted.exists():
                    shutil.rmtree(extracted)
                extracted.mkdir(parents=True)
                with zipfile.ZipFile(repo_zip) as zf:
                    zf.extractall(extracted)
                with zipfile.ZipFile(data_zip) as zf:
                    zf.extractall(extracted)
                if (extracted / "repo").exists() and (extracted / "data").exists():
                    return extracted
        raise FileNotFoundError("Expected a Kaggle dataset with repo/data folders or repo.zip/data.zip files")
    return candidates[0].parent


def write_yaml(path: Path, data: dict[str, object]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def find_data_dir(bundle: Path) -> Path:
    expected = bundle / "data" / "Данные"
    if expected.exists():
        return expected
    roots: dict[Path, int] = {}
    for csv_path in bundle.glob("**/*.csv"):
        if "repo" in csv_path.parts:
            continue
        video_path = csv_path.with_suffix(".mp4")
        if video_path.exists():
            roots[csv_path.parent.parent] = roots.get(csv_path.parent.parent, 0) + 1
    if roots:
        return max(roots, key=roots.get)
    mp4s = [p for p in bundle.glob("**/*.mp4") if "repo" not in p.parts]
    if mp4s:
        return mp4s[0].parent.parent if mp4s[0].parent.name != "data" else mp4s[0].parent
    raise FileNotFoundError(f"Cannot infer data dir under {bundle}")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def detection_recall_at_labels(
    repo_dir: Path,
    data_dir: Path,
    weights: Path,
    out_path: Path,
    yolo_only: bool = False,
    conf: float = 0.18,
    imgsz: int = 1600,
) -> dict[str, object]:
    import pandas as pd

    from lenta_shelf_ai.detectors import HybridDetector, YOLODetector
    from lenta_shelf_ai.utils import iou_xyxy, smart_float
    from lenta_shelf_ai.video import read_frame_at_ms

    detector = YOLODetector(str(weights), conf=conf, imgsz=imgsz) if yolo_only else HybridDetector(yolo_weights=str(weights), yolo_conf=conf, imgsz=imgsz)
    per_video = {}
    misses = []
    total = 0
    matched = 0
    for csv_path in sorted(data_dir.glob("*/*.csv")):
        video_path = csv_path.with_suffix(".mp4")
        if not video_path.exists():
            continue
        df = pd.read_csv(csv_path).rename(columns={"wholesale_level_1_coun": "wholesale_level_1_count"})
        video_total = 0
        video_matched = 0
        det_counts = []
        for ts, group in df.groupby("frame_timestamp"):
            frame = read_frame_at_ms(video_path, smart_float(ts))
            if frame is None:
                continue
            detections = detector.predict(frame)
            det_counts.append(len(detections))
            for _, row in group.iterrows():
                box = [smart_float(row[c]) for c in ["x_min", "y_min", "x_max", "y_max"]]
                best_iou = max([iou_xyxy(box, det.xyxy) for det in detections] or [0.0])
                video_total += 1
                if best_iou >= 0.35:
                    video_matched += 1
                else:
                    misses.append(
                        {
                            "video": video_path.name,
                            "timestamp_ms": float(smart_float(ts)),
                            "bbox": [round(float(v), 2) for v in box],
                            "best_iou": round(float(best_iou), 4),
                            "detections_on_frame": len(detections),
                        }
                    )
        total += video_total
        matched += video_matched
        per_video[video_path.name] = {
            "matched": video_matched,
            "total": video_total,
            "recall_at_iou_035": video_matched / max(1, video_total),
            "avg_detections": sum(det_counts) / max(1, len(det_counts)),
        }
    metrics = {
        "detector_mode": "yolo_only" if yolo_only else "hybrid",
        "weights": str(weights),
        "conf": conf,
        "imgsz": imgsz,
        "matched": matched,
        "total": total,
        "recall_at_iou_035": matched / max(1, total),
        "per_video": per_video,
        "misses": misses,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def build_pseudo_yolo_dataset_inprocess(
    data_dir: Path,
    base_dataset: Path,
    out_dir: Path,
    weights: Path,
    sample_fps: float,
    conf: float,
    imgsz: int,
    max_frames_per_video: int,
) -> dict[str, object]:
    import cv2

    from lenta_shelf_ai.detectors import YOLODetector
    from scripts.build_yolo_dataset import yolo_line

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for rel in ["images/train", "images/val", "labels/train", "labels/val"]:
        src = base_dataset / rel
        dst = out_dir / rel
        if src.exists():
            shutil.copytree(src, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector(str(weights), conf=conf, imgsz=imgsz)
    images_train = out_dir / "images/train"
    labels_train = out_dir / "labels/train"
    summary: dict[str, object] = {
        "weights": str(weights),
        "sample_fps": sample_fps,
        "conf": conf,
        "imgsz": imgsz,
        "videos": {},
    }
    pseudo_frames = 0
    pseudo_boxes = 0
    for video_path in sorted((data_dir / "Unlabeled").glob("*.mp4")):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        step = max(1, int(round(fps / max(0.1, sample_fps))))
        frame_idx = 0
        kept_frames = 0
        kept_boxes = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            if max_frames_per_video and kept_frames >= max_frames_per_video:
                break
            detections = detector.predict(frame)
            if detections:
                h, w = frame.shape[:2]
                stem = f"pseudo_{video_path.stem}_{frame_idx:08d}"
                cv2.imwrite(str(images_train / f"{stem}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                with open(labels_train / f"{stem}.txt", "w", encoding="utf-8") as f:
                    for det in detections:
                        f.write(yolo_line(det.xyxy, w, h) + "\n")
                kept_frames += 1
                kept_boxes += len(detections)
            frame_idx += 1
        cap.release()
        pseudo_frames += kept_frames
        pseudo_boxes += kept_boxes
        summary["videos"][video_path.name] = {
            "frames": kept_frames,
            "boxes": kept_boxes,
            "source_fps": fps,
            "step": step,
        }
    summary["pseudo_frames"] = pseudo_frames
    summary["pseudo_boxes"] = pseudo_boxes
    (out_dir / "pseudo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: price_tag\n",
        encoding="utf-8",
    )
    print("[PSEUDO]", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    start = time.time()
    gpu_runtime = validate_gpu_runtime()
    bundle = find_bundle()
    work = Path("/tmp/lenta_shelf_ai_solution")
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(bundle / "repo", work)
    sys.path.insert(0, str(work))

    data_dir = find_data_dir(bundle)
    print(f"[DATA] {data_dir}", flush=True)

    requirements_file = os.environ.get("EXP_REQUIREMENTS", "requirements-full.txt")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "False")
    os.environ.setdefault("FLAGS_use_onednn", "False")
    os.environ.setdefault("FLAGS_use_dnnl", "False")
    if os.environ.get("EXP_SKIP_TORCH_PIN", "0") != "1":
        torch_spec = os.environ.get("EXP_TORCH_SPEC", "torch==2.5.1+cu121 torchvision==0.20.1+cu121")
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
                *torch_spec.split(),
            ],
            cwd=work,
        )
    run([sys.executable, "-m", "pip", "install", "-q", "-r", requirements_file], cwd=work)
    if env_bool("EXP_REQUIRE_PADDLE", False):
        run(
            [
                sys.executable,
                "-c",
                (
                    "import paddle, paddleocr; "
                    "print('[PADDLE]', paddle.__version__, paddleocr.__version__, flush=True)"
                ),
            ],
            cwd=work,
        )
    run([sys.executable, "-m", "pytest", "-q"], cwd=work)

    model_dir = work / "models"
    model_dir.mkdir(exist_ok=True)
    final_weights = model_dir / "price_tag_yolo.pt"
    propagate = int(os.environ.get("EXP_PROPAGATE", "10"))
    epochs = int(os.environ.get("EXP_EPOCHS", "80"))
    imgsz = int(os.environ.get("EXP_IMGSZ", "1280"))
    batch = int(os.environ.get("EXP_BATCH", "4"))
    base_model = os.environ.get("EXP_MODEL", "yolo11n.pt")
    device = os.environ.get("EXP_DEVICE", "0,1")
    experiment_name = os.environ.get("EXP_NAME", "manual")
    best = final_weights

    if env_bool("EXP_SKIP_TRAIN", False):
        if not final_weights.exists():
            raise FileNotFoundError(f"EXP_SKIP_TRAIN=1 but {final_weights} is missing from the repo bundle")
        print(f"[SKIP] training disabled, using bundled weights: {final_weights}", flush=True)
    else:
        run(
            [
                sys.executable,
                "scripts/build_yolo_dataset.py",
                "--data-dir",
                str(data_dir),
                "--out-dir",
                "datasets/lenta_yolo",
                "--propagate",
                str(propagate),
            ],
            cwd=work,
        )
        run(
            [
                sys.executable,
                "scripts/train_detector.py",
                "--data",
                "datasets/lenta_yolo/data.yaml",
                "--model",
                base_model,
                "--epochs",
                str(epochs),
                "--imgsz",
                str(imgsz),
                "--batch",
                str(batch),
                "--device",
                device,
            ],
            cwd=work,
        )

        best = newest_file(work, "runs/**/weights/best.pt")
        shutil.copy2(best, final_weights)
    self_training_summary = None
    if env_bool("EXP_SELF_TRAIN", False):
        pseudo_dir = Path(os.environ.get("EXP_SELF_TRAIN_DATASET", "datasets/lenta_yolo_self"))
        pseudo_conf = os.environ.get("EXP_PSEUDO_CONF", "0.65")
        pseudo_sample_fps = os.environ.get("EXP_PSEUDO_SAMPLE_FPS", "1.0")
        pseudo_max_frames = os.environ.get("EXP_PSEUDO_MAX_FRAMES_PER_VIDEO", "0")
        self_epochs = os.environ.get("EXP_SELF_TRAIN_EPOCHS", "12")
        self_training_summary = build_pseudo_yolo_dataset_inprocess(
            data_dir=data_dir,
            base_dataset=work / "datasets/lenta_yolo",
            out_dir=work / pseudo_dir,
            weights=final_weights,
            sample_fps=float(pseudo_sample_fps),
            conf=float(pseudo_conf),
            imgsz=int(imgsz),
            max_frames_per_video=int(pseudo_max_frames),
        )
        run(
            [
                sys.executable,
                "scripts/train_detector.py",
                "--data",
                str(pseudo_dir / "data.yaml"),
                "--model",
                str(final_weights),
                "--epochs",
                self_epochs,
                "--imgsz",
                str(imgsz),
                "--batch",
                str(batch),
                "--device",
                device,
                "--name",
                "price_tag_yolo_selftrain",
            ],
            cwd=work,
        )
        best = newest_file(work, "runs/**/weights/best.pt")
        shutil.copy2(best, final_weights)

    cfg_path = work / "configs/kaggle_fast.yaml"
    pipeline_enable_ocr = env_bool("EXP_PIPELINE_ENABLE_OCR", False)
    pipeline_enable_qr = env_bool("EXP_PIPELINE_ENABLE_QR", True)
    pipeline_defer_ocr = env_bool("EXP_PIPELINE_DEFER_OCR", False)
    pipeline_prefer_paddle = env_bool("EXP_PIPELINE_PREFER_PADDLE", False)
    pipeline_use_gpu = env_bool("EXP_PIPELINE_USE_GPU", False)
    pipeline_enable_fallbacks = env_bool("EXP_PIPELINE_ENABLE_FALLBACKS", False)
    pipeline_conf = float(os.environ.get("EXP_PIPELINE_YOLO_CONF", os.environ.get("EXP_RECALL_CONF", "0.18")))
    pipeline_imgsz = int(os.environ.get("EXP_PIPELINE_DETECTOR_IMGSZ", "1600"))
    pipeline_sample_fps = float(os.environ.get("EXP_PIPELINE_SAMPLE_FPS", "2.0"))
    pipeline_max_detections = int(os.environ.get("EXP_PIPELINE_MAX_DETECTIONS", "120"))
    pipeline_max_frames = int(os.environ.get("EXP_PIPELINE_MAX_FRAMES", "0"))
    pipeline_save_crops = env_bool("EXP_PIPELINE_SAVE_CROPS", False)
    pipeline_enable_zonal_ocr = env_bool("EXP_PIPELINE_ENABLE_ZONAL_OCR", True)
    pipeline_qr_expansion_x = float(os.environ.get("EXP_PIPELINE_QR_EXPANSION_X", "0.55"))
    pipeline_qr_expansion_y = float(os.environ.get("EXP_PIPELINE_QR_EXPANSION_Y", "0.45"))
    pipeline_dedupe_visual_hash = int(os.environ.get("EXP_PIPELINE_DEDUPE_VISUAL_HASH", "14"))
    pipeline_dedupe_text_similarity = float(os.environ.get("EXP_PIPELINE_DEDUPE_TEXT_SIMILARITY", "0.86"))
    pipeline_dedupe_extended_window = int(os.environ.get("EXP_PIPELINE_DEDUPE_EXTENDED_WINDOW_MS", "12000"))
    pipeline_dedupe_row_y_threshold = float(os.environ.get("EXP_PIPELINE_DEDUPE_ROW_Y_THRESHOLD", "0.55"))
    pipeline_fallback_min_observations = int(os.environ.get("EXP_PIPELINE_FALLBACK_MIN_OBSERVATIONS", "3"))
    pipeline_fallback_require_evidence = env_bool("EXP_PIPELINE_FALLBACK_REQUIRE_EVIDENCE", True)
    representative_temporal_weight = float(os.environ.get("EXP_REPRESENTATIVE_TEMPORAL_WEIGHT", "0.0"))
    recall_conf = float(os.environ.get("EXP_RECALL_CONF", "0.18"))
    recall_imgsz = int(os.environ.get("EXP_RECALL_IMGSZ", "1600"))

    write_yaml(
        cfg_path,
        {
            "sample_fps": pipeline_sample_fps,
            "yolo_weights": str(final_weights),
            "yolo_conf": pipeline_conf,
            "detector_imgsz": pipeline_imgsz,
            "enable_fallback_detectors": pipeline_enable_fallbacks,
            "min_sharpness": 18.0,
            "max_frames": pipeline_max_frames,
            "max_detections_per_frame": pipeline_max_detections,
            "enable_ocr": pipeline_enable_ocr,
            "enable_qr": pipeline_enable_qr,
            "defer_ocr": pipeline_defer_ocr,
            "prefer_paddle": pipeline_prefer_paddle,
            "ocr_lang": "ru",
            "use_gpu": pipeline_use_gpu,
            "crop_pad_px": 8,
            "tracker_iou": 0.12,
            "tracker_center_threshold": 280.0,
            "max_lost": 8,
            "min_track_observations": 1,
            "dedupe_iou": 0.30,
            "dedupe_center_threshold": 90.0,
            "dedupe_time_window_ms": 1600,
            "representative_temporal_weight": representative_temporal_weight,
            "save_crops": pipeline_save_crops,
            "save_debug_json": True,
            "enable_zonal_ocr": pipeline_enable_zonal_ocr,
            "qr_expansion_x": pipeline_qr_expansion_x,
            "qr_expansion_y": pipeline_qr_expansion_y,
            "dedupe_visual_hash_threshold": pipeline_dedupe_visual_hash,
            "dedupe_text_similarity": pipeline_dedupe_text_similarity,
            "dedupe_extended_time_window_ms": pipeline_dedupe_extended_window,
            "dedupe_row_y_threshold_ratio": pipeline_dedupe_row_y_threshold,
            "fallback_min_observations": pipeline_fallback_min_observations,
            "fallback_require_evidence": pipeline_fallback_require_evidence,
        },
    )

    metrics_dir = work / "outputs/kaggle_metrics"
    det_metrics = detection_recall_at_labels(
        work,
        data_dir,
        final_weights,
        metrics_dir / "detection_recall_hybrid.json",
        conf=recall_conf,
        imgsz=recall_imgsz,
    )
    det_metrics_yolo = detection_recall_at_labels(
        work,
        data_dir,
        final_weights,
        metrics_dir / "detection_recall_yolo_only.json",
        yolo_only=True,
        conf=recall_conf,
        imgsz=recall_imgsz,
    )
    print("[METRICS] detection hybrid", json.dumps(det_metrics, ensure_ascii=False, indent=2), flush=True)
    print("[METRICS] detection yolo_only", json.dumps(det_metrics_yolo, ensure_ascii=False, indent=2), flush=True)

    if os.environ.get("EXP_SKIP_PUBLIC_EVAL", "0") == "1":
        print("[SKIP] public pipeline eval disabled for fast detector experiment", flush=True)
    else:
        run(
            [
                sys.executable,
                "scripts/evaluate_on_public.py",
                "--data-dir",
                str(data_dir),
                "--config",
                str(cfg_path),
                "--output-dir",
                "outputs/eval_public_fast",
            ],
            cwd=work,
        )

    artifacts = Path("/kaggle/working/artifacts")
    artifacts.mkdir(exist_ok=True)
    results_csv = None
    try:
        results_csv = newest_file(work, "runs/**/results.csv")
    except FileNotFoundError:
        pass
    for rel in ["models/price_tag_yolo.pt", "outputs/kaggle_metrics", "outputs/eval_public_fast"]:
        src = work / rel
        if src.is_dir():
            dst = artifacts / src.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        elif src.exists():
            shutil.copy2(src, artifacts / src.name)
    if results_csv is not None:
        shutil.copy2(results_csv, artifacts / "train_results.csv")
    summary = {
        "elapsed_sec": round(time.time() - start, 1),
        "experiment_name": experiment_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "propagate": propagate,
        "base_model": base_model,
        "device": device,
        "requirements_file": requirements_file,
        "torch_spec": os.environ.get("EXP_TORCH_SPEC", "torch==2.5.1+cu121 torchvision==0.20.1+cu121"),
        "self_training": self_training_summary,
        "recall_conf": recall_conf,
        "recall_imgsz": recall_imgsz,
        "pipeline_config": {
            "enable_ocr": pipeline_enable_ocr,
            "enable_qr": pipeline_enable_qr,
            "defer_ocr": pipeline_defer_ocr,
            "prefer_paddle": pipeline_prefer_paddle,
            "use_gpu": pipeline_use_gpu,
            "enable_fallback_detectors": pipeline_enable_fallbacks,
            "yolo_conf": pipeline_conf,
            "detector_imgsz": pipeline_imgsz,
            "sample_fps": pipeline_sample_fps,
            "max_frames": pipeline_max_frames,
            "max_detections_per_frame": pipeline_max_detections,
            "save_crops": pipeline_save_crops,
            "representative_temporal_weight": representative_temporal_weight,
            "enable_zonal_ocr": pipeline_enable_zonal_ocr,
            "qr_expansion_x": pipeline_qr_expansion_x,
            "qr_expansion_y": pipeline_qr_expansion_y,
            "dedupe_visual_hash_threshold": pipeline_dedupe_visual_hash,
            "dedupe_text_similarity": pipeline_dedupe_text_similarity,
            "dedupe_extended_time_window_ms": pipeline_dedupe_extended_window,
            "dedupe_row_y_threshold_ratio": pipeline_dedupe_row_y_threshold,
            "fallback_min_observations": pipeline_fallback_min_observations,
            "fallback_require_evidence": pipeline_fallback_require_evidence,
        },
        "gpu_runtime": gpu_runtime,
        "best_weights": str(final_weights),
        "source_best_weights": str(best),
        "detection_recall": det_metrics,
        "detection_recall_yolo_only": det_metrics_yolo,
    }
    (artifacts / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE]", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
