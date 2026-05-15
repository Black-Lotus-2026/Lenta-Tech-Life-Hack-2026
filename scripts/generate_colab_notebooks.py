#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = ROOT / "colab"


COMMON_SETUP = r'''
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

DATASET = "whitenigger/lenta-shelf-ai-bundle"
BUNDLE_DIR = Path("/content/lenta_bundle")
KAGGLE_INPUT = Path("/kaggle/input/lenta-shelf-ai-bundle")
KAGGLE_WORKING = Path("/kaggle/working")
RUN_TS = time.strftime("%Y%m%d_%H%M%S")
DRIVE_ROOT = Path("/content/drive/MyDrive/lenta_colab_runs")


def run(cmd, cwd=None, env=None, check=True):
    print("[RUN]", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=check)


def setup_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as exc:
        print("[WARN] Drive is not mounted:", repr(exc))
        return False


def setup_kaggle_credentials():
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    target = kaggle_dir / "kaggle.json"

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not (username and key):
        try:
            from google.colab import userdata
            username = username or userdata.get("KAGGLE_USERNAME")
            key = key or userdata.get("KAGGLE_KEY")
        except Exception:
            pass
    if username and key:
        target.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")
        target.chmod(0o600)
        print("[OK] Kaggle credentials loaded from env/Colab secrets")
        return

    candidates = list(Path("/content").glob("kaggle*.json"))
    if candidates:
        shutil.copy2(candidates[0], target)
        target.chmod(0o600)
        print("[OK] Kaggle credentials copied from", candidates[0])
        return

    from google.colab import files
    print("Upload kaggle.json now. The file stays inside this Colab runtime.")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("kaggle.json was not uploaded")
    src = Path("/content") / next(iter(uploaded.keys()))
    shutil.copy2(src, target)
    target.chmod(0o600)
    print("[OK] Kaggle credentials uploaded")


def prepare_bundle():
    run([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    setup_kaggle_credentials()
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    if not (BUNDLE_DIR / "repo.zip").exists() or not (BUNDLE_DIR / "data.zip").exists():
        run(["kaggle", "datasets", "download", "-d", DATASET, "-p", str(BUNDLE_DIR), "--unzip"])
    if not (BUNDLE_DIR / "repo.zip").exists() or not (BUNDLE_DIR / "data.zip").exists():
        raise FileNotFoundError(f"Expected repo.zip and data.zip in {BUNDLE_DIR}")

    KAGGLE_INPUT.mkdir(parents=True, exist_ok=True)
    KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUNDLE_DIR / "repo.zip", KAGGLE_INPUT / "repo.zip")
    shutil.copy2(BUNDLE_DIR / "data.zip", KAGGLE_INPUT / "data.zip")

    runner_root = Path("/content/lenta_runner_repo")
    if runner_root.exists():
        shutil.rmtree(runner_root)
    runner_root.mkdir(parents=True)
    with zipfile.ZipFile(BUNDLE_DIR / "repo.zip") as zf:
        zf.extractall(runner_root)
    script = runner_root / "repo" / "kaggle" / "gpu_experiment.py"
    if not script.exists():
        raise FileNotFoundError(script)
    print("[OK] runner:", script)
    return script


def run_experiment(script, exp_env):
    artifacts = KAGGLE_WORKING / "artifacts"
    if artifacts.exists():
        shutil.rmtree(artifacts)
    input_bundle = KAGGLE_WORKING / "input_bundle"
    if input_bundle.exists():
        shutil.rmtree(input_bundle)

    env = os.environ.copy()
    env.update({k: str(v) for k, v in exp_env.items()})
    run([sys.executable, str(script)], env=env)
    return artifacts


def run_error_analysis():
    work = Path("/tmp/lenta_shelf_ai_solution")
    eval_dir = work / "outputs" / "eval_public_fast"
    bundle = KAGGLE_WORKING / "input_bundle"
    if not eval_dir.exists() or not bundle.exists():
        print("[WARN] no eval outputs or input bundle for error analysis")
        return
    out_dir = KAGGLE_WORKING / "artifacts" / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_by_stem = {p.stem: p for p in bundle.glob("data/**/*/*.csv")}
    for pred_csv in sorted(eval_dir.glob("*_recognized.csv")):
        stem = pred_csv.name.replace("_recognized.csv", "")
        gt_csv = gt_by_stem.get(stem)
        if gt_csv is None:
            print("[WARN] no GT csv for", pred_csv)
            continue
        run([
            sys.executable,
            "scripts/analyze_final_errors.py",
            "--gt-csv",
            str(gt_csv),
            "--pred-csv",
            str(pred_csv),
            "--out-json",
            str(out_dir / f"{stem}_error_analysis.json"),
        ], cwd=work)


def save_artifacts(artifacts, experiment_name):
    run_dir = DRIVE_ROOT / f"{experiment_name}_{RUN_TS}"
    if DRIVE_ROOT.exists():
        if run_dir.exists():
            shutil.rmtree(run_dir)
        shutil.copytree(artifacts, run_dir / "artifacts")
        archive = shutil.make_archive(str(run_dir), "zip", run_dir)
        print("[OK] saved to Drive:", run_dir)
        print("[OK] zip:", archive)
        return run_dir
    local_dir = Path("/content") / f"{experiment_name}_{RUN_TS}"
    if local_dir.exists():
        shutil.rmtree(local_dir)
    shutil.copytree(artifacts, local_dir / "artifacts")
    archive = shutil.make_archive(str(local_dir), "zip", local_dir)
    print("[OK] saved locally:", local_dir)
    print("[OK] zip:", archive)
    return local_dir


def print_key_reports(artifacts):
    paths = [
        artifacts / "run_summary.json",
        artifacts / "kaggle_metrics" / "detection_recall_yolo_only.json",
        artifacts / "kaggle_metrics" / "detection_recall_hybrid.json",
        artifacts / "eval_public_fast" / "metrics.json",
    ]
    for path in paths:
        if not path.exists():
            print("[MISS]", path)
            continue
        print("\n" + "=" * 90)
        print(path)
        print("=" * 90)
        print(path.read_text(encoding="utf-8")[:12000])
'''


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(True),
    }


def markdown_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(True),
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "include_colab_link": True,
                "provenance": [],
            },
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def experiment_notebook(title: str, purpose: str, env: dict[str, str], critique: str) -> dict[str, object]:
    run_cell = (
        f"EXP_ENV = {json.dumps(env, indent=4)}\n"
        'EXPERIMENT_NAME = EXP_ENV["EXP_NAME"]\n\n'
        "setup_drive()\n"
        "script = prepare_bundle()\n"
        "artifacts = run_experiment(script, EXP_ENV)\n"
        "run_error_analysis()\n"
        "run_dir = save_artifacts(artifacts, EXPERIMENT_NAME)\n"
        "print_key_reports(artifacts)\n"
    )
    return notebook(
        [
            markdown_cell(
                f"""
                # {title}

                {purpose}

                Runtime: Google Colab GPU. Default guard requires one T4 GPU.
                If Colab gives L4/A100 instead, change `EXP_REQUIRE_GPU_NAME`.

                This notebook uses only local CV/OCR libraries inside Colab. It does not call cloud OCR/API/LLM during solution inference.
                """
            ),
            code_cell(
                """
                !nvidia-smi
                import sys
                print(sys.version)
                """
            ),
            code_cell(COMMON_SETUP),
            code_cell(run_cell),
            markdown_cell(
                f"""
                ## Required critique after run

                {critique}

                Minimum evidence to call this useful:
                - `eval_public_fast/metrics.json` has nonzero `good_rows_at_80`;
                - `26_12-20 pred_rows` moves closer to GT without collapsing `matched_rows`;
                - `field_fill` in error-analysis improves for prices, barcode, QR, SKU;
                - no fallback spam returns.
                """
            ),
        ]
    )


def compare_notebook() -> dict[str, object]:
    return notebook(
        [
            markdown_cell(
                """
                # Lenta Colab Runs Compare

                Reads artifacts saved by the experiment notebooks from `MyDrive/lenta_colab_runs`,
                builds a compact comparison table, and prints the next iteration targets.
                """
            ),
            code_cell(
                """
                import json
                from pathlib import Path

                import pandas as pd

                from google.colab import drive
                drive.mount("/content/drive")
                ROOT = Path("/content/drive/MyDrive/lenta_colab_runs")
                print(ROOT)
                """
            ),
            code_cell(
                """
                def read_json(path):
                    if not path.exists():
                        return {}
                    return json.loads(path.read_text(encoding="utf-8"))


                rows = []
                for run_dir in sorted(ROOT.glob("*")):
                    artifacts = run_dir / "artifacts"
                    if not artifacts.exists():
                        continue
                    summary = read_json(artifacts / "run_summary.json")
                    yolo = read_json(artifacts / "kaggle_metrics" / "detection_recall_yolo_only.json")
                    hybrid = read_json(artifacts / "kaggle_metrics" / "detection_recall_hybrid.json")
                    metrics = read_json(artifacts / "eval_public_fast" / "metrics.json")
                    row = {
                        "run": run_dir.name,
                        "experiment": summary.get("experiment_name", ""),
                        "elapsed_sec": summary.get("elapsed_sec", ""),
                        "yolo_recall": yolo.get("recall_at_iou_035", ""),
                        "hybrid_recall": hybrid.get("recall_at_iou_035", ""),
                    }
                    total_gt = total_pred = total_matched = total_good = 0
                    for video, m in metrics.items():
                        total_gt += int(m.get("gt_rows", 0))
                        total_pred += int(m.get("pred_rows", 0))
                        total_matched += int(m.get("matched_rows", 0))
                        total_good += int(m.get("good_rows_at_80", 0))
                        row[f"{video}_pred"] = m.get("pred_rows", "")
                        row[f"{video}_matched"] = m.get("matched_rows", "")
                        row[f"{video}_good80"] = m.get("good_rows_at_80", "")
                    row.update({
                        "gt_total": total_gt,
                        "pred_total": total_pred,
                        "matched_total": total_matched,
                        "good80_total": total_good,
                    })
                    rows.append(row)

                df = pd.DataFrame(rows)
                if df.empty:
                    print("No runs found. Run A/B/C notebooks first.")
                else:
                    display(df.sort_values(["good80_total", "matched_total"], ascending=False))
                """
            ),
            code_cell(
                """
                # Field-fill and duplicate/no-evidence diagnostics.
                diag_rows = []
                for run_dir in sorted(ROOT.glob("*")):
                    err_dir = run_dir / "artifacts" / "error_analysis"
                    for path in sorted(err_dir.glob("*_error_analysis.json")):
                        rep = read_json(path)
                        fill = rep.get("field_fill", {})
                        diag_rows.append({
                            "run": run_dir.name,
                            "video": path.name.replace("_error_analysis.json", ""),
                            "gt": rep.get("gt_rows"),
                            "pred": rep.get("pred_rows"),
                            "matched": rep.get("matched_rows"),
                            "no_evidence": len(rep.get("no_semantic_evidence_pred", [])),
                            "dup_clusters": len(rep.get("duplicate_clusters", [])),
                            "product_name": fill.get("product_name", 0),
                            "price_default": fill.get("price_default", 0),
                            "price_card": fill.get("price_card", 0),
                            "barcode": fill.get("barcode", 0),
                            "qr_code_barcode": fill.get("qr_code_barcode", 0),
                            "id_sku": fill.get("id_sku", 0),
                        })
                diag = pd.DataFrame(diag_rows)
                if diag.empty:
                    print("No error-analysis reports found.")
                else:
                    display(diag)
                """
            ),
            markdown_cell(
                """
                ## Harsh decision rule

                - If detector recall rises but `good80_total` stays zero, stop training detector and fix QR/OCR/parser/fusion.
                - If `pred_total` is above `1.35 * gt_total`, fallback/dedupe is still broken.
                - If `qr_code_barcode` remains near zero, QR cascade/layout crop is the next blocker.
                - If prices are still empty, zonal OCR/parser is still the first failure path.
                """
            ),
        ]
    )


def write_ipynb(name: str, data: dict[str, object]) -> None:
    COLAB_DIR.mkdir(parents=True, exist_ok=True)
    (COLAB_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    exp_a_env = {
        "EXP_NAME": "exp-colab-a-fallback-gated-noocr",
        "EXP_REQUIRE_GPU_NAME": "T4",
        "EXP_REQUIRE_GPU_COUNT": "1",
        "EXP_DEVICE": "0",
        "EXP_MODEL": "yolov8n.pt",
        "EXP_REQUIREMENTS": "requirements-train.txt",
        "EXP_SKIP_TORCH_PIN": "1",
        "EXP_SKIP_TRAIN": "1",
        "EXP_SKIP_PUBLIC_EVAL": "0",
        "EXP_RECALL_CONF": "0.08",
        "EXP_PIPELINE_ENABLE_OCR": "0",
        "EXP_PIPELINE_ENABLE_QR": "0",
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
    }
    exp_b_env = {
        "EXP_NAME": "exp-colab-b-zonal-qr-ocr-dedupe",
        "EXP_REQUIRE_GPU_NAME": "T4",
        "EXP_REQUIRE_GPU_COUNT": "1",
        "EXP_DEVICE": "0",
        "EXP_MODEL": "models/price_tag_yolo.pt",
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
        "EXP_PIPELINE_ENABLE_FALLBACKS": "0",
        "EXP_PIPELINE_YOLO_CONF": "0.12",
        "EXP_PIPELINE_SAMPLE_FPS": "4.0",
        "EXP_PIPELINE_MAX_DETECTIONS": "60",
        "EXP_PIPELINE_ENABLE_ZONAL_OCR": "1",
        "EXP_PIPELINE_QR_EXPANSION_X": "0.55",
        "EXP_PIPELINE_QR_EXPANSION_Y": "0.45",
        "EXP_PIPELINE_FALLBACK_REQUIRE_EVIDENCE": "1",
        "EXP_PIPELINE_DEDUPE_EXTENDED_WINDOW_MS": "12000",
        "EXP_PIPELINE_DEDUPE_VISUAL_HASH": "14",
        "EXP_PIPELINE_DEDUPE_TEXT_SIMILARITY": "0.86",
    }
    exp_c_env = {
        "EXP_NAME": "exp-colab-c-selftrain-detector",
        "EXP_REQUIRE_GPU_NAME": "T4",
        "EXP_REQUIRE_GPU_COUNT": "1",
        "EXP_DEVICE": "0",
        "EXP_MODEL": "models/price_tag_yolo.pt",
        "EXP_REQUIREMENTS": "requirements-train.txt",
        "EXP_SKIP_TORCH_PIN": "1",
        "EXP_SKIP_TRAIN": "1",
        "EXP_SELF_TRAIN": "1",
        "EXP_SELF_TRAIN_EPOCHS": "20",
        "EXP_PSEUDO_CONF": "0.62",
        "EXP_PSEUDO_SAMPLE_FPS": "1.5",
        "EXP_PSEUDO_MAX_FRAMES_PER_VIDEO": "180",
        "EXP_SKIP_PUBLIC_EVAL": "0",
        "EXP_RECALL_CONF": "0.10",
        "EXP_PIPELINE_ENABLE_OCR": "0",
        "EXP_PIPELINE_ENABLE_QR": "0",
        "EXP_PIPELINE_ENABLE_FALLBACKS": "0",
        "EXP_PIPELINE_YOLO_CONF": "0.10",
        "EXP_PIPELINE_SAMPLE_FPS": "4.0",
        "EXP_PIPELINE_MAX_DETECTIONS": "80",
        "EXP_PIPELINE_FALLBACK_REQUIRE_EVIDENCE": "1",
    }

    write_ipynb(
        "lenta_colab_exp_a_fallback_gated.ipynb",
        experiment_notebook(
            "Experiment A: fallback gated detector diagnostic",
            "Fast diagnostic for detector/fallback spam. OCR and QR are disabled deliberately.",
            exp_a_env,
            "If rows explode again, strict fallback gating is still insufficient. If rows collapse to zero, fallback evidence gate is too strict for no-OCR mode.",
        ),
    )
    write_ipynb(
        "lenta_colab_exp_b_zonal_qr_ocr.ipynb",
        experiment_notebook(
            "Experiment B: zonal OCR + QR cascade + dedupe",
            "Quality run for end-to-end CSV fields. This is the first Colab candidate for metric improvement.",
            exp_b_env,
            "If QR/barcode/price fill does not improve, the next fix is crop rectification and QR-zone localization, not detector training.",
        ),
    )
    write_ipynb(
        "lenta_colab_exp_c_selftrain_detector.ipynb",
        experiment_notebook(
            "Experiment C: self-training detector",
            "Detector improvement run using pseudo-labels from unlabeled videos. It is useful only if B shows detector misses are still a blocker.",
            exp_c_env,
            "If YOLO recall improves but final CSV does not, this branch is not the priority for top-1.",
        ),
    )
    write_ipynb("lenta_colab_compare_runs.ipynb", compare_notebook())


if __name__ == "__main__":
    main()
