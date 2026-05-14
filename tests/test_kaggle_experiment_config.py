from __future__ import annotations

from pathlib import Path

from scripts.prepare_kaggle_experiment_kernels import EXPERIMENTS


def _defaults(kernel_dir: str) -> dict[str, str]:
    for exp in EXPERIMENTS:
        if exp["dir"] == kernel_dir:
            return exp["defaults"]
    raise AssertionError(f"missing experiment {kernel_dir}")


def test_detector_diagnostic_kernel_disables_qr_to_avoid_native_crashes() -> None:
    defaults = _defaults("kernel_exp_a")

    assert defaults["EXP_PIPELINE_ENABLE_OCR"] == "0"
    assert defaults["EXP_PIPELINE_ENABLE_QR"] == "0"
    assert defaults["EXP_PIPELINE_ENABLE_FALLBACKS"] == "1"
    assert defaults["EXP_PIPELINE_FALLBACK_REQUIRE_EVIDENCE"] == "1"
    assert defaults["EXP_PIPELINE_DEDUPE_EXTENDED_WINDOW_MS"] == "15000"


def test_full_ocr_kernel_requires_paddle_and_uses_cpu_compatible_runtime() -> None:
    defaults = _defaults("kernel_exp_b")

    assert defaults["EXP_REQUIREMENTS"] == "requirements-full.txt"
    assert defaults["EXP_PIPELINE_ENABLE_OCR"] == "1"
    assert defaults["EXP_PIPELINE_DEFER_OCR"] == "1"
    assert defaults["EXP_PIPELINE_PREFER_PADDLE"] == "1"
    assert defaults["EXP_PIPELINE_USE_GPU"] == "0"
    assert defaults["EXP_REQUIRE_PADDLE"] == "1"
    assert defaults["EXP_PIPELINE_ENABLE_ZONAL_OCR"] == "1"
    assert defaults["EXP_PIPELINE_QR_EXPANSION_X"] == "0.55"
    assert defaults["EXP_PIPELINE_DEDUPE_TEXT_SIMILARITY"] == "0.86"


def test_full_requirements_pin_python312_compatible_paddleocr_stack() -> None:
    requirements = Path("requirements-full.txt").read_text(encoding="utf-8")

    assert "ultralytics" in requirements
    assert "paddleocr==3.3.3" in requirements
    assert "paddlepaddle==3.2.0" in requirements
    assert "langchain==0.3.27" in requirements
    assert "langchain-text-splitters" in requirements
    assert "rapidfuzz>=3.9.0" in requirements


def test_kaggle_experiment_disables_unstable_paddle_cpu_mkldnn_path() -> None:
    source = Path("kaggle/gpu_experiment.py").read_text(encoding="utf-8")

    assert "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT" in source
    assert "FLAGS_use_mkldnn" in source
    assert "FLAGS_use_onednn" in source


def test_kaggle_experiment_logs_hardened_pipeline_knobs() -> None:
    source = Path("kaggle/gpu_experiment.py").read_text(encoding="utf-8")

    assert "EXP_PIPELINE_ENABLE_ZONAL_OCR" in source
    assert "EXP_PIPELINE_QR_EXPANSION_X" in source
    assert "EXP_PIPELINE_DEDUPE_TEXT_SIMILARITY" in source
    assert "EXP_PIPELINE_FALLBACK_REQUIRE_EVIDENCE" in source
