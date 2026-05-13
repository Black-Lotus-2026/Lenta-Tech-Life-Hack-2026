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


def test_full_ocr_kernel_requires_paddle_and_uses_cpu_compatible_runtime() -> None:
    defaults = _defaults("kernel_exp_b")

    assert defaults["EXP_REQUIREMENTS"] == "requirements-full.txt"
    assert defaults["EXP_PIPELINE_ENABLE_OCR"] == "1"
    assert defaults["EXP_PIPELINE_PREFER_PADDLE"] == "1"
    assert defaults["EXP_PIPELINE_USE_GPU"] == "0"
    assert defaults["EXP_REQUIRE_PADDLE"] == "1"


def test_full_requirements_pin_python312_compatible_paddleocr_stack() -> None:
    requirements = Path("requirements-full.txt").read_text(encoding="utf-8")

    assert "ultralytics" in requirements
    assert "paddleocr==3.3.3" in requirements
    assert "paddlepaddle==3.2.0" in requirements
    assert "langchain-text-splitters" in requirements


def test_kaggle_experiment_disables_unstable_paddle_cpu_mkldnn_path() -> None:
    source = Path("kaggle/gpu_experiment.py").read_text(encoding="utf-8")

    assert "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT" in source
    assert "FLAGS_use_mkldnn" in source
    assert "FLAGS_use_onednn" in source
