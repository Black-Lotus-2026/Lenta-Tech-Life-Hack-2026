from pathlib import Path

import cv2
import numpy as np

from lenta_shelf_ai.detectors import _split_weight_paths
from lenta_shelf_ai.pipeline import PipelineConfig, PriceTagPipeline
from lenta_shelf_ai.qr import _default_wechat_model_paths


def test_split_weight_paths_accepts_multi_separator(monkeypatch):
    sep = ":"
    assert _split_weight_paths(f"a.pt{sep}b.pt,c.pt;d.pt\n") == ["a.pt", "b.pt", "c.pt", "d.pt"]


def test_split_weight_paths_preserves_windows_drive_prefixes():
    value = r"C:\models\a.pt;D:\weights\b.pt:/kaggle/model/c.pt"

    assert _split_weight_paths(value) == [
        r"C:\models\a.pt",
        r"D:\weights\b.pt",
        "/kaggle/model/c.pt",
    ]


def test_default_wechat_models_are_found_from_cwd(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "wechat_qr"
    model_dir.mkdir(parents=True)
    for name in ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]:
        (model_dir / name).write_text("x", encoding="utf-8")
    for env_name in [
        "LENTA_WECHAT_QR_DETECT_PROTOTXT",
        "LENTA_WECHAT_QR_DETECT_CAFFEMODEL",
        "LENTA_WECHAT_QR_SR_PROTOTXT",
        "LENTA_WECHAT_QR_SR_CAFFEMODEL",
    ]:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.chdir(tmp_path)
    paths = _default_wechat_model_paths()
    assert len(paths) == 4
    assert all(Path(path).exists() for path in paths)


def test_rectify_tag_crop_static_recovers_rotated_quad():
    image = np.zeros((220, 320, 3), dtype=np.uint8)
    quad = np.array([[70, 55], [250, 35], [270, 160], [55, 175]], dtype=np.int32)
    cv2.fillConvexPoly(image, quad, (245, 245, 245))
    cv2.rectangle(image, (105, 85), (170, 145), (20, 20, 220), -1)
    rectified, debug = PriceTagPipeline._rectify_tag_crop_static(image, min_area_ratio=0.10)
    assert debug["applied"] is True
    assert rectified.shape[0] >= 80
    assert rectified.shape[1] >= 140


def test_visual_qr_inference_is_opt_in(monkeypatch):
    cfg = PipelineConfig(enable_ocr=False, enable_qr=False, enable_field_zone_detector=False)
    pipe = PriceTagPipeline(cfg)
    row = {"barcode": "4690491121887", "price_default": "100.00", "price_card": "90.00"}
    monkeypatch.delenv("LENTA_INFER_QR_FROM_VISUAL", raising=False)
    pipe._infer_qr_fields_from_visual_evidence(row)
    assert row.get("qr_code_barcode", "") == ""
    monkeypatch.setenv("LENTA_INFER_QR_FROM_VISUAL", "1")
    pipe._infer_qr_fields_from_visual_evidence(row)
    assert row["qr_code_barcode"] == "4690491121887"
    assert row["price1_qr"] == "100.00"
    assert row["price4_qr"] == "90.00"
