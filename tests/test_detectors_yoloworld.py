from __future__ import annotations

import sys
import types

import numpy as np

from lenta_shelf_ai.detectors import YOLOWorldPromptDetector
from lenta_shelf_ai.pipeline import PriceTagPipeline


class _FakeBoxes:
    xyxy = np.array([
        [20, 20, 120, 80],      # plausible price tag
        [0, 0, 500, 500],       # too large, gated out
        [10, 10, 16, 16],       # too small, gated out
    ], dtype=float)
    conf = np.array([0.90, 0.99, 0.88], dtype=float)


class _FakeResult:
    boxes = _FakeBoxes()


class _FakeYOLOWorld:
    def __init__(self, weights):
        self.weights = weights
        self.classes = None

    def set_classes(self, prompts):
        self.classes = list(prompts)

    def predict(self, **kwargs):
        return [_FakeResult()]


def test_yoloworld_prompt_detector_is_size_gated(monkeypatch, tmp_path) -> None:
    fake_mod = types.SimpleNamespace(YOLOWorld=_FakeYOLOWorld)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_mod)
    weight = tmp_path / "world.pt"
    weight.write_bytes(b"fake")

    detector = YOLOWorldPromptDetector(weight, prompts=["small rectangular price sticker on shelf"], imgsz=640)
    detections = detector.predict(np.zeros((400, 600, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].source == "yolo_world"
    assert detector.model.classes == ["small rectangular price sticker on shelf"]


def test_yoloworld_tracks_are_not_trusted_without_semantic_evidence() -> None:
    pipe = PriceTagPipeline.__new__(PriceTagPipeline)
    pipe.config = type("Cfg", (), {"fallback_require_evidence": True, "fallback_min_observations": 3})()

    assert pipe._track_row_passes_source_gate({"_sources": "yolo:trained", "_observations": 1})
    assert not pipe._track_row_passes_source_gate({"_sources": "yolo_world:generic_open_vocab", "_observations": 1})
