from __future__ import annotations

import cv2
import numpy as np

from scripts.build_yolo_dataset import read_frame_at_ms_from_cap
from scripts.build_yolo_dataset import is_validation_sample


def test_validation_split_is_stable_for_propagated_timestamp() -> None:
    first = is_validation_sample("25_12-20", 1234.4, 0.2)
    second = is_validation_sample("25_12-20", 1234.49, 0.2)

    assert first == second


def test_validation_split_respects_extreme_ratios() -> None:
    assert not is_validation_sample("video", 1000, 0.0)
    assert is_validation_sample("video", 1000, 1.0)


def test_read_frame_at_ms_from_reused_capture(tmp_path) -> None:
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 16))
    for idx in range(3):
        writer.write(np.full((16, 16, 3), idx * 50, dtype=np.uint8))
    writer.release()

    cap = cv2.VideoCapture(str(path))
    try:
        frame = read_frame_at_ms_from_cap(cap, 100)
    finally:
        cap.release()

    assert frame is not None
    assert frame.shape[:2] == (16, 16)
