from __future__ import annotations

import cv2
import numpy as np

from lenta_shelf_ai.video import iter_video_frames, read_frame_at_ms


def test_video_reader_auto_falls_back_to_opencv_when_pyav_missing(tmp_path, monkeypatch):
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
    for idx in range(4):
        frame = np.full((24, 32, 3), 255, dtype=np.uint8)
        cv2.putText(frame, str(idx), (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        writer.write(frame)
    writer.release()

    monkeypatch.setenv("LENTA_VIDEO_READER", "auto")
    frames = list(iter_video_frames(path, sample_fps=5.0, max_frames=2, min_sharpness=0.0))

    assert len(frames) == 2
    assert frames[0].timestamp_ms >= 0
    assert read_frame_at_ms(path, frames[0].timestamp_ms) is not None

import numpy as np

from lenta_shelf_ai.video import _rotate_frame_bgr


def test_video_rotate_env(monkeypatch) -> None:
    image = np.arange(2 * 3 * 1, dtype=np.uint8).reshape(2, 3, 1).repeat(3, axis=2)

    monkeypatch.setenv("LENTA_VIDEO_ROTATE", "90cw")
    rotated = _rotate_frame_bgr(image)

    assert rotated.shape[:2] == (3, 2)
    assert int(rotated[0, 1, 0]) == int(image[0, 0, 0])
