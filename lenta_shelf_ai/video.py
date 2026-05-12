from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Iterator, Optional, Tuple

import cv2
import numpy as np

from .utils import sharpness_laplacian

@dataclass
class FramePacket:
    index: int
    timestamp_ms: int
    frame_bgr: np.ndarray
    sharpness: float

@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_ms: int


def get_video_meta(video_path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_ms = int(round(frame_count / fps * 1000)) if fps > 0 else 0
    cap.release()
    return VideoMeta(width, height, fps, frame_count, duration_ms)


def iter_video_frames(
    video_path: str | Path,
    sample_fps: float = 2.0,
    max_frames: Optional[int] = None,
    min_sharpness: float = 0.0,
) -> Iterator[FramePacket]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    step = max(1, int(round(fps / sample_fps))) if sample_fps > 0 else 1
    yielded = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            ts = int(round(cap.get(cv2.CAP_PROP_POS_MSEC)))
            sharp = sharpness_laplacian(frame)
            if sharp >= min_sharpness:
                yield FramePacket(index=idx, timestamp_ms=ts, frame_bgr=frame, sharpness=sharp)
                yielded += 1
                if max_frames and yielded >= max_frames:
                    break
        idx += 1
    cap.release()


def read_frame_at_ms(video_path: str | Path, timestamp_ms: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def make_video_thumbnail(video_path: str | Path, out_path: str | Path, timestamp_ms: int = 0, width: int = 1280) -> Optional[Path]:
    frame = read_frame_at_ms(video_path, timestamp_ms)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > width:
        new_h = int(h * width / w)
        frame = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return out_path
