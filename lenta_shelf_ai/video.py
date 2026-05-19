from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

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


def _video_reader_mode() -> str:
    return os.environ.get("LENTA_VIDEO_READER", "auto").strip().lower() or "auto"



# Camera orientation constants; disabled by default.
_CAM_DIAGONAL_MM = 16.0 / 2.8
_CAM_FOCAL_MM = 2.8
_CAM_DIST_COEFFS = np.array([-0.276, 0.06, 0.0084, -0.0016, -0.0044], dtype=np.float32)
_UNDISTORT_CACHE: dict[tuple[int, int, float], tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]] = {}


def _rotate_frame_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    mode = os.environ.get("LENTA_VIDEO_ROTATE", "0").strip().lower()
    if mode in {"", "0", "none", "no", "false"}:
        return frame_bgr
    if mode in {"90cw", "cw", "90", "+90"}:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    if mode in {"90ccw", "ccw", "270", "-90"}:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode in {"180", "flip"}:
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    print(f"[WARN] unknown LENTA_VIDEO_ROTATE={mode!r}; using unrotated frames")
    return frame_bgr


def _camera_matrix_for_size(width: int, height: int) -> np.ndarray:
    aspect = float(width) / max(1.0, float(height))
    h_mm = _CAM_DIAGONAL_MM / float(np.sqrt(aspect ** 2 + 1.0))
    w_mm = aspect * h_mm
    fx = _CAM_FOCAL_MM * float(width) / w_mm
    fy = _CAM_FOCAL_MM * float(height) / h_mm
    return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _undistort_frame_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    if os.environ.get("LENTA_ENABLE_UNDISTORT", "0") == "0":
        return frame_bgr
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    alpha = float(os.environ.get("LENTA_UNDISTORT_ALPHA", "0") or "0")
    key = (int(w), int(h), round(alpha, 4))
    maps = _UNDISTORT_CACHE.get(key)
    if maps is None:
        k = _camera_matrix_for_size(w, h)
        new_k, roi = cv2.getOptimalNewCameraMatrix(k, _CAM_DIST_COEFFS, (w, h), alpha, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(k, _CAM_DIST_COEFFS, None, new_k, (w, h), cv2.CV_32FC1)
        maps = (map1, map2, roi)
        _UNDISTORT_CACHE[key] = maps
    map1, map2, roi = maps
    fixed = cv2.remap(frame_bgr, map1, map2, cv2.INTER_LINEAR)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        fixed = fixed[y : y + rh, x : x + rw]
    return fixed


def _preprocess_frame_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    # Keep disabled by default. Use env knobs only for ablations because frame
    # rotation/undistortion changes the coordinate system expected by CSV.
    return _rotate_frame_bgr(_undistort_frame_bgr(frame_bgr))


def _iter_video_frames_opencv(
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
            frame_out = _preprocess_frame_bgr(frame)
            sharp = sharpness_laplacian(frame_out)
            if sharp >= min_sharpness:
                yield FramePacket(index=idx, timestamp_ms=ts, frame_bgr=frame_out, sharpness=sharp)
                yielded += 1
                if max_frames and yielded >= max_frames:
                    break
        idx += 1
    cap.release()


def _iter_video_frames_pyav(
    video_path: str | Path,
    sample_fps: float = 2.0,
    max_frames: Optional[int] = None,
    min_sharpness: float = 0.0,
) -> Iterator[FramePacket]:
    """Frame iterator with container PTS timestamps.

    OpenCV CAP_PROP_POS_MSEC is often coarse or container-dependent on long
    H.264 shelf videos. PyAV exposes frame.pts/time_base, so timestamps match
    the encoded timeline used by public CSVs more closely. This backend stays
    optional: when PyAV is not installed, iter_video_frames(..., auto) falls
    back to the original OpenCV reader.
    """
    import av  # type: ignore  # optional dependency

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
    except Exception as exc:  # pragma: no cover - corrupt/no-video file
        container.close()
        raise RuntimeError(f"Cannot open video stream: {video_path}") from exc

    time_base = float(getattr(stream, "time_base", 0.0) or 0.0)
    start_time = getattr(stream, "start_time", None)
    if start_time is None:
        start_time = 0
    period_ms = 1000.0 / float(sample_fps) if sample_fps and sample_fps > 0 else 0.0
    next_ts_ms = -1.0
    yielded = 0
    decoded_idx = 0

    try:
        for frame in container.decode(stream):
            pts = getattr(frame, "pts", None)
            if pts is not None and time_base > 0:
                ts = int(round((float(pts) - float(start_time)) * time_base * 1000.0))
            else:
                rate = float(getattr(stream, "average_rate", 0.0) or getattr(stream, "base_rate", 0.0) or 0.0)
                ts = int(round(decoded_idx / rate * 1000.0)) if rate > 0 else decoded_idx
            if period_ms > 0 and ts + 0.5 < next_ts_ms:
                decoded_idx += 1
                continue
            frame_bgr = _preprocess_frame_bgr(frame.to_ndarray(format="bgr24"))
            sharp = sharpness_laplacian(frame_bgr)
            if sharp >= min_sharpness:
                yield FramePacket(index=decoded_idx, timestamp_ms=max(0, ts), frame_bgr=frame_bgr, sharpness=sharp)
                yielded += 1
                next_ts_ms = float(ts) + period_ms if period_ms > 0 else -1.0
                if max_frames and yielded >= max_frames:
                    break
            decoded_idx += 1
    finally:
        container.close()


def iter_video_frames(
    video_path: str | Path,
    sample_fps: float = 2.0,
    max_frames: Optional[int] = None,
    min_sharpness: float = 0.0,
) -> Iterator[FramePacket]:
    mode = _video_reader_mode()
    if mode in {"pyav", "av", "auto"}:
        try:
            yield from _iter_video_frames_pyav(video_path, sample_fps=sample_fps, max_frames=max_frames, min_sharpness=min_sharpness)
            return
        except ImportError:
            if mode in {"pyav", "av"}:
                raise
        except Exception:
            if mode in {"pyav", "av"}:
                raise
    yield from _iter_video_frames_opencv(video_path, sample_fps=sample_fps, max_frames=max_frames, min_sharpness=min_sharpness)


def _read_frame_at_ms_pyav(video_path: str | Path, timestamp_ms: float) -> Optional[np.ndarray]:
    try:
        import av  # type: ignore
    except Exception:
        return None
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        time_base = float(getattr(stream, "time_base", 0.0) or 0.0)
        start_time = getattr(stream, "start_time", None) or 0
        if time_base <= 0:
            container.close()
            return None
        target_pts = int(round(float(timestamp_ms) / 1000.0 / time_base + float(start_time)))
        container.seek(max(0, target_pts), any_frame=False, backward=True, stream=stream)
        best_frame = None
        best_delta = float("inf")
        for idx, frame in enumerate(container.decode(stream)):
            pts = getattr(frame, "pts", None)
            if pts is None:
                continue
            ts = (float(pts) - float(start_time)) * time_base * 1000.0
            delta = abs(ts - float(timestamp_ms))
            if delta < best_delta:
                best_delta = delta
                best_frame = _preprocess_frame_bgr(frame.to_ndarray(format="bgr24"))
            if ts > float(timestamp_ms) + 250.0 or idx >= 12:
                break
        container.close()
        return best_frame
    except Exception:
        try:
            container.close()  # type: ignore[name-defined]
        except Exception:
            pass
        return None


def read_frame_at_ms(video_path: str | Path, timestamp_ms: float) -> Optional[np.ndarray]:
    mode = _video_reader_mode()
    if mode in {"pyav", "av", "auto"}:
        frame = _read_frame_at_ms_pyav(video_path, timestamp_ms)
        if frame is not None:
            return frame
        if mode in {"pyav", "av"}:
            return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
    ok, frame = cap.read()
    cap.release()
    return _preprocess_frame_bgr(frame) if ok else None


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
