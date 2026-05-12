from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .schema import Detection, TagObservation
from .utils import iou_xyxy

@dataclass
class Track:
    track_id: int
    last_detection: Detection
    last_timestamp_ms: int
    observations: List[TagObservation] = field(default_factory=list)
    lost: int = 0

    def add(self, obs: TagObservation) -> None:
        self.observations.append(obs)
        self.last_detection = obs.detection
        self.last_timestamp_ms = obs.timestamp_ms
        self.lost = 0

    @property
    def best_observation(self) -> Optional[TagObservation]:
        if not self.observations:
            return None
        def score(obs: TagObservation) -> float:
            qr_bonus = 100.0 if obs.qr_payloads else 0.0
            text_bonus = min(40.0, len(obs.text) * 0.2)
            return qr_bonus + text_bonus + obs.image_quality * 0.01 + obs.detection.score * 10
        return max(self.observations, key=score)

class SimpleTracker:
    """Greedy tracker for short shelf videos.

    It merges detections by IoU and center-distance. For duplicate tracks with the same
    decoded barcode/QR, downstream fusion collapses them again.
    """

    def __init__(self, iou_threshold: float = 0.12, center_threshold: float = 220.0, max_lost: int = 5):
        self.iou_threshold = iou_threshold
        self.center_threshold = center_threshold
        self.max_lost = max_lost
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    @staticmethod
    def _center(det: Detection) -> tuple[float, float]:
        return ((det.x_min + det.x_max) / 2.0, (det.y_min + det.y_max) / 2.0)

    def _match_score(self, det: Detection, track: Track) -> float:
        iou = iou_xyxy(det.xyxy, track.last_detection.xyxy)
        cx, cy = self._center(det)
        tx, ty = self._center(track.last_detection)
        dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        if iou < self.iou_threshold and dist > self.center_threshold:
            return -1.0
        return iou * 10.0 - dist / max(1.0, self.center_threshold)

    def update(self, observations: List[TagObservation]) -> None:
        unmatched_tracks = set(self.tracks.keys())
        for obs in sorted(observations, key=lambda o: o.detection.score, reverse=True):
            best_id = None
            best_score = -1.0
            for tid in list(unmatched_tracks):
                s = self._match_score(obs.detection, self.tracks[tid])
                if s > best_score:
                    best_id, best_score = tid, s
            if best_id is not None and best_score >= -0.2:
                self.tracks[best_id].add(obs)
                unmatched_tracks.discard(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                tr = Track(tid, obs.detection, obs.timestamp_ms)
                tr.add(obs)
                self.tracks[tid] = tr
        for tid in unmatched_tracks:
            self.tracks[tid].lost += 1
        for tid in [tid for tid, tr in self.tracks.items() if tr.lost > self.max_lost and not tr.observations]:
            del self.tracks[tid]

    def active_and_finished_tracks(self) -> List[Track]:
        return list(self.tracks.values())
