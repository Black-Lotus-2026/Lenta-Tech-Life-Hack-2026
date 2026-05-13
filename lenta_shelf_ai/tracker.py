from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional

from .schema import ABSENT_VALUE, Detection, TagObservation
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

    def select_best_observation(self, temporal_penalty_weight: float = 0.0) -> Optional[TagObservation]:
        if not self.observations:
            return None
        timestamps = sorted(obs.timestamp_ms for obs in self.observations)
        mid = len(timestamps) // 2
        median_ts = timestamps[mid] if len(timestamps) % 2 else (timestamps[mid - 1] + timestamps[mid]) / 2.0
        span_ms = max(1000.0, float(timestamps[-1] - timestamps[0]))

        def score(obs: TagObservation) -> float:
            qr_bonus = 100.0 if obs.qr_payloads else 0.0
            text_bonus = min(40.0, len(obs.text) * 0.2)
            quality_bonus = min(8.0, math.log1p(max(0.0, obs.image_quality)) * 0.8)
            area_bonus = min(6.0, obs.detection.area / 40000.0)
            detector_bonus = obs.detection.score * 6.0
            temporal_penalty = 0.0
            if temporal_penalty_weight > 0 and not obs.qr_payloads and not obs.text:
                temporal_penalty = abs(float(obs.timestamp_ms) - median_ts) / span_ms * float(temporal_penalty_weight)
            return qr_bonus + text_bonus + quality_bonus + area_bonus + detector_bonus - temporal_penalty

        return max(self.observations, key=score)

    @property
    def best_observation(self) -> Optional[TagObservation]:
        return self.select_best_observation()

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

    @staticmethod
    def _stable_id_groups(obs: TagObservation) -> list[set[str]]:
        def clean(value: object) -> str:
            text = str(value or "").strip()
            if not text or text.lower() == "nan" or text == ABSENT_VALUE:
                return ""
            return text

        groups = [
            {"barcode", "qr_code_barcode"},
            {"id_sku"},
            {"code"},
        ]
        values: list[set[str]] = []
        for group in groups:
            group_values = {clean(obs.parsed.get(col, "")) for col in group}
            values.append({value for value in group_values if value})
        return values

    def _has_conflicting_stable_ids(self, obs: TagObservation, track: Track) -> bool:
        obs_groups = self._stable_id_groups(obs)
        track_groups = [set() for _ in obs_groups]
        for prev in track.observations:
            for idx, values in enumerate(self._stable_id_groups(prev)):
                track_groups[idx].update(values)

        return any(
            obs_values and track_values and obs_values.isdisjoint(track_values)
            for obs_values, track_values in zip(obs_groups, track_groups)
        )

    def _match_score(self, obs: TagObservation, track: Track) -> float:
        if self._has_conflicting_stable_ids(obs, track):
            return -1.0
        det = obs.detection
        iou = iou_xyxy(det.xyxy, track.last_detection.xyxy)
        cx, cy = self._center(det)
        tx, ty = self._center(track.last_detection)
        dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        if iou < self.iou_threshold and dist > self.center_threshold:
            return -1.0
        return iou * 10.0 - dist / max(1.0, self.center_threshold)

    def update(self, observations: List[TagObservation]) -> None:
        matchable_tracks = {tid for tid, tr in self.tracks.items() if tr.lost <= self.max_lost}
        unmatched_tracks = set(matchable_tracks)
        for obs in sorted(observations, key=lambda o: o.detection.score, reverse=True):
            best_id = None
            best_score = -1.0
            for tid in list(unmatched_tracks):
                s = self._match_score(obs, self.tracks[tid])
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
