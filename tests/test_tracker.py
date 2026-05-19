from __future__ import annotations

from lenta_shelf_ai.schema import Detection, TagObservation
from lenta_shelf_ai.tracker import SimpleTracker


def _obs(
    ts: int,
    box: list[float],
    parsed: dict[str, str] | None = None,
    score: float = 0.9,
    quality: float = 100.0,
    text: str = "",
    qr_payloads: list[str] | None = None,
) -> TagObservation:
    return TagObservation(
        filename="shelf.mp4",
        timestamp_ms=ts,
        detection=Detection(*box, score=score),
        text=text,
        qr_payloads=qr_payloads or [],
        parsed=parsed or {},
        image_quality=quality,
    )


def test_tracker_does_not_match_tracks_lost_past_limit() -> None:
    tracker = SimpleTracker(iou_threshold=0.1, center_threshold=200.0, max_lost=1)

    tracker.update([_obs(0, [100, 100, 220, 220])])
    tracker.update([])
    tracker.update([])
    tracker.update([_obs(1500, [102, 102, 222, 222])])

    tracks = tracker.active_and_finished_tracks()

    assert len(tracks) == 2
    assert sorted(len(track.observations) for track in tracks) == [1, 1]


def test_tracker_keeps_conflicting_stable_ids_separate() -> None:
    tracker = SimpleTracker(iou_threshold=0.1, center_threshold=200.0, max_lost=8)

    tracker.update([_obs(0, [100, 100, 220, 220], {"barcode": "4600000000008"})])
    tracker.update([_obs(500, [102, 102, 222, 222], {"barcode": "4600000000015"})])

    tracks = tracker.active_and_finished_tracks()

    assert len(tracks) == 2
    assert sorted(len(track.observations) for track in tracks) == [1, 1]


def test_track_best_observation_can_prefer_temporal_middle_without_text_or_qr() -> None:
    tracker = SimpleTracker(iou_threshold=0.1, center_threshold=250.0, max_lost=8)

    tracker.update([_obs(1000, [100, 100, 220, 220], score=0.88, quality=100)])
    tracker.update([_obs(7000, [105, 105, 225, 225], score=0.88, quality=100)])
    tracker.update([_obs(14000, [110, 110, 230, 230], score=0.98, quality=300)])

    best = tracker.active_and_finished_tracks()[0].select_best_observation(temporal_penalty_weight=8.0)

    assert best is not None
    assert best.timestamp_ms == 7000


def test_track_best_observation_defaults_to_quality_without_temporal_penalty() -> None:
    tracker = SimpleTracker(iou_threshold=0.1, center_threshold=250.0, max_lost=8)

    tracker.update([_obs(1000, [100, 100, 220, 220], score=0.88, quality=100)])
    tracker.update([_obs(7000, [105, 105, 225, 225], score=0.88, quality=100)])
    tracker.update([_obs(14000, [110, 110, 230, 230], score=0.98, quality=300)])

    best = tracker.active_and_finished_tracks()[0].best_observation

    assert best is not None
    assert best.timestamp_ms == 14000


def test_track_best_observation_still_prioritizes_qr() -> None:
    tracker = SimpleTracker(iou_threshold=0.1, center_threshold=250.0, max_lost=8)

    tracker.update([_obs(1000, [100, 100, 220, 220], score=0.88, quality=100)])
    tracker.update([_obs(7000, [105, 105, 225, 225], score=0.88, quality=100)])
    tracker.update([_obs(14000, [110, 110, 230, 230], score=0.88, quality=100, qr_payloads=["b=4600000000008"])])

    best = tracker.active_and_finished_tracks()[0].best_observation

    assert best is not None
    assert best.timestamp_ms == 14000


def test_tracker_tightens_center_gate_for_stale_tracks() -> None:
    tracker = SimpleTracker(iou_threshold=0.1, center_threshold=200.0, max_lost=8)

    tracker.update([_obs(0, [100, 100, 220, 220])])
    tracker.update([])
    tracker.update([_obs(1000, [260, 100, 380, 220])])

    tracks = tracker.active_and_finished_tracks()

    assert len(tracks) == 2
    assert sorted(len(track.observations) for track in tracks) == [1, 1]
