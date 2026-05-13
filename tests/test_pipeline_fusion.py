from __future__ import annotations

from lenta_shelf_ai.pipeline import PipelineConfig, PriceTagPipeline
from lenta_shelf_ai.schema import Detection, TagObservation
from lenta_shelf_ai.tracker import Track


def _track(track_id: int, ts: int, box: list[float], parsed: dict[str, str] | None = None) -> Track:
    det = Detection(*box, score=0.8)
    obs = TagObservation(
        filename="shelf.mp4",
        timestamp_ms=ts,
        detection=det,
        parsed=parsed or {},
        image_quality=100.0,
    )
    tr = Track(track_id, det, ts)
    tr.add(obs)
    return tr


def test_spatial_dedupe_merges_fragmented_empty_tracks() -> None:
    pipe = PriceTagPipeline(PipelineConfig(enable_ocr=False))

    rows = pipe._tracks_to_rows(
        [
            _track(1, 1000, [100, 100, 220, 210]),
            _track(2, 1500, [106, 102, 226, 212]),
        ]
    )

    assert len(rows) == 1


def test_spatial_dedupe_keeps_conflicting_barcodes_separate() -> None:
    pipe = PriceTagPipeline(PipelineConfig(enable_ocr=False))

    rows = pipe._tracks_to_rows(
        [
            _track(1, 1000, [100, 100, 220, 210], {"barcode": "4600000000001"}),
            _track(2, 1500, [106, 102, 226, 212], {"barcode": "4600000000002"}),
        ]
    )

    assert len(rows) == 2
