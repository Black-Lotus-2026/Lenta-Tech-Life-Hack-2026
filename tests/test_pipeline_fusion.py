from __future__ import annotations

from pathlib import Path

import numpy as np

from lenta_shelf_ai.pipeline import PipelineConfig, PriceTagPipeline
from lenta_shelf_ai.schema import Detection, OCRLine, TagObservation
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


def test_deferred_ocr_enriches_best_track_observation(monkeypatch) -> None:
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    monkeypatch.setattr("lenta_shelf_ai.pipeline.read_frame_at_ms", lambda *_: image)

    class FakeOCR:
        def recognize(self, _crop):
            return [OCRLine("Молоко тестовое", confidence=0.9, engine="fake")]

    pipe = PriceTagPipeline(PipelineConfig(enable_ocr=False, enable_qr=False, defer_ocr=True))
    pipe.config.enable_ocr = True
    pipe.ocr = FakeOCR()
    tr = _track(1, 1000, [10, 10, 70, 50])

    pipe._enrich_representatives(Path("video.mp4"), [tr], None)
    rows = pipe._tracks_to_rows([tr])

    assert rows[0]["product_name"] == "Молоко тестовое"
