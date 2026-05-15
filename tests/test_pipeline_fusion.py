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


def test_name_price_alone_does_not_merge_far_apart_real_tags() -> None:
    pipe = PriceTagPipeline(PipelineConfig(enable_ocr=False))

    rows = pipe._tracks_to_rows(
        [
            _track(1, 1000, [100, 100, 220, 210], {"product_name": "Молоко тест", "price_default": "129.99"}),
            _track(2, 1000, [900, 100, 1020, 210], {"product_name": "Молоко тест", "price_default": "129.99"}),
        ]
    )

    assert len(rows) == 2


def test_visual_text_dedupe_merges_fragmented_tracks_without_ids() -> None:
    pipe = PriceTagPipeline(
        PipelineConfig(
            enable_ocr=False,
            dedupe_extended_time_window_ms=10000,
            dedupe_visual_hash_threshold=4,
            dedupe_text_similarity=0.80,
        )
    )
    t1 = _track(1, 1000, [100, 100, 220, 210], {"product_name": "Молоко тест", "price_default": "129.99"})
    t2 = _track(2, 5000, [118, 108, 238, 218], {"product_name": "Молоко тестовое", "price_default": "129.99"})
    t1.observations[0].visual_hash = "0f0f0f0f0f0f0f0f"
    t2.observations[0].visual_hash = "0f0f0f0f0f0f0f0e"
    t1.observations[0].text = "Молоко тест 129 99"
    t2.observations[0].text = "Молоко тестовое 129 99"

    rows = pipe._tracks_to_rows([t1, t2])

    assert len(rows) == 1


def test_dedupe_keeps_conflicting_sku_separate() -> None:
    pipe = PriceTagPipeline(PipelineConfig(enable_ocr=False))

    rows = pipe._tracks_to_rows(
        [
            _track(1, 1000, [100, 100, 220, 210], {"id_sku": "270207736530"}),
            _track(2, 1200, [104, 102, 224, 212], {"id_sku": "270207736531"}),
        ]
    )

    assert len(rows) == 2


def test_fallback_track_without_semantic_evidence_is_filtered() -> None:
    det = Detection(100, 100, 220, 210, score=0.5, source="heuristic")
    obs = TagObservation(
        filename="shelf.mp4",
        timestamp_ms=1000,
        detection=det,
        parsed={"color": "red"},
        image_quality=100.0,
    )
    tr = Track(1, det, 1000)
    tr.add(obs)
    pipe = PriceTagPipeline(PipelineConfig(enable_ocr=False, enable_fallback_detectors=True))

    rows = pipe._tracks_to_rows([tr])

    assert rows == []


def test_single_qr_seed_with_machine_id_is_kept() -> None:
    det = Detection(100, 100, 220, 210, score=0.6, source="qr_seed")
    obs = TagObservation(
        filename="shelf.mp4",
        timestamp_ms=1000,
        detection=det,
        parsed={"qr_code_barcode": "4670025474665"},
        image_quality=100.0,
    )
    tr = Track(1, det, 1000)
    tr.add(obs)
    pipe = PriceTagPipeline(PipelineConfig(enable_ocr=False, enable_fallback_detectors=True))

    rows = pipe._tracks_to_rows([tr])

    assert len(rows) == 1
    assert rows[0]["qr_code_barcode"] == "4670025474665"


def test_single_fallback_with_product_only_is_filtered() -> None:
    det = Detection(100, 100, 220, 210, score=0.5, source="red_white_tag")
    obs = TagObservation(
        filename="shelf.mp4",
        timestamp_ms=1000,
        detection=det,
        parsed={"product_name": "Молоко тестовое"},
        image_quality=100.0,
    )
    tr = Track(1, det, 1000)
    tr.add(obs)
    pipe = PriceTagPipeline(PipelineConfig(enable_ocr=False, enable_fallback_detectors=True))

    rows = pipe._tracks_to_rows([tr])

    assert rows == []


def test_tracks_debug_exports_coordinate_trajectory() -> None:
    pipe = PriceTagPipeline(PipelineConfig(enable_ocr=False))
    tr = _track(1, 1000, [100, 100, 220, 210])
    tr.add(
        TagObservation(
            filename="shelf.mp4",
            timestamp_ms=1500,
            detection=Detection(106, 102, 226, 212, score=0.7, source="yolo"),
            image_quality=120.0,
        )
    )

    debug = pipe._tracks_debug([tr])

    assert debug[0]["track_id"] == 1
    assert [point["timestamp_ms"] for point in debug[0]["trajectory"]] == [1000, 1500]
    assert debug[0]["trajectory"][1]["bbox"] == [106.0, 102.0, 226.0, 212.0]
