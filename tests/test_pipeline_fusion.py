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
            _track(1, 1000, [100, 100, 220, 210], {"barcode": "4600000000008"}),
            _track(2, 1500, [106, 102, 226, 212], {"barcode": "4600000000015"}),
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


def test_deferred_qr_tries_top_k_track_crops(monkeypatch) -> None:
    image = np.full((100, 160, 3), 255, dtype=np.uint8)
    monkeypatch.setattr("lenta_shelf_ai.pipeline.read_frame_at_ms", lambda *_: image)

    pipe = PriceTagPipeline(
        PipelineConfig(
            enable_ocr=False,
            enable_qr=True,
            defer_ocr=True,
            deferred_qr_top_k=2,
            deferred_ocr_top_k=0,
            deferred_recognition_top_k=2,
        )
    )

    calls: list[str] = []

    def fake_recognize(frame, det, output_dir, crop_name, run_qr=True, run_ocr=True, known_qr_payloads=None):
        calls.append(crop_name)
        payloads = ["4670025474665"] if crop_name.endswith("_qr1") else []
        return "", [], payloads, {"qr_code_barcode": "4670025474665"} if payloads else {}, 100.0, {"fake": crop_name}

    monkeypatch.setattr(pipe, "_recognize_detection", fake_recognize)

    det1 = Detection(10, 10, 80, 70, score=0.9, source="yolo")
    det2 = Detection(12, 12, 82, 72, score=0.8, source="yolo")
    tr = Track(1, det1, 1000)
    tr.add(TagObservation("shelf.mp4", 1000, det1, image_quality=120.0))
    tr.add(TagObservation("shelf.mp4", 1500, det2, image_quality=110.0))

    pipe._enrich_representatives(Path("video.mp4"), [tr], None)
    rows = pipe._tracks_to_rows([tr])

    assert len(calls) == 2
    assert rows[0]["qr_code_barcode"] == "4670025474665"


def test_temporal_qr_reconstruction_uses_multiple_track_zone_crops(monkeypatch) -> None:
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    monkeypatch.setattr("lenta_shelf_ai.pipeline.read_frame_at_ms", lambda *_: image)

    pipe = PriceTagPipeline(
        PipelineConfig(
            enable_ocr=False,
            enable_qr=True,
            defer_ocr=True,
            deferred_qr_top_k=2,
            deferred_ocr_top_k=0,
            deferred_recognition_top_k=2,
            temporal_qr_reconstruction=True,
            temporal_qr_min_crops=2,
            temporal_qr_top_k=2,
        )
    )

    from lenta_shelf_ai.zones import FieldZone

    class FakeZoneDetector:
        def predict(self, _crop):
            return [FieldZone("qr_code_barcode", 0.95, (20, 18, 76, 74), "fake_zone")]

    pipe.zone_detector = FakeZoneDetector()

    def fake_recognize(frame, det, output_dir, crop_name, run_qr=True, run_ocr=True, known_qr_payloads=None):
        return "", [], [], {}, 100.0, {"fake": crop_name}

    decode_labels: list[str] = []

    def fake_decode(_crop, force_native=False, debug_label=""):
        decode_labels.append(debug_label)
        if debug_label.startswith("temporal_qr_code_barcode"):
            return ["4670025474665"], {"payloads": 1, "debug_label": debug_label}
        return [], {"payloads": 0, "debug_label": debug_label}

    monkeypatch.setattr(pipe, "_recognize_detection", fake_recognize)
    monkeypatch.setattr("lenta_shelf_ai.pipeline.decode_qr_payloads_with_debug", fake_decode)

    det1 = Detection(10, 10, 90, 90, score=0.9, source="yolo")
    det2 = Detection(12, 12, 92, 92, score=0.8, source="yolo")
    tr = Track(1, det1, 1000)
    tr.add(TagObservation("shelf.mp4", 1000, det1, image_quality=120.0))
    tr.add(TagObservation("shelf.mp4", 1500, det2, image_quality=110.0))

    pipe._enrich_representatives(Path("video.mp4"), [tr], None)
    rows = pipe._tracks_to_rows([tr])

    assert any(label.startswith("temporal_qr_code_barcode") for label in decode_labels)
    assert rows[0]["qr_code_barcode"] == "4670025474665"


def test_final_row_bbox_uses_geometry_representative_not_qr_hit() -> None:
    pipe = PriceTagPipeline(PipelineConfig(enable_ocr=False, enable_qr=False))
    good_det = Detection(100, 100, 220, 210, score=0.95, source="yolo")
    qr_det = Detection(500, 500, 620, 610, score=0.10, source="yolo")
    tr = Track(1, good_det, 1000)
    tr.add(TagObservation("shelf.mp4", 1000, good_det, image_quality=150.0, visual_hash="aaaaaaaaaaaaaaaa"))
    tr.add(
        TagObservation(
            "shelf.mp4",
            1400,
            qr_det,
            qr_payloads=["4670025474665"],
            parsed={"qr_code_barcode": "4670025474665"},
            image_quality=80.0,
            visual_hash="bbbbbbbbbbbbbbbb",
        )
    )

    rows = pipe._tracks_to_rows([tr])

    assert len(rows) == 1
    assert rows[0]["qr_code_barcode"] == "4670025474665"
    assert rows[0]["x_min"] == 100.0
    assert rows[0]["frame_timestamp"] == 1000


def test_zone_detector_routes_machine_zone_before_full_crop(monkeypatch) -> None:
    import numpy as np
    from lenta_shelf_ai.pipeline import PipelineConfig, PriceTagPipeline
    from lenta_shelf_ai.schema import Detection
    from lenta_shelf_ai.zones import FieldZone

    image = np.full((100, 160, 3), 255, dtype=np.uint8)

    class FakeZoneDetector:
        def predict(self, _crop):
            return [FieldZone("barcode", 0.95, (10, 60, 140, 88), "fake_zone")]

    pipe = PriceTagPipeline(PipelineConfig(enable_ocr=False, enable_qr=True, enable_field_zone_detector=False))
    pipe.zone_detector = FakeZoneDetector()

    calls: list[str] = []

    def fake_decode(_crop, force_native=False, debug_label=""):
        calls.append(f"{debug_label}:{force_native}")
        if debug_label == "barcode":
            return ["4670025474665"], {"hit": "barcode"}
        return [], {"hit": "none"}

    monkeypatch.setattr("lenta_shelf_ai.pipeline.decode_qr_payloads_with_debug", fake_decode)

    _, _, payloads, parsed, _, debug = pipe._recognize_detection(
        image,
        Detection(0, 0, 159, 99, score=0.9, source="yolo"),
        None,
        "unit",
        run_qr=True,
        run_ocr=False,
    )

    assert payloads == ["4670025474665"]
    assert parsed["qr_code_barcode"] == "4670025474665"
    assert calls[0] == "barcode:True"
    assert debug["zone_attempts"][0]["zone_label"] == "barcode"


def test_field_zone_orientation_uses_upright_rotated_crop(monkeypatch) -> None:
    import numpy as np
    from lenta_shelf_ai.pipeline import PipelineConfig, PriceTagPipeline
    from lenta_shelf_ai.schema import Detection
    from lenta_shelf_ai.zones import FieldZone

    # Landscape crop from rotated video. The field-zone model expects the tag
    # upright, which in the team clean pipeline is a 270-degree crop rotation.
    image = np.full((100, 160, 3), 255, dtype=np.uint8)

    class OrientationAwareZoneDetector:
        def predict(self, crop):
            h, w = crop.shape[:2]
            if h > w:
                return [FieldZone("barcode", 0.95, (10, 95, 90, 130), "fake_upright")]
            return []

    pipe = PriceTagPipeline(
        PipelineConfig(
            enable_ocr=False,
            enable_qr=True,
            enable_field_zone_detector=False,
            field_zone_crop_rotations=(270, 0),
            field_zone_qr_full_crop_fallback=False,
            field_zone_qr_context_fallback=False,
        )
    )
    pipe.zone_detector = OrientationAwareZoneDetector()

    def fake_decode(_crop, force_native=False, debug_label=""):
        return (["4670025474665"], {"debug_label": debug_label}) if debug_label == "barcode" else ([], {})

    monkeypatch.setattr("lenta_shelf_ai.pipeline.decode_qr_payloads_with_debug", fake_decode)

    _, _, payloads, parsed, _, debug = pipe._recognize_detection(
        image,
        Detection(0, 0, 159, 99, score=0.9, source="yolo"),
        None,
        "unit",
        run_qr=True,
        run_ocr=False,
    )

    assert payloads == ["4670025474665"]
    assert parsed["qr_code_barcode"] == "4670025474665"
    assert debug["zones"]["semantic_rotation"] == 270
    assert debug["zones"]["orientation_attempts"][0]["zone_count"] == 1


def test_qr_full_crop_fallback_can_be_disabled_independently(monkeypatch) -> None:
    import numpy as np
    from lenta_shelf_ai.pipeline import PipelineConfig, PriceTagPipeline
    from lenta_shelf_ai.schema import Detection
    from lenta_shelf_ai.zones import FieldZone

    image = np.full((100, 160, 3), 255, dtype=np.uint8)

    class FakeZoneDetector:
        def predict(self, _crop):
            return [FieldZone("barcode", 0.95, (10, 60, 140, 88), "fake_zone")]

    pipe = PriceTagPipeline(
        PipelineConfig(
            enable_ocr=False,
            enable_qr=True,
            enable_field_zone_detector=False,
            field_zone_full_crop_fallback=True,
            field_zone_qr_full_crop_fallback=False,
        )
    )
    pipe.zone_detector = FakeZoneDetector()

    calls: list[str] = []

    def fake_decode(_crop, force_native=False, debug_label=""):
        calls.append(debug_label)
        return (["4670025474665"], {}) if debug_label == "barcode" else ([], {})

    monkeypatch.setattr("lenta_shelf_ai.pipeline.decode_qr_payloads_with_debug", fake_decode)

    _, _, payloads, _, _, _ = pipe._recognize_detection(
        image,
        Detection(0, 0, 159, 99, score=0.9, source="yolo"),
        None,
        "unit",
        run_qr=True,
        run_ocr=False,
    )

    assert payloads == ["4670025474665"]
    assert "full_crop" not in calls


def test_qr_full_and_context_fallbacks_can_be_disabled_when_zone_misses(monkeypatch) -> None:
    import numpy as np
    from lenta_shelf_ai.pipeline import PipelineConfig, PriceTagPipeline
    from lenta_shelf_ai.schema import Detection
    from lenta_shelf_ai.zones import FieldZone

    image = np.full((100, 160, 3), 255, dtype=np.uint8)

    class FakeZoneDetector:
        def predict(self, _crop):
            return [FieldZone("barcode", 0.95, (10, 60, 140, 88), "fake_zone")]

    pipe = PriceTagPipeline(
        PipelineConfig(
            enable_ocr=False,
            enable_qr=True,
            enable_field_zone_detector=False,
            field_zone_full_crop_fallback=True,
            field_zone_qr_full_crop_fallback=False,
            field_zone_qr_context_fallback=False,
        )
    )
    pipe.zone_detector = FakeZoneDetector()

    calls: list[str] = []

    def fake_decode(_crop, force_native=False, debug_label=""):
        calls.append(debug_label)
        return [], {}

    monkeypatch.setattr("lenta_shelf_ai.pipeline.decode_qr_payloads_with_debug", fake_decode)

    _, _, payloads, _, _, _ = pipe._recognize_detection(
        image,
        Detection(0, 0, 159, 99, score=0.9, source="yolo"),
        None,
        "unit",
        run_qr=True,
        run_ocr=False,
    )

    assert payloads == []
    assert calls == ["barcode"]


def test_ocr_full_crop_fallback_can_be_enabled_independently() -> None:
    import numpy as np
    from lenta_shelf_ai.pipeline import PipelineConfig, PriceTagPipeline
    from lenta_shelf_ai.schema import Detection, OCRLine

    image = np.full((100, 160, 3), 255, dtype=np.uint8)

    class NoZones:
        def predict(self, _crop):
            return []

    class FakeOCR:
        def recognize_zoned(self, _crop):
            return [OCRLine("Молоко 129 99", confidence=0.9, engine="fake")]

    pipe = PriceTagPipeline(
        PipelineConfig(
            enable_ocr=False,
            enable_qr=False,
            enable_field_zone_detector=False,
            field_zone_full_crop_fallback=False,
            field_zone_ocr_full_crop_fallback=True,
        )
    )
    pipe.zone_detector = NoZones()
    pipe.ocr = FakeOCR()

    text, lines, _, parsed, _, _ = pipe._recognize_detection(
        image,
        Detection(0, 0, 159, 99, score=0.9, source="yolo"),
        None,
        "unit",
        run_qr=False,
        run_ocr=True,
    )

    assert "Молоко" in text
    assert len(lines) == 1
    assert parsed["price_default"] == "129.99"


def test_field_aware_fusion_rejects_invalid_barcode_noise_and_keeps_valid_ean() -> None:
    pipe = PriceTagPipeline(PipelineConfig(enable_ocr=False))

    rows = pipe._tracks_to_rows(
        [
            _track(1, 1000, [100, 100, 220, 210], {"barcode": "11111108"}),
            _track(2, 1200, [104, 102, 224, 212], {"barcode": "4670025474665"}),
        ]
    )

    assert len(rows) == 1
    assert rows[0]["barcode"] == "4670025474665"


def test_ocr_zone_selector_includes_sku_date_and_code_labels() -> None:
    from lenta_shelf_ai.zones import FieldZone

    zones = [
        FieldZone("product_name", 0.99, (0, 0, 100, 20), "unit"),
        FieldZone("price_default", 0.98, (0, 20, 60, 40), "unit"),
        FieldZone("price_card", 0.97, (0, 40, 60, 60), "unit"),
        FieldZone("price_discount", 0.96, (0, 60, 60, 80), "unit"),
        FieldZone("discount_amount", 0.95, (0, 80, 60, 100), "unit"),
        FieldZone("id_sku", 0.40, (80, 80, 160, 100), "unit"),
        FieldZone("print_datetime", 0.39, (80, 100, 160, 120), "unit"),
        FieldZone("code", 0.38, (80, 120, 160, 140), "unit"),
    ]

    selected = PriceTagPipeline._select_ocr_zones(zones, budget=8, per_label=1)
    labels = [z.label for z in selected]

    assert "id_sku" in labels
    assert "print_datetime" in labels
    assert "code" in labels
