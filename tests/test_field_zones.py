from __future__ import annotations

import numpy as np
import cv2

from lenta_shelf_ai.zones import (
    CompositeFieldZoneDetector,
    FieldZone,
    HeuristicFieldZoneDetector,
    crop_zone,
    zones_to_debug,
)


def test_heuristic_field_zone_detector_finds_machine_code_regions() -> None:
    image = np.full((180, 320, 3), 245, dtype=np.uint8)
    # QR-like square texture on the right side.
    for y in range(40, 120, 14):
        for x in range(215, 295, 14):
            if (x + y) % 28 == 0:
                cv2.rectangle(image, (x, y), (x + 8, y + 8), (0, 0, 0), -1)
    # 1D barcode-like lower strip.
    for x in range(45, 235, 7):
        cv2.rectangle(image, (x, 138), (x + 3, 168), (0, 0, 0), -1)

    zones = HeuristicFieldZoneDetector(enable_priors=False).predict(image)
    labels = {z.label for z in zones}

    assert "qr_code_barcode" in labels or "barcode" in labels
    assert all(z.area > 0 for z in zones)


def test_composite_detector_keeps_yolo_zones_and_adds_missing_machine_fallback() -> None:
    image = np.full((120, 220, 3), 245, dtype=np.uint8)
    for x in range(30, 180, 6):
        cv2.rectangle(image, (x, 92), (x + 3, 112), (0, 0, 0), -1)

    class FakeYolo:
        names = {0: "product_name"}

        def predict(self, _image):
            return [FieldZone("product_name", 0.9, (10, 10, 160, 42), "fake_yolo")]

    detector = CompositeFieldZoneDetector(FakeYolo(), HeuristicFieldZoneDetector(enable_priors=False))
    zones = detector.predict(image)
    labels = {z.label for z in zones}

    assert "product_name" in labels
    assert "barcode" in labels or "qr_code_barcode" in labels


def test_crop_zone_adds_padding_and_clamps_to_image() -> None:
    image = np.full((50, 60, 3), 255, dtype=np.uint8)
    zone = FieldZone("price_default", 0.8, (0, 0, 20, 20), "test")

    cropped = crop_zone(image, zone, pad_ratio=0.5, min_pad=8)

    assert cropped.shape[0] > 20
    assert cropped.shape[1] > 20
    assert cropped.shape[0] <= image.shape[0]
    assert cropped.shape[1] <= image.shape[1]


def test_zones_to_debug_is_csv_json_safe() -> None:
    debug = zones_to_debug([FieldZone("barcode", 0.81234, (1.1, 2.2, 30.3, 40.4), "unit")])

    assert debug == [
        {
            "label": "barcode",
            "score": 0.8123,
            "source": "unit",
            "xyxy": [1.1, 2.2, 30.3, 40.4],
        }
    ]


def test_composite_detector_keeps_heuristic_machine_zone_even_when_yolo_has_same_label() -> None:
    image = np.full((140, 260, 3), 245, dtype=np.uint8)
    for x in range(45, 235, 7):
        cv2.rectangle(image, (x, 104), (x + 3, 132), (0, 0, 0), -1)

    class FakeYolo:
        names = {0: "barcode"}

        def predict(self, _image):
            return [FieldZone("barcode", 0.95, (5, 5, 60, 30), "zone_yolo")]

    detector = CompositeFieldZoneDetector(FakeYolo(), HeuristicFieldZoneDetector(enable_priors=False))
    zones = detector.predict(image)

    barcode_zones = [z for z in zones if z.label == "barcode"]
    assert len(barcode_zones) >= 2
    assert any("heuristic" in z.source for z in barcode_zones)
