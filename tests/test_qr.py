from __future__ import annotations

import cv2
import numpy as np

from lenta_shelf_ai.qr import _qr_candidate_regions, _qr_image_variants, parse_qr_payload


def test_qr_variants_include_upscale_and_thresholds() -> None:
    image = np.full((80, 100, 3), 220, dtype=np.uint8)
    image[20:60, 30:70] = 30

    variants = _qr_image_variants(image)

    assert len(variants) >= 8
    assert variants[0].shape == image.shape
    assert any(v.shape[0] > image.shape[0] and v.shape[1] > image.shape[1] for v in variants)
    assert all(v.ndim == 3 and v.shape[2] == 3 for v in variants)
    assert any(v.shape[:2] == (image.shape[1], image.shape[0]) for v in variants)


def test_qr_candidate_regions_include_layout_priors() -> None:
    image = np.full((120, 240, 3), 240, dtype=np.uint8)
    image[40:95, 155:210] = 255
    for offset in (0, 18, 36):
        cv2.rectangle(image, (160 + offset, 45), (170 + offset, 55), (0, 0, 0), -1)
        cv2.rectangle(image, (160 + offset, 65), (170 + offset, 75), (0, 0, 0), -1)

    regions = _qr_candidate_regions(image)

    assert regions
    assert regions[0].shape == image.shape
    assert any(region.shape[0] < image.shape[0] and region.shape[1] < image.shape[1] for region in regions)


def test_parse_qr_payload_accepts_raw_ean13() -> None:
    parsed = parse_qr_payload("4670025474665")

    assert parsed["qr_code_barcode"] == "4670025474665"


def test_parse_qr_payload_accepts_gs1_gtin() -> None:
    parsed = parse_qr_payload("(01)04670025474665")

    assert parsed["qr_code_barcode"] == "4670025474665"


def test_qr_candidate_regions_not_truncated_to_three_regions():
    image = np.full((180, 320, 3), 245, dtype=np.uint8)
    # Dense square-like QR surrogate in a late-prior/contour region.
    for y in range(115, 160, 12):
        for x in range(245, 295, 12):
            if (x + y) % 24 == 0:
                cv2.rectangle(image, (x, y), (x + 7, y + 7), (0, 0, 0), -1)

    regions = _qr_candidate_regions(image)

    assert len(regions) > 3
