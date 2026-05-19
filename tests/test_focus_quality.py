from __future__ import annotations

import cv2
import numpy as np

from lenta_shelf_ai.focus import compute_focus_quality


def test_focus_quality_prefers_sharp_crop_over_blur() -> None:
    image = np.full((160, 260, 3), 240, dtype=np.uint8)
    cv2.putText(image, "129 99", (14, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
    for x in range(165, 235, 8):
        cv2.rectangle(image, (x, 35), (x + 4, 115), (0, 0, 0), -1)
    blurred = cv2.GaussianBlur(image, (17, 17), 0)

    sharp_score = compute_focus_quality(image)["score"]
    blur_score = compute_focus_quality(blurred)["score"]

    assert sharp_score > blur_score
    assert 0.0 <= blur_score <= 1.0
    assert 0.0 <= sharp_score <= 1.0
