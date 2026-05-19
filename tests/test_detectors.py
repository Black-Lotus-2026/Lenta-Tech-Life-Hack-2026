import cv2
import numpy as np

from lenta_shelf_ai.detectors import ColorGeometryDetector, HybridDetector, QRSeedDetector, RedWhiteTagDetector
from lenta_shelf_ai.utils import iou_xyxy


def test_red_white_detector_expands_price_panel_to_full_tag():
    image = np.full((360, 640, 3), 45, dtype=np.uint8)

    # A Lenta-like tag: red price panel on the left, white text/QR area on the right.
    tag_box = [220, 90, 430, 250]
    cv2.rectangle(image, (220, 90), (295, 250), (45, 55, 220), -1)
    cv2.rectangle(image, (295, 90), (430, 250), (245, 245, 240), -1)
    cv2.putText(image, "12999", (228, 168), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    cv2.putText(image, "SKU 370204", (305, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    cv2.rectangle(image, (368, 172), (416, 220), (15, 15, 15), 2)
    for offset in range(376, 409, 12):
        cv2.line(image, (offset, 176), (offset, 216), (15, 15, 15), 2)
        cv2.line(image, (372, offset - 4), (412, offset - 4), (15, 15, 15), 2)

    # Red product-like distractor without adjacent white label evidence.
    cv2.rectangle(image, (40, 40), (120, 210), (30, 40, 210), -1)

    detections = RedWhiteTagDetector(max_width=640).predict(image)

    assert detections, "expected at least one tag candidate"
    best = max(detections, key=lambda det: iou_xyxy(det.xyxy, tag_box))
    assert best.source == "red_white_tag"
    assert iou_xyxy(best.xyxy, tag_box) >= 0.70


def test_hybrid_detector_keeps_fallbacks_when_weights_missing():
    detector = HybridDetector(yolo_weights="missing.pt", enable_fallbacks=False)

    assert any(isinstance(item, QRSeedDetector) for item in detector.detectors)
    assert any(isinstance(item, RedWhiteTagDetector) for item in detector.detectors)
    assert any(isinstance(item, ColorGeometryDetector) for item in detector.detectors)
