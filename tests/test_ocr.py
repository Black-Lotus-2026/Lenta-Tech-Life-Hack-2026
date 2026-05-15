import sys
import types

import numpy as np


def test_paddleocr_engine_supports_paddleocr_3_api(monkeypatch):
    calls = []

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            unexpected = {"use_gpu", "show_log", "use_angle_cls"}
            assert unexpected.isdisjoint(kwargs)

        def ocr(self, image):
            return [
                {
                    "rec_texts": ["Напиток тестовый", "129,99"],
                    "rec_scores": [0.91, 0.88],
                    "rec_polys": [
                        [[0, 0], [10, 0], [10, 10], [0, 10]],
                        [[0, 12], [10, 12], [10, 22], [0, 22]],
                    ],
                }
            ]

    fake_module = types.SimpleNamespace(PaddleOCR=FakePaddleOCR)
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    from lenta_shelf_ai.ocr import PaddleOCREngine

    engine = PaddleOCREngine(lang="ru", use_gpu=False)
    lines = engine.recognize(np.full((32, 64, 3), 255, dtype=np.uint8))

    assert calls == [
        {
            "lang": "ru",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
        }
    ]
    assert [line.text for line in lines] == ["Напиток тестовый", "129,99"]
    assert [line.confidence for line in lines] == [0.91, 0.88]


def test_ensemble_ocr_disables_repeatedly_failing_engine(monkeypatch):
    from lenta_shelf_ai.ocr import BaseOCREngine, EnsembleOCREngine
    from lenta_shelf_ai.schema import OCRLine

    class BrokenEngine(BaseOCREngine):
        engine = "broken"

        def __init__(self):
            self.calls = 0

        def recognize(self, image_bgr):
            self.calls += 1
            raise RuntimeError("boom")

    class WorkingEngine(BaseOCREngine):
        engine = "working"

        def recognize(self, image_bgr):
            return [OCRLine(text="Молоко", confidence=0.9, engine=self.engine)]

    monkeypatch.setenv("LENTA_OCR_MAX_ENGINE_FAILURES", "2")
    engine = EnsembleOCREngine(prefer_paddle=False)
    broken = BrokenEngine()
    engine.engines = [broken, WorkingEngine()]

    image = np.full((32, 64, 3), 255, dtype=np.uint8)
    assert [line.text for line in engine.recognize(image)] == ["Молоко"]
    assert [line.text for line in engine.recognize(image)] == ["Молоко"]
    assert [line.text for line in engine.recognize(image)] == ["Молоко"]

    assert broken.calls == 2


def test_zonal_ocr_runs_full_and_local_zones_without_duplicates():
    from lenta_shelf_ai.ocr import BaseOCREngine, EnsembleOCREngine
    from lenta_shelf_ai.schema import OCRLine

    class ZoneAwareEngine(BaseOCREngine):
        engine = "zoneaware"

        def recognize(self, image_bgr):
            h, w = image_bgr.shape[:2]
            if h < 80:
                return [OCRLine(text="129 99", confidence=0.7, engine=self.engine)]
            return [OCRLine(text="Товар тестовый", confidence=0.8, engine=self.engine)]

    engine = EnsembleOCREngine(prefer_paddle=False)
    engine.engines = [ZoneAwareEngine()]
    image = np.full((120, 200, 3), 255, dtype=np.uint8)

    lines = engine.recognize_zoned(image)

    assert "Товар тестовый" in [line.text for line in lines]
    assert "129 99" in [line.text for line in lines]
    assert any("|zone:" in line.engine for line in lines)


def test_suppress_code_artifacts_masks_dense_qr_block_but_keeps_text_area():
    import cv2

    from lenta_shelf_ai.ocr import suppress_code_artifacts

    image = np.full((120, 220, 3), 245, dtype=np.uint8)
    # Text-like strokes on the left should stay dark.
    cv2.putText(image, "129", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    # Dense QR-like block on the right should be suppressed for OCR.
    for y in range(25, 95, 10):
        for x in range(150, 205, 10):
            if ((x // 10) + (y // 10)) % 2 == 0:
                cv2.rectangle(image, (x, y), (x + 7, y + 7), (0, 0, 0), -1)

    cleaned = suppress_code_artifacts(image)

    left_before = image[35:85, 8:85].mean()
    left_after = cleaned[35:85, 8:85].mean()
    qr_before = image[25:95, 150:205].mean()
    qr_after = cleaned[25:95, 150:205].mean()

    assert qr_after > qr_before + 20
    assert abs(left_after - left_before) < 12
