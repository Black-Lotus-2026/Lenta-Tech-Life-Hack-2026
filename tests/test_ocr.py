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
