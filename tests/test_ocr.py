import sys
import types

import cv2
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


def test_easyocr_engine_is_opt_in_and_reads_fake_result(monkeypatch):
    calls = []

    class FakeReader:
        def __init__(self, langs, gpu=False, verbose=False):
            calls.append({"langs": langs, "gpu": gpu, "verbose": verbose})

        def readtext(self, image, detail=1, paragraph=False):
            return [([[0, 0], [20, 0], [20, 10], [0, 10]], "129 99", 0.86)]

    fake_module = types.SimpleNamespace(Reader=FakeReader)
    monkeypatch.setitem(sys.modules, "easyocr", fake_module)
    monkeypatch.setenv("LENTA_EASYOCR_LANGS", "ru,en")

    from lenta_shelf_ai.ocr import EasyOCREngine

    engine = EasyOCREngine(lang="ru", use_gpu=True)
    lines = engine.recognize(np.full((32, 64, 3), 255, dtype=np.uint8))

    assert calls == [{"langs": ["ru", "en"], "gpu": True, "verbose": False}]
    assert [line.text for line in lines] == ["129 99"]
    assert lines[0].engine == "easyocr-normal"


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


def test_rapidocr_engine_reads_fake_result(monkeypatch):
    calls = []

    class FakeRapidOCR:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __call__(self, image):
            return [([[0, 0], [30, 0], [30, 10], [0, 10]], "Кефир 1%", 0.93)]

    fake_module = types.SimpleNamespace(RapidOCR=FakeRapidOCR)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake_module)
    monkeypatch.setenv("LENTA_RAPIDOCR_REC_MODEL", "/tmp/rec.onnx")

    from lenta_shelf_ai.ocr import RapidOCREngine

    engine = RapidOCREngine(lang="ru", use_gpu=False)
    lines = engine.recognize(np.full((32, 64, 3), 255, dtype=np.uint8))

    assert calls == [{"rec_model_path": "/tmp/rec.onnx"}]
    assert [line.text for line in lines] == ["Кефир 1%"]
    assert lines[0].engine == "rapidocr"


def test_ensemble_ocr_can_disable_paddle_by_env(monkeypatch):
    from lenta_shelf_ai import ocr as ocr_module
    from lenta_shelf_ai.ocr import BaseOCREngine, EnsembleOCREngine
    from lenta_shelf_ai.schema import OCRLine

    class DummyEngine(BaseOCREngine):
        engine = "dummy"

        def recognize(self, image_bgr):
            return [OCRLine(text="Товар", confidence=0.9, engine=self.engine)]

    def fail_paddle(*args, **kwargs):
        raise AssertionError("Paddle should not be constructed")

    monkeypatch.setenv("LENTA_OCR_DISABLE_PADDLE", "1")
    monkeypatch.setenv("LENTA_OCR_ENABLE_RAPIDOCR", "0")
    monkeypatch.setenv("LENTA_OCR_ENABLE_EASYOCR", "0")
    monkeypatch.setattr(ocr_module, "PaddleOCREngine", fail_paddle)
    monkeypatch.setattr(ocr_module, "TesseractOCREngine", lambda *args, **kwargs: DummyEngine())

    engine = EnsembleOCREngine(prefer_paddle=True)
    lines = engine.recognize(np.full((32, 64, 3), 255, dtype=np.uint8))

    assert [line.engine for line in lines] == ["dummy"]


def test_ocr_glare_suppression_is_bounded() -> None:
    from lenta_shelf_ai.ocr import suppress_specular_glare_for_ocr

    image = np.full((96, 160, 3), 230, dtype=np.uint8)
    cv2.putText(image, "129 99", (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.rectangle(image, (70, 20), (118, 78), (255, 255, 255), -1)

    fixed = suppress_specular_glare_for_ocr(image)

    assert fixed.shape == image.shape
    assert np.mean(np.abs(fixed.astype(np.int16) - image.astype(np.int16))) > 0.1
