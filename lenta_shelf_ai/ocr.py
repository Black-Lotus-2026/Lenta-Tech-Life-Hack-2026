from __future__ import annotations

import re
import inspect
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .schema import OCRLine
from .utils import normalize_text


def enhance_crop(image_bgr: np.ndarray, max_side: int = 1600) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    scale = 1.0
    if max(h, w) < 650:
        scale = min(3.0, 850.0 / max(h, w))
    elif max(h, w) > max_side:
        scale = max_side / float(max(h, w))
    if abs(scale - 1.0) > 1e-3:
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    # Unsharp mask.
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    out = cv2.addWeighted(out, 1.45, blur, -0.45, 0)
    return out

class BaseOCREngine:
    def recognize(self, image_bgr: np.ndarray) -> List[OCRLine]:
        raise NotImplementedError

class PaddleOCREngine(BaseOCREngine):
    def __init__(self, lang: str = "ru", use_gpu: bool = False):
        # PaddlePaddle 3.3.x CPU oneDNN/PIR path is unstable for OCR inference.
        # Keep CPU inference on the plain backend unless the runtime explicitly opts in.
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
        os.environ.setdefault("FLAGS_use_mkldnn", "False")
        os.environ.setdefault("FLAGS_use_onednn", "False")
        os.environ.setdefault("FLAGS_use_dnnl", "False")
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError("Install paddleocr to enable Paddle OCR") from exc
        # Works across PaddleOCR 2.x and 3.x with minor API differences.
        params = inspect.signature(PaddleOCR).parameters
        self._legacy_api = "use_angle_cls" in params
        if self._legacy_api:
            kwargs = dict(use_angle_cls=True, lang=lang, show_log=False, use_gpu=use_gpu)
            try:
                self.ocr = PaddleOCR(**kwargs)
            except TypeError:
                kwargs.pop("show_log", None)
                self.ocr = PaddleOCR(**kwargs)
        else:
            kwargs = dict(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )
            if use_gpu:
                kwargs["device"] = "gpu:0"
            try:
                self.ocr = PaddleOCR(**kwargs)
            except TypeError:
                kwargs.pop("device", None)
                self.ocr = PaddleOCR(**kwargs)
        self.engine = "paddleocr"

    def recognize(self, image_bgr: np.ndarray) -> List[OCRLine]:
        image_bgr = enhance_crop(image_bgr)
        if self._legacy_api:
            try:
                result = self.ocr.ocr(image_bgr, cls=True)
            except TypeError:
                result = self.ocr.ocr(image_bgr)
        else:
            result = self.ocr.ocr(image_bgr)
        lines: List[OCRLine] = []
        if not result:
            return lines
        # PaddleOCR 3.x returns page-level mappings with rec_texts/rec_scores/rec_polys.
        if isinstance(result, list) and result and hasattr(result[0], "get"):
            for page in result:
                texts = list(page.get("rec_texts") or [])
                scores = list(page.get("rec_scores") or [])
                boxes = list(page.get("rec_polys") or page.get("rec_boxes") or [])
                for i, text in enumerate(texts):
                    text = normalize_text(str(text))
                    if not text:
                        continue
                    conf = float(scores[i]) if i < len(scores) else 0.0
                    box = boxes[i] if i < len(boxes) else None
                    lines.append(OCRLine(text=text, confidence=conf, box=box, engine=self.engine))
            return lines
        # PaddleOCR sometimes returns [lines] and sometimes lines directly.
        if len(result) == 1 and isinstance(result[0], list) and result[0] and isinstance(result[0][0], (list, tuple)):
            candidates = result[0]
        else:
            candidates = result
        for item in candidates:
            try:
                box = item[0]
                text = item[1][0]
                conf = float(item[1][1])
            except Exception:
                continue
            text = normalize_text(str(text))
            if text:
                lines.append(OCRLine(text=text, confidence=conf, box=box, engine=self.engine))
        return lines

class TesseractOCREngine(BaseOCREngine):
    def __init__(self, lang: str = "rus+eng"):
        import pytesseract  # noqa: F401
        self.lang = lang
        self.engine = "tesseract"

    def recognize(self, image_bgr: np.ndarray) -> List[OCRLine]:
        import pytesseract

        image_bgr = enhance_crop(image_bgr)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        lines: List[OCRLine] = []
        # Use data output to get confidences; PSM 6 for uniform block, 11 sparse as fallback.
        for psm in [6, 11, 12]:
            config = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
            try:
                data = pytesseract.image_to_data(rgb, lang=self.lang, config=config, output_type=pytesseract.Output.DICT)
            except Exception:
                continue
            n = len(data.get("text", []))
            row_acc = {}
            for i in range(n):
                text = normalize_text(str(data["text"][i]))
                if not text:
                    continue
                try:
                    conf = float(data["conf"][i]) / 100.0
                except Exception:
                    conf = 0.0
                if conf < 0.05:
                    continue
                key = (data.get("block_num", [0]*n)[i], data.get("par_num", [0]*n)[i], data.get("line_num", [i]*n)[i])
                row_acc.setdefault(key, []).append((text, conf, data.get("left", [0]*n)[i], data.get("top", [0]*n)[i], data.get("width", [0]*n)[i], data.get("height", [0]*n)[i]))
            for toks in row_acc.values():
                line_text = normalize_text(" ".join(t[0] for t in toks))
                if len(line_text) < 2:
                    continue
                conf = float(np.mean([t[1] for t in toks]))
                x1 = min(t[2] for t in toks); y1 = min(t[3] for t in toks)
                x2 = max(t[2]+t[4] for t in toks); y2 = max(t[3]+t[5] for t in toks)
                lines.append(OCRLine(text=line_text, confidence=conf, box=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]], engine=f"{self.engine}-psm{psm}"))
            if lines:
                break
        # Deduplicate near-identical lines.
        unique: List[OCRLine] = []
        seen = set()
        for line in sorted(lines, key=lambda x: x.confidence, reverse=True):
            key = re.sub(r"\W+", "", line.text.lower())
            if key and key not in seen:
                unique.append(line); seen.add(key)
        return unique

class EnsembleOCREngine(BaseOCREngine):
    def __init__(self, prefer_paddle: bool = True, lang: str = "ru", use_gpu: bool = False):
        self.engines: List[BaseOCREngine] = []
        self._failure_counts: dict[int, int] = {}
        self._disabled_engines: set[int] = set()
        self._max_engine_failures = max(1, int(os.environ.get("LENTA_OCR_MAX_ENGINE_FAILURES", "3")))
        if prefer_paddle:
            try:
                self.engines.append(PaddleOCREngine(lang=lang, use_gpu=use_gpu))
            except Exception as exc:
                print(f"[WARN] PaddleOCR disabled: {exc}")
        try:
            self.engines.append(TesseractOCREngine(lang="rus+eng"))
        except Exception as exc:
            print(f"[WARN] Tesseract disabled: {exc}")

    def recognize(self, image_bgr: np.ndarray) -> List[OCRLine]:
        all_lines: List[OCRLine] = []
        for idx, engine in enumerate(self.engines):
            if idx in self._disabled_engines:
                continue
            try:
                all_lines.extend(engine.recognize(image_bgr))
            except Exception as exc:
                print(f"[WARN] OCR engine failed: {exc}")
                self._failure_counts[idx] = self._failure_counts.get(idx, 0) + 1
                if self._failure_counts[idx] >= self._max_engine_failures:
                    self._disabled_engines.add(idx)
                    print(f"[WARN] OCR engine {getattr(engine, 'engine', idx)} disabled after {self._failure_counts[idx]} failures")
        # Keep high-confidence unique text first.
        unique: List[OCRLine] = []
        seen = set()
        for line in sorted(all_lines, key=lambda l: l.confidence, reverse=True):
            key = re.sub(r"\W+", "", line.text.lower())
            if not key or key in seen:
                continue
            unique.append(line)
            seen.add(key)
        return unique
