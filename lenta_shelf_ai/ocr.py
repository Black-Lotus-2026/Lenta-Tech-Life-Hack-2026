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


def suppress_specular_glare_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    """Bounded reflection cleanup before OCR.

    Unlike QR decoding, OCR receives one enhanced crop, so this is conservative:
    it only inpaints obvious large white glare blobs and leaves normal white
    price-tag background untouched.
    """
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if h < 40 or w < 40:
        return image_bgr
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((val > 242) & (sat < 38)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(min(h, w) * 0.025)),) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    ratio = float((mask > 0).mean())
    if ratio < 0.010 or ratio > 0.32:
        return image_bgr
    try:
        return cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)
    except Exception:
        return image_bgr


def suppress_code_artifacts(image_bgr: np.ndarray) -> np.ndarray:
    """Mask dense QR/barcode-like regions for OCR only.

    QR and 1D barcodes are decoded from the original crop before OCR. For OCR,
    their high-frequency black modules often dominate thresholding and create
    garbage tokens. This conservative mask targets dense machine-code blocks on
    the right/lower part of the tag and leaves normal text/price areas intact.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if h < 40 or w < 40:
        return image_bgr

    out = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, 120)
    edges = cv2.Canny(gray, 45, 160)
    texture = cv2.bitwise_or(dark, edges)
    texture = cv2.morphologyEx(texture, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(texture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_code_boxes: list[tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 6 or bh < 6:
            continue
        area_ratio = (bw * bh) / max(1.0, float(w * h))
        if not 0.002 <= area_ratio <= 0.45:
            continue
        roi_dark = dark[y : y + bh, x : x + bw]
        roi_edges = edges[y : y + bh, x : x + bw]
        dark_density = float((roi_dark > 0).mean())
        edge_density = float((roi_edges > 0).mean())
        aspect = bw / max(1, bh)

        right_or_lower = x > 0.42 * w or y > 0.48 * h
        qr_like = 0.55 <= aspect <= 1.85 and dark_density >= 0.16 and edge_density >= 0.045
        barcode_like = (
            (aspect >= 2.8 or aspect <= 0.36)
            and dark_density >= 0.11
            and edge_density >= 0.040
            and (y > 0.45 * h or x > 0.50 * w)
        )
        if right_or_lower and qr_like and area_ratio >= 0.002:
            small_code_boxes.append((x, y, x + bw, y + bh))
        if not right_or_lower or not (qr_like or barcode_like):
            continue

        pad = max(2, int(0.03 * max(bw, bh)))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        bg = np.percentile(out.reshape(-1, 3), 88, axis=0).astype(np.uint8).tolist()
        cv2.rectangle(out, (x1, y1), (x2, y2), bg, thickness=-1)
    if len(small_code_boxes) >= 6:
        x1 = min(box[0] for box in small_code_boxes)
        y1 = min(box[1] for box in small_code_boxes)
        x2 = max(box[2] for box in small_code_boxes)
        y2 = max(box[3] for box in small_code_boxes)
        bw, bh = x2 - x1, y2 - y1
        area_ratio = (bw * bh) / max(1.0, float(w * h))
        aspect = bw / max(1, bh)
        if 0.01 <= area_ratio <= 0.45 and 0.45 <= aspect <= 2.2 and (x1 > 0.42 * w or y1 > 0.45 * h):
            pad = max(2, int(0.025 * max(bw, bh)))
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            bg = np.percentile(out.reshape(-1, 3), 88, axis=0).astype(np.uint8).tolist()
            cv2.rectangle(out, (x1, y1), (x2, y2), bg, thickness=-1)
    return out


def enhance_crop(image_bgr: np.ndarray, max_side: int = 1600, suppress_artifacts: bool = True) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    if suppress_artifacts:
        image_bgr = suppress_code_artifacts(image_bgr)
    if os.environ.get("LENTA_OCR_ENABLE_GLARE_SUPPRESSION", "1") != "0":
        image_bgr = suppress_specular_glare_for_ocr(image_bgr)
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


def _crop_zone(image_bgr: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    ix1 = max(0, min(w - 1, int(round(x1 * w))))
    iy1 = max(0, min(h - 1, int(round(y1 * h))))
    ix2 = max(0, min(w, int(round(x2 * w))))
    iy2 = max(0, min(h, int(round(y2 * h))))
    if ix2 <= ix1 or iy2 <= iy1:
        return image_bgr[:0, :0].copy()
    return image_bgr[iy1:iy2, ix1:ix2].copy()


def split_price_tag_zones(image_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Heuristic OCR zones for Lenta tags.

    Full-crop OCR often spends capacity on QR/barcode texture and misses prices.
    These zones intentionally overlap: product/name top area, price-dominant left
    and right panels, and lower barcode/SKU/date strip. They require no template
    labels and keep a full-crop fallback.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    h, w = image_bgr.shape[:2]
    if h < 24 or w < 24:
        return [("full", image_bgr)]
    zones = [
        ("full", image_bgr),
        ("product_top", _crop_zone(image_bgr, (0.00, 0.00, 1.00, 0.62))),
        ("price_left", _crop_zone(image_bgr, (0.00, 0.18, 0.58, 0.92))),
        ("price_right", _crop_zone(image_bgr, (0.42, 0.18, 1.00, 0.92))),
        ("lower_codes", _crop_zone(image_bgr, (0.00, 0.55, 1.00, 1.00))),
        ("center", _crop_zone(image_bgr, (0.12, 0.12, 0.88, 0.88))),
    ]
    out: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[int, int]] = set()
    for name, crop in zones:
        if crop is None or crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        if ch < 18 or cw < 18:
            continue
        key = (ch, cw)
        if name != "full" and key in seen:
            continue
        out.append((name, crop))
        seen.add(key)
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


class EasyOCREngine(BaseOCREngine):
    """Optional EasyOCR backend used when PaddleOCR is unavailable or too heavy.

    The team clean pipeline currently gets price fields only through EasyOCR.
    Keep this backend opt-in so lightweight local runs do not download/load extra
    models unless Kaggle or the user explicitly requests it.
    """

    def __init__(self, lang: str = "ru", use_gpu: bool = False):
        try:
            import easyocr
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError("Install easyocr to enable EasyOCR") from exc
        langs_raw = os.environ.get("LENTA_EASYOCR_LANGS", "")
        if langs_raw:
            langs = [item.strip() for item in langs_raw.replace(";", ",").split(",") if item.strip()]
        else:
            langs = [lang] if lang else ["ru"]
            if "en" not in langs:
                langs.append("en")
        self.reader = easyocr.Reader(langs, gpu=bool(use_gpu), verbose=False)
        self.engine = "easyocr"
        self._variants = max(1, int(os.environ.get("LENTA_EASYOCR_VARIANTS", "1")))

    def recognize(self, image_bgr: np.ndarray) -> List[OCRLine]:
        image_bgr = enhance_crop(image_bgr)
        if image_bgr is None or image_bgr.size == 0:
            return []
        variants: list[tuple[str, np.ndarray]] = [("normal", image_bgr)]
        if self._variants >= 2:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
            variants.append(("inverted", cv2.bitwise_not(gray)))
        lines: List[OCRLine] = []
        for variant_name, variant in variants[: self._variants]:
            try:
                result = self.reader.readtext(variant, detail=1, paragraph=False)
            except Exception:
                continue
            for item in result or []:
                try:
                    box, text, conf = item
                except Exception:
                    continue
                text = normalize_text(str(text))
                if text:
                    lines.append(OCRLine(text=text, confidence=float(conf), box=box, engine=f"{self.engine}-{variant_name}"))
            if lines and os.environ.get("LENTA_EASYOCR_FAST_EXIT", "1") != "0":
                break
        return lines


class RapidOCREngine(BaseOCREngine):
    """Optional local RapidOCR/ONNXRuntime backend.

    It is useful on Kaggle/CPU when PaddleOCR is slow or unavailable. The class
    supports both rapidocr_onnxruntime and newer rapidocr packages and is
    failure-isolated by EnsembleOCREngine.
    """

    def __init__(self, lang: str = "ru", use_gpu: bool = False):  # noqa: ARG002 - RapidOCR package decides providers
        engine_cls = None
        import_error: Exception | None = None
        for module_name in ("rapidocr_onnxruntime", "rapidocr"):
            try:
                module = __import__(module_name, fromlist=["RapidOCR"])
                engine_cls = getattr(module, "RapidOCR")
                break
            except Exception as exc:  # pragma: no cover - optional dep
                import_error = exc
        if engine_cls is None:
            raise ImportError("Install rapidocr_onnxruntime or rapidocr to enable RapidOCR") from import_error
        kwargs = {}
        rec_model = os.environ.get("LENTA_RAPIDOCR_REC_MODEL", "").strip()
        det_model = os.environ.get("LENTA_RAPIDOCR_DET_MODEL", "").strip()
        cls_model = os.environ.get("LENTA_RAPIDOCR_CLS_MODEL", "").strip()
        if rec_model:
            kwargs["rec_model_path"] = rec_model
        if det_model:
            kwargs["det_model_path"] = det_model
        if cls_model:
            kwargs["cls_model_path"] = cls_model
        try:
            self.reader = engine_cls(**kwargs)
        except TypeError:
            self.reader = engine_cls()
        self.engine = "rapidocr"

    @staticmethod
    def _iter_result_items(result):
        if result is None:
            return []
        if isinstance(result, tuple) and result:
            result = result[0]
        if isinstance(result, dict):
            # Newer APIs can return dict-like rec_texts/rec_scores/rec_boxes.
            texts = result.get("rec_texts") or result.get("texts") or []
            scores = result.get("rec_scores") or result.get("scores") or []
            boxes = result.get("rec_boxes") or result.get("dt_polys") or result.get("boxes") or []
            return list(zip(boxes or [None] * len(texts), texts, scores or [0.0] * len(texts)))
        return result or []

    def recognize(self, image_bgr: np.ndarray) -> List[OCRLine]:
        image_bgr = enhance_crop(image_bgr)
        if image_bgr is None or image_bgr.size == 0:
            return []
        try:
            result = self.reader(image_bgr)
        except TypeError:
            result = self.reader.ocr(image_bgr)
        lines: List[OCRLine] = []
        for item in self._iter_result_items(result):
            box = None
            text = ""
            conf = 0.0
            try:
                if isinstance(item, dict):
                    text = str(item.get("text") or item.get("rec_text") or "")
                    conf = float(item.get("score") or item.get("confidence") or 0.0)
                    box = item.get("box") or item.get("points")
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    box, text, conf = item[0], item[1], item[2]
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    text, conf = item[0], item[1]
                else:
                    text = str(item)
            except Exception:
                continue
            text = normalize_text(str(text))
            if not text:
                continue
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.0
            lines.append(OCRLine(text=text, confidence=conf_f, box=box, engine=self.engine))
        return lines


class EnsembleOCREngine(BaseOCREngine):
    def __init__(self, prefer_paddle: bool = True, lang: str = "ru", use_gpu: bool = False):
        self.engines: List[BaseOCREngine] = []
        self._failure_counts: dict[int, int] = {}
        self._disabled_engines: set[int] = set()
        self._max_engine_failures = max(1, int(os.environ.get("LENTA_OCR_MAX_ENGINE_FAILURES", "3")))
        disable_paddle = os.environ.get("LENTA_OCR_DISABLE_PADDLE", "0").strip().lower() in {"1", "true", "yes", "y", "on"} or os.environ.get("LENTA_DISABLE_PADDLE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
        if prefer_paddle and not disable_paddle:
            try:
                self.engines.append(PaddleOCREngine(lang=lang, use_gpu=use_gpu))
            except Exception as exc:
                print(f"[WARN] PaddleOCR disabled: {exc}")
        elif prefer_paddle and disable_paddle:
            print("[WARN] PaddleOCR disabled by LENTA_OCR_DISABLE_PADDLE/LENTA_DISABLE_PADDLE")
        if os.environ.get("LENTA_OCR_ENABLE_RAPIDOCR", "1").strip().lower() in {"1", "true", "yes", "y", "on"}:
            try:
                self.engines.append(RapidOCREngine(lang=lang, use_gpu=use_gpu))
            except Exception as exc:
                print(f"[WARN] RapidOCR disabled: {exc}")
        if os.environ.get("LENTA_OCR_ENABLE_EASYOCR", "0").strip().lower() in {"1", "true", "yes", "y", "on"}:
            try:
                self.engines.append(EasyOCREngine(lang=lang, use_gpu=use_gpu))
            except Exception as exc:
                print(f"[WARN] EasyOCR disabled: {exc}")
        try:
            self.engines.append(TesseractOCREngine(lang="rus+eng"))
        except Exception as exc:
            print(f"[WARN] Tesseract disabled: {exc}")

    def _recognize_with_engines(self, image_bgr: np.ndarray) -> List[OCRLine]:
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
        return all_lines

    @staticmethod
    def _dedupe_lines(lines: List[OCRLine]) -> List[OCRLine]:
        unique: List[OCRLine] = []
        seen = set()
        for line in sorted(lines, key=lambda l: l.confidence, reverse=True):
            key = re.sub(r"\W+", "", line.text.lower())
            if not key or key in seen:
                continue
            unique.append(line)
            seen.add(key)
        return unique

    def recognize(self, image_bgr: np.ndarray) -> List[OCRLine]:
        return self._dedupe_lines(self._recognize_with_engines(image_bgr))

    def recognize_zoned(self, image_bgr: np.ndarray, max_zones: int = 6) -> List[OCRLine]:
        """Run OCR on full crop plus local zones, then deduplicate.

        Heavy engines are still optional and failure-isolated. Callers should use
        this mostly with deferred OCR, because it multiplies OCR calls by zones.
        """
        all_lines: List[OCRLine] = []
        for zone_name, zone_crop in split_price_tag_zones(image_bgr)[:max_zones]:
            zone_lines = self._recognize_with_engines(zone_crop)
            # Slightly downweight non-full duplicate zones, but keep them high
            # enough to win when full-crop OCR misses a price/product token.
            zone_weight = 1.0 if zone_name == "full" else 0.97
            for line in zone_lines:
                all_lines.append(
                    OCRLine(
                        text=line.text,
                        confidence=float(line.confidence) * zone_weight,
                        box=line.box,
                        engine=f"{line.engine}|zone:{zone_name}",
                    )
                )
        return self._dedupe_lines(all_lines)
