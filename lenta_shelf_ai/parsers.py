from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

from .schema import ABSENT_VALUE, OCRLine, OUTPUT_COLUMNS
from .utils import normalize_text, price_to_str
from .qr import parse_qr_payloads

PRICE_RE = re.compile(r"(?<![\d./-])(\d{1,5})\s*[.,]\s*(\d{2})(?![\d./-])")
DISCOUNT_RE = re.compile(r"[-−–]?\s*(\d{1,3})\s*%")
DATE_RE = re.compile(r"(\d{2}[./-]\d{2}[./-]\d{4}\s+\d{1,2}:\d{2})")
ZONE_RE = re.compile(r"(\d{2}_\d{6}\s*-\s*\d{6})")
EAN_RE = re.compile(r"(?<!\d)(\d{8,14})(?!\d)")
SKU_RE = re.compile(r"(?<!\d)(\d{9,13})(?!\d)")
SPECIAL_RE = re.compile(r"(?:^|\s)([ШШшКкЛл])(?:\s|$)")

KNOWN_INFO_WORDS = [
    "сухое", "полусухое", "полусладкое", "сладкое", "брют", "экстра", "удачная упаковка",
    "номер на весах", "цена за 1 кг", "цена за 100 г", "100г", "1кг", "1 кг",
]


def ean13_is_valid(code: str) -> bool:
    code = re.sub(r"\D", "", str(code))
    if len(code) != 13:
        return False
    digits = [int(c) for c in code]
    checksum = (10 - ((sum(digits[:-1:2]) + 3 * sum(digits[1:-1:2])) % 10)) % 10
    return checksum == digits[-1]


def classify_color(crop_bgr: np.ndarray) -> str:
    if crop_bgr is None or crop_bgr.size == 0:
        return ""
    h, w = crop_bgr.shape[:2]
    # Use saturated pixels to avoid white text/background. If none, classify by lightness.
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat > 45) & (val > 60)
    if mask.mean() < 0.01:
        return "white"
    hue = hsv[:, :, 0][mask]
    if len(hue) == 0:
        return "white"
    # Circular histogram in OpenCV hue units [0, 179].
    hist, bins = np.histogram(hue, bins=18, range=(0, 180))
    dominant = (bins[int(np.argmax(hist))] + bins[int(np.argmax(hist)) + 1]) / 2
    if dominant < 15 or dominant >= 165:
        return "red"
    if dominant < 38:
        return "yellow"
    if dominant < 90:
        return "green"
    if dominant < 135:
        return "blue"
    if dominant < 165:
        return "purple"
    return "red"


def _find_prices(text: str) -> List[str]:
    prices = []
    for a, b in PRICE_RE.findall(text.replace("\u00a0", " ")):
        try:
            value = float(f"{a}.{b}")
        except ValueError:
            continue
        if 0.01 <= value <= 999999:
            s = f"{value:.2f}"
            if s not in prices:
                prices.append(s)
    return prices


def _find_barcodes(text: str) -> List[str]:
    nums = [m.group(1) for m in EAN_RE.finditer(text)]
    nums = [re.sub(r"\D", "", n) for n in nums]
    valid = [n for n in nums if ean13_is_valid(n)]
    return valid or nums


def _candidate_product_lines(lines: Sequence[OCRLine]) -> List[str]:
    candidates: List[str] = []
    bad = re.compile(r"(руб|коп|цена|скид|карте|штрих|артикул|qr|код|дата|печати|итого|%|₽)", re.I)
    for line in lines:
        text = normalize_text(line.text)
        if len(text) < 5:
            continue
        if bad.search(text):
            continue
        if PRICE_RE.search(text) or DATE_RE.search(text) or ZONE_RE.search(text):
            continue
        digits = sum(ch.isdigit() for ch in text)
        letters = sum(ch.isalpha() for ch in text)
        if letters < 3 or digits > max(5, letters):
            continue
        candidates.append(text)
    return candidates


def _clean_product_name(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^[^A-Za-zА-Яа-яЁё]+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" -;,.|_")


def parse_text_fields(lines: Sequence[OCRLine], qr_fields: Dict[str, str], crop_bgr: Optional[np.ndarray] = None) -> Dict[str, str]:
    full_text = "\n".join(line.text for line in lines)
    full_text_norm = normalize_text(full_text)
    fields: Dict[str, str] = {}

    # QR-derived fields are the most reliable.
    fields.update(qr_fields)
    if fields.get("qr_code_barcode"):
        fields.setdefault("barcode", re.sub(r"\D", "", fields["qr_code_barcode"]))
    # In public labels, p1 ~= default, p4 often card/action card. Keep text OCR as secondary.
    if fields.get("price1_qr"):
        fields.setdefault("price_default", fields["price1_qr"])
    if fields.get("price4_qr"):
        fields.setdefault("price_card", fields["price4_qr"])
    if fields.get("action_price_qr"):
        fields.setdefault("price_discount", fields["action_price_qr"])

    # OCR prices, sorted by reading order/confidence. Use when QR missing.
    prices = _find_prices(full_text_norm)
    if prices:
        fields.setdefault("price_default", prices[0])
        if len(prices) >= 2:
            # Smallest displayed price is often card/action; default is usually first/larger.
            try:
                fields.setdefault("price_card", f"{min(float(p) for p in prices):.2f}")
            except Exception:
                fields.setdefault("price_card", prices[1])
        if len(prices) >= 3:
            fields.setdefault("price_discount", prices[-1])

    discount = DISCOUNT_RE.search(full_text_norm)
    if discount:
        fields["discount_amount"] = f"-{discount.group(1)}%"

    dt = DATE_RE.search(full_text_norm)
    if dt:
        fields["print_datetime"] = dt.group(1).replace("-", ".")

    zone = ZONE_RE.search(full_text_norm)
    if zone:
        fields["code"] = zone.group(1).replace(" ", "")

    barcodes = _find_barcodes(full_text_norm)
    if barcodes:
        # Prefer a valid EAN that differs from obvious SKU-like internal IDs.
        fields.setdefault("barcode", barcodes[0])

    # SKU: choose a long number that is not barcode and often starts with 2/3 in Lenta data.
    sku_candidates = [re.sub(r"\D", "", m.group(1)) for m in SKU_RE.finditer(full_text_norm)]
    sku_candidates = [s for s in sku_candidates if s != fields.get("barcode")]
    if sku_candidates:
        preferred = [s for s in sku_candidates if len(s) >= 10 and s.startswith(("2", "3"))]
        fields.setdefault("id_sku", (preferred or sku_candidates)[0])

    special = SPECIAL_RE.search(full_text_norm.replace("|", " "))
    if special:
        fields["special_symbols"] = special.group(1).upper()

    # Product name: merge 1-3 best textual lines. Avoid duplicating noisy OCR tokens.
    product_lines = _candidate_product_lines(lines)
    if product_lines:
        # Price tag names are often split into 2-3 lines; keep only plausible top long text.
        merged = " ".join(product_lines[:3])
        fields.setdefault("product_name", _clean_product_name(merged))

    low = full_text_norm.lower()
    info = [w for w in KNOWN_INFO_WORDS if w in low]
    if info:
        fields["additional_info"] = "; ".join(dict.fromkeys(info))

    if crop_bgr is not None:
        color = classify_color(crop_bgr)
        if color:
            fields["color"] = color

    # Mark fields absent only when we can infer absence from template/QR; otherwise leave empty if unrecognized.
    for col in ["price_discount", "discount_amount", "code", "additional_info", "special_symbols"]:
        if col not in fields:
            # Conservative default expected by statement: absent -> "нет". For product/price/barcode leave blank if not recognized.
            fields[col] = ABSENT_VALUE

    # Normalize prices from OCR/QR.
    for col in ["price_default", "price_card", "price_discount", "price1_qr", "price2_qr", "price3_qr", "price4_qr", "action_price_qr", "wholesale_level_1_price", "wholesale_level_2_price"]:
        if col in fields and fields[col] not in {"", ABSENT_VALUE}:
            fields[col] = price_to_str(fields[col])

    return fields


def parse_observation(lines: Sequence[OCRLine], qr_payloads: Iterable[str], crop_bgr: Optional[np.ndarray] = None) -> Dict[str, str]:
    qr_fields = parse_qr_payloads(qr_payloads)
    return parse_text_fields(lines, qr_fields, crop_bgr)


def merge_field_values(values: Iterable[str]) -> str:
    vals = [str(v).strip() for v in values if v is not None and str(v).strip() != ""]
    if not vals:
        return ""
    # Penalize OCR garbage; prefer non-"нет", longer names and valid-looking numbers.
    counts = Counter(vals)
    best = sorted(vals, key=lambda v: (counts[v], v != ABSENT_VALUE, len(v)), reverse=True)[0]
    return best
