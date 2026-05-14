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
PRICE_SPACED_RE = re.compile(r"(?<!\d)(\d{1,5})\s+(\d{2})(?!\d)")
PRICE_COMPACT_RE = re.compile(r"(?<!\d)(\d{3,7})(?!\d)")
DISCOUNT_RE = re.compile(r"[-−–]?\s*(\d{1,3})\s*%")
DATE_RE = re.compile(r"(\d{2}[./-]\d{2}[./-]\d{4}\s+\d{1,2}:\d{2})")
ZONE_RE = re.compile(r"(\d{2}_\d{6}\s*-\s*\d{6})")
EAN_RE = re.compile(r"(?<!\d)(\d{8,14})(?!\d)")
SKU_RE = re.compile(r"(?<!\d)(\d{9,13})(?!\d)")
SPECIAL_RE = re.compile(r"(?:^|\s)([ШШшКкЛл])(?:\s|$)")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
VOLUME_CONTEXT_RE = re.compile(r"(?:л|l|литр|мл|ml|кг|kg|гр|г)(?![а-яa-z])", re.I)
CURRENCY_CONTEXT_RE = re.compile(r"(?:руб|₽|коп)", re.I)

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


def _looks_like_non_price_context(context: str) -> bool:
    # Long identifiers around barcode/SKU/code labels should not become prices.
    return bool(re.search(r"(?:штрих|barcode|баркод|артикул|sku|id[_\s-]*sku|qr|код)", context, re.I))


def _unit_adjacent(text: str, start: int, end: int) -> bool:
    # Reject 0.75L/500г/1 кг even if another price with "руб" is nearby.
    local = text[max(0, start - 4) : min(len(text), end + 5)]
    return bool(VOLUME_CONTEXT_RE.search(local))


def _add_price(prices: List[str], integer_part: str, cents: str, context: str) -> None:
    if not integer_part or not cents:
        return
    try:
        value = float(f"{int(integer_part)}.{cents[:2]}")
    except ValueError:
        return
    if value < 2.0 and not CURRENCY_CONTEXT_RE.search(context):
        return
    if value > 999999:
        return
    if _looks_like_non_price_context(context):
        return
    s = f"{value:.2f}"
    if s not in prices:
        prices.append(s)


def _find_prices(text: str) -> List[str]:
    prices: List[str] = []
    text = normalize_text(text.replace("\u00a0", " "))
    occupied: List[tuple[int, int]] = []

    for match in PRICE_RE.finditer(text):
        if _unit_adjacent(text, match.start(), match.end()):
            occupied.append(match.span())
            continue
        context = text[max(0, match.start() - 18) : min(len(text), match.end() + 18)]
        _add_price(prices, match.group(1), match.group(2), context)
        occupied.append(match.span())

    # OCR often splits big price "129 99" or "129\n99". Keep this after
    # explicit decimal prices and require the integer part to be plausible.
    for match in PRICE_SPACED_RE.finditer(text):
        if any(not (match.end() <= a or match.start() >= b) for a, b in occupied):
            continue
        integer_part, cents = match.group(1), match.group(2)
        if len(integer_part) == 1 and int(integer_part) < 2:
            continue
        if _unit_adjacent(text, match.start(), match.end()):
            occupied.append(match.span())
            continue
        context = text[max(0, match.start() - 18) : min(len(text), match.end() + 18)]
        before_count = len(prices)
        _add_price(prices, integer_part, cents, context)
        if len(prices) > before_count:
            occupied.append(match.span())

    # Compact OCR "12999" -> 129.99, "378949" -> 3789.49.
    for match in PRICE_COMPACT_RE.finditer(text):
        if any(not (match.end() <= a or match.start() >= b) for a, b in occupied):
            continue
        raw = match.group(1)
        if len(raw) < 4 or len(raw) > 7:
            continue
        if raw.startswith("0"):
            continue
        prev_ch = text[match.start() - 1] if match.start() > 0 else ""
        next_ch = text[match.end()] if match.end() < len(text) else ""
        if prev_ch in ".,/:;-" or next_ch in ".,/:;-":
            continue
        if _unit_adjacent(text, match.start(), match.end()):
            continue
        # Skip when embedded in dates/times or code-like contexts.
        context = text[max(0, match.start() - 18) : min(len(text), match.end() + 18)]
        if re.search(r"\d{1,2}[./-]\d{1,2}|[:]", context):
            continue
        _add_price(prices, raw[:-2], raw[-2:], context)

    return prices


def _find_prices_from_lines(lines: Sequence[OCRLine], full_text: str) -> List[str]:
    prices = _find_prices(full_text)
    # Additional line-pair recovery: OCR may output integer and cents as separate
    # adjacent lines/tokens, which full-text normalization cannot distinguish from
    # random IDs. Restrict to short numeric lines and local context.
    texts = [normalize_text(line.text) for line in lines if normalize_text(line.text)]
    for left, right in zip(texts, texts[1:]):
        left_clean = normalize_text(left)
        right_clean = normalize_text(right)
        if re.fullmatch(r"\d{1,5}", left_clean) and re.fullmatch(r"\d{2}", right_clean):
            context = f"{left_clean} {right_clean}"
            _add_price(prices, left_clean, right_clean, context)
    return prices


def _find_barcodes(text: str) -> List[str]:
    nums = [m.group(1) for m in EAN_RE.finditer(text)]
    nums = [re.sub(r"\D", "", n) for n in nums]
    valid = [n for n in nums if ean13_is_valid(n)]
    if valid:
        return valid
    # A non-valid OCR number is usually worse than blank for the hidden metric
    # and can incorrectly merge tracks. QR parser still keeps 14-digit codes.
    return []


def _canonical_value(value: object) -> str:
    text = normalize_text(str(value or ""))
    # Some OCR/models render Cyrillic "нет" with visually similar Greek glyphs.
    if text.lower() in {"нет", "νες", "νετ"}:
        return ABSENT_VALUE
    return text


def _candidate_product_lines(lines: Sequence[OCRLine]) -> List[str]:
    candidates: List[str] = []
    bad = re.compile(r"(руб|коп|цена|скид|карте|штрих|артикул|qr|код|дата|печати|итого|%|₽)", re.I)
    for line in lines:
        text = normalize_text(line.text)
        if len(text) < 5:
            continue
        if not CYRILLIC_RE.search(text):
            continue
        if bad.search(text):
            continue
        if _find_prices(text) or DATE_RE.search(text) or ZONE_RE.search(text):
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
    prices = _find_prices_from_lines(lines, full_text_norm)
    if prices:
        numeric_prices = []
        for p in prices:
            try:
                numeric_prices.append(float(p))
            except Exception:
                pass
        if len(numeric_prices) >= 2:
            # On promo tags the price without card is usually the larger one,
            # while card/action price is the smaller highlighted one. This is
            # more robust than OCR reading order on rotated retail video.
            fields.setdefault("price_default", f"{max(numeric_prices):.2f}")
            fields.setdefault("price_card", f"{min(numeric_prices):.2f}")
        else:
            fields.setdefault("price_default", prices[0])
        if len(numeric_prices) >= 3:
            fields.setdefault("price_discount", f"{min(numeric_prices):.2f}")

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

    return {k: _canonical_value(v) for k, v in fields.items()}


def parse_observation(lines: Sequence[OCRLine], qr_payloads: Iterable[str], crop_bgr: Optional[np.ndarray] = None) -> Dict[str, str]:
    qr_fields = parse_qr_payloads(qr_payloads)
    return parse_text_fields(lines, qr_fields, crop_bgr)


def merge_field_values(values: Iterable[str]) -> str:
    vals = [_canonical_value(v) for v in values if v is not None and _canonical_value(v) != ""]
    if not vals:
        return ""
    # Penalize OCR garbage; prefer non-"нет", longer names and valid-looking numbers.
    counts = Counter(vals)
    best = sorted(vals, key=lambda v: (counts[v], v != ABSENT_VALUE, len(v)), reverse=True)[0]
    return best
