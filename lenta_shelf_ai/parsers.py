from __future__ import annotations

import os
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
DISCOUNT_PRICE_GLUE_RE = re.compile(r"(?<!\d)(\d{1,3})\s*%\s*(\d{3,7})(?!\d)")
DATE_RE = re.compile(r"(\d{2}[./-]\d{2}[./-]\d{4}\s+\d{1,2}:\d{2})")
ZONE_RE = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁё])("
    r"(?:\d{2,3}_\d{2,6}(?:\s*[-–]\s*\d{3,6})?(?:_[0-9A-ZА-ЯЁа-яё]+)*)"
    r"|(?:\d{2,3}\s+\d{2,6}(?:_[0-9A-ZА-ЯЁа-яё]+)+)"
    r"|(?:\d{4,6}\s*[-–]\s*\d{3,6})"
    r"|(?:\d{2,3}_[A-ZА-ЯЁ]{2,8}(?:_[0-9A-ZА-ЯЁа-яё]+)*)"
    r")(?![0-9A-Za-zА-Яа-яЁё])"
)
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

PRICE_OCR_TRANS = str.maketrans({
    "O": "0", "o": "0", "О": "0", "о": "0",
    "I": "1", "l": "1", "|": "1", "!": "1",
    "S": "5", "s": "5", "Ѕ": "5", "ѕ": "5",
    "Б": "6", "б": "6",
    "З": "3", "з": "3",
    "В": "8", "в": "8", "B": "8",
})


def _normalize_price_ocr_text(text: str) -> str:
    return normalize_text(str(text or "").replace("\u00a0", " ")).translate(PRICE_OCR_TRANS)


def _normalize_code_value(text: str) -> str:
    value = normalize_text(str(text or ""))
    value = re.sub(r"\s*[-–]\s*", "-", value)
    value = re.sub(r"^(\d{2,3})\s+(\d{2,6})", r"\1_\2", value)
    return value.replace(" ", "")


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
    # Long identifiers, dates and service labels around barcode/SKU/code labels
    # should not become prices. This directly targets public-run failures where
    # dates (16.04), barcodes and internal codes were parsed as price_default.
    return bool(
        re.search(
            r"(?:штрих|barcode|баркод|артикул|sku|id[_\s-]*sku|qr|код|дата|печати|весах|номер|партия|касса|терминал)",
            context,
            re.I,
        )
    )


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
    max_price = float(os.environ.get("LENTA_PRICE_MAX", "49999"))
    if value > max_price:
        return
    if _looks_like_non_price_context(context):
        return
    if DATE_RE.search(context) or re.search(r"\b\d{1,2}:\d{2}\b", context):
        return
    s = f"{value:.2f}"
    if s not in prices:
        prices.append(s)


def _find_prices(text: str) -> List[str]:
    prices: List[str] = []
    raw_text = normalize_text(str(text or "").replace("\u00a0", " "))
    text = _normalize_price_ocr_text(raw_text)
    occupied: List[tuple[int, int]] = []

    # Frequent glued OCR pattern: "28%1199" or "28%119999".
    # Treat the percent as discount evidence and the following run as a separate
    # price candidate; never parse the whole blob as a compact price.
    for match in DISCOUNT_PRICE_GLUE_RE.finditer(text):
        discount = int(match.group(1))
        digits = match.group(2)
        if 0 < discount <= 95:
            context = raw_text[max(0, match.start() - 18) : min(len(raw_text), match.end() + 18)]
            if len(digits) >= 5:
                _add_price(prices, digits[:-2], digits[-2:], context)
            else:
                _add_price(prices, digits, "00", context + " руб")
            occupied.append(match.span())

    for match in PRICE_RE.finditer(text):
        if _unit_adjacent(text, match.start(), match.end()):
            occupied.append(match.span())
            continue
        context = raw_text[max(0, match.start() - 18) : min(len(raw_text), match.end() + 18)]
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
        context = raw_text[max(0, match.start() - 18) : min(len(raw_text), match.end() + 18)]
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
        if (prev_ch and prev_ch in ".,/:;-") or (next_ch and next_ch in ".,/:;-"):
            continue
        if _unit_adjacent(text, match.start(), match.end()):
            continue
        # Skip when embedded in dates/times or code-like contexts.
        context = raw_text[max(0, match.start() - 18) : min(len(raw_text), match.end() + 18)]
        if re.search(r"\d{1,2}[./-]\d{1,2}|[:]", context):
            continue
        percent_local = text[max(0, match.start() - 4) : min(len(text), match.end() + 2)]
        if "%" in percent_local:
            continue
        _add_price(prices, raw[:-2], raw[-2:], context)

    return prices


def _line_box_xyxy(box: object) -> Optional[tuple[float, float, float, float]]:
    if box is None:
        return None
    try:
        arr = np.asarray(box, dtype=float)
        if arr.size == 4:
            flat = arr.reshape(-1)
            x1, y1, x2, y2 = [float(v) for v in flat]
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            return x1, y1, x2, y2
        if arr.ndim >= 2 and arr.shape[-1] == 2 and arr.size >= 8:
            pts = arr.reshape(-1, 2)
            xs = pts[:, 0]
            ys = pts[:, 1]
            return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    except Exception:
        return None
    return None


def _numeric_price_token(text: str, min_len: int, max_len: int) -> str:
    mapped = _normalize_price_ocr_text(text).strip()
    if re.search(r"[A-Za-zА-Яа-яЁё]", mapped):
        return ""
    # Pair recovery expects an integer token and a separate two-digit cents
    # token. Do not reuse already-decoded decimal prices or discount percents.
    if any(ch in mapped for ch in ",.%") or mapped.startswith(('-', '−', '–')):
        return ""
    digits = re.sub(r"\D", "", mapped)
    if not (min_len <= len(digits) <= max_len):
        return ""
    return digits


def _find_prices_from_geometry(lines: Sequence[OCRLine]) -> List[str]:
    prices: List[str] = []
    items: List[tuple[OCRLine, str, tuple[float, float, float, float]]] = []
    for line in lines:
        box = _line_box_xyxy(getattr(line, "box", None))
        if box is None:
            continue
        text = normalize_text(getattr(line, "text", ""))
        if not text or _looks_like_non_price_context(text):
            continue
        items.append((line, text, box))

    for left_line, left_text, left_box in items:
        integer_part = _numeric_price_token(left_text, 1, 5)
        if not integer_part:
            continue
        lx1, ly1, lx2, ly2 = left_box
        lw, lh = max(1.0, lx2 - lx1), max(1.0, ly2 - ly1)
        lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
        for right_line, right_text, right_box in items:
            if right_line is left_line:
                continue
            cents = _numeric_price_token(right_text, 2, 2)
            if not cents:
                continue
            rx1, ry1, rx2, ry2 = right_box
            rw, rh = max(1.0, rx2 - rx1), max(1.0, ry2 - ry1)
            rcx, rcy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
            if rcx <= lcx + 0.20 * lw:
                continue
            if rx1 > lx2 + 3.5 * max(lw, 16.0):
                continue
            if abs(rcy - lcy) > max(22.0, 0.75 * max(lh, rh)):
                continue
            if rh > 1.35 * lh:
                continue
            context = f"{left_text} {right_text}"
            _add_price(prices, integer_part, cents, context)
    return prices


def _find_prices_from_lines(lines: Sequence[OCRLine], full_text: str) -> List[str]:
    prices = _find_prices(full_text)
    # Additional line-pair recovery: OCR may output integer and cents as separate
    # adjacent lines/tokens, which full-text normalization cannot distinguish from
    # random IDs. Restrict to short numeric lines and local context.
    texts = [normalize_text(line.text) for line in lines if normalize_text(line.text)]
    for left, right in zip(texts, texts[1:]):
        left_clean = _numeric_price_token(left, 1, 5)
        right_clean = _numeric_price_token(right, 2, 2)
        if left_clean and right_clean:
            context = f"{left} {right}"
            _add_price(prices, left_clean, right_clean, context)
    for price in _find_prices_from_geometry(lines):
        if price not in prices:
            prices.append(price)
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



def _line_zone(line: OCRLine) -> str:
    engine = str(getattr(line, "engine", "") or "")
    m = re.search(r"\|zone:([A-Za-z0-9_\-]+)", engine)
    return m.group(1) if m else ""


def _lines_in_zone(lines: Sequence[OCRLine], *zones: str) -> List[OCRLine]:
    wanted = set(zones)
    return [line for line in lines if _line_zone(line) in wanted]


def _zone_text(lines: Sequence[OCRLine], *zones: str) -> str:
    return "\n".join(normalize_text(line.text) for line in _lines_in_zone(lines, *zones) if normalize_text(line.text))


NON_PRICE_ZONES = {"barcode", "qr_code_barcode", "id_sku", "print_datetime", "code", "lower_codes"}


def _price_candidate_lines(lines: Sequence[OCRLine]) -> List[OCRLine]:
    out: List[OCRLine] = []
    for line in lines:
        text = normalize_text(line.text)
        if not text:
            continue
        zone = _line_zone(line)
        if zone in NON_PRICE_ZONES:
            continue
        if _looks_like_non_price_context(text):
            continue
        if DATE_RE.search(text) or ZONE_RE.search(text) or re.search(r"\b\d{1,2}:\d{2}\b", text):
            continue
        out.append(line)
    return out


def _first_price_from_zone(lines: Sequence[OCRLine], zone: str) -> str:
    text = _zone_text(lines, zone)
    prices = _find_prices(text)
    return prices[0] if prices else ""


def _find_date_from_lines(lines: Sequence[OCRLine]) -> str:
    text = "\n".join(normalize_text(line.text) for line in lines if normalize_text(line.text))
    m = DATE_RE.search(text)
    return m.group(1).replace("-", ".") if m else ""


def _find_code_from_lines(lines: Sequence[OCRLine]) -> str:
    text = "\n".join(normalize_text(line.text) for line in lines if normalize_text(line.text))
    m = ZONE_RE.search(text)
    return _normalize_code_value(m.group(1)) if m else ""

def _canonical_value(value: object) -> str:
    text = normalize_text(str(value or ""))
    # Some OCR/models render Cyrillic "нет" with visually similar Greek glyphs.
    if text.lower() in {"нет", "νες", "νετ"}:
        return ABSENT_VALUE
    return text


def _candidate_product_lines(lines: Sequence[OCRLine]) -> List[str]:
    candidates: List[str] = []
    bad = re.compile(r"(руб|коп|цена|скид|карте|штрих|артикул|qr|код|дата|печати|итого|%|₽|товар\s+закончился|привез[её]м)", re.I)
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



def _price_as_float(value: object) -> Optional[float]:
    text = _canonical_value(value)
    if not text or text == ABSENT_VALUE:
        return None
    try:
        return float(price_to_str(text))
    except Exception:
        try:
            return float(str(text).replace(" ", "").replace(",", "."))
        except Exception:
            return None


def _normalize_default_card_pair(fields: Dict[str, str]) -> None:
    """Apply the Lenta promo rule from several team solutions.

    On discount/card tags, "Без карты" is the higher original/default price and
    "С картой" is the lower loyalty-card price. OCR zones sometimes swap these,
    especially after rotation, so normalize only when both fields are already
    present and numeric.
    """
    default_value = _price_as_float(fields.get("price_default"))
    card_value = _price_as_float(fields.get("price_card"))
    if default_value is None or card_value is None:
        return
    if default_value <= 0 or card_value <= 0:
        return
    if default_value + 0.005 < card_value:
        fields["price_default"], fields["price_card"] = fields.get("price_card", ""), fields.get("price_default", "")


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

    # Zone-detector hints: do not let a full-crop OCR line reorder
    # price_default/price_card/price_discount. QR remains higher priority
    # because it is machine-readable.
    zone_default = _first_price_from_zone(lines, "price_default")
    zone_card = _first_price_from_zone(lines, "price_card")
    zone_discount_price = _first_price_from_zone(lines, "price_discount")
    if zone_default:
        fields.setdefault("price_default", zone_default)
    if zone_card:
        fields.setdefault("price_card", zone_card)
    if zone_discount_price:
        fields.setdefault("price_discount", zone_discount_price)

    zone_discount_text = _zone_text(lines, "discount_amount")
    zone_discount = DISCOUNT_RE.search(zone_discount_text) if zone_discount_text else None
    if zone_discount:
        discount_value = int(zone_discount.group(1))
        if 0 < discount_value <= 95:
            fields.setdefault("discount_amount", f"-{discount_value}%")

    zone_dt = _find_date_from_lines(_lines_in_zone(lines, "print_datetime"))
    if zone_dt:
        fields.setdefault("print_datetime", zone_dt)
    zone_code = _find_code_from_lines(_lines_in_zone(lines, "code"))
    if zone_code:
        fields.setdefault("code", zone_code)

    zone_machine_text = _zone_text(lines, "barcode", "id_sku")
    if zone_machine_text:
        zbarcodes = _find_barcodes(zone_machine_text)
        if zbarcodes:
            fields.setdefault("barcode", zbarcodes[0])
        zskus = [re.sub(r"\D", "", m.group(1)) for m in SKU_RE.finditer(zone_machine_text)]
        zskus = [x for x in zskus if x and x != fields.get("barcode")]
        if zskus:
            preferred = [x for x in zskus if len(x) >= 10 and x.startswith(("2", "3"))]
            fields.setdefault("id_sku", (preferred or zskus)[0])

    product_zone_lines = _candidate_product_lines(_lines_in_zone(lines, "product_name"))
    if product_zone_lines:
        fields.setdefault("product_name", _clean_product_name(" ".join(product_zone_lines[:3])))

    # OCR prices, sorted by reading order/confidence. Use when QR/zone fields are missing.
    # Do not feed barcode/SKU/date/code zones into price recovery: public debug
    # showed false values such as 16.04, 23.45 and 62327.49 winning fusion.
    price_lines = _price_candidate_lines(lines)
    price_text_norm = normalize_text("\n".join(line.text for line in price_lines))
    prices = _find_prices_from_lines(price_lines, price_text_norm)
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
        # Do not infer price_discount just because three numbers were seen. In
        # the public labels price_discount is usually absent ("нет"); only QR
        # action_price_qr or explicit price_discount zone may populate it.

    discount = DISCOUNT_RE.search(full_text_norm)
    if discount:
        discount_value = int(discount.group(1))
        if 0 < discount_value <= 95:
            fields["discount_amount"] = f"-{discount_value}%"

    dt = DATE_RE.search(full_text_norm)
    if dt:
        fields["print_datetime"] = dt.group(1).replace("-", ".")

    zone = ZONE_RE.search(full_text_norm)
    if zone:
        fields["code"] = _normalize_code_value(zone.group(1))

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
    _normalize_default_card_pair(fields)

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
