# -*- coding: utf-8 -*-

import argparse
import logging
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
import easyocr

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
except Exception:
    pyzbar_decode = None


def rotate_image(img, deg):
    """Поворот изображения"""
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# ============================================================
# LOGGING
# ============================================================

class TimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

for handler in logging.root.handlers:
    handler.setFormatter(
        TimeFormatter("[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

logger = logging.getLogger("tracking")

# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_COLUMNS = [
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min", "y_min", "x_max", "y_max",
    "qr_code_barcode",
    "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]

FIELD_CLASSES = {
    "additional_info", "barcode", "code", "discount_amount", "id_sku",
    "price_card", "price_default", "price_discount", "print_datetime",
    "product_name", "qr_code_barcode",
}

PRICE_FIELDS = {"price_card", "price_default", "price_discount"}
QR_FIELDS = {"qr_code_barcode", "barcode", "code"}
TEXT_FIELDS = {"product_name", "additional_info", "print_datetime", "id_sku"}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Detection:
    class_name: str
    conf: float
    bbox: Tuple[int, int, int, int]


@dataclass
class TrackState:
    track_id: int
    first_frame: int
    first_bbox: Tuple[int, int, int, int]
    last_frame: int
    last_bbox: Tuple[int, int, int, int]
    best_score: float = -1.0
    field_votes: Dict[str, List[Tuple[str, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    lost_counter: int = 0


# ============================================================
# EasyOCR
# ============================================================

logger.info("Инициализация EasyOCR...")
try:
    ocr_en = easyocr.Reader(['en'], gpu=True)  # Для цифр и латиницы
    ocr_ru = easyocr.Reader(['ru'], gpu=True)  # Для русского текста
    logger.info("EasyOCR успешно инициализирован")
except Exception as e:
    logger.error(f"Ошибка EasyOCR: {e}")
    ocr_en = None
    ocr_ru = None


# ============================================================
# HELPERS
# ============================================================

def resolve_project_root():
    start = pathlib.Path.cwd().resolve()
    for p in [start, *start.parents]:
        if (p / "data").exists():
            return p
    return start


def expand_box(box, pad, w, h):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad)
    py = int(bh * pad)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(w - 1, x2 + px),
        min(h - 1, y2 + py)
    )


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def distance(box1, box2):
    x1, y1 = center(box1)
    x2, y2 = center(box2)
    return np.hypot(x1 - x2, y1 - y2)


def iou(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (ax2 - ax1) * (ay2 - ay1)
    area2 = (bx2 - bx1) * (by2 - by1)
    union = area1 + area2 - inter
    if union <= 0:
        return 0
    return inter / union


def preprocess_for_ocr(img, target_h=100):
    """Предобработка изображения для OCR"""
    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if h < target_h:
        scale = target_h / h
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


def preprocess_for_qr(img, target_size=300):
    """Предобработка для QR/штрихкодов с усилением контраста"""
    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Увеличиваем
    if min(h, w) < target_size:
        scale = target_size / min(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Усиливаем контраст
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced


def clean_text(txt):
    """Очистка текста"""
    if not txt:
        return ""
    txt = str(txt).replace("\n", " ").replace("\t", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def clean_price_text(text):
    """Извлечение цены из текста"""
    if not text:
        return ""

    cleaned = re.sub(r'[^\d.,]', '', text)
    cleaned = cleaned.replace(',', '.')

    parts = cleaned.split('.')
    if len(parts) > 2:
        cleaned = ''.join(parts[:-1]) + '.' + parts[-1]

    if '.' not in cleaned:
        if len(cleaned) >= 3:
            return f"{cleaned[:-2]}.{cleaned[-2:]}"
        elif len(cleaned) > 0:
            return f"{cleaned}.00"
        return ""

    int_part, frac_part = cleaned.split('.')
    if not frac_part:
        return f"{int_part}.00"
    elif len(frac_part) == 1:
        return f"{int_part}.{frac_part}0"
    else:
        return f"{int_part}.{frac_part[:2]}"


def clean_discount(text):
    """Очистка скидки в формате -число%"""
    if not text:
        return ""

    # Ищем число со знаком минус или просто число
    text = str(text).replace(",", ".").replace(" ", "")

    # Паттерн: -число% или число%
    match = re.search(r'-?\d+\.?\d*\s*%', text)
    if match:
        val = match.group()
        # Убираем пробел перед %
        val = val.replace(" ", "")
        # Если нет минуса, добавляем
        if not val.startswith('-'):
            val = '-' + val
        return val

    # Если нет знака %, ищем просто число
    match = re.search(r'-?\d+\.?\d*', text)
    if match:
        val = match.group()
        if not val.startswith('-'):
            val = '-' + val
        return f"{val}%"

    return ""


def clean_sku(text):
    """Очистка артикула (цифры, может быть длинным)"""
    if not text:
        return ""
    digits = re.sub(r'[^\d]', '', str(text))
    return digits if len(digits) >= 4 else ""


def clean_datetime(text):
    """Очистка даты/времени"""
    if not text:
        return ""
    text = str(text).strip()
    # Ищем паттерны даты: DD.MM.YYYY или DD/MM/YYYY и т.д.
    date_match = re.search(r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}', text)
    if date_match:
        return date_match.group()

    time_match = re.search(r'\d{1,2}:\d{2}(:\d{2})?', text)
    if time_match:
        return time_match.group()

    return clean_text(text)


def parse_qr_prices(qr_text):
    """Извлечение цен из QR-кода"""
    if not qr_text:
        return []
    vals = re.findall(r'\d+[.,]\d{2}', qr_text)
    return [v.replace(",", ".") for v in vals[:4]]


# ============================================================
# OCR
# ============================================================

def ocr_price(img):
    """Распознавание цены"""
    if img is None or img.size == 0 or ocr_en is None:
        return "", 0.0

    processed = preprocess_for_ocr(img)
    if processed is None:
        return "", 0.0

    h, w = processed.shape[:2]
    logger.info(f"  [price] размер {w}x{h}")

    variants = [
        ("normal", processed),
        ("inverted", cv2.bitwise_not(processed)),
    ]

    for vname, variant in variants:
        try:
            results = ocr_en.readtext(variant, detail=1, paragraph=False)

            if results:
                large_parts = []
                small_parts = []

                for bbox, text, score in results:
                    y_coords = [p[1] for p in bbox]
                    height = max(y_coords) - min(y_coords)
                    digit_text = re.sub(r'[^\d.]', '', text)

                    if digit_text:
                        logger.info(f"  [{vname}] '{digit_text}' h={height:.0f} conf={score:.3f}")

                        if height > h * 0.5:
                            large_parts.append((height, digit_text))
                        else:
                            small_parts.append((height, digit_text))

                if large_parts or small_parts:
                    large_parts.sort(key=lambda x: x[0], reverse=True)
                    small_parts.sort(key=lambda x: x[0], reverse=True)

                    int_part = ''.join([p[1] for p in large_parts])
                    frac_part = ''.join([p[1] for p in small_parts])

                    if int_part and frac_part:
                        price = f"{int_part}.{frac_part[:2]}"
                    elif int_part:
                        price = f"{int_part[:-2]}.{int_part[-2:]}" if len(int_part) >= 3 else f"{int_part}.00"
                    elif frac_part:
                        price = f"0.{frac_part[:2]}"
                    else:
                        continue

                    if re.match(r'^\d+\.\d{2}$', price):
                        logger.info(f"  ✓ Цена [{vname}]: {price}")
                        return price, 0.8

        except Exception as e:
            logger.error(f"  OCR error [{vname}]: {e}")

    logger.info(f"  ✗ Цена не распознана")
    return "", 0.0


def ocr_discount(img):
    """Распознавание скидки. Ищет число и форматирует как -число%"""
    if img is None or img.size == 0 or ocr_en is None:
        return "", 0.0

    processed = preprocess_for_ocr(img, target_h=80)
    if processed is None:
        return "", 0.0

    h, w = processed.shape[:2]
    logger.info(f"  [discount] размер {w}x{h}")

    variants = [
        ("normal", processed),
        ("inverted", cv2.bitwise_not(processed)),
    ]

    for vname, variant in variants:
        try:
            # Для скидки не используем allowlist, чтобы видеть минус и процент
            results = ocr_en.readtext(variant, detail=1, paragraph=False)

            if results:
                all_texts = []
                all_scores = []

                for bbox, text, score in results:
                    if text.strip():
                        all_texts.append(text.strip())
                        all_scores.append(score)
                        logger.info(f"  [{vname}] '{text.strip()}' conf={score:.3f}")

                if all_texts:
                    combined = " ".join(all_texts)
                    avg_score = sum(all_scores) / len(all_scores)

                    # Ищем число (возможно с минусом и процентом)
                    cleaned = clean_discount(combined)
                    if cleaned:
                        logger.info(f"  ✓ Скидка [{vname}]: {cleaned}")
                        return cleaned, float(avg_score)

        except Exception as e:
            logger.error(f"  OCR error [{vname}]: {e}")

    logger.info(f"  ✗ Скидка не распознана")
    return "", 0.0


def ocr_text_field(img, field_name):
    """Распознавание текстового поля с группировкой по размеру шрифта"""
    if img is None or img.size == 0:
        return "", 0.0

    # Для русского текста используем ocr_ru, для цифровых полей - ocr_en
    if field_name in {"id_sku", "print_datetime"}:
        reader = ocr_en
    else:
        reader = ocr_ru

    if reader is None:
        return "", 0.0

    processed = preprocess_for_ocr(img, target_h=80)
    if processed is None:
        return "", 0.0

    h, w = processed.shape[:2]
    logger.info(f"  [{field_name}] размер {w}x{h}")

    variants = [
        ("normal", processed),
        ("inverted", cv2.bitwise_not(processed)),
    ]

    for vname, variant in variants:
        try:
            # НЕ используем paragraph=True - он плохо работает с ценниками
            results = reader.readtext(variant, detail=1, paragraph=False)

            if results:
                # Группируем текст по Y-координате (строки)
                lines = defaultdict(list)

                for bbox, text, score in results:
                    if not text.strip():
                        continue

                    # Вычисляем центр Y для группировки по строкам
                    y_coords = [p[1] for p in bbox]
                    y_center = sum(y_coords) / len(y_coords)
                    height = max(y_coords) - min(y_coords)

                    # Группируем по Y с допуском в 30% высоты
                    line_key = round(y_center / (height * 0.7))
                    lines[line_key].append((bbox, text.strip(), score, y_center))

                    logger.info(f"  [{vname}] '{text.strip()}' y={y_center:.0f} h={height:.0f} conf={score:.3f}")

                if lines:
                    # Сортируем строки по Y (сверху вниз)
                    sorted_lines = sorted(lines.items(), key=lambda x: min(p[3] for p in x[1]))

                    all_texts = []
                    all_scores = []

                    for line_key, line_parts in sorted_lines:
                        # Сортируем части в строке по X (слева направо)
                        line_parts.sort(key=lambda p: min(pt[0] for pt in p[0]))
                        line_text = " ".join([p[1] for p in line_parts])
                        line_score = sum(p[2] for p in line_parts) / len(line_parts)

                        all_texts.append(line_text)
                        all_scores.append(line_score)

                    combined = " ".join(all_texts)
                    avg_score = sum(all_scores) / len(all_scores)

                    # Очистка в зависимости от типа поля
                    if field_name == "id_sku":
                        cleaned = clean_sku(combined)
                    elif field_name == "print_datetime":
                        cleaned = clean_datetime(combined)
                    else:
                        cleaned = clean_text(combined)

                    if cleaned and len(cleaned) > 1:
                        logger.info(f"  ✓ [{field_name}] [{vname}]: '{cleaned}'")
                        return cleaned, float(avg_score)

        except Exception as e:
            logger.error(f"  OCR error [{vname}]: {e}")

    logger.info(f"  ✗ [{field_name}] не распознано")
    return "", 0.0


def decode_qr_field(img):
    """Декодирование QR/штрихкода"""
    if img is None or img.size == 0 or pyzbar_decode is None:
        return ""

    processed = preprocess_for_qr(img)
    if processed is None:
        return ""

    h, w = processed.shape[:2]
    logger.info(f"  [QR] размер {w}x{h}")

    variants = [
        ("enhanced", processed),
        ("inverted", cv2.bitwise_not(processed)),
    ]

    # Добавляем увеличенный вариант
    big = cv2.resize(processed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    variants.append(("big", big))

    results = []
    for vname, variant in variants:
        try:
            decoded = pyzbar_decode(variant)
            if decoded:
                for d in decoded:
                    val = d.data.decode("utf-8", errors="ignore")
                    if val and val not in results:
                        results.append(val)
                        logger.info(f"  ✓ QR [{vname}]: {val[:100]}")
        except Exception as e:
            logger.debug(f"  QR error [{vname}]: {e}")

    if results:
        return " | ".join(results)

    logger.info(f"  ✗ QR не декодирован")
    return ""


# ============================================================
# YOLO
# ============================================================

def get_model_names(model):
    if hasattr(model, "names"):
        return model.names
    if hasattr(model.model, "names"):
        return model.model.names
    return {}


def detect(model, frame, conf, imgsz):
    names = get_model_names(model)
    results = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)
    out = []
    if not results:
        return out
    boxes = results[0].boxes
    if boxes is None:
        return out
    cls_arr = boxes.cls.cpu().numpy()
    conf_arr = boxes.conf.cpu().numpy()
    xyxy_arr = boxes.xyxy.cpu().numpy()
    for cls_id, cf, bb in zip(cls_arr, conf_arr, xyxy_arr):
        name = names.get(int(cls_id), str(int(cls_id)))
        x1, y1, x2, y2 = map(int, bb[:4])
        out.append(Detection(class_name=name, conf=float(cf), bbox=(x1, y1, x2, y2)))
    return out


# ============================================================
# TRACKING
# ============================================================

def update_votes(track, field_name, value, score):
    if not value:
        return
    track.field_votes[field_name].append((value, score))
    logger.info(f"  ✓ Track {track.track_id}: {field_name} = {value}")


def get_best_vote(votes):
    if not votes:
        return "нет"
    weighted = defaultdict(float)
    for val, score in votes:
        weighted[val] += score
    return max(weighted.items(), key=lambda x: x[1])[0]


def match_tracks(tracks, detections, frame_idx, next_track_id, max_lost=10):
    matched = []
    used_tracks = set()

    for det in detections:
        best_id = None
        best_score = -1

        for tid, track in tracks.items():
            if tid in used_tracks:
                continue
            i = iou(track.last_bbox, det.bbox)
            d = distance(track.last_bbox, det.bbox)
            score = i - d * 0.0001
            if score > best_score:
                best_score = score
                best_id = tid

        if best_id is not None and best_score > 0.05:
            tracks[best_id].last_bbox = det.bbox
            tracks[best_id].last_frame = frame_idx
            tracks[best_id].lost_counter = 0
            matched.append((best_id, det))
            used_tracks.add(best_id)
        else:
            tid = next_track_id
            next_track_id += 1
            tracks[tid] = TrackState(
                track_id=tid, first_frame=frame_idx, first_bbox=det.bbox,
                last_frame=frame_idx, last_bbox=det.bbox
            )
            matched.append((tid, det))

    for tid, track in tracks.items():
        if tid not in used_tracks:
            track.lost_counter += 1

    to_del = [tid for tid, track in tracks.items() if track.lost_counter > max_lost]
    for tid in to_del:
        del tracks[tid]

    return tracks, matched, next_track_id


# ============================================================
# MAIN
# ============================================================

def process_video(
        video_path,
        tag_model_path,
        field_model_path,
        out_csv,
        conf=0.1,
        imgsz=640,
        frame_stride=1,
        max_lost_frames=30,
        tag_rotation=270,
        debug_dir=None,
):
    """
    Логика:
    1. Видео повёрнуто на 90° по часовой
    2. Tag-модель ищет ценники на повёрнутом видео
    3. Вырезаем ценник, поворачиваем в нормальную ориентацию
    4. Field-модель ищет поля на нормальном ценнике
    5. OCR/QR для каждого поля
    """
    start_time = datetime.now()
    logger.info(f"=== НАЧАЛО ОБРАБОТКИ ВИДЕО ===")
    logger.info(f"Видео: {video_path}")
    logger.info(f"Поворот ценника: {tag_rotation}°")

    if debug_dir:
        debug_dir = pathlib.Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    tag_model = YOLO(str(tag_model_path))
    field_model = YOLO(str(field_model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Видео: {total} кадров, {fps:.2f} FPS")

    tracks = {}
    next_track_id = 1
    frame_idx = 0
    processed_frames = 0
    debug_saved = 0

    logger.info("Начинаю обработку...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        processed_frames += 1
        save_debug = debug_dir and debug_saved < 3

        # Ищем ценники на повёрнутом видео
        tag_dets = detect(tag_model, frame, conf=0.1, imgsz=imgsz)

        if tag_dets:
            if frame_idx % 50 == 0:
                logger.info(f"Кадр {frame_idx}: найдено {len(tag_dets)} ценников")

            tracks, matched, next_track_id = match_tracks(
                tracks, tag_dets, frame_idx, next_track_id, max_lost_frames
            )

            for track_id, det in matched:
                track = tracks[track_id]

                # Вырезаем ценник
                x1, y1, x2, y2 = expand_box(det.bbox, pad=0.2, w=frame.shape[1], h=frame.shape[0])
                tag_crop_rotated = frame[y1:y2, x1:x2]

                if tag_crop_rotated.size == 0:
                    continue

                # Поворачиваем в нормальную ориентацию
                if tag_rotation != 0:
                    tag_crop_normal = rotate_image(tag_crop_rotated, tag_rotation)
                else:
                    tag_crop_normal = tag_crop_rotated

                if save_debug and debug_saved == 0:
                    cv2.imwrite(str(debug_dir / f"frame{frame_idx}_track{track_id}_rotated.jpg"), tag_crop_rotated)
                    cv2.imwrite(str(debug_dir / f"frame{frame_idx}_track{track_id}_normal.jpg"), tag_crop_normal)

                # Ищем поля на нормальном ценнике
                field_dets = detect(field_model, tag_crop_normal, conf=conf, imgsz=640)

                if save_debug and debug_saved == 0 and field_dets:
                    debug_img = tag_crop_normal.copy()
                    for fd in field_dets:
                        fx1, fy1, fx2, fy2 = fd.bbox
                        cv2.rectangle(debug_img, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                        cv2.putText(debug_img, fd.class_name, (fx1, fy1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imwrite(str(debug_dir / f"frame{frame_idx}_track{track_id}_fields.jpg"), debug_img)

                for fd in field_dets:
                    field_name = fd.class_name

                    if field_name not in FIELD_CLASSES:
                        continue

                    # Вырезаем поле
                    fx1, fy1, fx2, fy2 = fd.bbox
                    field_crop = tag_crop_normal[fy1:fy2, fx1:fx2]

                    if field_crop.size == 0:
                        continue

                    if save_debug and debug_saved == 0:
                        cv2.imwrite(str(debug_dir / f"frame{frame_idx}_track{track_id}_{field_name}.jpg"), field_crop)

                        # Обработка в зависимости от типа поля
                        if field_name in QR_FIELDS:
                            # QR/штрихкод
                            qr_text = decode_qr_field(field_crop)
                            if qr_text:
                                if field_name == "qr_code_barcode":
                                    update_votes(track, "qr_code_barcode", qr_text, 3.0)
                                    # Парсим цены из QR
                                    qr_prices = parse_qr_prices(qr_text)
                                    for i, p in enumerate(qr_prices):
                                        if p:
                                            update_votes(track, f"price{i + 1}_qr", p, 2.0)
                                elif field_name == "barcode":
                                    update_votes(track, "barcode", qr_text, 2.5)
                                elif field_name == "code":
                                    update_votes(track, "code", qr_text, 2.0)

                        elif field_name in PRICE_FIELDS:
                            # Цена
                            price, ocr_conf = ocr_price(field_crop)
                            if price:
                                update_votes(track, field_name, price, ocr_conf)

                        elif field_name == "discount_amount":
                            # Скидка - специальная обработка
                            discount, ocr_conf = ocr_discount(field_crop)
                            if discount:
                                update_votes(track, field_name, discount, ocr_conf)

                        else:
                            # Текстовые поля
                            txt, ocr_conf = ocr_text_field(field_crop, field_name)
                            if txt:
                                update_votes(track, field_name, txt, ocr_conf)

            if save_debug:
                debug_saved += 1

        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            progress = (frame_idx / total) * 100 if total > 0 else 0
            logger.info(f"Прогресс: {progress:.1f}% | Треков: {len(tracks)} | Время: {elapsed:.1f}с")

    cap.release()

    elapsed_total = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n=== ГОТОВО ===")
    logger.info(f"Кадров: {processed_frames}, треков: {len(tracks)}, время: {elapsed_total:.1f}с")

    # Формируем результат
    rows = []
    for tid in sorted(tracks.keys()):
        tr = tracks[tid]
        row = {col: "нет" for col in EXPECTED_COLUMNS}
        row["filename"] = video_path.name if hasattr(video_path, 'name') else str(video_path)
        x1, y1, x2, y2 = tr.first_bbox
        row["x_min"] = x1
        row["y_min"] = y1
        row["x_max"] = x2
        row["y_max"] = y2
        row["frame_timestamp"] = tr.first_frame

        for field_name, votes in tr.field_votes.items():
            if votes:
                row[field_name] = get_best_vote(votes)
                logger.info(f"Track {tid}: {field_name} = {row[field_name]}")

        # Если нет скидочной цены, используем карточную
        if row["price_discount"] == "нет" and row["price_card"] != "нет":
            row["price_discount"] = row["price_card"]

        rows.append(row)

    out_csv = pathlib.Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Статистика
    non_empty = {col: (df[col] != "нет").sum() for col in EXPECTED_COLUMNS if (df[col] != "нет").sum() > 0}
    logger.info("\n=== СТАТИСТИКА ===")
    for col, count in sorted(non_empty.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {col}: {count}/{len(rows)}")

    logger.info(f"\nРезультат: {out_csv}")
    return df


# ============================================================
# ARGS
# ============================================================

def parse_args():
    root = resolve_project_root()
    parser = argparse.ArgumentParser(description="Распознавание ценников на видео")
    parser.add_argument("--video", type=pathlib.Path, required=True)
    parser.add_argument("--tag-model", type=pathlib.Path, default=root / "weight" / "best-price-tag.pt")
    parser.add_argument("--field-model", type=pathlib.Path, default=root / "weight" / "best.pt")
    parser.add_argument("--out-csv", type=pathlib.Path, default=root / "runs" / "result.csv")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--tag-rotation", type=int, default=270, choices=[0, 90, 180, 270],
                        help="Поворот ценника в нормальную ориентацию")
    parser.add_argument("--debug-dir", type=pathlib.Path, default=pathlib.Path("./debug"))
    return parser.parse_args()


def main():
    args = parse_args()
    process_video(
        video_path=args.video,
        tag_model_path=args.tag_model,
        field_model_path=args.field_model,
        out_csv=args.out_csv,
        conf=args.conf,
        imgsz=args.imgsz,
        frame_stride=args.frame_stride,
        tag_rotation=args.tag_rotation,
        debug_dir=args.debug_dir,
    )


if __name__ == "__main__":
    main()