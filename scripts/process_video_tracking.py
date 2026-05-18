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
    ocr_en = easyocr.Reader(['en'], gpu=True)
    ocr_ru = easyocr.Reader(['ru'], gpu=True)
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
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    if min(h, w) < target_size:
        scale = target_size / min(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def clean_text(txt):
    if not txt:
        return ""
    txt = str(txt).replace("\n", " ").replace("\t", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def clean_discount(text):
    if not text:
        return ""
    text = str(text).replace(",", ".").replace(" ", "")
    match = re.search(r'-?\d+\.?\d*\s*%', text)
    if match:
        val = match.group().replace(" ", "")
        if not val.startswith('-'):
            val = '-' + val
        return val
    match = re.search(r'-?\d+\.?\d*', text)
    if match:
        val = match.group()
        if not val.startswith('-'):
            val = '-' + val
        return f"{val}%"
    return ""


def clean_sku(text):
    if not text:
        return ""
    digits = re.sub(r'[^\d]', '', str(text))
    return digits if len(digits) >= 4 else ""


def clean_datetime(text):
    if not text:
        return ""
    text = str(text).strip()
    date_match = re.search(r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}', text)
    if date_match:
        return date_match.group()
    time_match = re.search(r'\d{1,2}:\d{2}(:\d{2})?', text)
    if time_match:
        return time_match.group()
    return clean_text(text)


def parse_qr_prices(qr_text):
    if not qr_text:
        return []
    vals = re.findall(r'\d+[.,]\d{2}', qr_text)
    return [v.replace(",", ".") for v in vals[:4]]


# ============================================================
# OCR
# ============================================================

def ocr_price(img):
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
                large_parts, small_parts = [], []
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
            results = ocr_en.readtext(variant, detail=1, paragraph=False)
            if results:
                all_texts, all_scores = [], []
                for bbox, text, score in results:
                    if text.strip():
                        all_texts.append(text.strip())
                        all_scores.append(score)
                        logger.info(f"  [{vname}] '{text.strip()}' conf={score:.3f}")
                if all_texts:
                    combined = " ".join(all_texts)
                    avg_score = sum(all_scores) / len(all_scores)
                    cleaned = clean_discount(combined)
                    if cleaned:
                        logger.info(f"  ✓ Скидка [{vname}]: {cleaned}")
                        return cleaned, float(avg_score)
        except Exception as e:
            logger.error(f"  OCR error [{vname}]: {e}")
    logger.info(f"  ✗ Скидка не распознана")
    return "", 0.0


def ocr_text_field(img, field_name):
    if img is None or img.size == 0:
        return "", 0.0
    reader = ocr_en if field_name in {"id_sku", "print_datetime"} else ocr_ru
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
            results = reader.readtext(variant, detail=1, paragraph=False)
            if results:
                lines = defaultdict(list)
                for bbox, text, score in results:
                    if not text.strip():
                        continue
                    y_coords = [p[1] for p in bbox]
                    y_center = sum(y_coords) / len(y_coords)
                    height = max(y_coords) - min(y_coords)
                    line_key = round(y_center / (height * 0.7))
                    lines[line_key].append((bbox, text.strip(), score, y_center))
                    logger.info(f"  [{vname}] '{text.strip()}' y={y_center:.0f} h={height:.0f} conf={score:.3f}")
                if lines:
                    sorted_lines = sorted(lines.items(), key=lambda x: min(p[3] for p in x[1]))
                    all_texts, all_scores = [], []
                    for line_key, line_parts in sorted_lines:
                        line_parts.sort(key=lambda p: min(pt[0] for pt in p[0]))
                        line_text = " ".join([p[1] for p in line_parts])
                        line_score = sum(p[2] for p in line_parts) / len(line_parts)
                        all_texts.append(line_text)
                        all_scores.append(line_score)
                    combined = " ".join(all_texts)
                    avg_score = sum(all_scores) / len(all_scores)
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
        ("big", cv2.resize(processed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
    ]
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


def match_tracks(tracks, detections, frame_idx, next_track_id, max_lost=30):
    IOU_THRESH = 0.03
    RECOVERY_FRAMES = 5
    RECOVERY_IOU = 0.1

    matched = []
    used_tracks = set()
    used_detections = set()

    for det_idx, det in enumerate(detections):
        best_id = None
        best_score = -1
        for tid, track in tracks.items():
            if tid in used_tracks:
                continue
            i = iou(track.last_bbox, det.bbox)
            d = distance(track.last_bbox, det.bbox)
            score = i - d * 0.00001
            if score > best_score:
                best_score = score
                best_id = tid

        if best_id is not None and best_score > IOU_THRESH:
            tracks[best_id].last_bbox = det.bbox
            tracks[best_id].last_frame = frame_idx
            tracks[best_id].lost_counter = 0
            matched.append((best_id, det))
            used_tracks.add(best_id)
            used_detections.add(det_idx)

    for det_idx, det in enumerate(detections):
        if det_idx in used_detections:
            continue
        best_lost_id = None
        best_lost_iou = -1
        for tid, track in tracks.items():
            if tid in used_tracks:
                continue
            if track.lost_counter <= RECOVERY_FRAMES:
                i = iou(track.last_bbox, det.bbox)
                if i > best_lost_iou:
                    best_lost_iou = i
                    best_lost_id = tid
        if best_lost_id is not None and best_lost_iou > RECOVERY_IOU:
            tracks[best_lost_id].last_bbox = det.bbox
            tracks[best_lost_id].last_frame = frame_idx
            tracks[best_lost_id].lost_counter = 0
            matched.append((best_lost_id, det))
            used_tracks.add(best_lost_id)
            used_detections.add(det_idx)

    for det_idx, det in enumerate(detections):
        if det_idx in used_detections:
            continue
        tid = next_track_id
        next_track_id += 1
        tracks[tid] = TrackState(
            track_id=tid, first_frame=frame_idx, first_bbox=det.bbox,
            last_frame=frame_idx, last_bbox=det.bbox
        )
        matched.append((tid, det))
        used_tracks.add(tid)

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
        tag_rotation=270,
        debug_dir=None,
):
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

        tag_dets = detect(tag_model, frame, conf=0.1, imgsz=imgsz)

        if tag_dets:
            if frame_idx % 50 == 0:
                logger.info(f"Кадр {frame_idx}: найдено {len(tag_dets)} ценников | треков: {len(tracks)}")

            tracks, matched, next_track_id = match_tracks(
                tracks, tag_dets, frame_idx, next_track_id
            )

            for track_id, det in matched:
                track = tracks[track_id]

                x1, y1, x2, y2 = expand_box(det.bbox, pad=0.2, w=frame.shape[1], h=frame.shape[0])
                tag_crop_rotated = frame[y1:y2, x1:x2]

                if tag_crop_rotated.size == 0:
                    continue

                if tag_rotation != 0:
                    tag_crop_normal = rotate_image(tag_crop_rotated, tag_rotation)
                else:
                    tag_crop_normal = tag_crop_rotated

                if save_debug and debug_saved == 0:
                    cv2.imwrite(str(debug_dir / f"frame{frame_idx}_track{track_id}_rotated.jpg"), tag_crop_rotated)
                    cv2.imwrite(str(debug_dir / f"frame{frame_idx}_track{track_id}_normal.jpg"), tag_crop_normal)

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

                    fx1, fy1, fx2, fy2 = fd.bbox
                    field_crop = tag_crop_normal[fy1:fy2, fx1:fx2]

                    if field_crop.size == 0:
                        continue

                    if save_debug and debug_saved == 0:
                        cv2.imwrite(str(debug_dir / f"frame{frame_idx}_track{track_id}_{field_name}.jpg"), field_crop)

                    if field_name in QR_FIELDS:
                        qr_text = decode_qr_field(field_crop)
                        if qr_text:
                            if field_name == "qr_code_barcode":
                                update_votes(track, "qr_code_barcode", qr_text, 3.0)
                                qr_prices = parse_qr_prices(qr_text)
                                for i, p in enumerate(qr_prices):
                                    if p:
                                        update_votes(track, f"price{i + 1}_qr", p, 2.0)
                            elif field_name == "barcode":
                                update_votes(track, "barcode", qr_text, 2.5)
                            elif field_name == "code":
                                update_votes(track, "code", qr_text, 2.0)

                    elif field_name in PRICE_FIELDS:
                        price, ocr_conf = ocr_price(field_crop)
                        if price:
                            update_votes(track, field_name, price, ocr_conf)

                    elif field_name == "discount_amount":
                        discount, ocr_conf = ocr_discount(field_crop)
                        if discount:
                            update_votes(track, field_name, discount, ocr_conf)

                    else:
                        txt, ocr_conf = ocr_text_field(field_crop, field_name)
                        if txt:
                            update_votes(track, field_name, txt, ocr_conf)

            if save_debug:
                debug_saved += 1

        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            progress = (frame_idx / total) * 100 if total > 0 else 0
            logger.info(
                f"Прогресс: {progress:.1f}% ({frame_idx}/{total}) | Треков: {len(tracks)} | Время: {elapsed:.1f}с")

    cap.release()

    elapsed_total = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n=== ГОТОВО ===")
    logger.info(f"Кадров: {processed_frames}, треков: {len(tracks)}, время: {elapsed_total:.1f}с")

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
                logger.info(f"Track {tid}: {field_name} = {row[field_name]} ({len(votes)} голосов)")

        # Взаимное дублирование цен:
        # Если price_card нет, берём из price_discount
        if row["price_card"] == "нет" and row["price_discount"] != "нет":
            row["price_card"] = row["price_discount"]
            logger.info(f"Track {tid}: price_card ← price_discount = {row['price_card']}")

        # Если price_discount нет, берём из price_card
        if row["price_discount"] == "нет" and row["price_card"] != "нет":
            row["price_discount"] = row["price_card"]
            logger.info(f"Track {tid}: price_discount ← price_card = {row['price_discount']}")

        # Если price_default нет, берём из price_card или price_discount
        if row["price_default"] == "нет":
            if row["price_card"] != "нет":
                row["price_default"] = row["price_card"]
                logger.info(f"Track {tid}: price_default ← price_card = {row['price_default']}")
            elif row["price_discount"] != "нет":
                row["price_default"] = row["price_discount"]
                logger.info(f"Track {tid}: price_default ← price_discount = {row['price_default']}")

        rows.append(row)

    out_csv = pathlib.Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

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