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

PRICE_FIELDS = {"price_card", "price_default", "price_discount"}

EXPECTED_COLUMNS = [
    "filename",
    "price_default",
    "price_card",
    "price_discount",
    "frame_timestamp",
    "x_min", "y_min", "x_max", "y_max",
]


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
    first_bbox: Tuple[int, int, int, int]  # Координаты в оригинальном (повёрнутом) видео
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
    ocr = easyocr.Reader(['en'], gpu=True)
    logger.info("EasyOCR успешно инициализирован")
except Exception as e:
    logger.error(f"Ошибка EasyOCR: {e}")
    ocr = None


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


def extract_price_from_text(text):
    """
    Извлекает цену из текста.
    """
    if not text:
        return ""

    # Убираем всё кроме цифр, точки и запятой
    cleaned = re.sub(r'[^\d.,]', '', text)
    cleaned = cleaned.replace(',', '.')

    # Убираем лишние точки, оставляем только последнюю
    parts = cleaned.split('.')
    if len(parts) > 2:
        cleaned = ''.join(parts[:-1]) + '.' + parts[-1]

    # Если нет точки
    if '.' not in cleaned:
        if len(cleaned) >= 3:
            return f"{cleaned[:-2]}.{cleaned[-2:]}"
        elif len(cleaned) > 0:
            return f"{cleaned}.00"
        return ""

    # Есть точка
    int_part, frac_part = cleaned.split('.')

    if not frac_part:
        return f"{int_part}.00"
    elif len(frac_part) == 1:
        return f"{int_part}.{frac_part}0"
    else:
        return f"{int_part}.{frac_part[:2]}"


def ocr_price(img):
    """Распознавание цены. Изображение уже в нормальной ориентации."""
    if img is None or img.size == 0 or ocr is None:
        return "", 0.0

    h, w = img.shape[:2]

    # Конвертируем в серый
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Увеличиваем для лучшего распознавания
    if h < 100:
        scale = 100 / h
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    logger.info(f"  OCR: размер {gray.shape[1]}x{gray.shape[0]}")

    # Варианты для распознавания
    variants = [
        ("normal", gray),
        ("inverted", cv2.bitwise_not(gray)),
    ]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(("clahe", clahe.apply(gray)))
    variants.append(("clahe_inv", cv2.bitwise_not(clahe.apply(gray))))

    for vname, variant in variants:
        try:
            results = ocr.readtext(variant, detail=1, paragraph=False)

            if results:
                # Группируем по размеру: большие цифры = целая часть, маленькие = копейки
                large_parts = []
                small_parts = []

                for bbox, text, score in results:
                    y_coords = [p[1] for p in bbox]
                    height = max(y_coords) - min(y_coords)

                    # Только цифры и точка
                    digit_text = re.sub(r'[^\d.]', '', text)

                    if digit_text:
                        logger.info(f"  [{vname}] '{digit_text}' h={height:.0f} conf={score:.3f}")

                        if height > gray.shape[0] * 0.5:
                            large_parts.append((height, digit_text))
                        else:
                            small_parts.append((height, digit_text))

                if large_parts or small_parts:
                    # Собираем целую часть
                    large_parts.sort(key=lambda x: x[0], reverse=True)
                    int_part = ''.join([p[1] for p in large_parts])

                    # Собираем копейки
                    small_parts.sort(key=lambda x: x[0], reverse=True)
                    frac_part = ''.join([p[1] for p in small_parts])

                    # Формируем цену
                    if int_part and frac_part:
                        price = f"{int_part}.{frac_part[:2]}"
                    elif int_part:
                        if len(int_part) >= 3:
                            price = f"{int_part[:-2]}.{int_part[-2:]}"
                        else:
                            price = f"{int_part}.00"
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
        max_lost_frames=20,
        tag_rotation=270,  # Поворот ценника в нормальную ориентацию (270 = против часовой)
        debug_dir=None,
):
    """
    Логика:
    1. Видео повёрнуто на 90° по часовой
    2. Tag-модель ищет ценники на повёрнутом видео
    3. Вырезаем ценник из повёрнутого видео
    4. Поворачиваем ценник на 270° → нормальная ориентация
    5. Field-модель ищет поля на нормально ориентированном ценнике
    6. Вырезаем поля из нормального ценника
    7. Передаём поля в OCR (они уже в нормальной ориентации)
    """
    start_time = datetime.now()
    logger.info(f"=== НАЧАЛО ОБРАБОТКИ ВИДЕО ===")
    logger.info(f"Видео: {video_path}")
    logger.info(f"Поворот ценника в нормальную ориентацию: {tag_rotation}°")

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

        # Шаг 2: Ищем ценники на повёрнутом видео (НЕ поворачиваем видео!)
        tag_dets = detect(tag_model, frame, conf=0.1, imgsz=imgsz)

        if tag_dets:
            if frame_idx % 50 == 0:
                logger.info(f"Кадр {frame_idx}: найдено {len(tag_dets)} ценников")

            tracks, matched, next_track_id = match_tracks(
                tracks, tag_dets, frame_idx, next_track_id, max_lost_frames
            )

            for track_id, det in matched:
                track = tracks[track_id]

                # Шаг 3: Вырезаем ценник из повёрнутого видео
                x1, y1, x2, y2 = expand_box(det.bbox, pad=0.2, w=frame.shape[1], h=frame.shape[0])
                tag_crop_rotated = frame[y1:y2, x1:x2]  # Это повёрнутый ценник

                if tag_crop_rotated.size == 0:
                    continue

                # Шаг 4: Поворачиваем ценник в нормальную ориентацию
                if tag_rotation != 0:
                    tag_crop_normal = rotate_image(tag_crop_rotated, tag_rotation)
                else:
                    tag_crop_normal = tag_crop_rotated

                if save_debug and debug_saved == 0:
                    cv2.imwrite(str(debug_dir / f"frame{frame_idx}_track{track_id}_rotated.jpg"), tag_crop_rotated)
                    cv2.imwrite(str(debug_dir / f"frame{frame_idx}_track{track_id}_normal.jpg"), tag_crop_normal)

                # Шаг 5: Ищем поля на нормально ориентированном ценнике
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

                    if field_name not in PRICE_FIELDS:
                        continue

                    # Шаг 6: Вырезаем поле из нормального ценника
                    fx1, fy1, fx2, fy2 = fd.bbox
                    field_crop = tag_crop_normal[fy1:fy2, fx1:fx2]  # Уже в нормальной ориентации!

                    if field_crop.size == 0:
                        continue

                    if save_debug and debug_saved == 0:
                        cv2.imwrite(str(debug_dir / f"frame{frame_idx}_track{track_id}_{field_name}.jpg"), field_crop)

                    # Шаг 7: Передаём в OCR (поле уже в нормальной ориентации)
                    price, ocr_conf = ocr_price(field_crop)

                    if price:
                        update_votes(track, field_name, price, ocr_conf)

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

        if row["price_discount"] == "нет" and row["price_card"] != "нет":
            row["price_discount"] = row["price_card"]

        rows.append(row)

    out_csv = pathlib.Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    for col in ["price_default", "price_card", "price_discount"]:
        count = (df[col] != "нет").sum()
        if count > 0:
            logger.info(f"  {col}: {count}/{len(rows)}")

    logger.info(f"Результат: {out_csv}")
    return df


# ============================================================
# ARGS
# ============================================================

def parse_args():
    root = resolve_project_root()
    parser = argparse.ArgumentParser(description="Распознавание цен на ценниках")
    parser.add_argument("--video", type=pathlib.Path, required=True)
    parser.add_argument("--tag-model", type=pathlib.Path, default=root / "weight" / "best-price-tag.pt")
    parser.add_argument("--field-model", type=pathlib.Path, default=root / "weight" / "best.pt")
    parser.add_argument("--out-csv", type=pathlib.Path, default=root / "runs" / "result.csv")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--tag-rotation", type=int, default=270, choices=[0, 90, 180, 270],
                        help="Поворот ценника в нормальную ориентацию (270 = против часовой)")
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