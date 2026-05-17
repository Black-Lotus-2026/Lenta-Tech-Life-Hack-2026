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
from paddleocr import PaddleOCR

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
except Exception:
    pyzbar_decode = None


# ============================================================
# LOGGING (с временем)
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
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]

FIELD_CLASSES = {
    "additional_info",
    "barcode",
    "code",
    "discount_amount",
    "id_sku",
    "price_card",
    "price_default",
    "price_discount",
    "print_datetime",
    "product_name",
    "qr_code_barcode",
}

# FIX: Поля, которые обрабатываются редко (не каждый кадр)
RARE_FIELDS = {"product_name", "additional_info", "code", "discount_amount", "id_sku", "print_datetime"}
# FIX: Поля цен - обрабатываем чаще, но не каждый кадр
PRICE_FIELDS = {"price_card", "price_default", "price_discount"}
# FIX: QR и штрихкод - каждый кадр
QR_FIELDS = {"qr_code_barcode", "barcode"}


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
    # FIX: последний кадр, когда обрабатывали каждое поле
    last_processed_frame: Dict[str, int] = field(default_factory=dict)
    # FIX: счётчик пропущенных кадров (для удаления старых треков)
    lost_counter: int = 0


# ============================================================
# OCR (ИНИЦИАЛИЗАЦИЯ)
# ============================================================

logger.info("Инициализация PaddleOCR (это может занять время)...")
try:
    ocr = PaddleOCR(
        lang='ru',
        use_angle_cls=True,
        show_log=False
    )
    logger.info("PaddleOCR успешно инициализирован")
except Exception as e:
    logger.error(f"Ошибка инициализации PaddleOCR: {e}")
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


def laplacian_score(img):
    if img is None or img.size == 0:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def clean_text(txt):
    if txt is None:
        return ""
    txt = str(txt)
    txt = txt.replace("\n", " ").replace("\t", " ")
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r'[^\w\s\.,\-\(\)№%]', '', txt)
    return txt.strip()


def clean_price(text):
    if not text:
        return ""
    text = text.replace(",", ".")
    patterns = [r'\d+\.\d{2}', r'\d+\.\d{1}', r'\d+']
    for pattern in patterns:
        vals = re.findall(pattern, text)
        if vals:
            return max(vals, key=len)
    return ""


def clean_barcode(text):
    if not text:
        return ""
    vals = re.findall(r"\d{8,14}", text.replace(" ", ""))
    if not vals:
        return ""
    return max(vals, key=len)


def parse_qr_prices(text):
    if not text:
        return []
    vals = re.findall(r"\d+[.,]\d{2}", text)
    return [v.replace(",", ".") for v in vals[:4]]


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


def enhance_image(img):
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    scale = min(2, max(1, int(600 / min(h, w))))  # FIX: уменьшил максимальный масштаб для скорости
    if scale > 1:
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = cv2.medianBlur(gray, 3)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


# ============================================================
# OCR ФУНКЦИИ (ускоренная)
# ============================================================

def paddle_ocr_enhanced(img):
    """Ускоренная OCR: только один вариант обработки"""
    if img is None or img.size == 0 or ocr is None:
        return "", 0.0

    # FIX: только одна, самая эффективная предобработка
    processed = enhance_image(img)
    if processed is None:
        processed = img

    try:
        # Конвертация в RGB для PaddleOCR
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        elif processed.shape[2] == 4:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGRA2RGB)
        elif processed.shape[2] == 3 and processed.dtype == np.uint8:
            if processed[0, 0, 0] > processed[0, 0, 2]:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        result = ocr.ocr(processed, cls=True)
        if result and result[0]:
            texts = []
            scores = []
            for line in result[0]:
                if line and len(line) >= 2:
                    texts.append(line[1][0])
                    scores.append(line[1][1])
            if texts:
                combined = " ".join(texts)
                avg_score = sum(scores) / len(scores)
                return clean_text(combined), float(avg_score)
    except Exception as e:
        logger.debug(f"OCR error: {e}")
    return "", 0.0


# ============================================================
# QR (ускоренный)
# ============================================================

def decode_qr(crop):
    if pyzbar_decode is None:
        return ""
    # FIX: только два варианта: серый и увеличенный (без адаптивного порога для скорости)
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    # сначала пробуем на оригинальном масштабе
    try:
        decoded = pyzbar_decode(gray)
        if decoded:
            vals = [d.data.decode("utf-8", errors="ignore") for d in decoded]
            if vals:
                return " | ".join(vals)
    except:
        pass
    # потом увеличиваем
    big = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    try:
        decoded = pyzbar_decode(big)
        if decoded:
            vals = [d.data.decode("utf-8", errors="ignore") for d in decoded]
            if vals:
                return " | ".join(vals)
    except:
        pass
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
# TRACKING (улучшенный)
# ============================================================

def update_votes(track, field_name, value, score):
    if not value or value == "нет":
        return
    track.field_votes[field_name].append((value, score))
    logger.debug(f"Track {track.track_id}: {field_name} = '{value}' (score: {score:.3f})")


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
            score = i - d * 0.001
            if score > best_score:
                best_score = score
                best_id = tid
        if best_id is not None and best_score > 0.1:  # FIX: повышен порог
            tracks[best_id].last_bbox = det.bbox
            tracks[best_id].last_frame = frame_idx
            tracks[best_id].lost_counter = 0
            matched.append((best_id, det))
            used_tracks.add(best_id)
        else:
            tid = next_track_id
            next_track_id += 1
            tracks[tid] = TrackState(
                track_id=tid,
                first_frame=frame_idx,
                first_bbox=det.bbox,
                last_frame=frame_idx,
                last_bbox=det.bbox
            )
            matched.append((tid, det))
    # Увеличиваем lost_counter для неиспользованных треков
    for tid, track in tracks.items():
        if tid not in used_tracks:
            track.lost_counter += 1
    # Удаляем давно потерянные треки
    to_del = [tid for tid, track in tracks.items() if track.lost_counter > max_lost]
    for tid in to_del:
        del tracks[tid]
    return tracks, matched, next_track_id


# ============================================================
# MAIN (оптимизированный)
# ============================================================

def process_video(
        video_path,
        tag_model_path,
        field_model_path,
        out_csv,
        conf,
        imgsz,
        frame_stride,
        field_process_interval=5,      # FIX: обрабатывать поля (кроме QR) каждые N кадров
        qr_process_interval=1,         # FIX: QR обрабатываем каждый кадр
        max_lost_frames=15,            # FIX: через сколько кадров удалить трек
):
    start_time = datetime.now()
    logger.info(f"=== НАЧАЛО ОБРАБОТКИ ВИДЕО ===")
    logger.info(f"Видео файл: {video_path}")
    logger.info(f"Модель ценников: {tag_model_path}")
    logger.info(f"Модель полей: {field_model_path}")

    logger.info("Загрузка моделей YOLO...")
    tag_model = YOLO(str(tag_model_path))
    field_model = YOLO(str(field_model_path))
    logger.info("Модели YOLO загружены")

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
    detected_tags = 0

    logger.info("Начинаю обработку кадров...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        processed_frames += 1

        # Детекция ценников
        tag_dets = detect(tag_model, frame, conf=0.15, imgsz=imgsz)
        if tag_dets:
            detected_tags += len(tag_dets)
            logger.debug(f"Кадр {frame_idx}: найдено {len(tag_dets)} ценников")

            tracks, matched, next_track_id = match_tracks(
                tracks, tag_dets, frame_idx, next_track_id, max_lost_frames
            )

            # Обработка каждого трека
            for track_id, det in matched:
                track = tracks[track_id]

                # Расширяем область ценника
                x1, y1, x2, y2 = expand_box(det.bbox, pad=0.1, w=frame.shape[1], h=frame.shape[0])
                tag_crop = frame[y1:y2, x1:x2]
                if tag_crop.size == 0:
                    continue

                # Проверка четкости (снизил порог)
                sharpness = laplacian_score(tag_crop)
                if sharpness < 15:
                    logger.debug(f"Ценник {track_id}: низкая четкость ({sharpness:.1f})")
                    continue

                score = det.conf + sharpness * 0.0005
                if score > track.best_score:
                    track.best_score = score

                # Увеличиваем для детекции полей (умеренно)
                scale = min(2, max(1, int(600 / min(tag_crop.shape[:2]))))
                if scale > 1:
                    tag_crop = cv2.resize(tag_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

                # Детекция полей на этом кадре (один раз на трек)
                field_dets = detect(field_model, tag_crop, conf=conf, imgsz=640)

                for fd in field_dets:
                    field_name = fd.class_name
                    if field_name not in FIELD_CLASSES:
                        continue

                    # FIX: Определяем, нужно ли обрабатывать это поле на текущем кадре
                    last_proc = track.last_processed_frame.get(field_name, -999)
                    if field_name in QR_FIELDS:
                        interval = qr_process_interval
                    elif field_name in PRICE_FIELDS:
                        interval = max(1, field_process_interval // 2)  # цены чуть чаще
                    else:
                        interval = field_process_interval

                    if frame_idx - last_proc < interval:
                        continue  # пропускаем, ещё не время

                    # Обновляем время обработки
                    track.last_processed_frame[field_name] = frame_idx

                    fx1, fy1, fx2, fy2 = fd.bbox
                    crop = tag_crop[fy1:fy2, fx1:fx2]
                    if crop.size == 0:
                        continue

                    # QR
                    if field_name == "qr_code_barcode":
                        qr_text = decode_qr(crop)
                        if qr_text:
                            logger.info(f"Track {track_id}: QR обновлён: {qr_text[:50]}")
                            update_votes(track, "qr_code_barcode", qr_text, 3.0)
                            qr_prices = parse_qr_prices(qr_text)
                            for i, p in enumerate(qr_prices):
                                if p:
                                    update_votes(track, f"price{i+1}_qr", p, 2.0)
                        continue

                    # Штрихкод
                    if field_name == "barcode":
                        barcode = decode_qr(crop)
                        if not barcode:
                            txt, score_ocr = paddle_ocr_enhanced(crop)
                            barcode = clean_barcode(txt)
                        if barcode:
                            logger.info(f"Track {track_id}: штрихкод {barcode}")
                            update_votes(track, "barcode", barcode, 2.5)
                        continue

                    # Текстовые поля (OCR)
                    txt, ocr_conf = paddle_ocr_enhanced(crop)
                    if txt and ocr_conf > 0.2:
                        if field_name in PRICE_FIELDS:
                            txt = clean_price(txt)
                            if txt:
                                logger.info(f"Track {track_id}: {field_name} = {txt} (conf {ocr_conf:.2f})")
                        else:
                            txt = clean_text(txt)
                            if txt:
                                logger.info(f"Track {track_id}: {field_name} = '{txt}'")
                        if txt:
                            boost = 2.0 if field_name in PRICE_FIELDS else 1.0
                            update_votes(track, field_name, txt, ocr_conf * boost)

        frame_idx += 1

        # Логирование прогресса
        if frame_idx % 200 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            progress = (frame_idx / total) * 100 if total > 0 else 0
            logger.info(f"Прогресс: {progress:.1f}% ({frame_idx}/{total}) | Треков: {len(tracks)} | Время: {elapsed:.1f}с")

    cap.release()

    elapsed_total = (datetime.now() - start_time).total_seconds()
    logger.info(f"=== ОБРАБОТКА ЗАВЕРШЕНА ===")
    logger.info(f"Обработано кадров: {processed_frames}")
    logger.info(f"Найдено ценников: {detected_tags}")
    logger.info(f"Отслеживаемых объектов: {len(tracks)}")
    logger.info(f"Общее время: {elapsed_total:.1f} секунд")

    # Формирование CSV
    logger.info("Формирование результатов...")
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
                best = get_best_vote(votes)
                row[field_name] = best
                logger.debug(f"Track {tid}: {field_name} = '{best}'")

        if row["price_discount"] == "нет" and row["price_card"] != "нет":
            row["price_discount"] = row["price_card"]
        rows.append(row)

    out_csv = pathlib.Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    non_empty_stats = {col: (df[col] != "нет").sum() for col in EXPECTED_COLUMNS if (df[col] != "нет").sum() > 0}
    logger.info("=== СТАТИСТИКА РАСПОЗНАВАНИЯ ===")
    for col, count in sorted(non_empty_stats.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {col}: {count} ценников ({count/len(rows)*100:.1f}%)")
    logger.info(f"Результаты сохранены: {out_csv}")
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
    parser.add_argument("--conf", type=float, default=0.25, help="Порог детекции полей")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер для YOLO")
    parser.add_argument("--frame-stride", type=int, default=2, help="Пропускать кадры")
    parser.add_argument("--field-interval", type=int, default=5, help="Обрабатывать поля раз в N кадров")
    parser.add_argument("--qr-interval", type=int, default=1, help="QR обрабатывать каждый N кадров")
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
        field_process_interval=args.field_interval,
        qr_process_interval=args.qr_interval,
    )


if __name__ == "__main__":
    main()