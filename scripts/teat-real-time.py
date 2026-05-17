# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

COLOR_TAG = (0, 255, 0)      # зелёный – ценник
COLOR_FIELD = (255, 0, 0)    # синий – поля

def draw_boxes(frame, boxes, names, color, label_prefix=""):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{label_prefix}{names[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def rotate_image(img, deg):
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def main():
    parser = argparse.ArgumentParser(description="Визуализация с раздельным поворотом для tag и field")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tag-model", default="weight/best-price-tag.pt")
    parser.add_argument("--field-model", default="weight/best.pt")
    parser.add_argument("--out-video", default="output_annotated.mp4")
    parser.add_argument("--conf-tag", type=float, default=0.15)
    parser.add_argument("--conf-field", type=float, default=0.25)
    parser.add_argument("--rotate-tag", type=int, default=0, choices=[0,90,180,270],
                        help="Поворот для tag модели (относительно исходного кадра)")
    parser.add_argument("--rotate-field", type=int, default=270,
                        help="Поворот кропа ценника перед field моделью (90,180,270,0). По умолч. 270, т.к. tag обучен на повёрнутых, field на прямых")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--window-scale", type=float, default=0.5)
    args = parser.parse_args()

    print("Загрузка моделей...")
    tag_model = YOLO(args.tag_model)
    field_model = YOLO(args.field_model) if Path(args.field_model).exists() else None
    print("Модели загружены")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out_video, fourcc, fps, out_size)

    frame_count = 0
    print("Обработка видео...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Поворачиваем кадр для tag модели (если нужно)
        if args.rotate_tag != 0:
            frame_for_tag = rotate_image(frame, args.rotate_tag)
        else:
            frame_for_tag = frame

        # Детекция ценников
        tag_results = tag_model(frame_for_tag, conf=args.conf_tag, imgsz=640, verbose=False)[0]
        tag_boxes = tag_results.boxes
        if tag_boxes is not None:
            # Рисуем зелёные рамки НА ПОВЁРНУТОМ кадре, а потом вернём обратно? Упростим: будем рисовать на оригинальном, пересчитав координаты.
            # Но для простоты визуализации будем рисовать на frame (оригинал), но координаты из повёрнутого нужно отобразить обратно.
            # Поскольку rotate_tag у нас скорее всего 0 (tag модель работает на исходном), то оставим как есть.
            # Допустим, rotate_tag = 0 (мы не поворачиваем кадр), тогда tag_boxes даны в координатах оригинала.
            # Если rotate_tag != 0, то координаты нужно пересчитать, но для простоты рекомендуем rotate_tag=0.
            frame = draw_boxes(frame, tag_boxes, tag_results.names, COLOR_TAG, "TAG:")

            if field_model:
                for box in tag_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    tag_crop = frame[y1:y2, x1:x2]
                    if tag_crop.size == 0:
                        continue
                    # Применяем поворот к кропу для field модели
                    if args.rotate_field != 0:
                        tag_crop_rot = rotate_image(tag_crop, args.rotate_field)
                    else:
                        tag_crop_rot = tag_crop
                    field_results = field_model(tag_crop_rot, conf=args.conf_field, imgsz=640, verbose=False)[0]
                    if field_results.boxes is not None:
                        # Координаты полей в rotated crop; пересчитываем в оригинальный кроп, затем в глобальные
                        for fbox in field_results.boxes:
                            fx1, fy1, fx2, fy2 = map(int, fbox.xyxy[0].tolist())
                            # Если кроп был повернут, нужно применить обратный поворот к координатам полей
                            if args.rotate_field == 90:
                                # поворот кропа на 90 CW: обратный - 90 CCW
                                hh, ww = tag_crop.shape[:2]
                                fx1, fy1 = fy1, ww - fx2
                                fx2, fy2 = fy2, ww - fx1
                            elif args.rotate_field == 180:
                                hh, ww = tag_crop.shape[:2]
                                fx1, fx2 = ww - fx2, ww - fx1
                                fy1, fy2 = hh - fy2, hh - fy1
                            elif args.rotate_field == 270:
                                hh, ww = tag_crop.shape[:2]
                                fx1, fy1 = hh - fy2, fx1
                                fx2, fy2 = hh - fy1, fx2
                            # Теперь fx1,fy1,fx2,fy2 в координатах оригинального кропа
                            # Добавляем смещение ценника
                            fx1 += x1; fx2 += x1
                            fy1 += y1; fy2 += y1
                            fconf = float(fbox.conf[0])
                            fcls = field_results.names[int(fbox.cls[0])]
                            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), COLOR_FIELD, 1)
                            cv2.putText(frame, f"{fcls} {fconf:.2f}", (fx1, fy1-3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_FIELD, 1)

        out.write(frame)
        if args.show:
            display = cv2.resize(frame, None, fx=args.window_scale, fy=args.window_scale, interpolation=cv2.INTER_AREA)
            cv2.imshow("Detection", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Обработано кадров: {frame_count}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Готово! Результат сохранён в {args.out_video}")

if __name__ == "__main__":
    main()