# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

COLOR_TAG = (0, 255, 0)      # зелёный
COLOR_FIELD = (255, 0, 0)    # синий

def rotate_image(img, deg):
    """Поворачивает изображение на заданное количество градусов (90, 180, 270)"""
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def invert_rotate_bbox(bbox, rot_deg, crop_w, crop_h):
    """
    Преобразует координаты bbox (x1,y1,x2,y2) из повёрнутого кропа в координаты исходного кропа.
    rot_deg – угол, на который был повёрнут кроп перед детекцией (90, 180, 270).
    crop_w, crop_h – ширина и высота исходного (неповёрнутого) кропа.
    """
    x1, y1, x2, y2 = bbox
    if rot_deg == 90:
        # Поворот на 90° CW, обратный – 90° CCW
        new_x1 = y1
        new_y1 = crop_h - x2
        new_x2 = y2
        new_y2 = crop_h - x1
    elif rot_deg == 180:
        new_x1 = crop_w - x2
        new_y1 = crop_h - y2
        new_x2 = crop_w - x1
        new_y2 = crop_h - y1
    elif rot_deg == 270:
        # Поворот на 90° CCW, обратный – 90° CW
        new_x1 = crop_h - y2
        new_y1 = x1
        new_x2 = crop_h - y1
        new_y2 = x2
    else:
        return (x1, y1, x2, y2)
    # Приводим к валидному порядку (min, max)
    return (min(new_x1, new_x2), min(new_y1, new_y2), max(new_x1, new_x2), max(new_y1, new_y2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Путь к видео")
    parser.add_argument("--tag-model", default="weight/best-price-tag.pt", help="Модель для детекции ценников")
    parser.add_argument("--field-model", default="weight/best.pt", help="Модель для детекции полей")
    parser.add_argument("--out-video", default="output_annotated.mp4", help="Выходное видео")
    parser.add_argument("--conf-tag", type=float, default=0.15, help="Порог для ценников")
    parser.add_argument("--conf-field", type=float, default=0.25, help="Порог для полей")
    parser.add_argument("--rotate-field", type=int, default=270, choices=[0,90,180,270],
                        help="Поворот кропа перед field моделью (270 = против часовой, чтобы сделать прямой из повёрнутого)")
    parser.add_argument("--imgsz-field", type=int, default=640, help="Размер изображения для field модели")
    parser.add_argument("--show", action="store_true", help="Показывать окно")
    parser.add_argument("--window-scale", type=float, default=0.5, help="Масштаб окна")
    args = parser.parse_args()

    # Загрузка моделей
    print("Загрузка моделей...")
    tag_model = YOLO(args.tag_model)
    field_model = YOLO(args.field_model) if Path(args.field_model).exists() else None
    if field_model:
        print("Модель полей загружена")
    else:
        print("Модель полей не найдена (будут только зелёные рамки)")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count = 0
    print("Обработка видео...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Детекция ценников
        tag_results = tag_model(frame, conf=args.conf_tag, imgsz=640, verbose=False)[0]
        if tag_results.boxes is not None:
            for tag_box in tag_results.boxes:
                # Координаты ценника в исходном кадре
                x1, y1, x2, y2 = map(int, tag_box.xyxy[0].tolist())
                # Зелёная рамка
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_TAG, 2)
                cv2.putText(frame, f"TAG {tag_box.conf[0]:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TAG, 2)

                if field_model:
                    # 2. Вырезаем область ценника
                    tag_crop = frame[y1:y2, x1:x2]
                    if tag_crop.size == 0:
                        continue
                    # 3. Поворачиваем кроп
                    crop_rot = rotate_image(tag_crop, args.rotate_field)
                    h_rot, w_rot = crop_rot.shape[:2]

                    # 4. Детекция полей на повёрнутом кропе
                    field_results = field_model(crop_rot, conf=args.conf_field, imgsz=args.imgsz_field, verbose=False)[0]
                    if field_results.boxes is not None:
                        for fbox in field_results.boxes:
                            # Координаты поля в системе координат model input (размер args.imgsz_field)
                            fx1, fy1, fx2, fy2 = map(int, fbox.xyxy[0].tolist())
                            # 5. Масштабируем к реальному размеру crop_rot
                            scale_x = w_rot / args.imgsz_field
                            scale_y = h_rot / args.imgsz_field
                            fx1 = int(fx1 * scale_x)
                            fx2 = int(fx2 * scale_x)
                            fy1 = int(fy1 * scale_y)
                            fy2 = int(fy2 * scale_y)
                            # 6. Преобразуем координаты из повёрнутого кропа в координаты исходного кропа
                            orig_w = tag_crop.shape[1]
                            orig_h = tag_crop.shape[0]
                            fx1, fy1, fx2, fy2 = invert_rotate_bbox((fx1, fy1, fx2, fy2), args.rotate_field, orig_w, orig_h)
                            # 7. Добавляем смещение ценника → глобальные координаты
                            gx1 = x1 + fx1
                            gy1 = y1 + fy1
                            gx2 = x1 + fx2
                            gy2 = y1 + fy2
                            # Отрисовка синей рамки поля (внутри зелёной)
                            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), COLOR_FIELD, 1)
                            fcls = field_results.names[int(fbox.cls[0])]
                            cv2.putText(frame, f"{fcls} {fbox.conf[0]:.2f}", (gx1, gy1-3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_FIELD, 1)

        out.write(frame)
        if args.show:
            disp = cv2.resize(frame, None, fx=args.window_scale, fy=args.window_scale, interpolation=cv2.INTER_AREA)
            cv2.imshow("Detection", disp)
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