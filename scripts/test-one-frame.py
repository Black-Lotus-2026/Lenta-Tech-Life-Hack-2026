import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Пути к моделям (укажите свои)
TAG_MODEL_PATH = "../weight/best-price-tag.pt"    # детектор ценников
FIELD_MODEL_PATH = "../weight/best.pt"            # детектор полей (если нужен)
VIDEO_PATH = r"D:\Lenta-Tech-Life-Hack-2026\data\Данные\43_15\43_15.mp4"          # замените на свой

# 1. Загружаем один кадр из видео
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()
if not ret:
    print("❌ Не удалось прочитать видео")
    exit()

# 🔄 ПОВОРОТ (добавить здесь)
frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)   # или CLOCKWISE

# 2. Загружаем модели
print("Загрузка моделей...")
tag_model = YOLO(TAG_MODEL_PATH)
field_model = YOLO(FIELD_MODEL_PATH) if FIELD_MODEL_PATH else None
print("Модели загружены")

# 3. Детекция ценников – ПРАВИЛЬНЫЙ вызов
results = tag_model.predict(source=frame, conf=0.15, imgsz=640, verbose=False)
boxes = results[0].boxes
if boxes is None or len(boxes) == 0:
    print("❌ Модель ценников не нашла ни одного ценника на кадре")
    print("   Возможные причины:")
    print("   - Веса модели не соответствуют вашим ценникам (обучены на других данных)")
    print("   - Кадр слишком темный/размытый/маленький")
    print("   - Ценники не видны или повернуты")
    # Попробуем понизить порог
    results = tag_model.predict(source=frame, conf=0.05, imgsz=1280, verbose=False)
    boxes = results[0].boxes
    if boxes and len(boxes) > 0:
        print(f"✅ При conf=0.05 найдено {len(boxes)} ценников!")
    else:
        print("❌ Даже при conf=0.05 ничего не найдено.")
        exit()
else:
    print(f"✅ Найдено ценников: {len(boxes)}")

# 4. Сохраняем кропы и пробуем OCR
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    crop = frame[y1:y2, x1:x2]
    cv2.imwrite(f"debug_tag_{i}.png", crop)
    print(f"Ценник {i}: сохранён как debug_tag_{i}.png, размер {crop.shape}")

    # Прямой OCR на всем ценнике (без детекции полей)
    # Увеличиваем
    scale = 800 / min(crop.shape[:2])
    crop_big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop_big, cv2.COLOR_BGR2GRAY)
    # Адаптивный порог
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # OCR
    ocr = PaddleOCR(lang='ru', use_textline_orientation=True)
    ocr_result = ocr.ocr(thresh, cls=True)
    if ocr_result and ocr_result[0]:
        texts = [line[1][0] for line in ocr_result[0] if line[1][1] > 0.3]
        print(f"   Распознанный текст: {' '.join(texts)}")
    else:
        print("   OCR не распознал текст")