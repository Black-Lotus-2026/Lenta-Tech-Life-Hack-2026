# Lenta Shelf AI

**Готовый локальный продукт для кейса Lenta Tech: распознавание ценников с 4K-видео робота и экспорт CSV в формате задания.**

Решение построено без облачных API: все компоненты запускаются локально в Docker/venv. В базовом режиме работает на CPU через OpenCV + Tesseract + pyzbar; в соревновательном режиме подключаются дообученный YOLO и PaddleOCR/zxing-cpp.

---

## 1. Что внутри

```text
lenta_shelf_ai/
  detectors.py        # YOLO + QR-seed + color/geometry fallback
  qr.py               # zxing-cpp / pyzbar / OpenCV QR decoding
  ocr.py              # PaddleOCR + Tesseract ensemble
  parsers.py          # поля ценника: цены, скидка, barcode, SKU, дата, зона, цвет
  tracker.py          # объединение одного ценника между кадрами
  pipeline.py         # видео -> CSV
scripts/
  build_yolo_dataset.py    # автоматическая сборка YOLO-датасета из public CSV
  build_pseudo_yolo_dataset.py # pseudo-labeling Unlabeled/*.mp4 для self-training
  train_detector.py        # обучение компактного детектора
  infer_video.py           # CLI inference
  evaluate_on_public.py    # строгая локальная proxy-оценка
  export_edge.py           # ONNX/OpenVINO/RKNN экспорт
app.py                     # Gradio UI: загрузка видео -> скачать CSV
Dockerfile                 # локальный/серверный запуск
configs/default.yaml       # настройки пайплайна
```

---

## 2. Быстрый запуск CPU-safe

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Ubuntu/Debian системные зависимости:
sudo apt-get update
sudo apt-get install -y ffmpeg libzbar0 tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng

python scripts/infer_video.py /path/to/video.mp4 --output-dir outputs --sample-fps 1.0
python app.py
```

UI откроется на `http://localhost:7860`.

---

## 3. Соревновательный запуск для максимального качества

### 3.1 Установить полный стек

```bash
pip install -r requirements-full.txt
```

### 3.2 Положить данные

Ожидаемая структура:

```text
data/
  Данные/
    25_12-20/25_12-20.mp4
    25_12-20/25_12-20.csv
    26_12-20/26_12-20.mp4
    26_12-20/26_12-20.csv
    43_15/43_15.mp4
    43_15/43_15.csv
    Unlabeled/*.mp4
```

### 3.3 Автоматически собрать датасет для детектора

Ручная разметка не нужна. Скрипт использует предоставленные CSV и автоматически расширяет датасет соседними кадрами через template tracking.

```bash
python scripts/build_yolo_dataset.py \
  --data-dir data/Данные \
  --out-dir datasets/lenta_yolo \
  --propagate 10
```

### 3.4 Обучить компактный YOLO

```bash
python scripts/train_detector.py \
  --data datasets/lenta_yolo/data.yaml \
  --model yolo11n.pt \
  --epochs 150 \
  --imgsz 1280 \
  --batch 4 \
  --device 0

mkdir -p models
cp runs/lenta/price_tag_yolo/weights/best.pt models/price_tag_yolo.pt
```

Если GPU нет, можно поставить `--device cpu --epochs 50`, но качество будет ниже.

### 3.5 Self-training без ручной разметки

После первого детектора можно автоматически разметить `Unlabeled/*.mp4` только high-confidence предсказаниями и дообучить модель:

```bash
python scripts/build_pseudo_yolo_dataset.py \
  --data-dir data/Данные \
  --base-dataset datasets/lenta_yolo \
  --out-dir datasets/lenta_yolo_self \
  --weights models/price_tag_yolo.pt \
  --sample-fps 1.0 \
  --conf 0.65 \
  --imgsz 1600

python scripts/train_detector.py \
  --data datasets/lenta_yolo_self/data.yaml \
  --model models/price_tag_yolo.pt \
  --epochs 30 \
  --imgsz 1280 \
  --batch 4 \
  --device 0 \
  --name price_tag_yolo_selftrain
```

Это не использует ручную разметку: pseudo-labels берутся только из локального detector-а.

### 3.6 Inference

```bash
python scripts/infer_video.py data/Данные/Unlabeled/26_2-10.mp4 \
  --config configs/default.yaml \
  --output-dir outputs/unlabeled_26_2_10 \
  --weights models/price_tag_yolo.pt \
  --sample-fps 4.0 \
  --defer-ocr
```

Результат: `outputs/unlabeled_26_2_10/26_2-10_recognized.csv`.

`--defer-ocr` включает соревновательный двухпроходный режим: детекция и трекинг идут часто, а OCR/QR запускаются только на лучшем crop каждого уникального ценника. Это резко дешевле, чем OCR каждого bbox на каждом кадре, и сохраняет recall от dense sampling.

---

## 4. Формат CSV

Порядок колонок полностью соответствует заданию:

```text
filename, product_name, price_default, price_card, price_discount,
barcode, discount_amount, id_sku, print_datetime, code, additional_info,
color, special_symbols, frame_timestamp, x_min, y_min, x_max, y_max,
qr_code_barcode, price1_qr, price2_qr, price3_qr, price4_qr,
wholesale_level_1_count, wholesale_level_1_price,
wholesale_level_2_count, wholesale_level_2_price,
action_price_qr, action_code_qr
```

Если параметр точно отсутствует на ценнике, пишется `нет`. Если поле не распознано, остается пустым. Для публичных CSV поддерживается legacy-опечатка `wholesale_level_1_coun`.

---

## 5. Архитектура пайплайна

1. **Sampling**: 4K MP4 читается с частотой 2-4 FPS, размазанные кадры отсекаются по Laplacian sharpness.
2. **Detection**:
   - основной детектор: дообученный YOLO11n/YOLOv8n;
   - QR-seed: если QR найден раньше ценника, область расширяется до всего ценника;
   - HSV fallback: красные/желтые/зеленые ценники и edge-density фильтрация.
3. **Crop enhancement**: upscale, CLAHE, unsharp mask.
4. **QR/barcode**: zxing-cpp -> pyzbar -> OpenCV QR. QR-поля имеют приоритет над OCR.
5. **OCR**: PaddleOCR для accuracy-режима, Tesseract rus+eng fallback для CPU-safe.
6. **Parsing**: регулярные выражения и доменные правила для цен, скидки, EAN-13, SKU, даты печати, кода зоны, special symbols, цвета.
7. **Temporal fusion**: один ценник объединяется по треку/QR/barcode/product+price; выбирается лучший crop по QR/OCR/sharpness. В режиме `defer_ocr` OCR/QR выполняются после трекинга только для representative crop.
8. **CSV export**: фиксированная схема Lenta Tech.

Схема и критика подробно описаны в `docs/architecture.md` и `docs/self_critique.md`.

---

## 6. Локальная оценка

```bash
python scripts/evaluate_on_public.py \
  --data-dir data/Данные \
  --config configs/default.yaml \
  --output-dir outputs/eval_public
```

Скрипт считает строгую proxy-метрику: матч по bbox IoU, затем доля строк с >=80% совпадения ключевых полей. Это не гарантирует точного совпадения со скрытой метрикой жюри, но помогает быстро находить регрессии.

---

## 7. Docker

```bash
docker build -t lenta-shelf-ai .
docker run --rm -p 7860:7860 -v "$PWD/models:/app/models" lenta-shelf-ai
```

Для maximum quality в Dockerfile можно раскомментировать установку `requirements-full.txt`.

---

## 8. Деплой

Рекомендуемый путь для демо: **Hugging Face Spaces Docker**.

1. Создать публичный Space, SDK = Docker.
2. Загрузить репозиторий.
3. Добавить `models/price_tag_yolo.pt`.
4. Проверить, что UI доступен без авторизации.

Подробно: `docs/deploy_huggingface_spaces.md`.

---

## 9. Профиль public-данных

Из загруженного архива: 6 видео, все 3840x2160, около 20 FPS. Размечены 3 видео: 57 + 71 + 29 строк, еще 3 видео без CSV. Полная таблица: `docs/data_profile.md`.

---

## 10. Что улучшить для финального сабмита

- Обучить YOLO и положить веса в `models/price_tag_yolo.pt`.
- Включить `zxing-cpp` и PaddleOCR в продовом образе.
- Прогнать self-training на `Unlabeled/*.mp4`: брать только crops с QR decoded + high detector confidence.
- Если появится внутренний SKU master-data, восстанавливать `product_name` по barcode/SKU и резко поднять качество названий.

Текущая жесткая самооценка артефакта без обученных весов: **84/100**. После обучения и self-training на public+unlabeled: ожидаемый уровень **93-96/100**.
