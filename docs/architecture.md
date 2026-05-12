# Архитектура Lenta Shelf AI

## Цель
Преобразовать 4K-видео робота в одну строку CSV на уникальный ценник с полями из задания: название, цены, QR-поля, barcode, SKU, дата печати, зона выкладки, цвет, специальные символы и координаты.

## Основной поток

1. **Video sampler**: читает MP4 через OpenCV, пропускает размазанные кадры по Laplacian sharpness, в обычном режиме берет 1-2 FPS, в accuracy режиме - до 5 FPS.
2. **Detector ensemble**:
   - основной путь: компактный YOLO11n/YOLOv8n, дообученный на автоматическом датасете из публичных CSV и соседних кадров через template tracking;
   - QR-seed detector: OpenCV QRCodeDetector находит QR и расширяет область до ценника;
   - color/geometry fallback: HSV-сегментация красных/желтых/зеленых ценников + edge-density фильтры.
3. **Crop enhancer**: upscale, CLAHE, unsharp masking. Это критично для 4K-кадров с мелкими ценниками и бликами.
4. **QR/barcode layer**: zxing-cpp -> pyzbar -> OpenCV QR. QR считается источником истины для barcode и price1-price4/action fields.
5. **OCR layer**: PaddleOCR при наличии, иначе Tesseract rus+eng. OCR запускается по crop, а не по полному кадру.
6. **Template parser**: регулярные выражения + бизнес-правила для цен, скидки, EAN-13, SKU, даты печати, кода зоны, special_symbols, цвета.
7. **Temporal fusion**: простой SORT-like tracker + объединение треков по barcode/QR/product+price. Из нескольких кадров выбирается лучший crop по QR/OCR/sharpness.
8. **CSV writer**: строго фиксированный порядок колонок, совместимость с опечаткой `wholesale_level_1_coun` в публичных CSV.

## Почему это сильнее одиночного OCR

- OCR всего кадра дает шум от товаров, этикеток и POS-материалов.
- QR обычно содержит структурированные цены и barcode, поэтому его нужно декодировать раньше OCR.
- Один ценник виден на десятках кадров: мультикадровое голосование резко повышает шанс получить резкий crop без блика.
- Цвет/геометрия полезны до обучения и как fallback, но для Top-решения detector должен быть дообучен.

## Режимы

- `CPU-safe`: без тяжелых зависимостей, Tesseract fallback, подходит для демо и слабого сервера.
- `Accurate`: YOLO + PaddleOCR + QR + fusion, основной соревновательный режим.
- `Fast`: детекция + QR без OCR, полезно для edge-оценки и больших видео.

## Edge/RKNN

После обучения:

```bash
python scripts/export_edge.py --weights runs/lenta/price_tag_yolo/weights/best.pt --format onnx --imgsz 1280
python scripts/export_edge.py --weights runs/lenta/price_tag_yolo/weights/best.pt --format rknn --name rk3588 --imgsz 1280
```

Для CPU можно использовать ONNX/OpenVINO. Для Rockchip - RKNN INT8 после калибровки на crop/frame наборе из public+unlabeled видео.
