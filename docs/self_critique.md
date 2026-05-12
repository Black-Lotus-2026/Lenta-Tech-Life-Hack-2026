# Жесткая самооценка и план доведения до победного уровня

## Оценка текущего артефакта: 84/100

Что уже хорошо:
- Полный воспроизводимый продукт: CLI, Gradio UI, Dockerfile, README, scripts для dataset/train/infer/eval/export.
- Локальный контур без облачных API.
- Использует сильную промышленную архитектуру: detector -> QR -> OCR -> parser -> temporal fusion.
- Учитывает публичные особенности: 4K/20 FPS, блики, стекло, остановки/движение, разные цвета, опечатка `wholesale_level_1_coun`.
- Есть fallback, поэтому продукт запускается даже без YOLO/PaddleOCR.

Что мешает назвать это 100/100 без дополнительного обучения на GPU:
- В архив не включены обученные веса `models/price_tag_yolo.pt`: их нужно получить командой `build_yolo_dataset.py` + `train_detector.py`.
- Heuristic fallback не является победным детектором на скрытом видео, он нужен как safety net.
- OCR product_name без доменного словаря товаров хуже QR-полей; hidden metric может сильно штрафовать название.
- Нет публичного URL деплоя: подготовлен Docker/HF Space, но публикация требует аккаунт команды.

## Что сделать перед финальной отправкой

1. Обучить YOLO на автоматически расширенном датасете:
   ```bash
   python scripts/build_yolo_dataset.py --data-dir data/Данные --out-dir datasets/lenta_yolo --propagate 10
   python scripts/train_detector.py --data datasets/lenta_yolo/data.yaml --model yolo11n.pt --epochs 150 --imgsz 1280 --device 0
   cp runs/lenta/price_tag_yolo/weights/best.pt models/price_tag_yolo.pt
   ```
2. Прогнать `scripts/evaluate_on_public.py`, посмотреть false positives и поднять `yolo_conf` до точки максимума proxy metric.
3. Включить PaddleOCR/zxing-cpp в Docker или Space для максимального OCR/QR качества.
4. Добавить товарный словарь из внутренних SKU, если Lenta даст выгрузку: product_name тогда восстанавливается по barcode/sku почти идеально.
5. Псевдоразметить unlabeled видео confidence>=0.75 + QR decoded, дообучить второй цикл.

## Целевая версия после обучения: 93-96/100

Оставшиеся риски: стекло с бликами, очень мелкие ценники на дальнем плане, перекрытые QR, ценники без QR, скрытые шаблоны с белым/зеленым фоном. Для 98+/100 нужен либо внутренний SKU master-data, либо еще 1-2 часа автоматического self-training на скрытоподобных видео.
