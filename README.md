## OCR для ценников с PaddleOCR

В репозитории добавлен ноутбук `notebooks/paddleocr_label_ocr.ipynb`, который:

1. читает CSV-разметку из `data/Данные/*/*.csv`;
2. вытаскивает нужный кадр по `frame_timestamp` из видео;
3. вырезает ценник по `x_min`, `y_min`, `x_max`, `y_max`;
4. распознаёт текст через PaddleOCR;
5. сохраняет результат в `outputs/ocr_recognized.csv`;
6. считает метрики совпадения полей и сохраняет их в `metrics/ocr_metrics.json`.

### Запуск

```powershell
pip install -r requirements.txt
```

Откройте `notebooks/paddleocr_label_ocr.ipynb` и выполните ячейки сверху вниз.

### Примечание

В ноутбуке используется предобученная PaddleOCR-модель. Под «обучением» здесь подразумевается калибровка OCR-пайплайна и проверка качества на train/test split.

