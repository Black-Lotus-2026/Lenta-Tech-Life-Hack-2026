# Lenta Shelf AI

Offline computer-vision pipeline for shelf price-tag detection and CSV field recognition.

The release contains:

- FastAPI web app with Jinja2 templates for video upload and result download.
- Whole-tag detector, field-zone detector, QR/barcode cascade, OCR parser, tracking and row fusion.
- Kaggle and local evaluation scripts for the public videos.
- Regression tests for parsing, QR/barcode recovery, tracking/fusion and deployment code paths.

No cloud OCR, remote API, or LLM inference is used by the runtime pipeline.

## Local Run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860`, upload an `.mp4`, then download the generated CSV.

## Higher-Quality Runtime

For GPU/Kaggle experiments install the extended requirements:

```bash
pip install -r requirements-full.txt
python scripts/evaluate_on_public.py --data-dir /path/to/data/Данные --config configs/default.yaml --output-dir outputs/eval_public
```

The main model files are:

- `models/price_tag_yolo.pt`: whole price-tag detector.
- `models/field_zone_yolo.pt`: field detector inside each tag.
- `models/wechat_qr/*`: local QR recovery models used by OpenCV when available.

## CSV Output

The output schema follows the task fields:

```text
frame_timestamp,x_min,y_min,x_max,y_max,product_name,price_default,price_card,price_discount,discount_amount,barcode,qr_code_barcode,id_sku,print_datetime,code
```

## Verification

```bash
python -m pytest -q
python -m compileall app.py lenta_shelf_ai scripts kaggle tests
```
