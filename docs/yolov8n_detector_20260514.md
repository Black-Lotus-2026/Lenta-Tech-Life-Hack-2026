# YOLOv8n detector artifact 2026-05-14

Source archive: `lenta_yolov8n_runs.zip`

Selected weights:

- `yolov8n_lenta_20260514_145955/price_tag_yolo.pt`
- copied locally to `models/price_tag_yolo.pt`
- mirrored as `models/price_tag_yolo_yolov8n_20260514_best.pt`

Training configuration from `args.yaml`:

- base model: `yolov8n.pt`
- image size: `1280`
- batch: `6`
- requested epochs: `120`
- device: `0`
- optimizer: `AdamW`
- cache: `disk`
- rect: `true`
- multi_scale: `true`
- close_mosaic: `10`
- mosaic: `1.0`
- mixup: `0.1`
- perspective: `0.0001`

Best row from `results.csv`:

| metric | value |
| --- | ---: |
| epoch | 12 |
| precision(B) | 0.71522 |
| recall(B) | 0.89223 |
| mAP50(B) | 0.75548 |
| mAP50-95(B) | 0.28991 |

Critical caveat:

These are YOLO validation metrics on the generated YOLO dataset, not the final
hackathon metric. The artifact still needs Colab/Kaggle end-to-end validation:

1. `detection_recall_yolo_only.json`
2. `detection_recall_hybrid.json`
3. `eval_public_fast/metrics.json`
4. recognized CSV field fill rates
5. duplicate/no-evidence analysis

Do not treat this artifact as top-1 proven until `good_rows_at_80` improves on
public evaluation and OCR/QR/parser fill rates improve without row explosion.
