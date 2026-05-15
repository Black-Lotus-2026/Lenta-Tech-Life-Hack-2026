Place trained detector weights here.

Current competitive detector artifact:

- architecture: YOLOv8n
- source run: `yolov8n_lenta_20260514_145955`
- train setup: 120 requested epochs, early-stopped after 37 logged epochs
- best validation epoch by mAP50: epoch 12
- best validation metrics from `results.csv`: precision `0.71522`, recall `0.89223`, mAP50 `0.75548`, mAP50-95 `0.28991`
- runtime path expected by config/Kaggle/Colab: `models/price_tag_yolo.pt`

```bash
cp runs/lenta/yolov8n_lenta/weights/best.pt models/price_tag_yolo.pt
```

Large weights are intentionally excluded from git by `.gitignore`.
