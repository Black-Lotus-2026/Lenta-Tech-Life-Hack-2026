# Model Artifacts

Runtime model paths used by the application and Kaggle scripts:

- `price_tag_yolo.pt`: whole price-tag detector.
- `field_zone_yolo.pt`: detector for fields inside a price-tag crop.
- `wechat_qr/`: local OpenCV QR recovery assets.

The default configuration uses `price_tag_yolo.pt` and `field_zone_yolo.pt`.
