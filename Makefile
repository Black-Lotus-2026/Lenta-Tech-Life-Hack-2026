.PHONY: install app infer dataset train eval docker

install:
	pip install -r requirements.txt

install-full:
	pip install -r requirements-full.txt

app:
	python app.py

infer:
	python scripts/infer_video.py $(VIDEO) --output-dir outputs --weights models/price_tag_yolo.pt

dataset:
	python scripts/build_yolo_dataset.py --data-dir data/Данные --out-dir datasets/lenta_yolo --propagate 10

train:
	python scripts/train_detector.py --data datasets/lenta_yolo/data.yaml --model yolo11n.pt --epochs 150 --imgsz 1280 --device 0

eval:
	python scripts/evaluate_on_public.py --data-dir data/Данные --config configs/default.yaml --output-dir outputs/eval_public

docker:
	docker build -t lenta-shelf-ai .
