from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple

import gradio as gr
import pandas as pd

from lenta_shelf_ai.pipeline import PipelineConfig, PriceTagPipeline


def process_video(video_file, mode: str, sample_fps: float, yolo_weights: str):
    if video_file is None:
        raise gr.Error("Загрузите mp4-видео с робота")
    video_path = video_file if isinstance(video_file, str) else video_file.name
    work_dir = Path(tempfile.mkdtemp(prefix="lenta_shelf_ai_"))
    cfg = PipelineConfig.from_file("configs/default.yaml")
    cfg.sample_fps = float(sample_fps)
    cfg.yolo_weights = yolo_weights.strip() or cfg.yolo_weights
    if mode == "Быстро: детекция + QR без OCR":
        cfg.enable_ocr = False
        cfg.sample_fps = min(cfg.sample_fps, 1.0)
    elif mode == "Точно: YOLO + QR + OCR + фьюжн":
        cfg.enable_ocr = True
        cfg.prefer_paddle = True
    elif mode == "CPU-safe: OCR Tesseract fallback":
        cfg.enable_ocr = True
        cfg.prefer_paddle = False
        cfg.detector_imgsz = min(int(cfg.detector_imgsz), 1280)
    pipe = PriceTagPipeline(cfg)
    out_csv = work_dir / f"{Path(video_path).stem}_recognized.csv"
    df = pipe.run_video(video_path, output_dir=work_dir, output_csv=out_csv)
    preview = df.head(200)
    return str(out_csv), preview, f"Готово: {len(df)} уникальных кандидатов ценников. CSV: {out_csv.name}"


with gr.Blocks(title="Lenta Shelf AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# Lenta Shelf AI - распознавание ценников с видео робота
Локальный пайплайн: детекция ценников, QR/barcode decoding, OCR, парсинг полей и экспорт CSV по формату Lenta Tech.
"""
    )
    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="Видео с робота (.mp4)")
            mode = gr.Radio(
                ["Точно: YOLO + QR + OCR + фьюжн", "CPU-safe: OCR Tesseract fallback", "Быстро: детекция + QR без OCR"],
                value="CPU-safe: OCR Tesseract fallback",
                label="Режим",
            )
            sample_fps = gr.Slider(0.2, 5.0, value=1.0, step=0.2, label="Частота обработки кадров, FPS")
            yolo_weights = gr.Textbox(value="models/price_tag_yolo.pt", label="Путь к весам YOLO (если есть)")
            run_btn = gr.Button("Распознать и сформировать CSV", variant="primary")
        with gr.Column(scale=2):
            status = gr.Textbox(label="Статус")
            csv_file = gr.File(label="Скачать CSV")
            table = gr.Dataframe(label="Предпросмотр результата", wrap=True, interactive=False)
    run_btn.click(process_video, inputs=[video, mode, sample_fps, yolo_weights], outputs=[csv_file, table, status])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=port)
