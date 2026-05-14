# Google Colab experiments

These notebooks run the same local pipeline code as the Kaggle kernels, but use a
Colab GPU runtime when Kaggle quota is exhausted.

Open from GitHub:

- [Experiment A: fallback gated detector diagnostic](https://colab.research.google.com/github/Black-Lotus-2026/Lenta-Tech-Life-Hack-2026/blob/main/colab/lenta_colab_exp_a_fallback_gated.ipynb)
- [Experiment B: zonal QR/OCR quality run](https://colab.research.google.com/github/Black-Lotus-2026/Lenta-Tech-Life-Hack-2026/blob/main/colab/lenta_colab_exp_b_zonal_qr_ocr.ipynb)
- [Experiment C: self-training detector run](https://colab.research.google.com/github/Black-Lotus-2026/Lenta-Tech-Life-Hack-2026/blob/main/colab/lenta_colab_exp_c_selftrain_detector.ipynb)
- [Compare saved Colab runs](https://colab.research.google.com/github/Black-Lotus-2026/Lenta-Tech-Life-Hack-2026/blob/main/colab/lenta_colab_compare_runs.ipynb)

Runtime requirements:

- Use `Runtime -> Change runtime type -> T4 GPU`.
- Upload `kaggle.json` when the notebook asks, or configure Colab secrets
  `KAGGLE_USERNAME` and `KAGGLE_KEY`.
- The notebooks download `whitenigger/lenta-shelf-ai-bundle`, run local
  dependencies only, and save artifacts to `MyDrive/lenta_colab_runs`.

Run order:

1. Run Experiment B first. It tests the end-to-end metric path: QR, OCR, parser,
   row fusion and dedupe.
2. Run Experiment A if B still has duplicate/fallback spam.
3. Run Experiment C only if detection recall is still the limiting blocker.
4. Run the compare notebook after each batch and inspect `good80_total`,
   `pred_total`, QR/barcode/price fill rates and duplicate/no-evidence counts.

Do not call cloud OCR/API/LLM from these notebooks. Colab is used only as GPU
compute for local code and local open-source libraries.
