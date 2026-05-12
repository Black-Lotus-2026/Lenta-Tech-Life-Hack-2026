from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .schema import LEGACY_COLUMN_ALIASES, OUTPUT_COLUMNS
from .utils import iou_xyxy, smart_float, normalize_text


def load_result_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=LEGACY_COLUMN_ALIASES)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[OUTPUT_COLUMNS]


def row_similarity(gt: pd.Series, pred: pd.Series) -> float:
    # Field-level exact/normalized similarity plus bbox IoU. Hidden metric may differ,
    # but this is a strict local proxy for self-critique.
    important = [
        "product_name", "price_default", "price_card", "price_discount", "barcode", "discount_amount",
        "id_sku", "print_datetime", "code", "additional_info", "color", "special_symbols",
        "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr", "action_price_qr", "action_code_qr",
    ]
    hits = 0
    total = 0
    for col in important:
        g = normalize_text(str(gt.get(col, ""))).lower().replace(",", ".")
        p = normalize_text(str(pred.get(col, ""))).lower().replace(",", ".")
        if g in {"", "nan"}:
            continue
        total += 1
        if g == p or (g == "нет" and p in {"нет", ""}):
            hits += 1
        elif col == "product_name" and g and p:
            common = len(set(g.split()) & set(p.split()))
            if common / max(1, len(set(g.split()))) >= 0.65:
                hits += 0.75
    field_score = hits / max(1, total)
    bbox_gt = [smart_float(gt.get(c)) for c in ["x_min", "y_min", "x_max", "y_max"]]
    bbox_pr = [smart_float(pred.get(c)) for c in ["x_min", "y_min", "x_max", "y_max"]]
    bbox_score = iou_xyxy(bbox_gt, bbox_pr) if not any(v != v for v in bbox_gt + bbox_pr) else 0.0
    return 0.85 * field_score + 0.15 * bbox_score


def evaluate_csv(gt_csv: str | Path, pred_csv: str | Path, iou_threshold: float = 0.35, row_threshold: float = 0.80) -> Dict[str, float]:
    gt = load_result_csv(gt_csv)
    pred = load_result_csv(pred_csv)
    matched_pred = set()
    good = 0
    matched = 0
    for gi, grow in gt.iterrows():
        best_j = None; best_iou = 0.0
        gbox = [smart_float(grow.get(c)) for c in ["x_min", "y_min", "x_max", "y_max"]]
        for pj, prow in pred.iterrows():
            if pj in matched_pred:
                continue
            pbox = [smart_float(prow.get(c)) for c in ["x_min", "y_min", "x_max", "y_max"]]
            iou = iou_xyxy(gbox, pbox)
            if iou > best_iou:
                best_iou = iou; best_j = pj
        if best_j is not None and best_iou >= iou_threshold:
            matched += 1
            matched_pred.add(best_j)
            sim = row_similarity(grow, pred.loc[best_j])
            if sim >= row_threshold:
                good += 1
    return {
        "gt_rows": float(len(gt)),
        "pred_rows": float(len(pred)),
        "matched_rows": float(matched),
        "good_rows_at_80": float(good),
        "success_metric_proxy": good / max(1, len(gt)),
        "detection_recall_proxy": matched / max(1, len(gt)),
    }
