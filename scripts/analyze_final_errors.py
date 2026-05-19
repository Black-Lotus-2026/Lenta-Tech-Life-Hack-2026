#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from lenta_shelf_ai.schema import LEGACY_COLUMN_ALIASES, OUTPUT_COLUMNS
from lenta_shelf_ai.utils import iou_xyxy, normalize_text, smart_float, text_similarity


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=LEGACY_COLUMN_ALIASES)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[OUTPUT_COLUMNS]


def _box(row: pd.Series) -> list[float]:
    return [smart_float(row.get(c)) for c in ["x_min", "y_min", "x_max", "y_max"]]


def _valid_box(box: list[float]) -> bool:
    return len(box) == 4 and not any(math.isnan(v) for v in box) and box[2] > box[0] and box[3] > box[1]


def _nonempty(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "none", "нет"}:
        return ""
    return text


def _evidence_columns(row: pd.Series) -> list[str]:
    cols = [
        "product_name",
        "price_default",
        "price_card",
        "barcode",
        "id_sku",
        "print_datetime",
        "qr_code_barcode",
        "price1_qr",
        "price4_qr",
    ]
    return [col for col in cols if _nonempty(row.get(col, ""))]


def _match(gt: pd.DataFrame, pred: pd.DataFrame, iou_threshold: float) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matches: list[dict[str, Any]] = []
    for gi, grow in gt.iterrows():
        gbox = _box(grow)
        best_j = None
        best_iou = 0.0
        for pj, prow in pred.iterrows():
            if pj in matched_pred:
                continue
            pbox = _box(prow)
            if not _valid_box(gbox) or not _valid_box(pbox):
                continue
            score = iou_xyxy(gbox, pbox)
            if score > best_iou:
                best_iou = score
                best_j = pj
        if best_j is not None and best_iou >= iou_threshold:
            matched_gt.add(gi)
            matched_pred.add(best_j)
            matches.append({"gt_index": int(gi), "pred_index": int(best_j), "iou": round(float(best_iou), 4)})
    return matches, matched_gt, matched_pred


def _duplicate_clusters(pred: pd.DataFrame, iou_threshold: float = 0.30, text_threshold: float = 0.86) -> list[list[int]]:
    parent = {int(i): int(i) for i in pred.index}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, row_i in pred.iterrows():
        box_i = _box(row_i)
        if not _valid_box(box_i):
            continue
        text_i = normalize_text(" ".join(_nonempty(row_i.get(c, "")) for c in ["product_name", "price_default", "price_card"]))
        for j, row_j in pred.loc[pred.index > i].iterrows():
            box_j = _box(row_j)
            if not _valid_box(box_j):
                continue
            same_id = False
            conflict_id = False
            for col in ["qr_code_barcode", "barcode", "id_sku"]:
                a = _nonempty(row_i.get(col, ""))
                b = _nonempty(row_j.get(col, ""))
                if a and b and a == b:
                    same_id = True
                elif a and b and a != b:
                    conflict_id = True
            if conflict_id:
                continue
            text_j = normalize_text(" ".join(_nonempty(row_j.get(c, "")) for c in ["product_name", "price_default", "price_card"]))
            if same_id or iou_xyxy(box_i, box_j) >= iou_threshold or (text_i and text_j and text_similarity(text_i, text_j) >= text_threshold):
                union(int(i), int(j))

    clusters: dict[int, list[int]] = {}
    for i in pred.index:
        clusters.setdefault(find(int(i)), []).append(int(i))
    return [items for items in clusters.values() if len(items) > 1]


def analyze(gt_csv: Path, pred_csv: Path, iou_threshold: float = 0.35) -> dict[str, Any]:
    gt = _load_csv(gt_csv)
    pred = _load_csv(pred_csv)
    matches, matched_gt, matched_pred = _match(gt, pred, iou_threshold)
    no_evidence = [int(i) for i, row in pred.iterrows() if not _evidence_columns(row)]
    qr_failed_like = [
        int(i)
        for i, row in pred.iterrows()
        if _nonempty(row.get("product_name", "")) and not _nonempty(row.get("qr_code_barcode", "")) and not _nonempty(row.get("barcode", ""))
    ]
    field_fill = {
        col: int(sum(1 for _, row in pred.iterrows() if _nonempty(row.get(col, ""))))
        for col in OUTPUT_COLUMNS
        if col not in {"filename", "frame_timestamp", "x_min", "y_min", "x_max", "y_max"}
    }
    field_fill = {k: v for k, v in field_fill.items() if v}

    return {
        "gt_csv": str(gt_csv),
        "pred_csv": str(pred_csv),
        "gt_rows": int(len(gt)),
        "pred_rows": int(len(pred)),
        "matched_rows": int(len(matches)),
        "unmatched_gt": [int(i) for i in gt.index if i not in matched_gt],
        "unmatched_pred": [int(i) for i in pred.index if i not in matched_pred],
        "no_semantic_evidence_pred": no_evidence,
        "qr_failed_like_pred": qr_failed_like[:200],
        "duplicate_clusters": _duplicate_clusters(pred),
        "field_fill": field_fill,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Final CSV error analysis for Lenta shelf predictions")
    parser.add_argument("--gt-csv", required=True, type=Path)
    parser.add_argument("--pred-csv", required=True, type=Path)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--iou", type=float, default=0.35)
    args = parser.parse_args()

    report = analyze(args.gt_csv, args.pred_csv, iou_threshold=args.iou)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
