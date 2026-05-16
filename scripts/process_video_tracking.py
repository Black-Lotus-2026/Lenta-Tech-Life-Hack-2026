import argparse
import logging
import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import easyocr
import numpy as np
import pandas as pd
from ultralytics import YOLO

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
except Exception:
    pyzbar_decode = None


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("video-tracking")


EXPECTED_COLUMNS = [
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]

MODEL_CLASSES = [
    "additional_info",
    "barcode",
    "code",
    "discount_amount",
    "id_sku",
    "price_card",
    "price_default",
    "price_discount",
    "print_datetime",
    "product_name",
    "qr_code_barcode",
]

ANCHOR_CLASSES = {
    "price_card",
    "price_default",
    "price_discount",
    "product_name",
    "barcode",
    "qr_code_barcode",
    "id_sku",
}

QRCODE_CLASSES = {"qr_code_barcode", "barcode", "code"}


@dataclass
class Detection:
    class_name: str
    conf: float
    bbox: Tuple[int, int, int, int]


@dataclass
class TrackState:
    track_id: int
    first_frame: int
    first_bbox: Tuple[int, int, int, int]
    last_frame: int
    last_bbox: Tuple[int, int, int, int]
    best_score: float = -1.0
    best_frame_idx: int = -1
    best_frame_img: Optional[np.ndarray] = None
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    color: str = "нет"
    fields: Dict[str, Tuple[str, float]] = field(default_factory=dict)


def resolve_project_root() -> pathlib.Path:
    start = pathlib.Path.cwd().resolve()
    for c in [start, *start.parents]:
        if (c / "notebook").exists() and (c / "data").exists():
            return c
    return start


def load_sr_model(project_root: pathlib.Path):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    candidates = [
        project_root / "weight" / "ESPCN_x4.pb",
        project_root / "weights" / "ESPCN_x4.pb",
        pathlib.Path("weight/ESPCN_x4.pb"),
        pathlib.Path("weights/ESPCN_x4.pb"),
    ]
    for p in candidates:
        if p.exists():
            try:
                sr.readModel(str(p))
                sr.setModel("espcn", 4)
                logger.info("ESPCN loaded: %s", p)
                return sr
            except Exception as e:
                logger.warning("ESPCN load failed (%s): %s", p, e)
    logger.warning("ESPCN model not found, fallback without SR")
    return None

# Thresholds and tuning
SHARPNESS_THRESHOLD = 80.0  # below this, skip OCR on crop
MIN_CROP_AREA_FOR_OCR = 28 * 28  # require minimal area for OCR


def enhance_crop(crop: np.ndarray, sr_model) -> np.ndarray:
    if crop is None or crop.size == 0:
        return crop
    if sr_model is None:
        return crop
    h, w = crop.shape[:2]
    if h < 24 or w < 24:
        return crop
    try:
        return sr_model.upsample(crop)
    except Exception:
        return crop


def clean_price(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"\d+[\d\s]*[.,]?\d{0,2}", text.replace("\xa0", " "))
    if not m:
        return ""
    return m.group(0).replace(" ", "").replace(",", ".")


def clean_date(text: str) -> str:
    if not text:
        return ""
    patterns = [r"\d{2}[./-]\d{2}[./-]\d{2,4}(?:\s+\d{1,2}:\d{2})?", r"\d{4}-\d{2}-\d{2}", r"\d{1,2}:\d{2}"]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(0)
    return ""


def clean_code(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"[A-Za-z0-9_\-]{4,}", text)
    return m.group(0) if m else ""


def clean_barcode(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"\b\d{8,14}\b", text)
    return m.group(0) if m else ""


def clean_generic(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def parse_qr_prices(text: str) -> List[str]:
    if not text:
        return []
    vals = re.findall(r"\d+[.,]\d{2}", text)
    return [v.replace(",", ".") for v in vals][:4]


def ocr_image(reader: easyocr.Reader, img: np.ndarray) -> Tuple[str, float]:
    if img is None or img.size == 0:
        return "", 0.0
    try:
        results = reader.readtext(img)
    except Exception:
        return "", 0.0
    if not results:
        return "", 0.0
    txt = [r[1] for r in results if r[2] > 0.2]
    conf = [float(r[2]) for r in results]
    return " ".join(txt), float(np.mean(conf)) if conf else 0.0


def decode_qr(crop: np.ndarray, sr_model) -> str:
    if crop is None or crop.size == 0:
        return ""
    if pyzbar_decode is None:
        return ""
    enhanced = enhance_crop(crop, sr_model)
    try:
        results = pyzbar_decode(enhanced)
    except Exception:
        return ""
    out = []
    for r in results:
        try:
            out.append(r.data.decode("utf-8", errors="ignore"))
        except Exception:
            continue
    return " | ".join([x for x in out if x])


def infer_color(crop: np.ndarray) -> str:
    if crop is None or crop.size == 0:
        return "нет"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = float(np.mean(hsv[:, :, 0]))
    s = float(np.mean(hsv[:, :, 1]))
    v = float(np.mean(hsv[:, :, 2]))
    if s < 25 and v > 170:
        return "white"
    if h < 10 or h > 170:
        return "red"
    if 15 <= h <= 40:
        return "yellow"
    if 40 < h <= 85:
        return "green"
    if 85 < h <= 135:
        return "blue"
    return "red"


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1, (ax2 - ax1) * (ay2 - ay1))
    ba = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(aa + ba - inter)


def center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return float(np.hypot(ax - bx, ay - by))


def expand_box(box: Tuple[int, int, int, int], pad: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    px = int(max(4, bw * pad))
    py = int(max(4, bh * pad))
    return max(0, x1 - px), max(0, y1 - py), min(w - 1, x2 + px), min(h - 1, y2 + py)


def get_model_names(model: YOLO) -> Dict[int, str]:
    try:
        if hasattr(model, "model") and hasattr(model.model, "names"):
            return dict(model.model.names)
        if hasattr(model, "names"):
            return dict(model.names)
    except Exception:
        pass
    return {i: n for i, n in enumerate(MODEL_CLASSES)}


def detect_frame(model: YOLO, frame: np.ndarray, conf: float, imgsz: int, names_map: Dict[int, str]) -> List[Detection]:
    out: List[Detection] = []
    res = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)
    if not res:
        return out
    r = res[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out

    try:
        cls_arr = boxes.cls.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
    except Exception:
        cls_arr = np.array(boxes.cls)
        xyxy = np.array(boxes.xyxy)
        confs = np.array(boxes.conf)

    for cls_id, bb, c in zip(cls_arr, xyxy, confs):
        cid = int(cls_id)
        name = names_map.get(cid, f"class_{cid}")
        if name not in MODEL_CLASSES:
            continue
        x1, y1, x2, y2 = [int(max(0, v)) for v in bb[:4]]
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(Detection(class_name=name, conf=float(c), bbox=(x1, y1, x2, y2)))
    return out


def update_track_field(track: TrackState, key: str, value: str, score: float):
    if not value:
        return
    old = track.fields.get(key)
    if old is None or score > old[1]:
        track.fields[key] = (value, score)


def detection_to_text(det: Detection, crop: np.ndarray, reader: easyocr.Reader, sr_model, sharpness: float) -> Tuple[str, float, bool]:
    """Return (text, confidence, from_qr)
    - from_qr==True means text was obtained strictly from QR decode (pyzbar)
    - For class 'qr_code_barcode' we ONLY accept decoded QR (no OCR fallback)
    - For other barcode/code classes we attempt decode then OCR
    - Skip OCR for very blurry/small crops according to SHARPNESS_THRESHOLD and MIN_CROP_AREA_FOR_OCR
    """
    # QR class: strict decode only
    if det.class_name == 'qr_code_barcode':
        decoded = decode_qr(crop, sr_model)
        if decoded:
            return decoded, max(0.5, det.conf), True
        logger.debug(f"QR decode empty for qr_code_barcode (conf={det.conf})")
        return "", 0.0, False

    # For barcode/code: try decode first
    if det.class_name in {'barcode', 'code'}:
        decoded = decode_qr(crop, sr_model)
        if decoded:
            return decoded, max(0.5, det.conf), True

    # Decide whether to run OCR
    area = crop.shape[0] * crop.shape[1]
    if sharpness < SHARPNESS_THRESHOLD or area < MIN_CROP_AREA_FOR_OCR:
        logger.debug(f"Skipping OCR for class={det.class_name} due to low sharpness={sharpness:.1f} or small area={area}")
        return "", 0.0, False

    txt, ocr_conf = ocr_image(reader, enhance_crop(crop, sr_model))
    return txt, max(ocr_conf, det.conf * 0.5), False


def apply_cleaner(class_name: str, text: str) -> str:
    if class_name in {"price_card", "price_default", "price_discount", "discount_amount"}:
        return clean_price(text)
    if class_name == "print_datetime":
        return clean_date(text)
    if class_name in {"barcode"}:
        return clean_barcode(text) or clean_code(text)
    if class_name in {"id_sku", "code", "qr_code_barcode"}:
        return clean_code(text)
    if class_name in {"product_name", "additional_info"}:
        return clean_generic(text)
    return clean_generic(text)


def within(box: Tuple[int, int, int, int], region: Tuple[int, int, int, int]) -> bool:
    bx, by = center(box)
    x1, y1, x2, y2 = region
    return x1 <= bx <= x2 and y1 <= by <= y2


def match_tracks(
    tracks: Dict[int, TrackState],
    anchors: List[Detection],
    frame_idx: int,
    next_id: int,
    max_age: int,
) -> Tuple[Dict[int, TrackState], List[Tuple[int, Detection]], int]:
    active = [t for t in tracks.values() if frame_idx - t.last_frame <= max_age]
    used_tracks = set()
    matched: List[Tuple[int, Detection]] = []

    for a in anchors:
        best_tid = None
        best_score = -1e9
        for t in active:
            if t.track_id in used_tracks:
                continue
            i = iou(t.last_bbox, a.bbox)
            d = distance(t.last_bbox, a.bbox)
            score = i * 2.5 - 0.002 * d
            if score > best_score:
                best_score = score
                best_tid = t.track_id

        if best_tid is not None:
            t = tracks[best_tid]
            if iou(t.last_bbox, a.bbox) >= 0.08 or distance(t.last_bbox, a.bbox) <= 220:
                tracks[best_tid].last_bbox = a.bbox
                tracks[best_tid].last_frame = frame_idx
                matched.append((best_tid, a))
                used_tracks.add(best_tid)
                continue

        tid = next_id
        next_id += 1
        tracks[tid] = TrackState(
            track_id=tid,
            first_frame=frame_idx,
            first_bbox=a.bbox,
            last_frame=frame_idx,
            last_bbox=a.bbox,
        )
        matched.append((tid, a))
        used_tracks.add(tid)

    return tracks, matched, next_id


def post_fill_row(row: Dict[str, str]):
    if row["barcode"] == "нет":
        val = clean_barcode(row["qr_code_barcode"])
        if val:
            row["barcode"] = val
    if row["id_sku"] == "нет":
        cand = re.search(r"\b\d{9,13}\b", row["code"]) if row["code"] != "нет" else None
        if cand:
            row["id_sku"] = cand.group(0)
    if row["price_discount"] == "нет" and row["price_card"] != "нет":
        row["price_discount"] = row["price_card"]

    qr_text = row["qr_code_barcode"] if row["qr_code_barcode"] != "нет" else ""
    qr_prices = parse_qr_prices(qr_text)
    qr_targets = ["price1_qr", "price2_qr", "price3_qr", "price4_qr"]
    for i, p in enumerate(qr_prices):
        if i < len(qr_targets) and row[qr_targets[i]] == "нет":
            row[qr_targets[i]] = p

    if row["action_price_qr"] == "нет" and row["price_card"] != "нет":
        row["action_price_qr"] = row["price_card"]


def process_video(
    video_path: pathlib.Path,
    weights_path: pathlib.Path,
    out_csv: pathlib.Path,
    video_label: str,
    frame_stride: int,
    conf: float,
    imgsz: int,
    rotate_ccw90: bool,
    max_age: int,
    examples_dir: Optional[pathlib.Path],
    examples_count: int,
):
    model = YOLO(str(weights_path))
    names_map = get_model_names(model)
    logger.info("Weights: %s", weights_path)

    reader = easyocr.Reader(["ru", "en"], gpu=False)
    sr_model = load_sr_model(resolve_project_root())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Video opened: fps=%.3f frames=%d", fps, total)

    tracks: Dict[int, TrackState] = {}
    next_id = 1

    frame_idx = 0
    last_logged_pct = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if rotate_ccw90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        dets = detect_frame(model, frame, conf=conf, imgsz=imgsz, names_map=names_map)
        if not dets:
            frame_idx += 1
            continue

        anchors = [d for d in dets if d.class_name in ANCHOR_CLASSES]
        if not anchors:
            frame_idx += 1
            continue

        tracks, matched, next_id = match_tracks(tracks, anchors, frame_idx, next_id, max_age=max_age)

        h, w = frame.shape[:2]

        for tid, anchor in matched:
            t = tracks[tid]
            x1, y1, x2, y2 = anchor.bbox
            x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            acrop = frame[y1c:y2c, x1c:x2c]

            if t.color == "нет":
                t.color = infer_color(acrop)

            sharp = 0.0
            if acrop is not None and acrop.size > 0:
                gray = cv2.cvtColor(acrop, cv2.COLOR_BGR2GRAY)
                sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            obs_score = float(anchor.conf) + 0.0004 * sharp

            if obs_score > t.best_score:
                t.best_score = obs_score
                t.best_frame_idx = frame_idx
                t.best_bbox = anchor.bbox
                t.best_frame_img = frame.copy()

            region = expand_box(anchor.bbox, pad=0.9, w=w, h=h)
            related = [d for d in dets if within(d.bbox, region)]

            for d in related:
                dx1, dy1, dx2, dy2 = d.bbox
                dx1, dy1 = max(0, dx1), max(0, dy1)
                dx2, dy2 = min(w - 1, dx2), min(h - 1, dy2)
                crop = frame[dy1:dy2, dx1:dx2]
                if crop is None or crop.size == 0:
                    continue
                raw, txt_conf, from_qr = detection_to_text(d, crop, reader=reader, sr_model=sr_model, sharpness=sharp)
                cleaned = apply_cleaner(d.class_name, raw)
                score = float(d.conf) * max(0.2, txt_conf)

                # Handle qr_code_barcode strictly from decoded QR only
                if d.class_name == "qr_code_barcode":
                    if from_qr and raw:
                        update_track_field(t, "qr_code_barcode", raw, score)
                    else:
                        logger.debug(f"qr_code_barcode not decoded in frame {frame_idx} track={t.track_id}")

                elif d.class_name == "barcode":
                    # barcode: accept decoded payload first, else attempt cleaned OCR barcode
                    if from_qr and raw:
                        update_track_field(t, "barcode", raw, score)
                    else:
                        val = clean_barcode(cleaned) or clean_barcode(raw) or cleaned
                        if val:
                            update_track_field(t, "barcode", val, score)

                else:
                    # other classes: use cleaned OCR if available
                    if raw and not from_qr:
                        update_track_field(t, d.class_name, cleaned, score)

                # Keep extra fields from QR payloads only if from actual QR decode
                if from_qr and raw:
                    prices = parse_qr_prices(raw)
                    for i, p in enumerate(prices[:4], start=1):
                        update_track_field(t, f"price{i}_qr", p, score * (0.98 - 0.05 * i))

        # Progress logging
        if total and total > 0:
            pct = int(frame_idx / float(total) * 100)
            if pct - last_logged_pct >= 2:
                logger.info(f"Processing video: {pct}% ({frame_idx}/{total} frames)")
                last_logged_pct = pct

        frame_idx += 1

    cap.release()

    rows: List[Dict[str, str]] = []
    for tid in sorted(tracks.keys(), key=lambda k: tracks[k].first_frame):
        t = tracks[tid]
        row = {k: "нет" for k in EXPECTED_COLUMNS}
        row["filename"] = video_label

        x1, y1, x2, y2 = t.first_bbox
        row["frame_timestamp"] = str(t.first_frame)
        row["x_min"] = str(x1)
        row["y_min"] = str(y1)
        row["x_max"] = str(x2)
        row["y_max"] = str(y2)
        row["color"] = t.color if t.color else "нет"

        for k, (v, _) in t.fields.items():
            if k in row and v:
                row[k] = v

        post_fill_row(row)
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info("Saved CSV: %s (%d rows)", out_csv, len(rows))

    if examples_dir is not None and examples_count > 0:
        examples_dir.mkdir(parents=True, exist_ok=True)
        ranked = sorted(tracks.values(), key=lambda t: t.best_score, reverse=True)
        saved = 0
        for t in ranked:
            if t.best_frame_img is None or t.best_bbox is None:
                continue
            img = t.best_frame_img.copy()
            x1, y1, x2, y2 = t.best_bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"track={t.track_id} frame={t.best_frame_idx}",
                (max(0, x1), max(16, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            out_img = examples_dir / f"example_track_{t.track_id:03d}.jpg"
            cv2.imwrite(str(out_img), img)
            saved += 1
            if saved >= examples_count:
                break
        logger.info("Saved examples: %d -> %s", saved, examples_dir)


def parse_args():
    root = resolve_project_root()
    default_video = root / "data" / "Данные" / "25_12-20" / "25_12-20.mp4"
    default_weights = root / "weight" / "best.pt"
    default_csv = root / "runs" / "visualizations" / "25_12-20_tracking.csv"
    default_examples = root / "runs" / "visualizations" / "tracking_examples"

    p = argparse.ArgumentParser(description="Video -> tracked price tags -> CSV")
    p.add_argument("--video", type=pathlib.Path, default=default_video)
    p.add_argument("--weights", type=pathlib.Path, default=default_weights)
    p.add_argument("--out-csv", type=pathlib.Path, default=default_csv)
    p.add_argument("--video-label", type=str, default="25_12-20/25_12-20.mp4")
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--max-age", type=int, default=45)
    p.add_argument(
        "--no-rotate",
        action="store_true",
        help="Do not rotate frames. By default frames are rotated 90 CCW.",
    )
    p.add_argument("--examples-dir", type=pathlib.Path, default=default_examples)
    p.add_argument("--examples-count", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    process_video(
        video_path=args.video,
        weights_path=args.weights,
        out_csv=args.out_csv,
        video_label=args.video_label,
        frame_stride=max(1, args.frame_stride),
        conf=args.conf,
        imgsz=args.imgsz,
        rotate_ccw90=not bool(args.no_rotate),
        max_age=max(1, args.max_age),
        examples_dir=args.examples_dir,
        examples_count=max(0, args.examples_count),
    )


if __name__ == "__main__":
    main()



