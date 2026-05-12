from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

OUTPUT_COLUMNS: List[str] = [
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

ABSENT_VALUE = "нет"

QR_FIELD_ALIASES: Dict[str, List[str]] = {
    "qr_code_barcode": ["barcode", "b"],
    "price1_qr": ["price1", "p1"],
    "price2_qr": ["price2", "p2"],
    "price3_qr": ["price3", "p3"],
    "price4_qr": ["price4", "p4"],
    "wholesale_level_1_count": ["wholesaleLevel1Count", "wL1C"],
    "wholesale_level_1_price": ["wholesaleLevel1Price", "wL1P"],
    "wholesale_level_2_count": ["wholesaleLevel2Count", "wL2C"],
    "wholesale_level_2_price": ["wholesaleLevel2Price", "wL2P"],
    "action_price_qr": ["actionPrice", "aP"],
    "action_code_qr": ["actionCode", "aC"],
}

LEGACY_COLUMN_ALIASES: Dict[str, str] = {
    "wholesale_level_1_coun": "wholesale_level_1_count",
}

@dataclass
class Detection:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float = 0.0
    label: str = "price_tag"
    source: str = "unknown"

    @property
    def xyxy(self) -> List[float]:
        return [float(self.x_min), float(self.y_min), float(self.x_max), float(self.y_max)]

    @property
    def area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def clamp(self, width: int, height: int) -> "Detection":
        self.x_min = max(0.0, min(float(width - 1), self.x_min))
        self.x_max = max(0.0, min(float(width - 1), self.x_max))
        self.y_min = max(0.0, min(float(height - 1), self.y_min))
        self.y_max = max(0.0, min(float(height - 1), self.y_max))
        return self

    def expanded(self, width: int, height: int, px: float = 0.08, py: float = 0.10) -> "Detection":
        w = self.x_max - self.x_min
        h = self.y_max - self.y_min
        return Detection(
            max(0.0, self.x_min - w * px),
            max(0.0, self.y_min - h * py),
            min(float(width - 1), self.x_max + w * px),
            min(float(height - 1), self.y_max + h * py),
            self.score,
            self.label,
            self.source,
        )

@dataclass
class OCRLine:
    text: str
    confidence: float = 0.0
    box: Any = None
    engine: str = "unknown"

@dataclass
class TagObservation:
    filename: str
    timestamp_ms: int
    detection: Detection
    text: str = ""
    ocr_lines: List[OCRLine] = field(default_factory=list)
    qr_payloads: List[str] = field(default_factory=list)
    parsed: Dict[str, Any] = field(default_factory=dict)
    image_quality: float = 0.0

    def to_row(self) -> Dict[str, Any]:
        row = {col: "" for col in OUTPUT_COLUMNS}
        row["filename"] = self.filename
        row["frame_timestamp"] = int(self.timestamp_ms)
        row["x_min"] = round(float(self.detection.x_min), 1)
        row["y_min"] = round(float(self.detection.y_min), 1)
        row["x_max"] = round(float(self.detection.x_max), 1)
        row["y_max"] = round(float(self.detection.y_max), 1)
        for k, v in self.parsed.items():
            if k in row and v is not None:
                row[k] = v
        return normalize_row(row)


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {col: row.get(col, "") for col in OUTPUT_COLUMNS}
    for key, value in list(normalized.items()):
        if value is None:
            normalized[key] = ""
        elif isinstance(value, float):
            if value != value:
                normalized[key] = ""
            else:
                normalized[key] = value
    return normalized


def ensure_columns(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_row(row) for row in rows]
