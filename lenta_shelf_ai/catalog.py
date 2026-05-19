from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

from .utils import normalize_text, text_similarity, smart_float

_DIGITS_RE = re.compile(r"\D+")
_CYR_OR_LATIN_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")
_BAD_PRODUCT_RE = re.compile(
    r"\b(?:товар\s+закончился|нет\s+товара|stock\s*out|service|служебн|ошибк|ценник|barcode|qr)\b",
    re.IGNORECASE,
)

BARCODE_COLUMNS = (
    "barcode",
    "bar_code",
    "ean",
    "ean13",
    "gtin",
    "gtin13",
    "штрихкод",
    "штрих_код",
    "шк",
    "barcodes",
)
NAME_COLUMNS = (
    "fullname",
    "full_name",
    "product_name",
    "name",
    "title",
    "Наименование",
    "наименование",
    "товар",
    "product",
)
PRICE_COLUMNS = (
    "price",
    "price_regular",
    "regular_price",
    "cost",
    "cost_regular",
    "main_price",
    "old_price",
    "цена",
)
_STOP_TOKENS = {
    "товар", "цена", "скидка", "карта", "картой", "лент", "лента", "руб", "коп", "шт", "кг", "гр", "г", "мл", "л",
    "нет", "product", "price", "discount", "barcode", "qr",
}

# Reused across short-lived Pipeline objects in tests/ablation loops.  Loading
# the bundled 35-40 MB catalogs repeatedly was a hidden runtime bug.
_GLOBAL_BARCODE_CACHE: dict[str, Dict[str, str]] = {}
_GLOBAL_TEXT_ROWS_CACHE: dict[str, list[dict[str, object]]] = {}


def _digits(value: object) -> str:
    return _DIGITS_RE.sub("", str(value or ""))


def _normal_key(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(value or "").strip().lower())


def _candidate_paths(root: Path) -> list[Path]:
    env = os.environ.get("LENTA_PRODUCT_CATALOG", "").strip()
    paths: list[Path] = []
    if env:
        for item in env.replace(";", os.pathsep).split(os.pathsep):
            item = item.strip()
            if item:
                paths.append(Path(item))
    paths.extend(
        [
            root / "data" / "catalogs" / "products_a.csv",
            root / "data" / "catalogs" / "products_b.csv",
            root / "data" / "catalog" / "products.csv",
            root / "data" / "products.csv",
            root / "data" / "db_hack.csv",
            root / "db_hack.csv",
            root / "input" / "db_hack.csv",
            root / "../input" / "db_hack.csv",
            Path("/kaggle/input") / "db_hack.csv",
            Path("/kaggle/input") / "products.csv",
        ]
    )
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        try:
            resolved = str(p.expanduser().resolve())
        except Exception:
            resolved = str(p)
        if resolved not in seen:
            out.append(Path(resolved))
            seen.add(resolved)
    return out


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        if sample.count(";") >= sample.count(","):
            return ";"
        return ","


def _open_csv_dicts(path: Path):
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                sample = fh.read(4096)
                fh.seek(0)
                delimiter = _sniff_delimiter(sample)
                reader = csv.DictReader(fh, delimiter=delimiter)
                if reader.fieldnames:
                    yield from reader
                    return
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            return
        except Exception:
            return


def _pick_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    columns_list = list(columns)
    normalized = {_normal_key(c): c for c in columns_list}
    for candidate in candidates:
        key = _normal_key(candidate)
        if key in normalized:
            return normalized[key]
    for column in columns_list:
        key = _normal_key(column)
        if any(_normal_key(candidate) in key or key in _normal_key(candidate) for candidate in candidates):
            return column
    return None


def _clean_product_name(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" ;,\t\n\r")
    if len(text) < 3:
        return ""
    if not _CYR_OR_LATIN_RE.search(text):
        return ""
    return text


def _normalize_product_text(text: object) -> str:
    value = normalize_text(str(text or "")).lower().replace("ё", "е")
    value = value.translate(str.maketrans({
        "0": "о", "1": "l", "3": "з", "4": "ч", "5": "s", "6": "б", "8": "в",
    }))
    value = re.sub(r"[^0-9a-zа-я]+", " ", value, flags=re.I)
    return normalize_text(value)


def _tokens(text: object) -> set[str]:
    norm = _normalize_product_text(text)
    out = set()
    for tok in norm.split():
        if len(tok) < 3 or tok in _STOP_TOKENS:
            continue
        out.add(tok)
    return out


def _price_values(row: dict[str, object], columns: Iterable[str]) -> list[float]:
    values: list[float] = []
    for col in columns:
        value = smart_float(row.get(col, ""), default=float("nan"))
        if not math.isnan(value) and 1.0 <= value <= 99999:
            values.append(float(round(value, 2)))
    return values


class ProductCatalog:
    """Local product catalog lookup.

    Barcode lookup is deterministic and safe.  Text+price matching is optional
    and high-threshold only; it is intended for local catalog recovery
    when OCR sees a noisy product name but no barcode evidence.
    """

    def __init__(self, paths: Iterable[str | Path] = ()):  # noqa: D401
        self.paths = [Path(p) for p in paths]
        self._by_barcode: Optional[Dict[str, str]] = None
        self._product_rows: Optional[list[dict[str, object]]] = None
        self.loaded_paths: list[str] = []
        self.loaded_text_paths: list[str] = []

    @classmethod
    def from_env_or_default(cls, root: str | Path | None = None) -> "ProductCatalog":
        base = Path(root or ".")
        return cls(p for p in _candidate_paths(base) if p.exists() and p.is_file())

    def _load_one(self, path: Path, *, include_text_rows: bool = False) -> tuple[Dict[str, str], list[dict[str, object]]]:
        try:
            cache_key = str(path.expanduser().resolve())
        except Exception:
            cache_key = str(path)
        if not include_text_rows and cache_key in _GLOBAL_BARCODE_CACHE:
            if cache_key not in self.loaded_paths and _GLOBAL_BARCODE_CACHE[cache_key]:
                self.loaded_paths.append(cache_key)
            return _GLOBAL_BARCODE_CACHE[cache_key], []
        if include_text_rows and cache_key in _GLOBAL_TEXT_ROWS_CACHE and cache_key in _GLOBAL_BARCODE_CACHE:
            if cache_key not in self.loaded_paths and _GLOBAL_BARCODE_CACHE[cache_key]:
                self.loaded_paths.append(cache_key)
            if cache_key not in self.loaded_text_paths and _GLOBAL_TEXT_ROWS_CACHE[cache_key]:
                self.loaded_text_paths.append(cache_key)
            return _GLOBAL_BARCODE_CACHE[cache_key], _GLOBAL_TEXT_ROWS_CACHE[cache_key]
        rows_iter = _open_csv_dicts(path)
        if rows_iter is None:
            return {}, []
        rows = list(rows_iter or [])
        if not rows:
            return {}, []
        columns = list(rows[0].keys())
        barcode_col = _pick_column(columns, BARCODE_COLUMNS)
        name_col = _pick_column(columns, NAME_COLUMNS)
        price_cols = [c for c in columns if _pick_column([c], PRICE_COLUMNS)]
        if not name_col:
            return {}, []
        mapping: Dict[str, str] = {}
        product_rows: list[dict[str, object]] = []
        for row in rows:
            name = _clean_product_name(row.get(name_col, ""))
            if not name:
                continue
            if barcode_col:
                raw = row.get(barcode_col, "")
                for token in re.split(r"[;,|\s]+", str(raw or "")):
                    code = _digits(token)
                    if len(code) < 8:
                        continue
                    mapping.setdefault(code, name)
                    if len(code) > 13:
                        mapping.setdefault(code[:13], name)
                        mapping.setdefault(code[-13:], name)
            if include_text_rows:
                prices = _price_values(row, price_cols)
                toks = _tokens(name)
                if toks:
                    product_rows.append({"name": name, "tokens": toks, "norm": _normalize_product_text(name), "prices": prices, "source": str(path)})
        _GLOBAL_BARCODE_CACHE[cache_key] = mapping
        if include_text_rows:
            _GLOBAL_TEXT_ROWS_CACHE[cache_key] = product_rows
        if mapping and cache_key not in self.loaded_paths:
            self.loaded_paths.append(cache_key)
        if product_rows and cache_key not in self.loaded_text_paths:
            self.loaded_text_paths.append(cache_key)
        return mapping, product_rows

    def _load(self) -> Dict[str, str]:
        if self._by_barcode is not None:
            return self._by_barcode
        merged: Dict[str, str] = {}
        for path in self.paths:
            mapping, _ = self._load_one(path, include_text_rows=False)
            merged.update(mapping)
        self._by_barcode = merged
        return merged

    def _rows(self) -> list[dict[str, object]]:
        if self._product_rows is None:
            rows: list[dict[str, object]] = []
            if self._by_barcode is None:
                self._by_barcode = {}
            for path in self.paths:
                mapping, product_rows = self._load_one(path, include_text_rows=True)
                self._by_barcode.update(mapping)
                rows.extend(product_rows)
            self._product_rows = rows
        return self._product_rows or []

    def __len__(self) -> int:
        return len(self._load())

    def name_for_barcode(self, barcode: object) -> str:
        code = _digits(barcode)
        if len(code) < 8:
            return ""
        mapping = self._load()
        keys = [code]
        if len(code) > 13:
            keys.extend([code[:13], code[-13:]])
        if len(code) == 14 and code.startswith("0"):
            keys.append(code[1:])
        for key in keys:
            if key in mapping:
                return mapping[key]
        return ""

    def best_text_price_match(self, query: object, prices: Iterable[object] = (), min_score: float | None = None) -> dict[str, object]:
        query_text = _clean_product_name(query)
        if not query_text:
            return {}
        q_tokens = _tokens(query_text)
        if len(q_tokens) < 2:
            return {}
        q_norm = _normalize_product_text(query_text)
        q_prices = [smart_float(p, default=float("nan")) for p in prices]
        q_prices = [float(round(p, 2)) for p in q_prices if not math.isnan(p) and 1.0 <= p <= 99999]
        threshold = float(min_score if min_score is not None else os.environ.get("LENTA_CATALOG_TEXT_MATCH_MIN_SCORE", "0.84"))
        best: dict[str, object] = {}
        second_score = 0.0
        # Keep this bounded for Kaggle runtime; high-quality candidates almost
        # always share at least two content tokens with OCR text.
        for row in self._rows():
            rtokens = row.get("tokens", set())
            if not isinstance(rtokens, set) or not rtokens:
                continue
            exact_inter = q_tokens & rtokens
            fuzzy_inter: set[str] = set(exact_inter)
            for qt in q_tokens - exact_inter:
                if any((len(qt) >= 4 and rt.startswith(qt)) or (len(rt) >= 4 and qt.startswith(rt)) for rt in rtokens):
                    fuzzy_inter.add(qt)
            if len(fuzzy_inter) < 2 and (len(fuzzy_inter) < 1 or len(q_tokens) < 4):
                continue
            union = q_tokens | rtokens
            jaccard = len(fuzzy_inter) / max(1, len(union))
            coverage = len(fuzzy_inter) / max(1, min(len(q_tokens), len(rtokens)))
            token_score = 0.65 * coverage + 0.35 * jaccard
            if token_score < 0.18:
                continue
            seq = text_similarity(q_norm, str(row.get("norm", "")))
            price_score = 0.0
            row_prices = [float(v) for v in row.get("prices", []) or []]
            if q_prices and row_prices:
                best_diff = min(abs(a - b) / max(1.0, b) for a in q_prices for b in row_prices)
                price_score = max(0.0, 1.0 - min(1.0, best_diff / 0.08))
            elif not q_prices:
                price_score = 0.15
            score = 0.58 * token_score + 0.27 * seq + 0.15 * price_score
            if score > float(best.get("score", 0.0) or 0.0):
                second_score = float(best.get("score", 0.0) or 0.0)
                best = {"name": row.get("name", ""), "score": float(score), "source": row.get("source", ""), "token_score": float(token_score), "seq_score": float(seq), "price_score": float(price_score)}
            elif score > second_score:
                second_score = float(score)
        if not best or float(best.get("score", 0.0)) < threshold:
            return {}
        # Avoid ambiguous catalog overwrites when two products score almost equal.
        if second_score and float(best["score"]) - second_score < float(os.environ.get("LENTA_CATALOG_TEXT_MATCH_MIN_MARGIN", "0.025")):
            return {}
        return best


def should_replace_product_name(value: object) -> bool:
    text = str(value or "").strip()
    if not text or text.lower() in {"нет", "nan", "none", "null"}:
        return True
    if len(text) < 5:
        return True
    if _BAD_PRODUCT_RE.search(text):
        return True
    alpha_count = len(_CYR_OR_LATIN_RE.findall(text))
    digit_count = sum(ch.isdigit() for ch in text)
    if alpha_count == 0:
        return True
    if digit_count > alpha_count * 2:
        return True
    return False
