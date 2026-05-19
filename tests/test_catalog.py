from __future__ import annotations

from pathlib import Path

from lenta_shelf_ai.catalog import ProductCatalog, should_replace_product_name
from lenta_shelf_ai.pipeline import PriceTagPipeline


def test_product_catalog_loads_barcode_to_name(tmp_path: Path) -> None:
    path = tmp_path / "products.csv"
    path.write_text("barcode;fullname\n4670025474665;Молоко тестовое 3.2%\n", encoding="utf-8")

    catalog = ProductCatalog([path])

    assert catalog.name_for_barcode("4670025474665") == "Молоко тестовое 3.2%"
    assert catalog.name_for_barcode("04670025474665") == "Молоко тестовое 3.2%"
    assert len(catalog) == 1


def test_should_replace_product_name_rejects_service_noise() -> None:
    assert should_replace_product_name("нет") is True
    assert should_replace_product_name("11111108") is True
    assert should_replace_product_name("товар закончился") is True
    assert should_replace_product_name("Напиток апельсиновый 1 л") is False


def test_pipeline_catalog_enrichment_does_not_change_schema(tmp_path: Path) -> None:
    path = tmp_path / "products.csv"
    path.write_text("ean,name\n4670025474665,Сыр тестовый 200 г\n", encoding="utf-8")
    pipe = PriceTagPipeline.__new__(PriceTagPipeline)
    pipe.product_catalog = ProductCatalog([path])

    row = {"qr_code_barcode": "4670025474665", "barcode": "", "product_name": "нет"}
    pipe._enrich_row_from_catalog(row)

    assert row["product_name"] == "Сыр тестовый 200 г"
    assert row["barcode"] == "4670025474665"


def test_product_catalog_text_price_match_for_local_goods_style(tmp_path: Path) -> None:
    path = tmp_path / "goods.csv"
    path.write_text(
        "name,price,price_regular\n"
        "Вино игристое Santo Stefano Rosso 0.75 л,1199.99,1684.20\n"
        "Молоко другое 3.2%,99.99,129.99\n",
        encoding="utf-8",
    )
    catalog = ProductCatalog([path])

    match = catalog.best_text_price_match("Вино игристое Santo Stefano Rosso", prices=["1199.99"])

    assert match["name"] == "Вино игристое Santo Stefano Rosso 0.75 л"
    assert float(match["score"]) >= 0.84


def test_pipeline_catalog_text_match_is_explicit_opt_in(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "goods.csv"
    path.write_text("name,price\nКефир тестовый 1 процент,88.50\n", encoding="utf-8")
    pipe = PriceTagPipeline.__new__(PriceTagPipeline)
    pipe.product_catalog = ProductCatalog([path])

    row = {"product_name": "кефир тестовый", "price_card": "88.50", "barcode": "", "qr_code_barcode": ""}
    pipe._enrich_row_from_catalog(row)
    assert row["product_name"] == "кефир тестовый"

    monkeypatch.setenv("LENTA_CATALOG_TEXT_MATCH", "1")
    pipe._enrich_row_from_catalog(row)
    assert row["product_name"] == "Кефир тестовый 1 процент"
    assert row.get("_catalog_match_score", 0) >= 0.84
