from lenta_shelf_ai.qr import parse_qr_payload
from lenta_shelf_ai.parsers import ean13_is_valid, merge_field_values, parse_text_fields
from lenta_shelf_ai.schema import OCRLine


def test_qr_query_aliases():
    out = parse_qr_payload("b=4670025474665&p1=252.63&p4=129.99&aC=ABC")
    assert out["qr_code_barcode"] == "4670025474665"
    assert out["price1_qr"] == "252.63"
    assert out["price4_qr"] == "129.99"
    assert out["action_code_qr"] == "ABC"


def test_ean13():
    assert ean13_is_valid("4670025474665")


def test_text_parser_prices_discount():
    lines = [OCRLine("Напиток SANTO STEFANO Rosso"), OCRLine("252,63"), OCRLine("129,99"), OCRLine("-48%"), OCRLine("03.04.2026 3:08")]
    out = parse_text_fields(lines, {})
    assert out["price_default"] == "252.63"
    assert out["price_card"] == "129.99"
    assert out["discount_amount"] == "-48%"


def test_text_parser_does_not_treat_volume_as_price():
    lines = [
        OCRLine("Вино красное сухое Франция 0.75L"),
        OCRLine("Цена 599,99 руб"),
    ]

    out = parse_text_fields(lines, {})

    assert out["price_default"] == "599.99"
    assert out.get("price_card", "") != "0.75"


def test_text_parser_rejects_invalid_ocr_barcode_noise():
    lines = [OCRLine("270207716054"), OCRLine("Товар тестовый")]

    out = parse_text_fields(lines, {})

    assert "barcode" not in out


def test_parser_rejects_non_cyrillic_product_garbage():
    lines = [OCRLine("NI KE afi | i\\ 3)/ fel | ee All")]

    out = parse_text_fields(lines, {})

    assert "product_name" not in out


def test_merge_canonicalizes_greek_absent_noise():
    assert merge_field_values(["\u03bd\u03b5\u03c2", "нет"]) == "нет"
