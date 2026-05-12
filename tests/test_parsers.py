from lenta_shelf_ai.qr import parse_qr_payload
from lenta_shelf_ai.parsers import ean13_is_valid, parse_text_fields
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
