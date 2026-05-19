from lenta_shelf_ai.parsers import parse_text_fields
from lenta_shelf_ai.schema import OCRLine


def _parse(text: str):
    return parse_text_fields([OCRLine(text=text, confidence=0.9, engine="test")], {})


def test_extended_code_patterns_from_external_solutions():
    assert _parse("код 13_043015")["code"] == "13_043015"
    assert _parse("код 026005 - 026007")["code"] == "026005-026007"
    assert _parse("код 024 017_1_6_2")["code"] == "024_017_1_6_2"
    assert _parse("код 21_ЦПУ")["code"] == "21_ЦПУ"


def test_code_parser_rejects_short_ocr_alpha_garbage():
    for text in ["469AK", "133SE", "99VA", "99LUBC", "214HONG", "код 469AK"]:
        assert _parse(text)["code"] == "нет"


def test_price_ocr_digit_confusions_are_recovered():
    fields = _parse("Цена 1O9 99 руб")
    assert fields["price_default"] == "109.99"


def test_geometric_rubles_kopecks_pair_is_recovered():
    lines = [
        OCRLine(text="129", confidence=0.95, box=[[0, 0], [75, 0], [75, 55], [0, 55]], engine="test|zone:price_default"),
        OCRLine(text="99", confidence=0.95, box=[[82, 7], [112, 7], [112, 34], [82, 34]], engine="test|zone:price_default"),
    ]
    fields = parse_text_fields(lines, {})
    assert fields["price_default"] == "129.99"


def test_discount_glued_to_integer_price_is_split_not_compact_noise():
    fields = _parse("Скидка 28%1199 руб")
    assert fields["discount_amount"] == "-28%"
    assert fields["price_default"] == "1199.00"
    assert fields.get("price_card", "") != "28.11"


def test_discount_glued_to_compact_price_with_cents_is_split():
    fields = _parse("-28%119999")
    assert fields["discount_amount"] == "-28%"
    assert fields["price_default"] == "1199.99"
