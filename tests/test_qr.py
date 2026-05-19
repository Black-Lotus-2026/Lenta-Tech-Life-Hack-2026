from __future__ import annotations

import cv2
import numpy as np

from lenta_shelf_ai.qr import _qr_candidate_regions, _qr_image_variants, decode_qr_payloads, parse_qr_payload


def test_qr_variants_include_upscale_and_thresholds() -> None:
    image = np.full((80, 100, 3), 220, dtype=np.uint8)
    image[20:60, 30:70] = 30

    variants = _qr_image_variants(image)

    assert len(variants) >= 8
    assert variants[0].shape == image.shape
    assert any(v.shape[0] > image.shape[0] and v.shape[1] > image.shape[1] for v in variants)
    assert all(v.ndim == 3 and v.shape[2] == 3 for v in variants)
    assert any(v.shape[:2] == (image.shape[1], image.shape[0]) for v in variants)


def test_qr_candidate_regions_include_layout_priors() -> None:
    image = np.full((120, 240, 3), 240, dtype=np.uint8)
    image[40:95, 155:210] = 255
    for offset in (0, 18, 36):
        cv2.rectangle(image, (160 + offset, 45), (170 + offset, 55), (0, 0, 0), -1)
        cv2.rectangle(image, (160 + offset, 65), (170 + offset, 75), (0, 0, 0), -1)

    regions = _qr_candidate_regions(image)

    assert regions
    assert regions[0].shape == image.shape
    assert any(region.shape[0] < image.shape[0] and region.shape[1] < image.shape[1] for region in regions)


def test_parse_qr_payload_accepts_raw_ean13() -> None:
    parsed = parse_qr_payload("4670025474665")

    assert parsed["qr_code_barcode"] == "4670025474665"


def test_parse_qr_payload_accepts_gs1_gtin() -> None:
    parsed = parse_qr_payload("(01)04670025474665")

    assert parsed["qr_code_barcode"] == "4670025474665"


def test_qr_candidate_regions_not_truncated_to_three_regions():
    image = np.full((180, 320, 3), 245, dtype=np.uint8)
    # Dense square-like QR surrogate in a late-prior/contour region.
    for y in range(115, 160, 12):
        for x in range(245, 295, 12):
            if (x + y) % 24 == 0:
                cv2.rectangle(image, (x, y), (x + 7, y + 7), (0, 0, 0), -1)

    regions = _qr_candidate_regions(image)

    assert len(regions) > 3


def test_qr_decoder_respects_safety_env(monkeypatch):
    monkeypatch.setenv("LENTA_QR_MAX_REGIONS", "1")
    monkeypatch.setenv("LENTA_QR_MAX_VARIANTS", "1")
    monkeypatch.setenv("LENTA_QR_ENABLE_ZXING", "0")
    monkeypatch.setenv("LENTA_QR_ENABLE_PYZBAR", "0")
    monkeypatch.setenv("LENTA_QR_ENABLE_OPENCV", "0")
    image = np.full((80, 80, 3), 255, dtype=np.uint8)

    assert decode_qr_payloads(image) == []


def test_qr_decoder_can_use_crash_isolated_zxing_subprocess(monkeypatch):
    qrcode = __import__("pytest").importorskip("qrcode")
    __import__("pytest").importorskip("zxingcpp")
    from PIL import Image
    from lenta_shelf_ai.qr import decode_qr_payloads_with_debug

    payload = "barcode=4670025474665&price1=129.99&price4=99.99"
    qr = qrcode.QRCode(border=1, box_size=6)
    qr.add_data(payload)
    qr.make(fit=True)
    pil = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    monkeypatch.setenv("LENTA_QR_ENABLE_OPENCV", "0")
    monkeypatch.setenv("LENTA_QR_ENABLE_ZXING", "1")
    monkeypatch.setenv("LENTA_QR_ENABLE_PYZBAR", "0")
    monkeypatch.setenv("LENTA_QR_NATIVE_SUBPROCESS", "1")
    monkeypatch.setenv("LENTA_QR_NATIVE_MAX_VARIANTS", "4")
    monkeypatch.setenv("LENTA_QR_MAX_REGIONS", "1")
    monkeypatch.setenv("LENTA_QR_MAX_VARIANTS", "4")

    decoded, stats = decode_qr_payloads_with_debug(image)

    assert payload in decoded
    assert stats.get("native_processpool") is True or stats.get("native_subprocess") is True
    assert stats.get("native_worker_stats", {}).get("payloads", 0) >= 1 or stats.get("native_returncode") == 0


def test_qr_quiet_zone_variant_decodes_tight_generated_qr(monkeypatch):
    qrcode = __import__("pytest").importorskip("qrcode")
    from lenta_shelf_ai.qr import decode_qr_payloads

    payload = "4670025474665"
    qr = qrcode.QRCode(border=0, box_size=5)
    qr.add_data(payload)
    qr.make(fit=True)
    pil = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    monkeypatch.setenv("LENTA_QR_ENABLE_ZXING", "0")
    monkeypatch.setenv("LENTA_QR_ENABLE_PYZBAR", "0")
    monkeypatch.setenv("LENTA_QR_ENABLE_OPENCV", "1")
    monkeypatch.setenv("LENTA_QR_MAX_REGIONS", "1")
    monkeypatch.setenv("LENTA_QR_MAX_VARIANTS", "12")

    assert payload in decode_qr_payloads(image)


def test_native_priority_prefers_barcode_like_lower_strip() -> None:
    from lenta_shelf_ai.qr import _native_priority_variants, _qr_candidate_regions

    image = np.full((160, 300, 3), 245, dtype=np.uint8)
    # Synthetic 1D barcode-like texture in lower strip.
    for x in range(45, 255, 8):
        cv2.rectangle(image, (x, 112), (x + 3, 150), (0, 0, 0), -1)

    regions = _qr_candidate_regions(image, max_regions=6)
    native = _native_priority_variants(regions, [])

    assert native
    h, w = native[0].shape[:2]
    assert w / max(1, h) >= 1.4


def test_barcode_texture_allows_native_without_opencv_points(monkeypatch) -> None:
    from lenta_shelf_ai import qr as qr_module

    image = np.full((160, 300, 3), 245, dtype=np.uint8)
    for x in range(45, 255, 8):
        cv2.rectangle(image, (x, 112), (x + 3, 150), (0, 0, 0), -1)

    monkeypatch.setenv("LENTA_QR_NATIVE_ALWAYS", "0")
    monkeypatch.setenv("LENTA_QR_ENABLE_OPENCV", "1")
    monkeypatch.setenv("LENTA_QR_MAX_REGIONS", "6")
    monkeypatch.setenv("LENTA_QR_MAX_VARIANTS", "8")
    monkeypatch.setenv("LENTA_QR_NATIVE_MAX_VARIANTS", "2")

    def fake_opencv(_variants, stats):
        stats["opencv_attempts"] = 1
        return []

    def fake_native(_variants, stats):
        stats["fake_native_called"] = True
        return ["4670025474665"]

    monkeypatch.setattr(qr_module, "_decode_opencv_variants", fake_opencv)
    monkeypatch.setattr(qr_module, "_decode_native_variants_subprocess", fake_native)
    monkeypatch.setattr(qr_module, "_decode_native_variants_inprocess", lambda _variants, _stats: [])

    payloads, stats = qr_module.decode_qr_payloads_with_debug(image)

    assert payloads == ["4670025474665"]
    assert stats["native_allowed"] is True
    assert stats["native_allowed_reason"] == "machine_code_texture"
    assert stats["fake_native_called"] is True


def test_barcode_preprocess_adds_wide_strip_variants() -> None:
    from lenta_shelf_ai.qr import _barcode_image_variants

    image = np.full((42, 180, 3), 255, dtype=np.uint8)
    for x in range(12, 168, 7):
        width = 2 if (x // 7) % 3 else 4
        cv2.rectangle(image, (x, 7), (x + width, 35), (0, 0, 0), -1)

    variants = _barcode_image_variants(image)

    assert len(variants) >= 5
    assert max(v.shape[0] for v in variants) > image.shape[0]
    assert max(v.shape[1] for v in variants) > image.shape[1]


def test_qr_variant_dedupe_keeps_same_size_different_content() -> None:
    from lenta_shelf_ai.qr import _variant_signature

    a = np.full((32, 64, 3), 255, dtype=np.uint8)
    b = a.copy()
    b[:, 8:16] = 0
    b[0, 0] = a[0, 0]

    assert _variant_signature(a) != _variant_signature(b)


def test_parse_qr_payload_rejects_invalid_short_numeric_noise() -> None:
    from lenta_shelf_ai.qr import _has_structured_payload

    assert parse_qr_payload("11111108") == {}
    assert _has_structured_payload(["11111108"]) is False


def test_native_backend_defaults_to_file_subprocess(monkeypatch) -> None:
    from lenta_shelf_ai import qr as qr_module

    monkeypatch.delenv("LENTA_QR_NATIVE_BACKEND", raising=False)
    monkeypatch.setenv("LENTA_QR_NATIVE_SUBPROCESS", "1")

    def fail_processpool(_variants, _stats):
        raise AssertionError("processpool must not be the default native backend")

    monkeypatch.setattr(qr_module, "_decode_native_variants_processpool", fail_processpool)
    stats = {}
    assert qr_module._decode_native_variants_subprocess([], stats) == []


def test_qr_decoder_uses_wechat_fallback(monkeypatch):
    import types
    from lenta_shelf_ai import qr as qr_module

    class FakeWeChatQRCode:
        def __init__(self, *args):
            self.args = args

        def detectAndDecode(self, image):
            return (["barcode=4670025474665"], None)

    monkeypatch.setenv("LENTA_QR_ENABLE_OPENCV", "0")
    monkeypatch.setenv("LENTA_QR_ENABLE_OPENCV_BARCODE", "0")
    monkeypatch.setenv("LENTA_QR_ENABLE_WECHAT", "1")
    monkeypatch.setenv("LENTA_QR_ENABLE_ZXING", "0")
    monkeypatch.setenv("LENTA_QR_ENABLE_PYZBAR", "0")
    monkeypatch.setenv("LENTA_QR_MAX_REGIONS", "1")
    monkeypatch.setenv("LENTA_QR_MAX_VARIANTS", "2")
    monkeypatch.setattr(qr_module.cv2, "wechat_qrcode", types.SimpleNamespace(WeChatQRCode=FakeWeChatQRCode), raising=False)

    payloads, stats = qr_module.decode_qr_payloads_with_debug(np.full((64, 64, 3), 255, dtype=np.uint8))

    assert payloads == ["barcode=4670025474665"]
    assert stats["wechat_available"] is True
    assert stats["wechat_attempts"] == 1


def test_parse_qr_payload_accepts_ean_prices_array() -> None:
    parsed = parse_qr_payload('{"ean":"4670025474665","prices":["129.99","99,49"]}')

    assert parsed["qr_code_barcode"] == "4670025474665"
    assert parsed["price1_qr"] == "129.99"
    assert parsed["price2_qr"] == "99.49"


def test_parse_qr_payload_accepts_delimited_checksum_payload() -> None:
    parsed = parse_qr_payload("4670025474665|129.99|99.49")

    assert parsed["qr_code_barcode"] == "4670025474665"
    assert parsed["price1_qr"] == "129.99"
    assert parsed["price2_qr"] == "99.49"


def test_parse_qr_payload_rejects_delimited_invalid_numeric_noise() -> None:
    parsed = parse_qr_payload("11111108|129.99")

    assert "qr_code_barcode" not in parsed
    assert parsed["price1_qr"] == "129.99"


def test_opencv_qr_warp_fallback_runs_when_points_detected(monkeypatch):
    from lenta_shelf_ai import qr as qr_module

    payload = "4670025474665"

    class FakeDetector:
        def setEpsX(self, value):
            pass

        def setEpsY(self, value):
            pass

        def detectAndDecodeMulti(self, image):
            return False, [], None, None

        def detectMulti(self, image):
            pts = np.array([[[10, 10], [50, 10], [50, 50], [10, 50]]], dtype=np.float32)
            return True, pts

        def decode(self, image, points):
            return "", None

        def detectAndDecode(self, image):
            return payload, None, None

    monkeypatch.setattr(qr_module.cv2, "QRCodeDetector", FakeDetector)
    monkeypatch.setenv("LENTA_QR_ENABLE_ZXING", "0")
    monkeypatch.setenv("LENTA_QR_ENABLE_PYZBAR", "0")
    monkeypatch.setenv("LENTA_QR_ENABLE_WECHAT", "0")
    monkeypatch.setenv("LENTA_QR_ENABLE_OPENCV_BARCODE", "0")
    monkeypatch.setenv("LENTA_QR_MAX_REGIONS", "1")
    monkeypatch.setenv("LENTA_QR_MAX_VARIANTS", "2")
    monkeypatch.setenv("LENTA_QR_WARP_MAX_VARIANTS", "2")

    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    decoded, stats = qr_module.decode_qr_payloads_with_debug(image)

    assert payload in decoded
    assert stats.get("opencv_warp_attempts", 0) >= 1


def test_qr_glare_suppression_variant_preserves_shape_and_changes_glare() -> None:
    from lenta_shelf_ai import qr as qr_module

    image = np.full((96, 96, 3), 230, dtype=np.uint8)
    for y in range(8, 88, 12):
        cv2.line(image, (8, y), (88, y), (0, 0, 0), 2)
    cv2.circle(image, (48, 48), 18, (255, 255, 255), -1)

    fixed = qr_module._suppress_specular_glare(image)

    assert fixed.shape == image.shape
    assert np.mean(np.abs(fixed.astype(np.int16) - image.astype(np.int16))) > 0.1
