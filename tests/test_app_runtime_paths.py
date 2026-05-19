from __future__ import annotations

import importlib
import sys


def test_vercel_runtime_uses_tmp_for_writable_dirs(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("LENTA_RUNTIME_DIR", raising=False)
    sys.modules.pop("app", None)

    app_module = importlib.import_module("app")

    assert str(app_module.UPLOAD_DIR).replace("\\", "/").startswith("/tmp/")
    assert str(app_module.OUTPUT_DIR).replace("\\", "/").startswith("/tmp/")
