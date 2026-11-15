"""Smoke test for the Streamlit app: the script must render without raising."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / ".env").exists(), reason="the app needs .env to build settings"
)


def test_app_renders_the_upload_screen():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.run()

    assert not app.exception
    assert app.title[0].value == "📄 PDF Reader"
    assert len(app.file_uploader) == 1
    assert any("Upload a PDF" in info.value for info in app.info)
    assert len(app.sidebar.selectbox) >= 2
