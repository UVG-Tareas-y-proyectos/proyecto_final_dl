import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_guides_when_checkpoint_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["app.py", "--checkpoint", str(tmp_path / "missing.pt")])
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert "Primero entrena" in app.warning[0].value


def test_app_loads_pilot_checkpoint(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.setattr(sys, "argv", ["app.py", "--checkpoint", str(root / "artifacts/pilot_cnn.pt")])
    app = AppTest.from_file(str(root / "app.py")).run(timeout=20)
    assert not app.exception
    assert len(app.warning) == 0
    assert any("3 clases" in caption.value for caption in app.caption)

