import os
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")

def test_app_renders_and_runs_query(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    monkeypatch.setenv("PG_AUDIT_PATH", str(tmp_path / "a.jsonl"))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.button(key="run").click().run()
    assert not at.exception
    assert any("Guarded" in h.value for h in at.subheader)
