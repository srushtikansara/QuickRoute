"""Smoke test: every dashboard page renders and the main buttons work (uses Streamlit's AppTest)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest   # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_all_pages_render_and_buttons_work():
    app = AppTest.from_file(APP, default_timeout=240)
    app.run()
    assert not app.exception
    pages = app.sidebar.radio[0].options
    for page in pages:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page

    # Shrink the ACO so the test is quick, then run page 3.
    app.sidebar.radio[0].set_value(pages[2]).run()
    app.session_state["n_ants"] = 8
    app.session_state["n_iterations"] = 10
    app.selectbox[0].set_value("Hybrid ACO + Fuzzy").run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.session_state["last_run"] is not None

    # Page 5 comparison and page 6 export.
    app.sidebar.radio[0].set_value(pages[4]).run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.session_state["comparison"] is not None
    app.sidebar.radio[0].set_value(pages[5]).run()
    assert not app.exception and not app.error


def test_dataset_page_generate_button():
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    app.sidebar.radio[0].set_value("2. Dataset Management").run()
    generate = [b for b in app.button if b.label == "Generate dataset"][0]
    generate.click().run()
    assert not app.exception
    assert "Synthetic" in app.session_state["data_source"]
