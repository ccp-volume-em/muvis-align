import logging

import pytest

from muvis_align.ui._utils import catch_run_errors


def test_catch_run_errors_returns_result_on_success():
    class Dummy:
        @catch_run_errors
        def run_thing(self):
            return "ok"

    assert Dummy().run_thing() == "ok"


def test_catch_run_errors_shows_popup_and_logs_on_failure(monkeypatch, caplog):
    """A failing run_*() method must not propagate - it shows a napari popup, logs the full
    traceback to the main log file, and returns None so the caller (e.g. a *_process() handler)
    can bail out instead of showing a bogus 'completed' dialog."""
    shown = []
    monkeypatch.setattr(
        "muvis_align.ui._utils.show_error", lambda message: shown.append(message)
    )

    class Dummy:
        @catch_run_errors
        def run_thing(self):
            raise ValueError("boom")

    with caplog.at_level(logging.ERROR):
        result = Dummy().run_thing()

    assert result is None
    assert len(shown) == 1
    assert "run_thing failed" in shown[0]
    assert "boom" in shown[0]
    assert any("run_thing failed" in record.message for record in caplog.records)
