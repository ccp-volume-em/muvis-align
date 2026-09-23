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


def test_activity_dock_keeps_welcome_screen_off_while_open():
    """napari's welcome screen (an empty viewer) is drawn over the activity dock, hiding the bar
    for a project's whole first refresh - it is off while the dock is up, then back as it was."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from muvis_align.ui._utils import VisibleActivityDock

    qt_viewer = SimpleNamespace(show_welcome_screen=True)
    status_bar = MagicMock()
    viewer = SimpleNamespace(window=SimpleNamespace(_qt_viewer=qt_viewer, _status_bar=status_bar))

    with VisibleActivityDock(viewer):
        assert qt_viewer.show_welcome_screen is False
        status_bar._toggle_activity_dock.assert_called_once_with(True)

    assert qt_viewer.show_welcome_screen is True
    status_bar._toggle_activity_dock.assert_called_with(False)
