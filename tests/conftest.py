import pytest


@pytest.fixture(autouse=True)
def no_blocking_failure_dialog(monkeypatch):
    """A failing run_*() shows a modal dialog on the Qt thread: in a test, nothing would ever close it."""
    monkeypatch.setattr('muvis_align.ui._utils.QMessageBox.critical', lambda *args, **kwargs: None)
