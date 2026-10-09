import pytest
import zarr

import muvis_align.constants  # noqa: F401  sets its own zarr pool size on import, which this overrides

# zarr's threaded zstd decode crashed natively on the Windows py3.14 runner; tests don't need its throughput
zarr.config.set({'threading.max_workers': 1})


@pytest.fixture(autouse=True)
def no_blocking_failure_dialog(monkeypatch):
    """A failing run_*() shows a modal dialog on the Qt thread: in a test, nothing would ever close it."""
    monkeypatch.setattr('muvis_align.ui._utils.QMessageBox.critical', lambda *args, **kwargs: None)
