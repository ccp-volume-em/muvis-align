import logging

import pytest

from muvis_align.ui._utils import VisibleActivityDock, catch_run_errors, clear_napari_dask_cache, \
    patch_multiscale_label_show, patch_shapes_text_coords


def test_catch_run_errors_returns_result_on_success():
    class Dummy:
        @catch_run_errors
        def run_thing(self):
            return "ok"

    assert Dummy().run_thing() == "ok"


def test_catch_run_errors_shows_a_dialog_and_logs_on_failure(monkeypatch, caplog, qapp):
    """A failing run_*() method must not propagate - it shows a dialog naming the step (napari's corner
    notification fades unseen), logs the full traceback, and returns None so the caller (e.g. a *_process()
    handler) can bail out instead of showing a bogus 'completed' dialog."""
    shown = []
    monkeypatch.setattr("muvis_align.ui._utils.QMessageBox.critical",
                        lambda parent, title, message: shown.append(message))

    class Dummy:
        @catch_run_errors
        def run_pair_registration(self):
            raise ValueError("boom")

    with caplog.at_level(logging.ERROR):
        result = Dummy().run_pair_registration()

    assert result is None
    assert len(shown) == 1
    assert shown[0].startswith("Pair registration failed:\nboom")
    assert any("run_pair_registration failed" in record.message for record in caplog.records)


def test_running_out_of_memory_says_what_may_help():
    from muvis_align.ui._utils import failure_message

    assert "Out of memory" in failure_message("Pair registration", MemoryError("Unable to allocate 6.75 MiB"))
    assert "Out of memory" not in failure_message("Pair registration", ValueError("boom"))


def test_a_failure_off_the_qt_thread_falls_back_to_a_notification(monkeypatch, qapp):
    import threading
    from muvis_align.ui._utils import report_failure

    dialogs, notifications = [], []
    monkeypatch.setattr("muvis_align.ui._utils.QMessageBox.critical", lambda *args: dialogs.append(args))
    monkeypatch.setattr("muvis_align.ui._utils.show_error", notifications.append)

    worker = threading.Thread(target=report_failure, args=("Fusion", ValueError("boom")))
    worker.start()
    worker.join()

    assert not dialogs
    assert notifications[0].startswith("Fusion failed:")


def test_patched_shapes_text_coords_match_napari():
    import numpy as np
    from napari.components import ViewerModel
    from napari.layers import Shapes

    patch_shapes_text_coords()
    patch_shapes_text_coords()
    patched = Shapes._view_text_coords
    assert patched.fget._muvis_patched
    original = patched.fget._muvis_original
    assert not getattr(original.fget, '_muvis_patched', False)

    base = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=float)
    shapes = [np.column_stack([np.full(4, index % 3), base + index * 5]) for index in range(30)]
    labels = [str(index) for index in range(30)]
    viewer = ViewerModel()
    layer = viewer.add_shapes(shapes, shape_type='polygon', text={'string': '{labels}'}, features={'labels': labels})
    viewer.dims.set_current_step(0, 1)
    expected, actual = original.fget(layer), layer._view_text_coords
    indices = layer._view_indices if hasattr(layer, '_view_indices') else layer._indices_view
    assert len(indices) == 10
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]


def test_activity_dock_toggle_raises_no_private_access_warning():
    import warnings
    from types import SimpleNamespace
    from napari.utils._proxies import PublicOnlyProxy

    toggles = []
    qt_viewer = SimpleNamespace(show_welcome_screen=True)
    status_bar = PublicOnlyProxy(SimpleNamespace(_toggle_activity_dock=toggles.append))
    window = PublicOnlyProxy(SimpleNamespace(_qt_viewer=PublicOnlyProxy(qt_viewer), _status_bar=status_bar))
    viewer = SimpleNamespace(window=window)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.warns(FutureWarning):
            window._status_bar
        with VisibleActivityDock(viewer):
            assert not qt_viewer.show_welcome_screen
    assert qt_viewer.show_welcome_screen
    assert toggles == [True, False]


def test_the_multiscale_label_shows_only_once_the_layer_controls_hold_it(qtbot):
    from qtpy.QtWidgets import QVBoxLayout, QWidget
    from napari._qt.layer_controls.dynamic.widgets import qt_multiscale_level_control as control_module

    patch_multiscale_label_show()
    label_class = control_module.QtWrappedLabel
    shown, hidden = label_class('resolution:'), label_class('resolution:')

    shown.show()
    hidden.hide()
    assert not shown.isVisible()

    parent = QWidget()
    qtbot.addWidget(parent)
    layout = QVBoxLayout(parent)
    layout.addWidget(shown)
    layout.addWidget(hidden)
    parent.show()
    qtbot.waitExposed(parent)

    assert shown.isVisible() and shown.parent() is parent
    assert not hidden.isVisible()


def test_clearing_napari_dask_cache_drops_the_chunks_it_kept():
    import dask.array as da
    import numpy as np
    from napari.utils import _dask_utils

    cache = _dask_utils.resize_dask_cache(64 * 2**20).cache
    data = da.from_array(np.arange(4 * 256 * 256, dtype=np.float64).reshape(4, 256, 256), chunks=(1, 256, 256))
    with _dask_utils.configure_dask(data)():
        (data[1] * 2).compute()
    assert cache.total_bytes > 0

    clear_napari_dask_cache()

    assert cache.total_bytes == 0
