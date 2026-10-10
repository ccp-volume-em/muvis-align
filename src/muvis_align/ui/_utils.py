# Based on https://github.com/multiview-stitcher/napari-stitcher/blob/main/src/napari_stitcher/_stitcher_widget.py

import contextlib
import functools
import logging
import warnings

from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import QCoreApplication, QThread
from qtpy.QtWidgets import QMessageBox

from muvis_align.util import OperationCancelled


def operation_label(func_name):
    """'run_pair_registration' -> 'Pair registration'."""
    words = func_name.removeprefix('run_').replace('_', ' ')
    return words[:1].upper() + words[1:]


def log_file():
    return next((handler.baseFilename for handler in logging.getLogger().handlers
                 if isinstance(handler, logging.FileHandler)), None)


def failure_message(label, error):
    message = f'{label} failed:\n{error}'
    if isinstance(error, MemoryError):
        message += '\n\nOut of memory: fewer registration threads, or closing other programs, may help.'
    if log_file():
        message += f'\n\nDetails in {log_file()}'
    return message


def report_failure(label, error):
    # a dialog, as a success gets: napari's corner notification fades within seconds, unseen behind the activity dock
    message = failure_message(label, error)
    application = QCoreApplication.instance()
    if application is not None and QThread.currentThread() is application.thread():
        QMessageBox.critical(None, 'muvis-align', message)
    else:
        show_error(message)


def catch_run_errors(func):
    """Wrap a run_*() method so a failure shows a dialog and logs the full traceback to the main log
    file, instead of surfacing as an opaque signal-emission error - and returns None instead of
    propagating, so callers can bail out (e.g. skip a 'completed' dialog) by checking the return value.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except OperationCancelled:
            logging.info(f'{func.__name__} cancelled')
            show_info('Cancelled')
            return None
        except Exception as e:
            logging.exception(f'{func.__name__} failed')
            report_failure(operation_label(func.__name__), e)
            return None
    return wrapper


class TemporarilyDisabledWidgets(object):
    """
    Context manager to temporarily disable widgets during long computation
    """
    def __init__(self, enable_plugin_widget=None):
        self.enable_plugin_widget = enable_plugin_widget

    def __enter__(self):
        if self.enable_plugin_widget:
            self.enable_plugin_widget(False)

    def __exit__(self, type, value, traceback):
        if self.enable_plugin_widget:
            self.enable_plugin_widget()

    def disable(self, widgets):
        self.widgets = widgets
        self.enabled_states = {name: widget.enabled for name, widget in widgets.items()}
        for widget in self.widgets.values():
            widget.enabled = False

    def restore(self):
        for name, widget in self.widgets.items():
            widget.enabled = self.enabled_states.get(name, True)


class VisibleActivityDock(object):
    """
    Context manager to temporarily show the activity dock during long computation.

    napari's welcome screen, shown whenever the viewer has no layers, is drawn over the dock -
    the bar disappeared for the whole first refresh of a project. It is kept off meanwhile.
    """
    def __init__(self, viewer):
        self.viewer = viewer
        self._welcome_shown = None

    def __enter__(self):
        with _private_napari_access():
            qt_viewer = getattr(self.viewer.window, '_qt_viewer', None)
            if qt_viewer is not None and hasattr(qt_viewer, 'show_welcome_screen'):
                self._welcome_shown = qt_viewer.show_welcome_screen
                qt_viewer.show_welcome_screen = False
            self.viewer.window._status_bar._toggle_activity_dock(True)

    def __exit__(self, type, value, traceback):
        with _private_napari_access():
            self.viewer.window._status_bar._toggle_activity_dock(False)
            if self._welcome_shown is not None:
                self.viewer.window._qt_viewer.show_welcome_screen = self._welcome_shown
                self._welcome_shown = None


@contextlib.contextmanager
def _private_napari_access():
    # napari shows each private-access FutureWarning as a notification: under xpra a window of its own
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Private attribute access', category=FutureWarning)
        yield


def flush_paint_events():
    """Let Qt repaint what was just changed, without delivering pending user input.

    Excluding input is what makes this safe to call from inside an event handler: a plain
    processEvents() also delivers queued mouse events, and a ButtonRelease consumed by that
    nested pass leaves X's implicit pointer grab stuck - the cursor keeps whatever shape it
    had (an I-beam, over a text field) and later clicks never reach the widget under it.
    """
    try:
        from qtpy.QtCore import QEventLoop
        from qtpy.QtWidgets import QApplication
    except ImportError:  # pragma: no cover - Qt is always present in the napari plugin
        return
    app = QApplication.instance()
    if app is not None:
        app.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)


def patch_shapes_text_coords():
    """Make napari's Shapes label positions read the layer's shape list once, not once per shape
    in view - quadratic otherwise: 12.6s of a 22.8s add_shapes at 150k shapes (napari 0.9.0)."""
    from napari.layers import Shapes

    original = getattr(Shapes, '_view_text_coords', None)
    if not isinstance(original, property) or getattr(original.fget, '_muvis_patched', False):
        return

    def view_text_coords(self):
        data = self._data_view.data
        displayed = self._slice_input.displayed
        # napari 0.8 names it _indices_view, 0.9 _view_indices
        indices = self._view_indices if hasattr(self, '_view_indices') else self._indices_view
        coords = [data[index][:, displayed] for index in indices]
        return self.text.compute_text_coords(coords, self._slice_input.ndisplay, self._slice_input.order)

    view_text_coords._muvis_patched = True
    view_text_coords._muvis_original = original
    Shapes._view_text_coords = property(view_text_coords)


def patch_multiscale_label_show():
    """Keep napari's multiscale 'resolution:' label from showing before it has a parent: shown parentless, it is a
    window of its own (under xpra a tiny one flashing up) until the layer controls take it."""
    import importlib

    # napari 0.9.2 has the control in both places, 0.9.0 only in dynamic
    for module_name in ('napari._qt.layer_controls.dynamic.widgets.qt_multiscale_level_control',
                        'napari._qt.layer_controls.widgets.qt_multiscale_level_control'):
        try:
            control_module = importlib.import_module(module_name)
        except ImportError:
            control_module = None
        label_class = getattr(control_module, 'QtWrappedLabel', None)
        if label_class is not None and not getattr(label_class, '_muvis_patched', False):
            control_module.QtWrappedLabel = _parented_label_class(label_class)


def _parented_label_class(label_class):
    class ParentedLabel(label_class):
        _muvis_patched = True

        def setVisible(self, visible):
            # a layout shows a not explicitly hidden child itself once it adds it
            if not (visible and self.parent() is None):
                super().setVisible(visible)

    return ParentedLabel


def clear_napari_dask_cache():
    """Empty napari's dask cache: it keeps removed layers' computed chunks, up to a quarter of RAM (~500GB on a 2TB
    HPC node), and the lazy overview has its own plane cache."""
    try:
        from napari.utils._dask_utils import _DASK_CACHE
    except ImportError:
        return
    cache = getattr(_DASK_CACHE, 'cache', None)
    if cache is not None and getattr(cache, 'total_bytes', 0):
        cache.clear()
