# Based on https://github.com/multiview-stitcher/napari-stitcher/blob/main/src/napari_stitcher/_stitcher_widget.py

import functools
import logging

from napari.utils.notifications import show_error


def catch_run_errors(func):
    """Wrap a run_*() method so a failure shows a napari popup and logs the full traceback to
    the main log file, instead of surfacing as an opaque signal-emission error - and returns
    None instead of propagating, so callers can bail out (e.g. skip a 'completed' dialog) by
    checking the return value.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logging.exception(f'{func.__name__} failed')
            show_error(f'{func.__name__} failed: {e}')
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
        qt_viewer = getattr(self.viewer.window, '_qt_viewer', None)
        if qt_viewer is not None and hasattr(qt_viewer, 'show_welcome_screen'):
            self._welcome_shown = qt_viewer.show_welcome_screen
            qt_viewer.show_welcome_screen = False
        self.viewer.window._status_bar._toggle_activity_dock(True)

    def __exit__(self, type, value, traceback):
        self.viewer.window._status_bar._toggle_activity_dock(False)
        if self._welcome_shown is not None:
            self.viewer.window._qt_viewer.show_welcome_screen = self._welcome_shown
            self._welcome_shown = None


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
