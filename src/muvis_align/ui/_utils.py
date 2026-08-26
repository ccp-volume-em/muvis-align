# Based on https://github.com/multiview-stitcher/napari-stitcher/blob/main/src/napari_stitcher/_stitcher_widget.py


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
    Context manager to temporarily show the activity dock during long computation
    """
    def __init__(self, viewer):
        self.viewer = viewer

    def __enter__(self):
        self.viewer.window._status_bar._toggle_activity_dock(True)

    def __exit__(self, type, value, traceback):
        self.viewer.window._status_bar._toggle_activity_dock(False)
