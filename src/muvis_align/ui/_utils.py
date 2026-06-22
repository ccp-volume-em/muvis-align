# Based on https://github.com/multiview-stitcher/napari-stitcher/blob/main/src/napari_stitcher/_stitcher_widget.py


class TemporarilyDisabledWidgets(object):
    """
    Context manager to temporarily disable widgets during long computation
    """
    def __init__(self, widgets):
        self.store(widgets)

    def __enter__(self):
        self.init()

    def __exit__(self, type, value, traceback):
        self.restore()

    def store(self, widgets):
        self.widgets = widgets
        self.enabled_states = {name: widget.enabled for name, widget in widgets.items()}

    def init(self):
        for widget in self.widgets.values():
            widget.enabled = False

    def restore(self):
        for name, widget in self.widgets.items():
            #widget.enabled = self.enabled_states[name]
            widget.enabled = True


class VisibleActivityDock(object):
    """
    Context manager to temporarily disable widgets during long computation
    """
    def __init__(self, viewer):
        self.viewer = viewer

    def __enter__(self):
        self.viewer.window._status_bar._toggle_activity_dock(True)

    def __exit__(self, type, value, traceback):
        self.viewer.window._status_bar._toggle_activity_dock(False)
