from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

from qtpy.QtWidgets import QTabWidget

from muvis_align.ui.create_widgets import create_project_widget, create_widgets
from muvis_align.ui.Interface import Interface


class MainWidget(QTabWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.interface = Interface()

        self.widgets = self.create_widgets()
        for label, widget in self.widgets.items():
            self.addTab(widget.native, label)
        #viewer.window.add_dock_widget(self.main_output_widget, name='MASS', area='left')

    def create_widgets(self):
        project_widget = {'project': create_project_widget(self.interface)}
        section_widgets = create_widgets(self.interface)
        return project_widget | section_widgets
