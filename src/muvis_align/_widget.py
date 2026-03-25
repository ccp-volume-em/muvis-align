from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

import os.path
from qtpy.QtWidgets import QTabWidget

from muvis_align.file.project_yaml import read_params, write_params
from muvis_align.resources import get_project_template
from muvis_align.ui.create_widgets import create_project_widget, create_widgets


class MainWidget(QTabWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.template = get_project_template()
        if not self.template:
            raise FileNotFoundError('Project template not found')
        self.params = {}
        self.path_controls = {}
        self.widgets = self.create_widgets()
        for label, widget in self.widgets.items():
            self.addTab(widget, label)
        #viewer.window.add_dock_widget(self.main_output_widget, name='MASS', area='left')

    def set_params_path(self, path):
        if os.path.exists(path):
            self.params = read_params(path)
        else:
            write_params(path, self.template['parameters'], self.params)

    def create_widgets(self):
        project_widget = {'project': create_project_widget(self.set_params_path)}
        section_widgets = create_widgets(self.template, self.params)
        return project_widget | section_widgets

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")
