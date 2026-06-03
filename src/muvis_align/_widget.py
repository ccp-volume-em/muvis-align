from magicclass.ext.napari.viewer import ViewerWidget
from qtpy.QtWidgets import QTabWidget, QSizePolicy

from muvis_align.ui.create_widgets import create_project_widget, create_template_widgets
from muvis_align.ui.Interface import Interface
from muvis_align.logging import init_logging


class MainWidget(QTabWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.verbose = True
        init_logging()
        self.viewer = viewer

        self.overview = ViewerWidget()
        viewer.window.add_dock_widget(self.overview, name='muvis-align', area='left')
        self.overview.native.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))

        self.interface = Interface(viewer, self.overview, self.enable_tabs, self.verbose)

        self.tab_labels = []
        self.widgets = self.create_widgets()
        for label, widget in self.widgets.items():
            self.addTab(widget.native, label.replace('_', ' '))
            self.tab_labels.append(label)
        self.enable_tabs(False, 1)
        self.currentChanged.connect(self.tab_changed)
        #viewer.window.add_dock_widget(self.main_output_widget, name='MASS', area='left')

    def create_widgets(self):
        project_widget = {'project': create_project_widget(self.interface, self.project_path_set)}
        section_widgets = create_template_widgets(self.interface)
        return project_widget | section_widgets

    def tab_changed(self, index):
        self.interface.tab_changed(self.tab_labels[index])

    def enable_tabs(self, set=True, tab_index=-1):
        for index in range(self.count()):
            if (set and (tab_index < 0 or index <= tab_index)) or (not set and index >= tab_index):
                self.setTabEnabled(index, set)

    def project_path_set(self):
        self.enable_tabs(True, 1)
