from magicclass.ext.napari.viewer import ViewerWidget
from qtpy.QtWidgets import QApplication, QTabWidget

from muvis_align.ui.create_widgets import (
    create_project_widget,
    create_template_widgets,
)
from muvis_align.ui.Interface import Interface
from muvis_align.logging import init_logging


class MainWidget(QTabWidget):
    """Full UI construction that deliberately skips Interface.reset()."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.verbose = True
        init_logging(verbose=self.verbose)
        self.viewer = viewer

        self.overview = ViewerWidget()
        self.overview.min_height = 200
        viewer.window.add_dock_widget(
            self.overview,
            name="muvis-align",
            area="left",
            add_vertical_stretch=False,
        )
        self.interface = Interface(
            viewer,
            overview=self.overview,
            enable_plugin_widget=self.enable_plugin_widget,
            enable_tabs=self.enable_tabs,
            select_tab=self.select_tab,
            is_tab_enabled=self.is_tab_enabled,
            enable_tab=self.enable_tab,
            verbose=self.verbose,
        )

        self.tab_labels = []
        self.widgets = self.create_widgets()
        for label, widget in self.widgets.items():
            self.addTab(widget.native, label.replace("_", " "))
            self.tab_labels.append(label)
        self.enable_tabs(False, 1)
        self.currentChanged.connect(self.tab_changed)

    def create_widgets(self):
        project_widget = {
            "project": create_project_widget(
                self.interface,
                self.project_path_set,
            )
        }
        section_widgets = create_template_widgets(self.interface)
        return project_widget | section_widgets

    def tab_changed(self, index):
        self.interface.tab_changed(self.tab_labels[index])

    def enable_plugin_widget(self, enabled=True):
        self.setEnabled(enabled)
        QApplication.processEvents()

    def enable_tabs(self, enabled=True, tab_index=-1):
        for index in range(self.count()):
            if (
                enabled and (tab_index < 0 or index <= tab_index)
            ) or (
                not enabled and index >= tab_index
            ):
                self.setTabEnabled(index, enabled)
        # a tab enabled while hidden (e.g. fusion, right before a blocking QMessageBox) can be
        # left showing stale disabled styling under a slow/remote display (xpra) until the next
        # natural event-loop idle - flush immediately so it's interactive as soon as it's enabled
        QApplication.processEvents()

    def select_tab(self, tab_index):
        self.setCurrentIndex(tab_index)

    def is_tab_enabled(self, section_id):
        return self.isTabEnabled(self.tab_labels.index(section_id))

    def enable_tab(self, section_id, enabled=True):
        self.setTabEnabled(self.tab_labels.index(section_id), enabled)
        QApplication.processEvents()

    def project_path_set(self):
        self.enable_tabs(True, 1)
