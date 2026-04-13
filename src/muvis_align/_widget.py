from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

import logging
import os.path
from qtpy.QtWidgets import QTabWidget

from muvis_align.ui.create_widgets import create_project_widget, create_template_widgets
from muvis_align.ui.Interface import Interface
from muvis_align._version import __version__


class MainWidget(QTabWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.verbose = True
        self.init_logging()
        self.viewer = viewer
        self.interface = Interface(viewer, self.verbose)

        self.widgets = self.create_widgets()
        for label, widget in self.widgets.items():
            self.addTab(widget.native, label.replace('_', ' '))
        self.enable_tabs(False, 1)
        #viewer.window.add_dock_widget(self.main_output_widget, name='MASS', area='left')

    def init_logging(self, verbose=False):
        self.log_filename = 'log/muvis-align.log'
        log_format = '%(asctime)s %(levelname)s: %(message)s'
        basepath = os.path.dirname(self.log_filename)
        if basepath and not os.path.exists(basepath):
            os.makedirs(basepath)

        handlers = [logging.FileHandler(self.log_filename, encoding='utf-8')]
        if self.verbose:
            handlers += [logging.StreamHandler()]
        logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers, encoding='utf-8')

        # verbose external modules
        if verbose:
            # expose multiview_stitcher.registration logger and make more verbose
            mvsr_logger = logging.getLogger('multiview_stitcher.registration')
            mvsr_logger.setLevel(logging.DEBUG)
            if len(mvsr_logger.handlers) == 0:
                mvsr_logger.addHandler(logging.StreamHandler())
        else:
            # reduce verbose level
            for module in ['multiview_stitcher', 'multiview_stitcher.registration', 'multiview_stitcher.fusion']:
                logging.getLogger(module).setLevel(logging.WARNING)

        for module in ['ome_zarr']:
            logging.getLogger(module).setLevel(logging.WARNING)

        logging.info(f'muvis-align version {__version__}')

    def create_widgets(self):
        project_widget = {'project': create_project_widget(self.interface, self.project_path_set)}
        section_widgets = create_template_widgets(self.interface)
        return project_widget | section_widgets

    def enable_tabs(self, set=True, tab_index=-1):
        for index in range(self.count()):
            if (set and (tab_index < 0 or index <= tab_index)) or (not set and index >= tab_index):
                self.setTabEnabled(index, set)

    def project_path_set(self):
        self.enable_tabs()
