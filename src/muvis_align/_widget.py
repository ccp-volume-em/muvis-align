from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

import os.path
from qtpy.QtWidgets import QWidget, QTabWidget, QGridLayout, QLabel, QLineEdit, QPushButton
from qtpy.QtCore import Qt

from muvis_align.bilayers_util import get_section_ids, get_section_params
from muvis_align.file.project_yaml import read_params, write_params
from muvis_align.PathControl import PathControl
from muvis_align.resources import get_project_template


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
        widgets = {'project': self.create_project_widget()}
        template = self.template.get('inputs', []) + self.template.get('parameters', []) + self.template.get('outputs', [])
        for section_id in get_section_ids(template):
            section_template = get_section_params(template, section_id)
            section_params = self.params.get(section_id, {})
            widgets[section_id] = self.create_section_widget(section_template, section_params)
        return widgets

    def create_project_widget(self):
        project_template = [
            {'label': 'project_path',
             'type': 'textbox',
             'path_type': 'set',
             'default': 'muvis_align_project.yml',
             'function': self.set_params_path}
        ]
        return self.create_section_widget(project_template, {})

    def create_section_widget(self, section_template, section_params):
        # https://bilayers.org/understanding-config
        section_widget = QWidget()
        layout = QGridLayout()
        layout.setAlignment(Qt.AlignTop)
        section_widget.setLayout(layout)

        index = 0
        for index, template in enumerate(section_template):
            param_label = template.get('label')
            param_type = template.get('type').lower()
            value = section_params.get('label', template.get('default'))
            description = template.get('description')
            function = template.get('function')

            label_widget = QLabel(param_label)
            var_widget = None
            if param_type == 'checkbox':
                pass
            elif param_type == 'radio':
                pass
            elif param_type == 'dropdown':
                pass
            elif param_type == 'button':
                var_widget = QPushButton(param_label)
            else:
                var_widget = QLineEdit()
                if param_type in ['files', 'image']:
                    path_control = PathControl(template, var_widget, section_template, param_label, function=function)
                    for button_index, button in enumerate(path_control.get_button_widgets()):
                        layout.addWidget(button, index, 2 + button_index)
                    self.path_controls[param_label] = path_control

            if var_widget and value is not None:
                var_widget.setText(value)

            if label_widget:
                layout.addWidget(label_widget, index, 0)
                layout.addWidget(var_widget, index, 1)
            else:
                layout.addWidget(var_widget, index, 0, 1, -1)

            if description:
                if label_widget is not None:
                    label_widget.setToolTip(description)
                if var_widget is not None:
                    var_widget.setToolTip(description)

        layout.addWidget(QPushButton('run'), index + 1, 0, 1, -1)

        return section_widget

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")
