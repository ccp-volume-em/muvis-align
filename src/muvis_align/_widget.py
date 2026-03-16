from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import napari

import os.path
from qtpy.QtWidgets import QWidget, QTabWidget, QGridLayout, QLabel, QLineEdit
from qtpy.QtCore import Qt
import yaml

from muvis_align.bilayers_util import get_section_ids, get_section_params
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
            self.read_params(path)
        else:
            self.write_params(path, self.params)

    def read_params(self, path):
        with open(path, 'r') as infile:
            self.params = yaml.load(infile, Loader=yaml.Loader)

    def write_params(self, path, params):
        template = self.template['parameters']
        all_params = {}
        for section_id in get_section_ids(template):
            for template_param in get_section_params(template, section_id):
                if template_param.get('label') and template_param.get('default'):
                    params[template_param['label']] = template_param['default']
            if params:
                all_params[section_id] = params
        with open(path, 'w') as outfile:
            yaml.dump(all_params, outfile, default_flow_style=False)


    def create_widgets(self):
        widgets = {'project': self.create_project_widget()}
        template = self.template['parameters']
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
        section_widget = QWidget()
        layout = QGridLayout()
        layout.setAlignment(Qt.AlignTop)
        section_widget.setLayout(layout)

        for index, template in enumerate(section_template):
            param_label = template.get('label')
            param_type = template.get('type').lower()
            value = section_params.get('label', template.get('default'))
            description = template.get('description')
            function = template.get('function')

            label_widget = QLabel(param_label)
            var_widget = None
            if 'text' in param_type:
                var_widget = QLineEdit()
                if 'path' or 'file' in param_type:
                    path_control = PathControl(template, var_widget, section_template, param_label, function=function)
                    for button_index, button in enumerate(path_control.get_button_widgets()):
                        layout.addWidget(button, index, 2 + button_index)
                    self.path_controls[param_label] = path_control

            if value is not None:
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

        return section_widget

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")
