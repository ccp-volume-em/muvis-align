from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QGridLayout, QLabel, QLineEdit, QPushButton

from muvis_align.PathControl import PathControl
from muvis_align.ui.bilayers_util import get_section_dict


def create_project_widget(set_params_path):
    project_template = [
        {'label': 'project_path',
         'type': 'textbox',
         'path_type': 'set',
         'default': 'muvis_align_project.yml',
         'function': set_params_path}
    ]
    return create_section_widget(project_template, {})


def create_widgets(template, params):
    widgets = {}
    template = get_section_dict(template, 'inputs') | get_section_dict(template, 'parameters')
    for section_id, section_items in template.items():
        section_params = params.get(section_id, {})
        widgets[section_id] = create_section_widget(section_items, section_params)
    return widgets


def create_section_widget(section_template, section_params):
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
                #path_controls[param_label] = path_control

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
