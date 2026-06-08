# https://bilayers.org/understanding-config
# https://forum.image.sc/t/napari-widgets-from-bilayers/119800

import logging
from magicgui.widgets import Container, create_widget
import os.path

from muvis_align.ui.ParamWidget import ParamWidget
from muvis_align.ui.bilayers_util import bilayers_to_magicgui_field


map_bilayers_to_widget_type = {
    'textbox': 'LineEdit',
    'checkbox': 'CheckBox',
    'radio': 'Radio',
    'dropdown': 'Dropdown',
    'integer': 'SpinBox',
    'float': 'FloatSpinBox',
    'table': 'Table',
    'file': 'FileEdit',
    'image': 'FileEdit',
    'array': 'FileEdit',
    'measurement': 'FileEdit',
    'button': 'PushButton',
    'pushbutton': 'PushButton',
}


def create_project_widget(interface, function):
    project_template = [
        {'name': 'project_path',
         'label': 'Project path',
         'type': 'file',
         'output_dir_set': True,
         'default': 'muvis_align_project.yml'}
    ]
    return create_section_container('project', project_template, interface,
                                    connect_changed=False, function=function, add_button=False)


def create_template_widgets(interface):
    widgets = {}
    for section_id, section_items in interface.template.items():
        widgets[section_id] = create_section_container(section_id, section_items, interface)
    return widgets


def create_section_container(section_id, section_template, interface,
                             connect_changed=True, function=None, add_button=True):
    # https://pyapp-kit.github.io/magicgui/widgets/
    # https://pyapp-kit.github.io/magicgui/api/widgets/create_widget/
    widgets = []
    for template in section_template:
        try:
            section_params = interface.params.get(section_id, {})
            spec = bilayers_to_magicgui_field(
                template,
                section_id=section_id,
                section_params=section_params,
                widget_type_map=map_bilayers_to_widget_type,
            )

            full_name = section_id + '.' + spec.param_name
            param_widget = ParamWidget(full_name, None, interface, to_str=spec.is_file_type)

            options = {}
            if spec.widget_type == 'Dropdown' and spec.choices:
                options['choices'] = spec.to_magicgui_choices()

            if spec.is_file_type:
                options['mode'] = get_file_dialog_mode(spec.is_output, spec.file_count)
                ext = os.path.splitext(str(spec.default))[1]
                if ext:
                    options['filter'] = '*' + ext

            if spec.value is not None:
                widget = create_widget(
                    name=full_name,
                    label=spec.param_label,
                    widget_type=spec.widget_type,
                    options=options,
                    value=spec.value,
                )
            else:
                widget = create_widget(
                    name=full_name,
                    label=spec.param_label,
                    widget_type=spec.widget_type,
                    options=options,
                )
                # explicitly setting value=None triggers type deduction, and results in an error
            param_widget.widget = widget
            if spec.description:
                widget.tooltip = spec.description
            if function is not None:
                widget.changed.connect(function)
            # check if function with same name exists
            interface_function = interface.get_function(spec.param_name)
            if interface_function is not None:
                widget.changed.connect(interface_function)
            interface.param_widgets[full_name] = param_widget
            if connect_changed and spec.section_key != 'display_only':
                widget.changed.connect(param_widget.value_changed)
            widgets.append(widget)
        except Exception as e:
            logging.error(f'Error creating widget from {template}\n{e}')

    if add_button:
        name = section_id + '_process'
        widget = create_widget(name=name, label='Process', widget_type='PushButton')
        interface_function = interface.get_function(name)
        if interface_function is not None:
            widget.clicked.connect(interface_function)
        widgets.append(widget)

    return Container(widgets=widgets)


def get_file_dialog_mode(is_output, file_count):
    # https://pyapp-kit.github.io/magicgui/api/widgets/FileEdit/
    if file_count and 'multiple' in file_count:
        mode = 'd'
    elif is_output:
        mode = 'w'
    else:
        mode = 'r'
    return mode
