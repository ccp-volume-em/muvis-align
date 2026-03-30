# https://bilayers.org/understanding-config
# https://forum.image.sc/t/napari-widgets-from-bilayers/119800


from magicgui.widgets import Container, create_widget

from muvis_align.ui.bilayers_util import get_section_dict


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
    'measurement': 'FileEdit'
}


def create_project_widget(interface):
    project_template = [
        {'name': 'project_path',
         'label': 'Project path',
         'type': 'file',
         'output_dir_set': True,
         'default': 'muvis_align_project.yml'}
    ]
    return create_section_widget('', project_template, {}, interface)


def create_widgets(interface):
    widgets = {}
    sections = get_section_dict(interface.template, ['inputs', 'parameters', 'display_only', 'outputs'])
    for section_id, section_items in sections.items():
        section_params = interface.params.get(section_id, {})
        widgets[section_id] = create_section_widget(section_id, section_items, section_params, interface)
    return widgets


def create_section_widget(section_id, section_template, section_params, interface):
    # https://pyapp-kit.github.io/magicgui/widgets/
    widgets = []
    for index, template in enumerate(section_template):
        param_name = template.get('name')
        param_label = template.get('label')
        param_type = template.get('type').lower()
        value = section_params.get(param_label, template.get('default'))
        is_output = (section_id == 'output' or template.get('output_dir_set'))
        description = template.get('description')
        choices = template.get('options')
        has_action = False

        widget_type = map_bilayers_to_widget_type.get(param_type)
        if widget_type is None:
            print(f'Unsupported type {param_type}')
        options = {}
        if choices is not None:
            options['choices'] = list(choices)
        if widget_type == 'FileEdit':
            file_count = template.get('file_count')
            options['mode'] = get_file_dialog_mode(is_output, file_count)
            has_action = True
        widget = create_widget(name=param_name, value=value, label=param_label, widget_type=widget_type, options=options)
        if description:
            widget.tooltip = description
        if has_action:
            interface_function = interface.get_function(param_name)
            if interface_function is not None:
                widget.changed.connect(interface_function)
        widgets.append(widget)
    return Container(widgets=widgets)


def get_file_dialog_mode(is_output, file_count):
    if file_count and 'multiple' in file_count:
        mode = 'd'
    elif is_output:
        mode = 'w'
    else:
        mode = 'r'
    return mode
