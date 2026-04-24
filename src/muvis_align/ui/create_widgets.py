# https://bilayers.org/understanding-config
# https://forum.image.sc/t/napari-widgets-from-bilayers/119800


from magicgui.widgets import Container, create_widget
import os.path

from muvis_align.ui.ParamWidget import ParamWidget

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
    for index, template in enumerate(section_template):
        section_params = interface.params.get(section_id, {})
        section_key = template.get('section_key')
        param_name = template.get('name')
        param_label = template.get('label')
        param_type = template.get('type').lower()
        value = section_params.get(param_label, template.get('default'))
        is_output = (section_id == 'output' or template.get('output_dir_set'))
        description = template.get('description')
        choices = template.get('options')

        widget_type = map_bilayers_to_widget_type.get(param_type)
        is_file_type = (widget_type == 'FileEdit')
        if widget_type is None:
            print(f'Unsupported type {param_type}')

        full_name = section_id + '.' + param_name
        param_widget = ParamWidget(full_name, None, interface, to_str=is_file_type)

        options = {}
        if widget_type == 'Dropdown':
            options['choices'] = param_widget.create_choices({item['value']: item['label'] for item in choices})

        if is_file_type:
            file_count = template.get('file_count')
            options['mode'] = get_file_dialog_mode(is_output, file_count)
            ext = os.path.splitext(str(template.get('default')))[1]
            if ext:
                options['filter'] = '*' + ext
        widget = create_widget(name=full_name, value=value, label=param_label, widget_type=widget_type, options=options)
        param_widget.widget = widget
        if description:
            widget.tooltip = description
        if function is not None:
            widget.changed.connect(function)
        # check if function with same name exists
        interface_function = interface.get_function(param_name)
        if interface_function is not None:
            widget.changed.connect(interface_function)
        interface.param_widgets[full_name] = param_widget
        if connect_changed and section_key != 'display_only':
            widget.changed.connect(param_widget.value_changed)
        widgets.append(widget)

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
