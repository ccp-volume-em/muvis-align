# https://bilayers.org/understanding-config
# https://forum.image.sc/t/napari-widgets-from-bilayers/119800

import functools
import logging
from magicgui.widgets import Container, create_widget
import os.path

from muvis_align.ui.ParamWidget import ParamWidget
from muvis_align.ui.bilayers_util import bilayers_to_magicgui_field
from muvis_align.util import (path_param_to_list, path_param_to_text, relativize_to_project_dir,
                              resolve_to_project_dir, to_posix_path)


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
    'color': 'muvis_align.ui.MagicColorPicker.MagicColorPicker',
}


def _show_optional_file_dialog_without_overwrite_confirm(
        mode='w',
        caption=None,
        start_path=None,
        filter=None,
):
    from qtpy.QtWidgets import QFileDialog

    dialog = QFileDialog()
    dialog.setAcceptMode(QFileDialog.AcceptSave)
    dialog.setFileMode(QFileDialog.AnyFile)
    dialog.setOption(QFileDialog.DontConfirmOverwrite, True)
    if caption:
        dialog.setWindowTitle(caption)
    if filter:
        dialog.setNameFilter(filter)
    if start_path:
        path = os.path.expanduser(start_path)
        if os.path.isdir(path):
            dialog.setDirectory(path)
        else:
            dialog.setDirectory(os.path.dirname(path) or '.')
            dialog.selectFile(os.path.basename(path))

    if dialog.exec():
        selected_files = dialog.selectedFiles()
        if selected_files:
            return selected_files[0]
    return None


def _project_relative_file_dialog(widget, show_dialog, get_project_dir):
    """`widget`'s file dialog opened at its path as resolved against the project, the choice shown relative to it:
    FileEdit itself resolves against the working directory and shows the choice absolute."""
    def show(mode, caption=None, start_path=None, filter=None):
        project_dir = get_project_dir()
        start_path = resolve_to_project_dir(widget.line_edit.value, project_dir) or project_dir
        # an output not written yet: its nearest existing folder
        while start_path and not os.path.exists(start_path) and os.path.dirname(start_path) != start_path:
            start_path = os.path.dirname(start_path)
        result = show_dialog(mode, caption=caption, start_path=start_path, filter=filter)
        if result:
            widget.line_edit.value = relativize_to_project_dir(to_posix_path(path_param_to_text(result)), project_dir)
        # nothing for FileEdit to set: it would make the path absolute again
        return None
    return show


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

            widget_type = spec.widget_type
            if widget_type == 'Table' and spec.section_key == 'display_only':
                # read-only, and possibly a row per source or pair: a view that draws only the rows shown
                widget_type = 'muvis_align.ui.ModelTable.ModelTable'
            is_path_list = (spec.is_file_type and spec.section_key == 'inputs'
                            and 'multiple' in (spec.file_count or ''))
            if is_path_list:
                widget_type = 'muvis_align.ui.PathList.PathList'
            full_name = section_id + '.' + spec.param_name
            param_widget = ParamWidget(full_name, None, interface, to_str=spec.is_file_type,
                                       number_type=spec.number_type)

            options = {}
            if spec.widget_type == 'Dropdown' and spec.choices:
                options['choices'] = spec.to_magicgui_choices()

            if is_path_list:
                spec.value = path_param_to_list(spec.value) or None
            elif spec.is_file_type:
                options['mode'] = get_file_dialog_mode(spec.is_output, spec.file_count)
                ext = os.path.splitext(str(spec.default))[1]
                if ext:
                    options['filter'] = '*' + ext

            if spec.value is not None:
                widget = create_widget(
                    name=full_name,
                    label=spec.param_label,
                    widget_type=widget_type,
                    options=options,
                    value=spec.value,
                )
            else:
                widget = create_widget(
                    name=full_name,
                    label=spec.param_label,
                    widget_type=widget_type,
                    options=options,
                )
            if spec.is_file_type and options.get('mode') == 'w' and hasattr(widget, '_show_file_dialog'):
                widget._show_file_dialog = _show_optional_file_dialog_without_overwrite_confirm
            if spec.is_file_type and not is_path_list and section_id != 'project' \
                    and hasattr(widget, '_show_file_dialog'):
                widget._show_file_dialog = _project_relative_file_dialog(
                    widget, widget._show_file_dialog, getattr(interface, 'get_project_dir', lambda: None))
            if is_path_list:
                widget.native.base_dir = getattr(interface, 'get_project_dir', lambda: None)
            # explicitly setting value=None triggers type deduction, and results in an error
            param_widget.widget = widget
            if spec.description:
                widget.tooltip = spec.description
            if function is not None:
                widget.changed.connect(function)
            interface_function = interface.get_function(spec.param_name)
            if interface_function is not None:
                widget.changed.connect(interface_function)
            selected_function = interface.get_function(spec.param_name + '_selected')
            if selected_function is not None and hasattr(widget.native, 'itemSelectionChanged'):
                widget.native.itemSelectionChanged.connect(selected_function)
            interface.param_widgets[full_name] = param_widget
            if connect_changed and spec.section_key != 'display_only':
                widget.changed.connect(param_widget.value_changed)
            else:
                # https://github.com/pyapp-kit/magicgui/issues/348
                widget.read_only = True
            widgets.append(widget)
        except Exception as e:
            logging.error(f'Error creating widget from {template}\n{e}')

    if add_button:
        name = section_id + '_process'
        full_name = section_id + '.' + name
        widget = create_widget(name=full_name, label='Process', widget_type='PushButton')
        interface_function = interface.get_function(name)
        if interface_function is not None:
            # while an operation runs the button reads Cancel
            if hasattr(interface, 'process_or_cancel'):
                widget.clicked.connect(functools.partial(interface.process_or_cancel, interface_function))
            else:
                widget.clicked.connect(interface_function)
        interface.param_widgets[full_name] = ParamWidget(full_name, widget, interface)
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
