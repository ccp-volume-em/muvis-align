import os.path
from qtpy.QtWidgets import QPushButton, QFileDialog, QStyle

from muvis_align.util import get_label_element


class PathControl:
    def __init__(self, template, path_widget, params, param_label, function=None):
        self.template = template
        self.path_widget = path_widget
        self.params = params
        self.param_label = param_label
        self.function = function
        self.description = template.get('description')
        self.path_type = 'dir' if template.get('file_count', '') == 'multiple' else 'file'

        parts = template.get('default').split('.')
        self.default_ext = parts[1] if len(parts) > 1 else ''

        icon = path_widget.style().standardIcon(QStyle.SP_FileIcon)
        path_button = QPushButton(icon, '')
        path_button.clicked.connect(lambda: self.show_dialog(self.path_type))
        self.path_buttons = [path_button]
        if 'image' in self.path_type:
            # Add dir button for zarr
            icon = path_widget.style().standardIcon(QStyle.SP_DirIcon)
            dir_button = QPushButton(icon, '')
            dir_button.clicked.connect(lambda: self.show_dialog('dir'))
            self.path_buttons.append(dir_button)

    def get_button_widgets(self):
        return self.path_buttons

    def show_dialog(self, path_type):
        value = self.path_widget.text()
        if not value:
            param = get_label_element(self.params, self.param_label)
            if param and 'value' in param:
                value = param['value']
        if not value and 'default' in self.template:
            value = self.template['default']

        caption = self.description
        filter = None

        self.is_folder = False
        if path_type == 'dir':
            self.is_folder = True
            result = QFileDialog.getExistingDirectory(
                caption=caption, directory=value
            )
            self.process_result(result)
        elif path_type in ['save', 'set']:
            if path_type == 'set':
                options = QFileDialog.DontConfirmOverwrite  # only works on Windows?
            else:
                options = None
            result = QFileDialog.getSaveFileName(
                caption=caption, directory=value, filter=filter, options=options,
            )
            self.process_result(result[0])
        else:
            # open file
            result = QFileDialog.getOpenFileName(
                caption=caption, directory=value, filter=filter
            )
            # for zarr take parent path
            filepath = result[0]
            filename = os.path.basename(filepath)
            if filename in ['.zattrs', '.zgroup']:
                filepath = os.path.dirname(filepath)
            self.process_result(filepath)

    def process_result(self, filepath):
        if filepath:
            if not self.is_folder and os.path.splitext(filepath)[1] == '':
                filepath += self.default_ext
            self.path_widget.setText(filepath)
            if self.function is not None:
                self.function(filepath)
