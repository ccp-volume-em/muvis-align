import os.path

from muvis_align.file.project_yaml import read_params, write_params
from muvis_align.resources import get_project_template


class Interface:
    def __init__(self):
        self.template = get_project_template()
        if not self.template:
            raise FileNotFoundError('Project template not found')
        self.params = {}

    def get_function(self, function_label):
        if hasattr(self, function_label):
            return eval(f'self.{function_label}')
        else:
            return None

    def project_path(self, path):
        if os.path.exists(path):
            print('reading params...')
            self.params = read_params(path)
        else:
            print('writing params...')
            write_params(path, self.template.get('parameters', []), self.params)

    def input_images(self, path):
        print(path)
