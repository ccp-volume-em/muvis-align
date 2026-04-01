import logging
import os.path

from muvis_align.file.project_yaml import read_params, get_template_params, write_params
from muvis_align.MVSRegistrationNapari import MVSRegistrationNapari
from muvis_align.resources import get_project_template
from muvis_align.ui.bilayers_util import get_section_dict
from muvis_align.util import dir_regex, find_all_numbers


class Interface:
    def __init__(self, viewer, verbose=False):
        self.viewer = viewer
        self.verbose = verbose
        self.raw_template = get_project_template()
        if not self.raw_template:
            raise FileNotFoundError('Project template not found')
        self.template = get_section_dict(self.raw_template, ['inputs', 'parameters', 'display_only', 'outputs'])
        self.params_general = {}
        self.params = get_template_params(self.template)
        self.reg = MVSRegistrationNapari(self.params_general, self.viewer)

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
            write_params(path, self.template, self.params)

    def param_changed(self, instance):
        print(instance.name, instance.value)

    def input_images(self, path):
        path = str(path)
        self.params['operation'] = 'register'
        self.params['output'] = '/output'
        self.params['input'] = path

        filenames = dir_regex(path)
        filenames = sorted(filenames, key=lambda file: list(find_all_numbers(file)))  # sort first key first
        fileset_label = os.path.basename(path)
        if len(filenames) == 0:
            logging.warning(f'No files found for path: {path}')
            return
        elif self.verbose:
            logging.info(f'# total files: {len(filenames)}')
        self.reg.init_operation(fileset_label, filenames, self.params)
        self.reg.init_sims()
