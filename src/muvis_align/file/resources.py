#from pathlib import Path
#import pkgutil
import importlib.resources
import yaml

from muvis_align.constants import NAPARI_PROJECT_TEMPLATE


def get_project_template():
    # method 1: local path
    #RESOURCE_DIR = Path(__file__).parent.parent.parent / 'resources'
    #template_path = RESOURCE_DIR / PROJECT_TEMPLATE
    #file = open(template_path, 'r'):

    # method 2: pkgutil (old)
    #file = pkgutil.get_data('muvis_align', '../../resources/' + PROJECT_TEMPLATE)

    # method 3: importlib.resources (new)
    project_template_res = importlib.resources.files('muvis_align')
    project_template_file_res = project_template_res.joinpath('ui/' + NAPARI_PROJECT_TEMPLATE)
    file = project_template_file_res.read_text()

    # load file content
    template = yaml.load(file, Loader=yaml.Loader)

    return template
