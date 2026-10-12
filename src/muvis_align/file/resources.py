import importlib.resources
import yaml

from muvis_align.constants import NAPARI_PROJECT_TEMPLATE


def get_project_template_text():
    return importlib.resources.files('muvis_align').joinpath(NAPARI_PROJECT_TEMPLATE).read_text()


def get_project_template():
    file = get_project_template_text()
    template = yaml.load(file, Loader=yaml.Loader)

    return template
