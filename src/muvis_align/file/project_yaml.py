import yaml

from muvis_align.bilayers_util import get_section_ids, get_section_params


def read_params(path):
    with open(path, 'r') as infile:
        return yaml.load(infile, Loader=yaml.Loader)


def write_params(path, template, params):
    all_params = {}
    for section_id in get_section_ids(template):
        for template_param in get_section_params(template, section_id):
            if template_param.get('label') and template_param.get('default'):
                params[template_param['label']] = template_param['default']
        if params:
            all_params[section_id] = params
    with open(path, 'w') as outfile:
        yaml.dump(all_params, outfile, default_flow_style=False)
