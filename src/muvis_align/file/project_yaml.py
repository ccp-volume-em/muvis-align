import yaml

from muvis_align.ui.bilayers_util import get_section_dict


def read_params(path):
    with open(path, 'r') as infile:
        return yaml.load(infile, Loader=yaml.Loader)


def write_params(path, template, params):
    all_params = {}
    template = get_section_dict(template, 'inputs') | get_section_dict(template, 'parameters')
    for section_id, section_items in template.items():
        for param_id, template_param in section_items.items():
            if template_param.get('label') and template_param.get('default'):
                params[template_param['label']] = template_param['default']
        if params:
            all_params[section_id] = params
    with open(path, 'w') as outfile:
        yaml.dump(all_params, outfile, default_flow_style=False)
