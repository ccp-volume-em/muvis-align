import yaml

from muvis_align.ui.bilayers_util import get_section_dict


def read_params(path):
    with open(path, 'r') as infile:
        return yaml.load(infile, Loader=yaml.Loader)


def write_params(path, template0, params):
    # copy template default values to unset param values
    template = get_section_dict(template0)
    for section_id, section_items in template.items():
        section_params = params.get(section_id, {})
        for section_item in section_items:
            if section_item.get('label') and section_item.get('default'):
                label = section_item['name']
                if section_params.get(label) is None:
                    section_params[label] = section_item['default']
    with open(path, 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)
