import yaml


def read_params(path):
    with open(path, 'r') as infile:
        return yaml.load(infile, Loader=yaml.Loader)


def get_template_params(section_template):
    params = {}
    for section_id, section_items in section_template.items():
        section_params = {}
        for section_item in section_items:
            if section_item.get('label') and section_item.get('default'):
                label = section_item['name']
                if section_params.get(label) is None:
                    section_params[label] = section_item['default']
        params[section_id] = section_params
    return params


def write_params(path, section_template, params):
    # copy template default values to unset param values
    for section_id, section_items in section_template.items():
        section_params = params.get(section_id, {})
        for section_item in section_items:
            if section_item.get('label') and section_item.get('default'):
                label = section_item['name']
                if section_params.get(label) is None:
                    section_params[label] = section_item['default']
    with open(path, 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)
