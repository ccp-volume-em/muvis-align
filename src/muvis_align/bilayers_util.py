def get_section_ids(parameters):
    section_ids = []
    for param in parameters:
        section_id = param.get('section_id')
        if section_id not in section_ids:
            section_ids.append(section_id)
    return section_ids


def get_with_secion_id(parameters, section_id):
    section_params = []
    for param in parameters:
        if param.get('section_id') == section_id:
            section_params.append(param)
    return section_params
