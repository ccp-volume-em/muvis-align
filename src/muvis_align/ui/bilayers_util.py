def get_section_dict(template, keys=None):
    sections = {}
    if keys is None:
        keys = [None]
    for key in keys:
        if key is not None:
            items = template.get(key, [])
        else:
            items = template
        for item in items:
            section_id = item.get('section_id')
            if not section_id:
                section_id = key
            if section_id not in sections:
                sections[section_id] = []
            sections[section_id].append(item)
    return sections
