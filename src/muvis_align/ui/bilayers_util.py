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
            item['section_key'] = key
            if not section_id:
                section_id = key
            if section_id not in sections:
                sections[section_id] = []
            sections[section_id].append(item)
    sorted_sections = {}
    for section_id, section_items in sections.items():
        sorted_sections[section_id] = sorted(section_items, key=lambda item: item.get('section_id'))
    return sorted_sections
