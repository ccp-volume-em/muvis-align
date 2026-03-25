def get_section_dict(template, key):
    sections = {}
    for item in template.get(key, []):
        section_id = item.get('section_id')
        if not section_id:
            section_id = key
        if section_id not in sections:
            sections[section_id] = []
        sections[section_id].append(item)
    return sections
