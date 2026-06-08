from src.muvis_align.ui.bilayers_util import (
    bilayers_to_magicgui_field,
    get_section_dict,
)
from src.muvis_align.ui.create_widgets import map_bilayers_to_widget_type


def test_get_section_dict_normalizes_and_sorts():
    template = {
        'parameters': [
            {'name': 'beta', 'type': 'textbox', 'section_id': 's1', 'cli_order': '2'},
            {'name': 'alpha', 'type': 'textbox', 'section_id': 's1', 'cli_order': 1},
        ]
    }

    sections = get_section_dict(template, ['parameters'])

    assert list(sections.keys()) == ['s1']
    assert [item['name'] for item in sections['s1']] == ['alpha', 'beta']
    assert all(item['section_key'] == 'parameters' for item in sections['s1'])


def test_bilayers_to_magicgui_field_uses_param_name_for_value_lookup():
    template_item = {
        'name': 'pairing',
        'label': 'Pairing',
        'type': 'dropdown',
        'section_key': 'parameters',
        'options': [
            {'label': 'Default', 'value': 'default'},
            {'label': 'Overlay', 'value': 'overlay'},
        ],
        'default': 'default',
    }

    section_params = {'pairing': 'overlay'}
    spec = bilayers_to_magicgui_field(
        template_item,
        section_id='registration',
        section_params=section_params,
        widget_type_map=map_bilayers_to_widget_type,
    )

    assert spec.value == 'overlay'
    assert spec.widget_type == 'Dropdown'
    assert spec.choices == {'default': 'Default', 'overlay': 'Overlay'}


def test_bilayers_spec_builds_magicgui_choice_mapping():
    template_item = {
        'name': 'method',
        'label': 'Method',
        'type': 'dropdown',
        'options': [
            {'label': 'SIFT', 'value': 'sift'},
            {'label': 'ORB', 'value': 'orb'},
        ],
        'default': 'sift',
    }

    spec = bilayers_to_magicgui_field(
        template_item,
        section_id='registration',
        section_params={},
        widget_type_map=map_bilayers_to_widget_type,
    )
    magic_choices = spec.to_magicgui_choices()

    assert tuple(magic_choices['choices']) == ('sift', 'orb')
    assert magic_choices['key']('sift') == 'SIFT'


