import pytest

from muvis_align.ui.bilayers_util import (
    bilayers_to_magicgui_field,
    get_section_dict,
)
from muvis_align.ui.create_widgets import map_bilayers_to_widget_type


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
            {'label': 'Stack', 'value': 'stack'},
        ],
        'default': 'default',
    }

    section_params = {'pairing': 'stack'}
    spec = bilayers_to_magicgui_field(
        template_item,
        section_id='registration',
        section_params=section_params,
        widget_type_map=map_bilayers_to_widget_type,
    )

    assert spec.value == 'stack'
    assert spec.widget_type == 'Dropdown'
    assert spec.choices == {'default': 'Default', 'stack': 'Stack'}


@pytest.mark.parametrize('field_type, default, saved, widget_type, value', [
    ('float', None, {}, 'LineEdit', ''),
    ('float', None, {'gaussian_sigma': 1.5}, 'LineEdit', '1.5'),
    ('integer', None, {'gaussian_sigma': 4.0}, 'LineEdit', '4'),
    ('float', 2.0, {}, 'FloatSpinBox', 2.0),
])
def test_a_number_field_without_default_is_a_blank_able_text_field(field_type, default, saved, widget_type, value):
    template_item = {'name': 'gaussian_sigma', 'type': field_type, 'default': default}

    spec = bilayers_to_magicgui_field(template_item, section_id='pre_processing', section_params=saved,
                                      widget_type_map=map_bilayers_to_widget_type)

    assert spec.widget_type == widget_type
    assert spec.value == value
    assert spec.number_type == (field_type if widget_type == 'LineEdit' else None)


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


