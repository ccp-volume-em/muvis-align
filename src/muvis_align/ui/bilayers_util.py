"""Helpers to validate bilayers config and derive magicgui-ready field specs."""

# docs https://bilayers.org/understanding-config/
# example https://github.com/bilayer-containers/bilayers/blob/main/algorithms/cellpose_inference/config.yaml

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BilayersOption(BaseModel):
    model_config = ConfigDict(extra='allow')

    label: str
    value: Any


class BilayersField(BaseModel):
    model_config = ConfigDict(extra='allow')

    name: str
    type: str
    label: str | None = None
    default: Any = None
    description: str | None = None
    section_id: str | None = None
    section_key: str | None = None
    cli_order: int = 0
    options: list[BilayersOption] | None = None
    file_count: str | None = None
    output_dir_set: bool = False

    @field_validator('type', mode='before')
    @classmethod
    def _normalize_type(cls, value):
        return str(value).lower()

    @field_validator('cli_order', mode='before')
    @classmethod
    def _normalize_cli_order(cls, value):
        if value is None or value == '':
            return 0
        return int(value)


class MagicGuiFieldSpec(BaseModel):
    section_id: str
    section_key: str | None = None
    param_name: str
    param_label: str
    widget_type: str
    value: Any = None
    description: str | None = None
    choices: dict[Any, str] = Field(default_factory=dict)
    file_count: str | None = None
    default: Any = None
    is_output: bool = False
    is_file_type: bool = False

    def get_choice_label(self, choice):
        return self.choices[choice]

    def to_magicgui_choices(self):
        return to_magicgui_choices(self.choices)


def to_magicgui_choices(choices):
    if isinstance(choices, dict):
        def _get_choice_label(choice):
            return choices[choice]

        return {
            'choices': choices.keys(),
            'key': _get_choice_label,
        }
    else:
        return choices


def get_section_dict(template, keys=None):
    """Group bilayers fields by section and return normalized dict items."""
    sections = {}
    if keys is None:
        keys = [None]

    for key in keys:
        items = template.get(key, []) if key is not None else template
        for raw_item in items:
            item = BilayersField.model_validate({**raw_item, 'section_key': key})
            section_id = item.section_id or key
            sections.setdefault(section_id, []).append(item)

    return {
        section_id: [item.model_dump() for item in sorted(section_items, key=lambda item: item.cli_order)]
        for section_id, section_items in sections.items()
    }


def bilayers_to_magicgui_field(template_item, section_id, section_params, widget_type_map):
    """Convert one bilayers field definition to a magicgui widget spec."""
    item = BilayersField.model_validate(template_item)
    widget_type = widget_type_map.get(item.type)
    if widget_type is None:
        raise ValueError(f'Unsupported type {item.type}')

    is_file_type = widget_type == 'FileEdit'
    choices = {}
    if widget_type == 'Dropdown' and item.options:
        choices = {option.value: option.label for option in item.options}

    return MagicGuiFieldSpec(
        section_id=section_id,
        section_key=item.section_key,
        param_name=item.name,
        param_label=item.label or item.name,
        widget_type=widget_type,
        value=section_params.get(item.name, item.default),
        description=item.description,
        choices=choices,
        file_count=item.file_count,
        default=item.default,
        is_output=(section_id == 'output' or item.output_dir_set),
        is_file_type=is_file_type,
    )
