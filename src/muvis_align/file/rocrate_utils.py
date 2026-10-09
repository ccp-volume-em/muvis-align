# https://pypi.org/project/rocrate/
# https://github.com/ome/ome2024-ngff-challenge/tree/main/src/ome2024_ngff_challenge/zarr_crate
# https://www.researchobject.org/workflow-run-crate/profiles/workflow_run_crate/

from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from io import StringIO
import json
import os.path
import re
from pathlib import Path
from rocrate.model import ContextEntity
from rocrate.rocrate import ROCrate
from rocrate.model.computerlanguage import ComputerLanguage

from muvis_align.constants import NAPARI_PROJECT_TEMPLATE
from muvis_align.file.resources import get_project_template, get_project_template_text
from muvis_align.file.zarr_extension import ZarrCrate
from muvis_align.ui.bilayers_util import get_section_dict
from muvis_align.util import get_filetitle


ZARR_ENCODING_FORMAT = [
    'application/vnd.zarr',
    {'@id': 'https://openminds.docs.om-i.org/en/v3.0/instance_libraries/contentTypes.html#application-vnd-zarr'}
]
WORKFLOW_RUN_PROFILES = {
    'https://w3id.org/ro/wfrun/process/0.5': 'Process Run Crate',
    'https://w3id.org/ro/wfrun/workflow/0.5': 'Workflow Run Crate',
}
MUVIS_ALIGN_URL = 'https://github.com/folterj/muvis-align'
INPUT_PARAMETER_ID = '#input_output.input_path'
OUTPUT_PARAMETER_ID = '#input_output.output_path'
# the template's widget types as schema.org types; buttons are no parameters
PARAMETER_TYPES = {'checkbox': 'Boolean', 'integer': 'Integer', 'float': 'Float', 'textbox': 'Text',
                   'dropdown': 'Text', 'table': 'StructuredValue', 'image': 'File'}
INSTRUMENT_CONTEXTS = ['instrument', 'microscope', 'device', 'system', '']
DATETIME_LABELS = ['acquisitiondate', 'datetimeoriginal', 'datetime']


def create_zarr_ro_crate(dest_path, sources=None, source_paths=(), metrics=None):
    crate = ZarrCrate()
    # the crate is written as the fusion ends
    fusion_time = to_iso_datetime(datetime.now())
    properties = {'name': get_filetitle(dest_path),
                  'description': f'OME-Zarr image fused by muvis-align from {len(source_paths)} sources',
                  'datePublished': fusion_time,
                  'encodingFormat': ZARR_ENCODING_FORMAT}
    dataset_entity = crate.add_dataset(dest_path='.', properties=properties)
    source_entities = [crate.add(ContextEntity(crate, to_crate_path(source_path, dest_path), {
        '@type': 'Dataset' if os.path.isdir(source_path) else 'File'})) for source_path in source_paths]
    if source_entities:
        dataset_entity['isBasedOn'] = source_entities
    if sources:
        add_data_capture(crate, dataset_entity, sources, source_entities)
    if metrics:
        dataset_entity['variableMeasured'] = [crate.add(ContextEntity(crate, f'#metric-{name}', {
            '@type': 'PropertyValue', 'name': name, 'value': value,
            'description': f'Registration {name}, over the overlaps of all registered pairs'}))
            for name, value in metrics.items()]

    # the zarr describes how it was made also once it is moved away from the output's workflow run crate
    fusion_properties = {'@type': 'CreateAction', 'name': 'Fusion to OME-Zarr', 'endTime': fusion_time,
                         'instrument': add_software(crate), 'result': dataset_entity}
    if source_entities:
        fusion_properties['object'] = source_entities
    fusion_entity = crate.add(ContextEntity(crate, '#fusion-001', fusion_properties))
    dataset_entity.append_to('mentions', fusion_entity)
    crate.write(dest_path)
    return crate


def add_data_capture(crate, dataset_entity, sources, source_entities):
    # the first and last sources stand for the acquisition: reading every source's metadata is too slow
    metadatas = [source.metadata for source in (sources[0], sources[-1])]

    action_properties = {'@type': 'CreateAction', 'name': 'Image acquisition'}
    instrument_entity = add_instrument(crate, metadatas[0])
    if instrument_entity:
        action_properties['instrument'] = instrument_entity
    # the microscope made the sources, muvis-align the zarr from them
    action_properties['result'] = source_entities or dataset_entity
    times = [to_iso_datetime(find_metadata_value(metadata, DATETIME_LABELS)) for metadata in metadatas]
    times = sorted(time for time in times if time)
    if times:
        action_properties['startTime'] = times[0]
        action_properties['endTime'] = times[-1]
    action_entity = crate.add(ContextEntity(crate, '#data-capture-001', action_properties))
    # listed on the root, so it can be found from there
    dataset_entity.append_to('mentions', action_entity)


def add_instrument(crate, metadata):
    """The instrument named in the metadata, identified by what names it; None if the metadata does not."""
    instrument_properties = {}
    manufacturer = find_metadata_value(metadata, ['manufacturer', 'make'], INSTRUMENT_CONTEXTS)
    if manufacturer:
        instrument_properties['manufacturer'] = str(manufacturer)
    model = find_metadata_value(metadata, ['model'], INSTRUMENT_CONTEXTS)
    if model:
        instrument_properties['name'] = str(model)
    serial = find_metadata_value(metadata, ['serialnumber', 'serial'], INSTRUMENT_CONTEXTS)
    if serial:
        # schema.org serialNumber is text
        instrument_properties['serialNumber'] = str(serial)
    if not instrument_properties:
        return None
    # the same instrument gets the same id in every crate
    label = re.sub(r'[^a-z0-9]+', '-', ' '.join(instrument_properties.values()).lower()).strip('-')
    return crate.add(ContextEntity(crate, f'#instrument-{label}', {'@type': 'IndividualProduct',
                                                                   **instrument_properties}))


def create_workflow_run_crate(dest_path, params_path, params, source_paths, result_paths,
                              start_time=None, end_time=None):
    crate = ROCrate()
    root = crate.root_dataset
    root['name'] = f'muvis-align run: {get_filetitle(os.path.normpath(dest_path))}'
    for profile_id, profile_name in WORKFLOW_RUN_PROFILES.items():
        profile = crate.add(ContextEntity(crate, profile_id, {'@type': ['CreativeWork', 'Profile'],
                                                              'name': profile_name, 'version': '0.5'}))
        root.append_to('conformsTo', profile)

    workflow_entity = add_workflow(crate)
    parameters = add_formal_parameters(crate, workflow_entity)

    objects = []
    params_entity = None
    if params_path:
        params_entity = crate.add_file(source=params_path, dest_path=to_crate_path(params_path, dest_path),
                                       properties={'name': 'Project parameters', 'encodingFormat': 'application/yaml'})
        objects.append(params_entity)
    for source_path in source_paths:
        objects.append(crate.add(ContextEntity(crate, to_crate_path(source_path, dest_path), {
            '@type': 'Dataset' if os.path.isdir(source_path) else 'File',
            'exampleOfWork': {'@id': INPUT_PARAMETER_ID}})))
    objects.extend(add_parameter_values(crate, params, params_entity, parameters))

    results = []
    for result_path in result_paths:
        result_properties = {'exampleOfWork': {'@id': OUTPUT_PARAMETER_ID}}
        crate_path = to_crate_path(result_path, dest_path)
        # the outputs are already in place: a dataset without source, or a file as its own source, is not copied
        if os.path.isdir(result_path):
            result_properties['encodingFormat'] = ZARR_ENCODING_FORMAT
            results.append(crate.add_dataset(dest_path=crate_path, properties=result_properties))
        else:
            results.append(crate.add_file(source=result_path, dest_path=crate_path, properties=result_properties))

    action_properties = {'name': 'muvis-align run'}
    if start_time:
        action_properties['startTime'] = to_iso_datetime(start_time)
    action_properties['endTime'] = to_iso_datetime(end_time or datetime.now())
    action_entity = crate.add_action(workflow_entity, identifier='#run-001', object=objects, result=results,
                                     properties=action_properties)
    root.append_to('mentions', action_entity)

    crate.write(dest_path)
    return crate


def add_software(crate):
    try:
        software_version = version('muvis-align')
    except PackageNotFoundError:
        software_version = None
    software_properties = {'@type': 'SoftwareApplication', 'name': 'muvis-align', 'url': MUVIS_ALIGN_URL}
    if software_version:
        software_properties['softwareVersion'] = software_version
    return crate.add(ContextEntity(crate, MUVIS_ALIGN_URL, software_properties))


def add_workflow(crate):
    software_entity = add_software(crate)

    language = crate.add(ComputerLanguage(crate, '#muvis-align-project', {
        'name': 'muvis-align project YAML',
        'url': {'@id': MUVIS_ALIGN_URL}}))
    # the template is the workflow's description, shipped with muvis-align: written from the package
    workflow_entity = crate.add_workflow(source=StringIO(get_project_template_text()),
                                         dest_path=os.path.basename(NAPARI_PROJECT_TEMPLATE),
                                         main=True, lang=language,
                                         properties={'name': 'muvis-align', 'encodingFormat': 'application/yaml'})
    workflow_entity['isPartOf'] = software_entity
    return workflow_entity


def add_formal_parameters(crate, workflow_entity):
    sections = get_section_dict(get_project_template(), ['inputs', 'parameters', 'outputs'])
    parameters = {}
    for section_id, items in sections.items():
        for item in items:
            additional_type = PARAMETER_TYPES.get(item['type'])
            if additional_type:
                identifier = f'#{section_id}.{item["name"]}'
                properties = {}
                if item.get('label'):
                    properties['alternateName'] = item['label']
                parameter = crate.add_formal_parameter(
                    name=item['name'], additionalType=additional_type, identifier=identifier,
                    description=item.get('description'), valueRequired=not item.get('optional', True),
                    defaultValue=to_property_value(item.get('default')), properties=properties)
                workflow_entity.append_to('output' if item['section_key'] == 'outputs' else 'input', parameter)
                parameters[identifier] = parameter
    return parameters


def add_parameter_values(crate, params, params_entity, parameters):
    values = []
    for name, value in flatten_params(params, parameters).items():
        properties = {'@type': 'PropertyValue', 'name': name, 'value': to_property_value(value)}
        if params_entity is not None:
            properties['subjectOf'] = params_entity
        parameter = parameters.get(f'#{name}')
        if parameter is not None:
            properties['exampleOfWork'] = parameter
        values.append(crate.add(ContextEntity(crate, f'#value-{name}', properties)))
    return values


def flatten_params(params, parameters, prefix=''):
    # a parameter's own dict value (a table) stays whole
    flat = {}
    for key, value in params.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict) and f'#{name}' not in parameters:
            flat.update(flatten_params(value, parameters, name + '.'))
        else:
            flat[name] = value
    return flat


def to_property_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return json.dumps(value, default=str)


def to_crate_path(path, crate_path):
    # paths outside the crate stay relative where possible, so the crate moves along with its data
    try:
        return Path(os.path.relpath(path, crate_path)).as_posix()
    except ValueError:
        return Path(os.path.abspath(path)).as_uri()


def to_iso_datetime(value):
    if not value:
        return None
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    text = str(value)
    try:
        # TIFF's DateTime tag
        return datetime.strptime(text, '%Y:%m:%d %H:%M:%S').isoformat()
    except ValueError:
        return text


def find_metadata_value(metadata, labels, contexts=('',)):
    """The first field named one of the labels, preferring each context in turn: a dict key on its path."""
    for context in contexts:
        for label in labels:
            value = find_field(metadata, label, context)
            if value is not None:
                return value
    return None


def find_field(metadata, label, context, path=''):
    if isinstance(metadata, list):
        for item in metadata:
            match = find_field(item, label, context, path)
            if match is not None:
                return match
    elif isinstance(metadata, dict):
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                match = find_field(value, label, context, f'{path}/{normalise_key(key)}')
                if match is not None:
                    return match
            elif normalise_key(key) == label and context in path and value not in (None, ''):
                return value
    return None


def normalise_key(key):
    # Serial Number, serial_number and SerialNumber name the same field
    return re.sub(r'[^a-z0-9]', '', str(key).lower())
