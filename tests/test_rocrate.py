import json
import os

import pytest

from muvis_align.MVSRegistration import MVSRegistration
from muvis_align.file.rocrate_utils import (create_workflow_run_crate, create_zarr_ro_crate, find_metadata_value,
                                            flatten_params)


class StubSource:
    def __init__(self, metadata):
        self.metadata = metadata


OME_METADATA = {'Instrument': {'Microscope': {'Manufacturer': 'Zeiss', 'Model': 'Crossbeam', 'SerialNumber': 1234}},
                'Image': {'Name': 'tile', 'AcquisitionDate': '2026-01-02T03:04:05'}}
TIFF_METADATA = {'Make': 'FEI', 'Model': 'Helios', 'DateTime': '2026:01:02 04:00:00'}


def read_graph(crate_dir):
    with open(os.path.join(crate_dir, 'ro-crate-metadata.json'), encoding='utf-8') as file:
        return {entity['@id']: entity for entity in json.load(file)['@graph']}


def test_zarr_crate_data_capture(tmp_path):
    zarr_path = tmp_path / 'output' / 'registered.ome.zarr'
    zarr_path.mkdir(parents=True)
    source_paths = [str(tmp_path / 'tile1.tiff'), str(tmp_path / 'tile2.tiff')]
    create_zarr_ro_crate(str(zarr_path), [StubSource(OME_METADATA), StubSource(TIFF_METADATA)], source_paths,
                         metrics={'ncc': 0.75, 'quality': 0.5})
    graph = read_graph(zarr_path)

    sources = [{'@id': '../../tile1.tiff'}, {'@id': '../../tile2.tiff'}]
    root = graph['./']
    assert root['name'] == 'registered'
    assert 'application/vnd.zarr' in root['encodingFormat']
    assert root['isBasedOn'] == sources
    assert root['description'] == 'OME-Zarr image fused by muvis-align from 2 sources'
    assert root['datePublished']
    assert root['mentions'] == [{'@id': '#data-capture-001'}, {'@id': '#fusion-001'}]
    action = graph['#data-capture-001']
    assert action['@type'] == 'CreateAction'
    assert action['instrument'] == {'@id': '#instrument-zeiss-crossbeam-1234'}
    assert action['result'] == sources
    assert action['startTime'] == '2026-01-02T03:04:05'
    assert action['endTime'] == '2026-01-02T04:00:00'
    assert graph['#instrument-zeiss-crossbeam-1234'] == {
        '@id': '#instrument-zeiss-crossbeam-1234', '@type': 'IndividualProduct',
        'manufacturer': 'Zeiss', 'name': 'Crossbeam', 'serialNumber': '1234'}
    assert graph['../../tile1.tiff']['@type'] == 'File'
    fusion = graph['#fusion-001']
    assert fusion['instrument'] == {'@id': 'https://github.com/folterj/muvis-align'}
    assert fusion['object'] == sources
    assert fusion['result'] == {'@id': './'}
    assert fusion['endTime'] == root['datePublished']
    assert graph['https://github.com/folterj/muvis-align']['@type'] == 'SoftwareApplication'
    assert root['variableMeasured'] == [{'@id': '#metric-ncc'}, {'@id': '#metric-quality'}]
    assert graph['#metric-ncc']['@type'] == 'PropertyValue'
    assert graph['#metric-ncc']['value'] == 0.75
    assert graph['#metric-quality']['value'] == 0.5


def test_zarr_crate_without_instrument_metadata(tmp_path):
    zarr_path = tmp_path / 'registered.ome.zarr'
    zarr_path.mkdir()
    create_zarr_ro_crate(str(zarr_path), [StubSource({})])
    graph = read_graph(zarr_path)

    action = graph['#data-capture-001']
    assert 'instrument' not in action
    assert 'variableMeasured' not in graph['./']
    assert action['result'] == {'@id': './'}
    assert not [entity for entity in graph.values() if entity.get('@type') == 'IndividualProduct']


def test_find_metadata_value_prefers_context():
    metadata = {'Detector': {'Model': 'camera'}, 'Microscope': {'Model': 'scope'}}
    assert find_metadata_value(metadata, ['model'], ['microscope', '']) == 'scope'
    assert find_metadata_value(metadata, ['model'], ['instrument']) is None
    assert find_metadata_value(TIFF_METADATA, ['manufacturer', 'make'], ['microscope', '']) == 'FEI'
    # keys differing only in case, spaces or separators
    metadata = {'Microscope Info': {'Serial_Number': 7}}
    assert find_metadata_value(metadata, ['serialnumber'], ['microscope']) == 7


def test_flatten_params_keeps_parameter_tables():
    params = {'input_output': {'channels_table': {'label': ['0']}, 'overwrite': True},
              'general': {'logging': {'verbose': True}}}
    flat = flatten_params(params, {'#input_output.channels_table': None})
    assert flat == {'input_output.channels_table': {'label': ['0']}, 'input_output.overwrite': True,
                    'general.logging.verbose': True}


@pytest.mark.parametrize('params', [
    {'registration': {'method': 'orb', 'unknown_param': 3}},                  # project (UI) format
    {'general': {'overwrite': True}, 'registration': {'method': 'orb', 'unknown_param': 3}},   # CLI format
])
def test_workflow_run_crate(tmp_path, params):
    source_dir = tmp_path / 'data'
    source_dir.mkdir()
    source_paths = [str(source_dir / f'tile{index}.tiff') for index in range(2)]
    output_dir = tmp_path / 'data' / 'output'
    output_dir.mkdir()
    zarr_path = output_dir / 'registered.ome.zarr'
    zarr_path.mkdir()
    mappings_path = output_dir / 'mappings.json'
    mappings_path.write_text('{}')
    params_path = output_dir / 'project.yml'
    params_path.write_text('registration:\n  method: orb\n')

    create_workflow_run_crate(str(output_dir), str(params_path), params, source_paths,
                              [str(zarr_path), str(mappings_path)])
    graph = read_graph(output_dir)

    root = graph['./']
    assert {'@id': 'https://w3id.org/ro/wfrun/workflow/0.5'} in root['conformsTo']
    assert root['mainEntity'] == {'@id': 'project_template.yaml'}
    assert root['mentions'] == [{'@id': '#run-001'}]
    workflow = graph['project_template.yaml']
    assert 'ComputationalWorkflow' in workflow['@type']
    assert workflow['programmingLanguage'] == {'@id': '#muvis-align-project'}
    assert {'@id': '#registration.method'} in workflow['input']
    assert {'@id': '#input_output.output_path'} in workflow['output']
    assert (output_dir / 'project_template.yaml').exists()

    action = graph['#run-001']
    assert action['instrument'] == {'@id': 'project_template.yaml'}
    objects = [item['@id'] for item in action['object']]
    assert objects[:3] == ['project.yml', '../tile0.tiff', '../tile1.tiff']
    assert graph['../tile0.tiff']['exampleOfWork'] == {'@id': '#input_output.input_path'}
    assert [item['@id'] for item in action['result']] == ['registered.ome.zarr/', 'mappings.json']
    assert graph['registered.ome.zarr/']['exampleOfWork'] == {'@id': '#input_output.output_path'}

    method_value = graph['#value-registration.method']
    assert method_value['value'] == 'orb'
    assert method_value['subjectOf'] == {'@id': 'project.yml'}
    assert method_value['exampleOfWork'] == {'@id': '#registration.method'}
    assert 'exampleOfWork' not in graph['#value-registration.unknown_param']


def test_write_ro_crates_copies_params_to_output(tmp_path):
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    (output_dir / 'registered.ome.zarr').mkdir()
    params_path = tmp_path / 'params.yml'
    params_path.write_text('general: {}\n')

    reg = MVSRegistration()
    reg.output = str(output_dir) + '/'
    reg.output_params = {}
    reg.filenames = [str(tmp_path / 'tile.tiff')]
    reg.sources = [StubSource(TIFF_METADATA)]
    reg.start_time = None
    reg.write_ro_crates(['registered.ome.zarr'], str(params_path), {'general': {}})

    assert (output_dir / 'params.yml').read_text() == 'general: {}\n'
    assert read_graph(output_dir)['params.yml']['name'] == 'Project parameters'
    assert read_graph(output_dir / 'registered.ome.zarr')['#instrument-fei-helios']['manufacturer'] == 'FEI'
