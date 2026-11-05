import json
import os
import warnings

import yaml

from muvis_align.MVSRegistration import MVSRegistration
from muvis_align.constants import MUVIS_ALIGN_URL


class StubSource:
    def __init__(self, metadata):
        self.metadata = metadata


def make_registration(tmp_path, metrics=None):
    output_dir = tmp_path / 'output'
    store = output_dir / 'registered.ome.zarr'
    store.mkdir(parents=True)
    (store / 'zarr.json').write_text(json.dumps({'zarr_format': 3, 'node_type': 'group',
                                                 'attributes': {'ome': {'version': '0.5'}}}))
    (output_dir / 'mappings.json').write_text('{}')
    reg = MVSRegistration()
    reg.output = str(output_dir) + '/'
    reg.output_params = {}
    reg.filenames = [str(tmp_path / f'tile{index}.tiff') for index in range(3)]
    reg.sources = [StubSource({'index': index}) for index in range(3)]
    reg.start_time = None
    reg.summary_metrics = lambda: metrics or {}
    return reg, output_dir


def test_write_ro_crates_passes_the_run_to_fair_metadata_integration(tmp_path, monkeypatch):
    import muvis_align.MVSRegistration as registration_module
    calls = {}
    monkeypatch.setattr(registration_module, 'create_zarr_ro_crate',
                        lambda path, **kwargs: calls.setdefault('zarr', []).append((path, kwargs)))
    monkeypatch.setattr(registration_module, 'create_ro_crate',
                        lambda *args, **kwargs: calls.setdefault('run', []).append((args, kwargs)))
    reg, output_dir = make_registration(tmp_path, metrics={'ncc': 0.75})
    params = {'general': {'verbose': True}, 'registration': {'method': 'orb'}}

    reg.write_ro_crates(['registered.ome.zarr', 'registered.ome.tiff'], params)

    # only the zarr gets its own crate
    [(zarr_path, zarr_kwargs)] = calls['zarr']
    assert zarr_path == reg.output + 'registered.ome.zarr'
    assert zarr_kwargs['source_paths'] == reg.filenames
    assert zarr_kwargs['acquisition_metadata'] == [{'index': 0}, {'index': 2}]
    assert zarr_kwargs['measurements'] == {'ncc': {
        'value': 0.75, 'description': 'Registration ncc, over the overlaps of all registered pairs'}}
    assert zarr_kwargs['description'] == 'OME-Zarr image fused by muvis-align from 3 sources'
    derivation = zarr_kwargs['derivation']
    assert derivation['name'] == 'Fusion to OME-Zarr'
    assert derivation['software']['name'] == 'muvis-align' and derivation['software']['url'] == MUVIS_ALIGN_URL

    [(run_args, run_kwargs)] = calls['run']
    dest, template_path, input_paths, result_paths = run_args
    assert dest == str(output_dir)
    assert template_path.endswith('project_template.yaml')
    assert input_paths == reg.filenames
    assert result_paths == [reg.output + 'registered.ome.zarr', reg.output + 'registered.ome.tiff',
                            reg.output + 'mappings.json']
    assert run_kwargs['parameters'] is params
    assert run_kwargs['name'] == 'muvis-align run: output'
    assert run_kwargs['end_time'] == derivation['end_time']
    assert run_kwargs['software'] == derivation['software']


def test_project_parameters_fit_the_template(tmp_path):
    """The crates are written from muvis-align's own template and a project's values, all kept, without warning
    about any value the template has."""
    reg, output_dir = make_registration(tmp_path)
    with open(os.path.join(os.path.dirname(__file__), 'muvis_align_project.yml'), encoding='utf-8') as file:
        params = yaml.safe_load(file)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        reg.write_ro_crates(['registered.ome.zarr'], params)

    assert yaml.safe_load((output_dir / 'parameters.yaml').read_text(encoding='utf-8')) == params
    with open(output_dir / 'ro-crate-metadata.json', encoding='utf-8') as file:
        graph = {entity['@id']: entity for entity in json.load(file)['@graph']}
    assert graph['./']['mainEntity'] == {'@id': 'project_template.yaml'}
    assert graph['parameters.yaml']['about'] == {'@id': '#run'}
    assert (output_dir / 'registered.ome.zarr' / 'ro-crate-metadata.json').exists()
