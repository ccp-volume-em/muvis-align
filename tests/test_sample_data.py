import json
import os
from unittest.mock import MagicMock, patch

import pytest

from muvis_align import _sample_data
from muvis_align._sample_data import REPO_FILE_URL, REPO_TREE_URL

REPO_FILES = [
    'data/S000/S000_000.zarr/zarr.json',
    'data/S000/S000_000.zarr/0/c/0/0',
    'data/S000/notes.txt',
    'data/S001/S001_000.zarr/zarr.json',
    'data/3d/tile.ome.zarr/zarr.json',
    'tests/muvis_align_project.yml',
]


def fake_download(input_path):
    project_text = f'input_output:\n  input_path: {input_path}\n'.encode()
    tree = {'tree': [{'path': path, 'type': 'blob'} for path in REPO_FILES] + [{'path': 'data', 'type': 'tree'}]}

    def download(url):
        if url == REPO_TREE_URL:
            return json.dumps(tree).encode()
        if url == REPO_FILE_URL + 'tests/muvis_align_project.yml':
            return project_text
        return url.removeprefix(REPO_FILE_URL).encode()
    return download


def cached_files(cache_dir):
    return sorted(os.path.relpath(os.path.join(root, name), cache_dir).replace(os.sep, '/')
                  for root, _, names in os.walk(cache_dir) for name in names)


@pytest.mark.parametrize('input_path, expected', [
    ('../data/S000/*.zarr', ['data/S000/S000_000.zarr/0/c/0/0', 'data/S000/S000_000.zarr/zarr.json']),
    ('../data/S*/*.zarr', ['data/S000/S000_000.zarr/0/c/0/0', 'data/S000/S000_000.zarr/zarr.json',
                           'data/S001/S001_000.zarr/zarr.json']),
    ('../data/S000/*.zarr, ../data/3d/*.ome.zarr', ['data/3d/tile.ome.zarr/zarr.json',
                                                   'data/S000/S000_000.zarr/0/c/0/0',
                                                   'data/S000/S000_000.zarr/zarr.json']),
])
def test_download_sample_takes_the_project_and_its_input_files(tmp_path, input_path, expected):
    with patch.object(_sample_data, 'download', side_effect=fake_download(input_path)):
        _sample_data.download_sample('tests/muvis_align_project.yml', str(tmp_path))
    assert cached_files(tmp_path) == expected + ['tests/muvis_align_project.yml']
    assert (tmp_path / expected[0]).read_bytes() == expected[0].encode()


def test_download_sample_without_matching_data_caches_nothing(tmp_path):
    with patch.object(_sample_data, 'download', side_effect=fake_download('data/S000/*.zarr')):
        with pytest.raises(FileNotFoundError, match='data/S000/\\*.zarr'):
            _sample_data.download_sample('tests/muvis_align_project.yml', str(tmp_path))
    assert cached_files(tmp_path) == []


def test_get_sample_downloads_only_when_not_cached(tmp_path):
    cache_dir = MagicMock(user_cache_dir=MagicMock(return_value=str(tmp_path)))
    with patch.object(_sample_data, 'platformdirs', cache_dir), \
            patch.object(_sample_data, 'download_sample') as download_sample:
        project_path = _sample_data.get_sample('tests/muvis_align_project.yml')
        download_sample.assert_called_once_with('tests/muvis_align_project.yml', str(tmp_path))
        (tmp_path / 'tests').mkdir()
        (tmp_path / 'tests' / 'muvis_align_project.yml').write_text('input_output: {}\n')
        assert _sample_data.get_sample('tests/muvis_align_project.yml') == project_path
        download_sample.assert_called_once()
    assert project_path == os.path.join(str(tmp_path), 'tests/muvis_align_project.yml')


def test_download_retries_a_reset_connection():
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b'data'
    with patch.object(_sample_data.urllib.request, 'urlopen',
                      side_effect=[ConnectionResetError(), ConnectionResetError(), response]) as urlopen:
        assert _sample_data.download('https://example.org/file') == b'data'
    assert urlopen.call_count == 3
    with patch.object(_sample_data.urllib.request, 'urlopen', side_effect=ConnectionResetError()):
        with pytest.raises(ConnectionResetError):
            _sample_data.download('https://example.org/file', attempts=2)


def test_get_reader_accepts_project_files_and_sample_urls_only(tmp_path):
    other_yaml = tmp_path / 'other.yml'
    other_yaml.write_text('repos: []\n')
    project = os.path.join(os.path.dirname(__file__), 'muvis_align_project.yml')
    assert _sample_data.get_reader(project) is _sample_data.read_project
    assert _sample_data.get_reader(REPO_FILE_URL + 'tests/muvis_align_project.yml') is _sample_data.read_project
    assert _sample_data.get_reader(str(other_yaml)) is None
    assert _sample_data.get_reader(str(tmp_path / 'missing.yml')) is None
    assert _sample_data.get_reader([project]) is None


@pytest.mark.parametrize('path, local_path', [
    ('C:/projects/muvis_align_project.yml', 'C:/projects/muvis_align_project.yml'),
    (REPO_FILE_URL + 'tests/muvis_align_project3d.yml', 'cache/tests/muvis_align_project3d.yml'),
])
def test_read_project_sets_the_project_path_without_reading_sources(path, local_path):
    viewer = MagicMock()
    widget = MagicMock()
    viewer.window.add_plugin_dock_widget.return_value = (MagicMock(), widget)
    with patch.object(_sample_data.napari, 'current_viewer', return_value=viewer), \
            patch.object(_sample_data, 'get_sample', side_effect=lambda project_file: 'cache/' + project_file):
        assert _sample_data.read_project(path) == [(None,)]
    viewer.window.add_plugin_dock_widget.assert_called_once_with('muvis-align')
    widget.interface.param_widgets['project.project_path'].set_value.assert_called_once_with(local_path)
    widget.interface.input_output_process.assert_not_called()
