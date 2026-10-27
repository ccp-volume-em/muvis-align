import fnmatch
import json
import logging
import os
import posixpath
import urllib.request

import napari
import platformdirs
import yaml

from muvis_align.file.project_yaml import read_params
from muvis_align.util import path_param_to_text

REPO_TREE_URL = 'https://api.github.com/repos/ccp-volume-em/muvis-align/git/trees/main?recursive=1'
REPO_FILE_URL = 'https://raw.githubusercontent.com/ccp-volume-em/muvis-align/main/'


def get_reader(path):
    if isinstance(path, str) and (path.startswith(REPO_FILE_URL) or is_project_file(path)):
        return read_project
    return None


def is_project_file(path):
    try:
        params = read_params(path)
    except Exception:
        return False
    return isinstance(params, dict) and 'input_output' in params


def read_project(path):
    if path.startswith(REPO_FILE_URL):
        path = get_sample(path.removeprefix(REPO_FILE_URL))
    viewer = napari.current_viewer()
    _, widget = viewer.window.add_plugin_dock_widget('muvis-align')
    # as choosing the file on the project tab: its settings load, sources are only read on Process
    widget.interface.param_widgets['project.project_path'].set_value(path)
    return [(None,)]


def get_sample(project_file):
    cache_dir = platformdirs.user_cache_dir('muvis-align', appauthor=False)
    project_path = os.path.join(cache_dir, project_file)
    if not os.path.exists(project_path):
        download_sample(project_file, cache_dir)
    return project_path


def download(url, attempts=20):
    # some networks reset many new HTTPS connections from Python (seen at the Crick)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, ConnectionError) as error:
            logging.info(f'Download of {url} failed ({error}), attempt {attempt + 1}/{attempts}')
            if attempt == attempts - 1:
                raise


def download_sample(project_file, cache_dir):
    repo_paths = [item['path'] for item in json.loads(download(REPO_TREE_URL))['tree'] if item['type'] == 'blob']
    project_text = download(REPO_FILE_URL + project_file)
    input_path = path_param_to_text(yaml.safe_load(project_text)['input_output']['input_path'])
    project_dir = posixpath.dirname(project_file)
    patterns = [posixpath.normpath(posixpath.join(project_dir, part.strip())).split('/')
                for part in input_path.split(',')]
    input_paths = [repo_path for repo_path in repo_paths if is_in_input(repo_path, patterns)]
    if not input_paths:
        # nothing is cached, so a corrected input_path on GitHub is picked up next time
        raise FileNotFoundError(f'No data files on GitHub match the input path {input_path} of {project_file}')
    logging.info(f'Downloading sample {project_file}: {len(input_paths)} files')
    for repo_path in input_paths:
        local_path = os.path.join(cache_dir, repo_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as file:
            file.write(download(REPO_FILE_URL + repo_path))
    # written last: a cached project file means its data downloaded completely
    project_path = os.path.join(cache_dir, project_file)
    os.makedirs(os.path.dirname(project_path), exist_ok=True)
    with open(project_path, 'wb') as file:
        file.write(project_text)


def is_in_input(repo_path, patterns):
    # a file inside a folder the glob matches (e.g. a zarr chunk) belongs to the input too
    parts = repo_path.split('/')
    return any(len(parts) >= len(pattern)
               and all(fnmatch.fnmatchcase(part, pattern_part) for part, pattern_part in zip(parts, pattern))
               for pattern in patterns)
