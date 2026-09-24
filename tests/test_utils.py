import os

import numpy as np
import pytest

from muvis_align.util import calculate_rigid_difference, create_transform, \
    pattern_base_dir, resolve_to_project_dir, relativize_to_project_dir, \
    find_sbemimage_meta_dir, to_posix_path, get_process_memory, print_memory_usage, timed_calls, \
    timed_module_functions, rolling_map, describe_live_buffers


@pytest.mark.parametrize(
    'transform1, transform2, expected',
    [
        (
            np.eye(3),
            create_transform((0, 0), 0, translation=[1, 2], matrix_size=3),
            create_transform((0, 0), 0, translation=[1, 2], matrix_size=3),
        ),
        (
            create_transform((0, 0), 0, translation=[-1, -2], matrix_size=3),
            np.eye(3),
            create_transform((0, 0), 0, translation=[1, 2], matrix_size=3),
        ),
        (
            np.eye(3),
            create_transform((0, 0), 10, translation=[0, 0], matrix_size=3),
            create_transform((0, 0), 10, translation=[0, 0], matrix_size=3),
        ),
        (
            create_transform((0, 0), -10, translation=[0, 0], matrix_size=3),
            np.eye(3),
            create_transform((0, 0), 10, translation=[0, 0], matrix_size=3),
        ),
        (
            create_transform((0, 0), 10, translation=[1, 2], matrix_size=3),
            create_transform((0, 0), 35, translation=[4, 6], matrix_size=3),
            create_transform((0, 0), 25, translation=[2.25983055, 4.46017555], matrix_size=3),
        ),
        (
            create_transform((0, 0), 15, translation=[1, 2, 3], matrix_size=4),
            create_transform((0, 0), 40, translation=[5, 7, 11], matrix_size=4),
            create_transform((0, 0), 25, translation=[2.56960808, 5.86490531, 8], matrix_size=4),
        ),
    ],
)
def test_calculate_rigid_difference(transform1, transform2, expected):
    """Calculate rigid differences for 2D and 3D affine transforms."""
    np.testing.assert_allclose(calculate_rigid_difference(transform1, transform2), expected)


def test_resolve_to_project_dir_joins_relative_path():
    base_dir = os.path.abspath('project')
    resolved = resolve_to_project_dir('data/input', base_dir)
    assert resolved == os.path.normpath(os.path.join(base_dir, 'data/input')).replace('\\', '/')


def test_resolve_to_project_dir_leaves_absolute_path_unchanged():
    absolute = os.path.abspath('somewhere/else')
    assert resolve_to_project_dir(absolute, os.path.abspath('project')) == absolute.replace('\\', '/')


def test_resolve_to_project_dir_handles_multiple_comma_separated_paths():
    base_dir = os.path.abspath('project')
    absolute = os.path.abspath('somewhere/else')
    resolved = resolve_to_project_dir(f'data/a, {absolute}', base_dir)
    expected_joined = os.path.normpath(os.path.join(base_dir, "data/a")).replace('\\', '/')
    assert resolved == f'{expected_joined}, {absolute.replace(chr(92), "/")}'


def test_resolve_to_project_dir_always_returns_forward_slashes():
    """Even on Windows, os.path.join()/normpath() naturally produce backslashes - the result
    must be normalised to forward slashes so the same value is safe to show in the UI and to
    store in the (OS-portable) project file."""
    base_dir = os.path.abspath('project')
    resolved = resolve_to_project_dir('data/input', base_dir)
    assert '\\' not in resolved


@pytest.mark.parametrize('path, base_dir', [('', os.path.abspath('project')), ('data/input', None)])
def test_resolve_to_project_dir_no_op_without_path_or_base_dir(path, base_dir):
    assert resolve_to_project_dir(path, base_dir) == path


def test_relativize_to_project_dir_converts_absolute_path_under_base_dir():
    base_dir = os.path.abspath('project')
    absolute = os.path.join(base_dir, 'data', 'input')
    assert relativize_to_project_dir(absolute, base_dir) == 'data/input'


def test_relativize_to_project_dir_leaves_relative_path_unchanged():
    assert relativize_to_project_dir('data/input', os.path.abspath('project')) == 'data/input'


def test_relativize_to_project_dir_round_trips_with_resolve_to_project_dir():
    """The pair together must be idempotent: display-resolving a stored relative path and then
    relativizing the (now absolute) value the widget reports back must reproduce the original -
    otherwise every project load would silently rewrite the project file's paths to absolute."""
    base_dir = os.path.abspath('project')
    original = 'data/input'

    resolved = resolve_to_project_dir(original, base_dir)
    round_tripped = relativize_to_project_dir(resolved, base_dir)

    assert round_tripped == original


def test_relativize_to_project_dir_falls_back_to_absolute_on_different_drive(monkeypatch):
    def raise_value_error(path, start):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(os.path, 'relpath', raise_value_error)

    absolute = os.path.abspath('somewhere/else')
    assert relativize_to_project_dir(absolute, os.path.abspath('project')) == absolute.replace('\\', '/')


@pytest.mark.parametrize('pattern, expected', [
    ('data/tiles/*.tiff', 'data/tiles'),
    ('data/*/*.tiff', 'data'),          # a wildcard directory is not a directory
    ('data/**/*.ome.zarr', 'data'),
    ('data/tile_?.tif', 'data'),
    ('data/set[0-9]/*.tif', 'data'),
    ('data/file.tiff', 'data'),
    ('*/*.tiff', ''),
    ('file.tiff', ''),
])
def test_pattern_base_dir_skips_wildcard_components(pattern, expected):
    """A relative output path is taken relative to this (MVSRegistration.init), so a wildcard
    left in it makes an output directory that cannot be created - on Windows, WinError 123."""
    assert pattern_base_dir(pattern) == expected


def test_find_sbemimage_meta_dir_walks_up_to_a_deeply_nested_project_root(tmp_path):
    (tmp_path / 'meta').mkdir()
    tile_dir = tmp_path / 'tiles' / 'r0004' / 't0000'
    tile_dir.mkdir(parents=True)
    filename = str(tile_dir / 'sample_r0004_t0000_s00823.ome.tif')

    assert find_sbemimage_meta_dir(filename) == os.path.join(str(tile_dir), '..', '..', '..', 'meta')


def test_find_sbemimage_meta_dir_returns_none_when_not_found(tmp_path):
    filename = str(tmp_path / 'subset' / 'sample.ome.tif')

    assert find_sbemimage_meta_dir(filename) is None


@pytest.mark.parametrize('path, expected', [
    ('C:\\proj\\data', 'C:/proj/data'),
    ('data\\input\\', 'data/input/'),
    ('data/input/', 'data/input/'),
    ('C:/proj/data', 'C:/proj/data'),
    ('', ''),
])
def test_to_posix_path_converts_separators_and_keeps_a_trailing_one(path, expected):
    assert to_posix_path(path) == expected


@pytest.mark.parametrize('value', [None, 3, ['a\\b']])
def test_to_posix_path_passes_non_strings_through(value):
    # a path param can hold a list (a comma-separated input_path, once eval_path has split it)
    assert to_posix_path(value) is value


def test_resolve_to_project_dir_normalises_separators_without_a_base_dir():
    # nothing to resolve against (an unsaved project), but the separators are still ours
    assert resolve_to_project_dir('C:\\proj\\data', None) == 'C:/proj/data'


def test_relativize_to_project_dir_normalises_separators_without_a_base_dir():
    assert relativize_to_project_dir('C:\\proj\\data', None) == 'C:/proj/data'


def test_process_memory_tracks_an_allocation_or_says_it_cannot():
    """Memory reporting is what a killed run leaves behind, so it has to work where the run
    happens (Linux) and be harmless where it does not - never raising, never guessing.
    """
    before = get_process_memory()
    assert len(before) == 2

    if before[0] is None and before[1] is None:
        # a platform with neither /proc nor resource nor the Windows API: say nothing, quietly
        assert print_memory_usage() == ''
        return

    block = np.ones(200 * 1024 * 1024 // 8)
    after = get_process_memory()
    try:
        if before[0] is not None:
            assert after[0] - before[0] > 100 * 1024 ** 2
        if before[1] is not None:
            # peak is the figure an OOM post-mortem needs, so it must not fall back
            assert after[1] >= after[0] if after[0] is not None else True
            assert after[1] >= before[1]
    finally:
        del block
    assert 'rss' in print_memory_usage() or 'peak' in print_memory_usage()


def test_timed_calls_records_each_call_and_keeps_the_signature():
    from dask.utils import has_keyword

    def register(fixed_data, moving_data, scale=1):
        return fixed_data * scale

    times, cpu_times = [], []
    timed = timed_calls(register, times, cpu_times)
    assert timed(2, 3, scale=4) == 8
    assert timed(1, 1) == 1
    assert len(times) == len(cpu_times) == 2
    assert all(time_ >= 0 for time_ in times + cpu_times)
    # multiview_stitcher picks how to call a registration function by its keywords
    assert has_keyword(timed, 'fixed_data') and has_keyword(timed, 'moving_data')


def test_timed_module_functions_times_calls_and_restores():
    import types
    module = types.SimpleNamespace(score=lambda value: value + 1)
    original = module.score
    with timed_module_functions(module, ['score']) as cpu_times:
        assert module.score(1) == 2
        module.score(2)
    assert len(cpu_times['score']) == 2
    assert module.score is original


def test_rolling_map_yields_every_item_with_a_bounded_number_submitted():
    import threading
    import time
    lock = threading.Lock()
    started = []
    consumed = []

    def work(item):
        with lock:
            started.append(item)
        time.sleep(0.01 * (item % 3))
        return item * 10

    for item, result in rolling_map(work, range(40), workers=4):
        assert result == item * 10
        consumed.append(item)
        # never more than 2x workers started ahead of what has been handed back
        assert len(started) <= len(consumed) + 2 * 4
    assert sorted(consumed) == list(range(40))


def test_rolling_map_raises_and_skips_what_was_still_queued():
    import time
    started = []

    def work(item):
        started.append(item)
        if item == 0:
            raise ValueError('pair failed')
        time.sleep(0.05)
        return item

    with pytest.raises(ValueError, match='pair failed'):
        list(rolling_map(work, range(100), workers=2))
    assert len(started) < 10


class _TileKeeper:
    def __init__(self):
        # a dict of arrays only is not tracked by the garbage collector itself
        self.tiles = {index: np.zeros(3 << 20, np.uint8) for index in range(4)}
        self.views = [tile[::2] for tile in self.tiles.values()]


def test_describe_live_buffers_names_the_holder_and_counts_a_view_once():
    keeper = _TileKeeper()

    line = describe_live_buffers(min_bytes=2 << 20)

    # 3.14 tracks a dict of arrays itself, and names its owner as a referrer instead
    assert ('_TileKeeper>dict x4 12.0MB' in line
            or 'dict x4 12.0MB (held by' in line and '_TileKeeper' in line)
    # the views share their tiles' memory, so they add nothing
    assert '_TileKeeper>list' not in line
    del keeper
    assert '_TileKeeper' not in describe_live_buffers(min_bytes=2 << 20)
