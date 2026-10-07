import logging
import os
import subprocess
import sys

import numpy as np
import pytest

from muvis_align.util import calculate_rigid_difference, create_transform, \
    pattern_base_dir, resolve_to_project_dir, relativize_to_project_dir, \
    find_sbemimage_meta_dir, to_posix_path, get_process_memory, print_memory_usage, timed_calls, \
    timed_module_functions, rolling_map, get_filetitle, find_labelled_numbers, print_dict_simple, print_significants, \
    eval_context, convert_to_um, get_unique_file_labels, strip_common_path_prefix, format_phase_timing, \
    path_param_to_text


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


BASE_DIR = os.path.abspath('project')
ELSEWHERE = os.path.abspath('somewhere/else')
JOINED = os.path.normpath(os.path.join(BASE_DIR, 'data/input')).replace(os.sep, '/')


@pytest.mark.parametrize('path, base_dir, expected', [
    ('data/input', BASE_DIR, JOINED),
    (ELSEWHERE, BASE_DIR, ELSEWHERE.replace(os.sep, '/')),
    (f'data/input, {ELSEWHERE}', BASE_DIR, f"{JOINED}, {ELSEWHERE.replace(os.sep, '/')}"),
    ('', BASE_DIR, ''),
    ('data/input', None, 'data/input'),
    # nothing to resolve against (an unsaved project), but the separators are still ours
    ('C:\\proj\\data', None, 'C:/proj/data'),
])
def test_resolve_to_project_dir(path, base_dir, expected):
    assert resolve_to_project_dir(path, base_dir) == expected


@pytest.mark.parametrize('path, base_dir, expected', [
    (os.path.join(BASE_DIR, 'data', 'input'), BASE_DIR, 'data/input'),
    ('data/input', BASE_DIR, 'data/input'),
    ('C:\\proj\\data', None, 'C:/proj/data'),
])
def test_relativize_to_project_dir(path, base_dir, expected):
    assert relativize_to_project_dir(path, base_dir) == expected


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


def _get_pairs_every_pair(positions, sizes):
    """get_pairs() as it was, testing every pair - the reference its candidates must reproduce."""
    import math
    pairs, angles = [], []
    z_positions = [position['z'] for position in positions if 'z' in position]
    ordered_z = sorted(set(z_positions))
    is_mixed_3dstack = len(ordered_z) < len(z_positions)
    for first, second in np.transpose(np.triu_indices(len(positions), 1)):
        posi, posj, sizei, sizej = positions[first], positions[second], sizes[first], sizes[second]
        if is_mixed_3dstack:
            distance = math.dist([posi[dim] for dim in 'xy'], [posj[dim] for dim in 'xy'])
            min_distance = max([size[dim] for size in [sizei, sizej] for dim in 'xy'])
            if abs(ordered_z.index(posi['z']) - ordered_z.index(posj['z'])) > 1:
                min_distance = 0
            elif posi['z'] != posj['z']:
                min_distance *= 0.8
        else:
            distance = math.dist(posi.values(), posj.values())
            min_distance = max(list(sizei.values()) + list(sizej.values()))
        if distance < min_distance:
            pairs.append((int(first), int(second)))
            vector = np.array(list(posi.values())) - np.array(list(posj.values()))
            angle = math.degrees(math.atan2(vector[1], vector[0]))
            if distance < min(list(sizei.values()) + list(sizej.values())):
                angle += 90
            while angle < -90:
                angle += 180
            while angle > 90:
                angle -= 180
            angles.append(angle)
    return pairs, angles


def _tile_stack(sections, rows=4, columns=5, with_overview=True, seed=0):
    """Overlapping tiles jittered off a grid in several sections, each with a much larger overview."""
    rng = np.random.default_rng(seed)
    positions, sizes = [], []
    for section in range(sections):
        for row in range(rows):
            for column in range(columns):
                positions.append({'y': row * 90 + rng.uniform(-8, 8), 'x': column * 90 + rng.uniform(-8, 8),
                                  'z': section * 0.05})
                sizes.append({'y': 100.0, 'x': 120.0})
        if with_overview:
            positions.append({'y': 150.0, 'x': 200.0, 'z': section * 0.05})
            sizes.append({'y': 900.0, 'x': 1100.0})
    return positions, sizes


@pytest.mark.parametrize('positions, sizes', [
    _tile_stack(sections=5),
    _tile_stack(sections=1, with_overview=False),
    # every z different: not a mixed stack, so distances are over all dims
    ([{'y': y * 90.0, 'x': x * 90.0, 'z': float(y * 7 + x)} for y in range(4) for x in range(7)],
     [{'y': 100.0, 'x': 100.0}] * 28),
    # 2D, no z at all
    ([{'y': y * 95.0, 'x': x * 80.0} for y in range(5) for x in range(6)], [{'y': 100.0, 'x': 110.0}] * 30),
])
def test_get_pairs_matches_testing_every_pair(positions, sizes):
    from muvis_align.util import get_pairs

    assert get_pairs(positions, sizes) == _get_pairs_every_pair(positions, sizes)
    assert get_pairs(positions, sizes)[0]  # the cases do have pairs


def test_get_pairs_scales_with_neighbours_not_all_pairs():
    from muvis_align.util import _get_pairs_candidates

    def candidate_count(sections):
        positions, sizes = _tile_stack(sections=sections)
        z_index = {z_value: index for index, z_value in enumerate(sorted({position['z'] for position in positions}))}
        return len(_get_pairs_candidates(positions, sizes, True, z_index))

    # twice the sections, about twice the candidates: all pairs would be four times as many
    assert candidate_count(40) < 2.2 * candidate_count(20)


@pytest.mark.parametrize('value, expected', [
    (2, 2), (2.5, 2.5), ('2', 2), (' 16 ', 16), ('0.5', 0.5), (None, 1), ('', 1),
    ('10um', '10um'), (' 0.5 mm ', '0.5 mm'), ('250nm', '250nm'), ('1e-3mm', '1e-3mm'),
])
def test_parse_scale_takes_a_factor_or_a_pixel_size(value, expected):
    from muvis_align.util import parse_scale

    assert parse_scale(value) == expected


@pytest.mark.parametrize('text, um', [('10um', 10), ('0.5 mm', 500), ('250nm', 0.25), ('1e-3mm', 1), ('2µm', 2)])
def test_pixel_size_to_um(text, um):
    from muvis_align.util import pixel_size_to_um

    assert pixel_size_to_um(text) == pytest.approx(um)


@pytest.mark.parametrize('value', ['ten', '10 parsecs', 'um10'])
def test_parse_scale_rejects_what_is_neither(value):
    from muvis_align.util import parse_scale

    with pytest.raises(ValueError, match='neither a downscale factor nor a pixel size'):
        parse_scale(value)


@pytest.mark.parametrize('filename, expected', [
    ('data/slide_one.ome.tiff', 'slide_one'),
    ('data/tile_home.tif', 'tile_home'),
    ('data/scan.ome.zarr', 'scan'),
])
def test_get_filetitle_strips_only_the_ome_suffix(filename, expected):
    assert get_filetitle(filename) == expected


def test_rolling_map_stops_submitting_once_cancelled():
    """A cancel stops the map at the next finished item; what was never submitted never runs."""
    import pytest
    from muvis_align.util import OperationCancelled, cancellable, request_cancel

    started = []

    def work(item):
        started.append(item)
        return item

    with cancellable():
        with pytest.raises(OperationCancelled):
            for item, _ in rolling_map(work, range(100), workers=2):
                if item == 3:
                    request_cancel()
    assert len(started) < 100


def test_a_cancel_left_from_before_does_not_stop_the_next_operation():
    from muvis_align.util import cancellable, raise_if_cancelled, request_cancel

    request_cancel()
    with cancellable():
        raise_if_cancelled()


def test_labelled_numbers_are_keyed_by_their_lower_case_label_and_unlabelled_ones_left_out():
    assert find_labelled_numbers('EM04652-02_slice17_r0005_t0002_s00399.ome.tif') == \
        {'em': 4652, 'slice': 17, 'r': 5, 't': 2, 's': 399}
    assert find_labelled_numbers('S000_000_001.ome.zarr') == {'s': 0}


@pytest.mark.skipif(sys.platform == 'win32', reason='off on Windows: it also reports access violations handled there')
def test_a_native_crash_leaves_every_threads_stack_in_the_log(tmp_path):
    log_filename = tmp_path / 'muvis-align.log'
    code = ('import faulthandler; from muvis_align.logging import enable_fault_log;'
            f' enable_fault_log({str(log_filename)!r}); faulthandler._sigsegv()')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True)

    assert result.returncode != 0
    log = log_filename.read_text(encoding='utf-8', errors='replace')
    assert 'Fatal Python error' in log and 'most recent call first' in log


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows only')
def test_the_crash_log_is_off_on_windows(tmp_path):
    from muvis_align.logging import enable_fault_log

    enable_fault_log(str(tmp_path / 'muvis-align.log'))

    assert not (tmp_path / 'muvis-align.log').exists()


@pytest.mark.parametrize('value, expected', [
    (0.0025505462087219684, '0.00255'), (0.004, '0.004'), (1.244, '1.24'), (1.5, '1.5'),
    (-64863.2422089573, '-64900'), (12374.5, '12400'), (999.6, '1000'), (0.0, '0'), (-3.5, '-3.5'),
])
def test_print_significants_keeps_at_most_3_significant_digits(value, expected):
    assert print_significants(value, 3) == expected


def test_print_dict_simple_rounds_floats_only_in_zyx_order():
    assert print_dict_simple({'x': 18820.7, 'y': 0.0025505, 'z': 2}) == 'z: 2 y: 0.00255 x: 18800'
    # other keys follow, a rotation's or a pair's (the notebooks print pair qualities)
    assert print_dict_simple({'x': 1.0, 'r': 90.0}) == 'x: 1 r: 90'
    assert print_dict_simple({(0, 1): 0.912345, (1, 2): 0.5}) == '(0, 1): 0.912 (1, 2): 0.5'


def test_an_invalid_source_metadata_expression_warns_and_falls_back_to_the_default():
    context = {'fn': [0, 1, 2]}
    assert eval_context({'x': 'fn[-2]*24'}, 'x', 0, context) == 24
    with pytest.warns(UserWarning, match=r"Invalid source metadata x: 'fn\[-9\]' \(IndexError"):
        assert eval_context({'x': 'fn[-9]'}, 'x', 0, context) == 0


def test_highs_solves_on_the_calling_thread_alone_so_no_workers_are_torn_down_at_its_exit():
    import threading
    import psutil
    from scipy.optimize import _linprog_highs, linprog

    assert getattr(_linprog_highs._highs_wrapper, '_muvis_single_threaded', False)
    process = psutil.Process()
    before = len(process.threads())
    seen = []

    def solve():
        result = linprog(c=[1, 1], A_ub=[[-1, 0], [0, -1]], b_ub=[0, 0], bounds=(None, None))
        seen.append((len(process.threads()), result.status))

    worker = threading.Thread(target=solve)
    worker.start()
    worker.join()

    threads_during, status = seen[0]
    assert status == 0
    # HiGHS' own pool would add ~11 workers here; the calling thread (plus any unrelated one) is all
    assert threads_during <= before + 2


def _worker_blas_threads(_):
    return os.environ.get('OPENBLAS_NUM_THREADS'), os.environ.get('OMP_NUM_THREADS')


def test_workers_start_with_one_blas_thread_and_leave_the_parent_as_it_was():
    """Each OpenBLAS commits a buffer per thread as numpy/scipy load, before any initializer runs."""
    from muvis_align.util import worker_process_pool

    before = os.environ.get('OPENBLAS_NUM_THREADS')
    with worker_process_pool(2) as pool:
        seen = list(pool.map(_worker_blas_threads, range(2)))

    assert seen == [('1', '1'), ('1', '1')]
    assert os.environ.get('OPENBLAS_NUM_THREADS') == before


# (unit, um per unit): readers hand over either an OME abbreviation or ngff_zarr's spelled-out NGFF name
@pytest.mark.parametrize('unit, factor', [
    ('Å', 1e-4), ('A', 1e-4), ('angstrom', 1e-4),
    ('pm', 1e-6), ('picometer', 1e-6),
    ('nm', 1e-3), ('nanometer', 1e-3), ('NM', 1e-3),
    ('µm', 1.0), ('um', 1.0), ('micrometer', 1.0), ('Micrometer', 1.0), ('micron', 1.0),
    ('mm', 1e3), ('millimeter', 1e3),
    ('cm', 1e4), ('centimeter', 1e4),
    ('m', 1e6), ('meter', 1e6),
])
def test_convert_to_um_knows_every_spelling_of_a_unit(unit, factor):
    assert convert_to_um(1.0, unit) == pytest.approx(factor)
    assert convert_to_um(2.5, unit) == pytest.approx(2.5 * factor)


def test_convert_to_um_covers_every_unit_ngff_zarr_can_produce():
    from ngff_zarr.tiff_to_ngff_image import OME_UNIT_TO_NGFF
    from muvis_align.util import um_conversions

    for ome_unit, ngff_name in OME_UNIT_TO_NGFF.items():
        assert ngff_name in um_conversions, f'{ngff_name!r} (from OME {ome_unit!r}) is unhandled'
        assert ome_unit in um_conversions, f'OME unit {ome_unit!r} is unhandled'
        assert convert_to_um(1.0, ome_unit) == pytest.approx(convert_to_um(1.0, ngff_name))


@pytest.mark.parametrize('unit, logged', [(None, False), ('', False), ('furlong', True)])
def test_convert_to_um_leaves_an_unknown_unit_unscaled_and_logs_only_a_named_one(unit, logged, caplog):
    # an unscaled unit silently mis-sizes the image, so a named one it does not know is logged
    with caplog.at_level(logging.WARNING):
        assert convert_to_um(3.0, unit) == 3.0
    assert ('Unrecognised' in caplog.text) == logged


@pytest.mark.parametrize('filenames, expected', [
    (['/data/proj/subset/sample_ov000_s00400.ome.tif',
      '/data/proj/subset/sample_r0005_t0002_s00400.ome.tif',
      '/data/proj/subset/sample_r0005_t0003_s00400.ome.tif'],
     ['ov000', 'r0005_t0002', 'r0005_t0003']),
    # same basenames, no digits in the subdirectory names: the fallback is the relative path
    ([f'/nemo/proj/EM04652_02_slice017/{subdir}/EM04652-02_slice17_ov000_s00400.ome.tif'
      for subdir in ['subset', 'tiles', 'stitched', 'stitched_hpc']],
     [f'{subdir}/EM04652-02_slice17_ov000_s00400.ome.tif' for subdir in ['subset', 'tiles', 'stitched', 'stitched_hpc']]),
    # 's' first appears in the overview's label; it must not move ahead of 'r'/'t' in the tiles' own
    (['overviews/sample_ov000_s00025.ome.tif',
      'tiles/r0004/t0000/sample_r0004_t0000_s00823.ome.tif',
      'tiles/r0004/t0001/sample_r0004_t0001_s00824.ome.tif'],
     ['ov000_s00025', 'r0004_t0000_s00823', 'r0004_t0001_s00824']),
])
def test_get_unique_file_labels(filenames, expected):
    assert get_unique_file_labels(filenames) == expected


@pytest.mark.parametrize('filenames, expected', [
    (['/a/b/c/x.tif', '/a/b/c/y.tif', '/a/b/d/x.tif'], ['c/x.tif', 'c/y.tif', 'd/x.tif']),
    (['a/x.tif', 'b/x.tif'], ['a/x.tif', 'b/x.tif']),
])
def test_strip_common_path_prefix(filenames, expected):
    assert strip_common_path_prefix(filenames) == expected


# the timing line must say which regime a run is in: summed per-item wall time under a pool counts GIL waits
CPU_BOUND = 'more workers will not help'
IO_BOUND = 'more workers can overlap it'


@pytest.mark.parametrize('label, wall, item_times, cpu_times, workers, expected', [
    # summed per-item wall time far above wall, but CPU at wall: the threads only queued for the GIL
    ('threads queued on the GIL', 16.64, [1027.9 / 328] * 328, [16.5 / 328] * 328, 64, CPU_BOUND),
    # each item waits 3.0s but needs 0.05s of CPU, and 8 workers cannot overlap it all
    ('under-parallelised I/O', 328 * 3.0 / 8, [3.0] * 328, [0.05] * 328, 8, IO_BOUND),
    # the same work with enough workers reaches the CPU floor, which no arrangement of threads
    # beats - so it must stop advising more
    ('I/O at the floor', 16.4, [3.0] * 328, [0.05] * 328, 64, CPU_BOUND),
    # a caller that cannot measure CPU time must not get a made-up regime
    ('no cpu times', 5.0, [1.0, 3.0], [], 2, None),
])
def test_the_regime_is_named_from_wall_against_cpu(label, wall, item_times, cpu_times, workers,
                                                   expected):
    line = format_phase_timing(wall, item_times, cpu_times, workers)

    assert [verdict for verdict in (CPU_BOUND, IO_BOUND) if verdict in line] == ([expected] if expected else [])


def test_the_numbers_themselves_are_reported():
    line = format_phase_timing(10.0, [1.0, 3.0, 1.0, 3.0], [0.5] * 4, 4)

    assert 'wall 10.0s' in line
    assert 'per-item total 8.0s' in line
    assert 'cpu 2.0s' in line
    assert 'with 4 workers' in line
    assert 'mean 2000ms' in line and 'max 3000ms' in line    # from the wall times
    assert 'process cpu' not in line                          # optional, and not given here


def test_process_cpu_separates_this_phase_from_the_rest_of_the_process():
    """time.thread_time counts only the thread running the item: without the process figure, CPU burnt
    elsewhere in the process makes the phase look merely slow."""
    busy = format_phase_timing(598.5, [38254.3 / 4733] * 4733, [468.5 / 4733] * 4733, 64,
                               process_cpu_time=2268.0)
    assert 'process cpu 2268.0s' in busy
    assert '3.8 cores' in busy
    assert 'the rest is elsewhere in the process' in busy

    # ...and a process whose CPU is all this phase is not flagged as elsewhere
    contained = format_phase_timing(20.0, [1.0] * 16, [1.0] * 16, 16, process_cpu_time=17.0)
    assert 'process cpu 17.0s' in contained
    assert 'the rest is elsewhere in the process' not in contained


def test_items_in_worker_processes_report_busy_workers_not_a_gil_verdict():
    """Worker processes share no GIL: the items' cpu over the wall is how many were busy."""
    line = format_phase_timing(100.0, [8.0] * 100, [6.0] * 100, 8, process_cpu_time=5.0, processes=True)

    assert '6.0 of 8 busy' in line
    assert CPU_BOUND not in line and IO_BOUND not in line
