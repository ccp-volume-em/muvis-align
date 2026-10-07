import glob
import threading
import logging
import time
import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from muvis_align.constants import default_fusion_workers
from muvis_align.MVSRegistration import MVSRegistration, RegState
from tests.data_builders import DATA_DIR, TIFF_FILES, ZARR_FILES, make_msim, prepared_registration, recording_phase_factory

S000_TILES = [(DATA_DIR / name).as_posix() for name in ZARR_FILES]
TWO_TILES = S000_TILES[:2]
TWO_TIFFS = [(DATA_DIR / name).as_posix() for name in TIFF_FILES[:2]]
ALL_TILES = sorted(glob.glob('data/S*/*.ome.zarr'))


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (RegState.UNINIT, (False, False, False, False)),
        (RegState.INIT, (True, False, False, False)),
        (RegState.PAIRS_REG, (True, True, False, False)),
        (RegState.GLOBAL_REG, (True, True, True, False)),
        (RegState.FUSED, (True, True, True, True)),
    ],
    ids=lambda value: value.name if isinstance(value, RegState) else None,
)
def test_registration_state_predicates(state, expected):
    registration = MVSRegistration()
    registration.state = state

    actual = (
        registration.is_initialised(),
        registration.is_pairs_registered(),
        registration.is_global_registered(),
        registration.is_fused(),
    )

    assert actual == expected


def test_reset_clears_registration_state():
    registration = MVSRegistration()
    registration.state = RegState.FUSED
    registration.msims = [object()]
    registration.metrics = {"quality": 1}
    # a per-scale build is as stale as the full one once positions/transforms are rebuilt
    registration._scaled_msims = {'4': ['msim0']}

    registration.reset()

    assert registration.state is RegState.UNINIT
    assert registration.msims == []
    assert registration.register_msims is None
    assert registration.sources == []
    assert registration.metrics == {}
    assert registration.register_indices is None
    assert registration._scaled_msims == {}


def test_init_with_explicit_files_sets_labels_and_output(tmp_path):
    # a list input_path is glob-expanded (it may also be unexpanded patterns, e.g. from a
    # comma-separated UI input path), so the files need to actually exist here
    inputs = [str(tmp_path / "tile_01.tif"), str(tmp_path / "tile_02.tif")]
    for path in inputs:
        Path(path).touch()

    registration = MVSRegistration()
    result = registration.init(
        operation="register",
        label="sample",
        input_path=inputs,
        input_labels=["left", "right"],
        output_path="results/",
    )

    assert result is True
    assert registration.state is RegState.INIT
    assert registration.filenames == [Path(path).as_posix() for path in inputs]
    assert registration.file_labels == ["left", "right"]
    assert registration.output == str(tmp_path / "results") + "/"


def test_init_returns_false_when_pattern_has_no_files(tmp_path):
    registration = MVSRegistration()

    with patch("muvis_align.MVSRegistration.dir_regex", return_value=[]):
        result = registration.init(
            input_path=str(tmp_path / "*.tif"),
            output_path="results/",
        )

    assert result is False


def test_init_params_normalises_sections_and_forwards_options():
    registration = MVSRegistration()
    params = {
        "operation": "register",
        "input": "images/*.tif",
        "output": "results/",
        "preprocessing": {"scale": 2},
        "registration": {"method": "phase"},
        "fusion": {"method": "max"},
    }
    general = {
        "overwrite": True,
        "clear": True,
        "ui": "napari",
        "verbose": True,
        "debug": True,
    }

    with patch.object(registration, "init", return_value=True) as init:
        result = registration.init_params(general, params, label="sample")

    assert result is True
    assert registration.input_params == {"path": "images/*.tif"}
    assert registration.output_params == {"path": "results/"}
    assert registration.preprocess_params == {"scale": 2}
    assert init.call_args.kwargs["overwrite"] is True
    assert init.call_args.kwargs["label"] == "sample"


def _make_msims(count=1, size=8, pixel_size=1.0):
    # small single-level msims, cheap enough for preprocess() to really run its per-level steps
    return [make_msim(np.full((size, size), index + 1, dtype=np.uint16),
                      scale={'x': pixel_size, 'y': pixel_size}) for index in range(count)]


@pytest.mark.parametrize(
    ("kwargs", "expected_modified"),
    [
        ({}, False),
        ({"scale": 1}, False),
        ({"normalisation": "none"}, False),
        ({"normalisation": False}, False),
        ({"gaussian_sigma": 1}, True),
    ],
)
def test_preprocess_sets_modified_flag_for_enabled_steps(kwargs, expected_modified):
    registration = MVSRegistration()
    registration.scales = [{"x": 1.0, "y": 1.0}]
    registration.source_transform_key = "source_metadata"
    msims = _make_msims()

    _, _, modified = registration.preprocess(msims, **kwargs)

    assert modified is expected_modified


def test_preprocess_reports_an_option_it_does_not_recognise(caplog):
    """A project file may carry options preprocess() does not implement, so an unknown one is
    not fatal - but passing the whole section as one keyword (params=...) instead of expanding
    it leaves every step at its default, and registration then runs at full resolution with
    nothing to say why."""
    registration = MVSRegistration()
    registration.scales = [{"x": 1.0, "y": 1.0}]
    registration.source_transform_key = "source_metadata"
    msims = _make_msims()

    with caplog.at_level(logging.WARNING):
        _, _, modified = registration.preprocess(msims, params={"scale": 8}, typo=1)

    assert modified is False
    assert 'unknown pre-processing option' in caplog.text
    assert 'params' in caplog.text and 'typo' in caplog.text


def test_preprocess_applies_scale_via_select_msim_subpyramid():
    # preprocess()'s `scale` override selects a real (smaller) sub-pyramid directly from the
    # msims it's given (every native level at or coarser than `scale`), rather than resizing to
    # an exact match or re-running the whole init_data() pipeline a second time
    registration = MVSRegistration()
    registration.scales = [{"x": 1.0, "y": 1.0}]
    registration.source_transform_key = "source_metadata"
    registration.sources = [object()]
    msims = _make_msims()

    with patch(
        "muvis_align.MVSRegistration.select_msim_subpyramid_at_scale", return_value=msims,
    ) as select_msim_subpyramid_at_scale:
        _, _, modified = registration.preprocess(msims, scale=2)

    assert modified is True
    select_msim_subpyramid_at_scale.assert_called_once_with(msims, registration.sources, 2)


def test_validate_overlap_reports_near_images():
    registration = MVSRegistration()
    sims = [object(), object()]
    positions = [{"x": 0.0, "y": 0.0}, {"x": 0.5, "y": 0.0}]

    with (
        patch(
            "muvis_align.MVSRegistration.get_sim_position_final",
            side_effect=positions,
        ),
        patch(
            "muvis_align.MVSRegistration.get_sim_physical_size",
            return_value={"x": 1.0, "y": 1.0},
        ),
    ):
        distances, overlaps = registration.validate_overlap(
            sims, ["left", "right"]
        )

    assert len(distances) == 2
    assert overlaps == [True, True]


def test_get_metrics_supports_summary_tuple_and_numpy_pair():
    registration = MVSRegistration()
    registration.metrics = {
        "summary": {"source": {"quality": 0.5}},
        "pairs": {(0, 1): {"registered": {"quality": 0.8}}},
    }

    assert registration.get_metrics("quality") == 0.5
    assert registration.get_metrics(
        "quality", np.array([0, 1])
    ) == pytest.approx(0.8)
    assert registration.get_metrics(pair=(3, 4)) == {}


def test_output_exists_supports_zarr_and_regular_files(tmp_path):
    registration = MVSRegistration()
    registration.output = str(tmp_path) + "/"
    zarr_output = tmp_path / "fused.zarr"
    zarr_output.mkdir()
    (zarr_output / "zarr.json").write_text("{}", encoding="utf-8")
    (tmp_path / "preview.tif").write_bytes(b"image")

    assert registration.output_exists("fused", ".zarr")
    assert registration.output_exists("preview", "tif")
    assert not registration.output_exists("missing", ".zarr")


@pytest.mark.parametrize(
    ("output_exists", "existing_path", "expected_state"),
    [
        (True, None, RegState.FUSED),
        (False, "mappings.json", RegState.GLOBAL_REG),
        (False, "pairs.json", RegState.PAIRS_REG),
    ],
)
def test_check_progress_uses_most_advanced_available_state(
    output_exists, existing_path, expected_state
):
    registration = MVSRegistration()
    registration.output = "output/"
    registration.output_params = {
        "pair_mappings": "pairs.json",
        "mappings": "mappings.json",
    }

    def path_exists(path):
        return existing_path is not None and path.endswith(existing_path)

    with (
        patch.object(
            registration, "output_exists", return_value=output_exists
        ),
        patch(
            "muvis_align.MVSRegistration.os.path.exists",
            side_effect=path_exists,
        ),
    ):
        registration.check_progress("fused", ".zarr")

    assert registration.state is expected_state


def test_init_data_defers_msim_construction_to_first_msims_read(tmp_path):
    """init_data() should only resolve cheap per-source metadata (position/scale/rotation) -
    the expensive per-source msim build (build_source_msim(), the actual bottleneck when
    loading many tiles) must not run until something genuinely reads reg.msims."""
    reg = prepared_registration(TWO_TIFFS, tmp_path, preprocess=False)

    # cheap metadata is already fully resolved...
    assert reg._msims is None  # ... but the expensive msims list is not built yet
    assert len(reg.positions) == 2
    assert len(reg.scales) == 2
    assert len(reg.rotations) == 2
    assert all(source._msim is None for source in reg.sources)  # per-source msim deferred too

    # reading reg.msims (the property) triggers the deferred build, on demand
    msims = reg.msims
    assert len(msims) == 2
    assert reg._msims is msims
    # and not even then is each source's own msim built: the run's msims come straight off the
    # source's arrays, so the get_sim_from_array per level that one would cost never happens
    assert all(source._msim is None for source in reg.sources)


def test_select_pair_overlap_then_register_overlap_matches_register_pairs(tmp_path):
    """select_pair_overlap()/register_overlap() together replace the old single register_pair()
    call, split so a caller (e.g. a UI preview) can cache the overlap crop select_pair_overlap()
    returns and re-run register_overlap() on it for parameter-only changes, without re-selecting
    resolution or re-cropping from the (possibly large) source data every time."""
    reg = prepared_registration(TWO_TILES, tmp_path)

    params = {'method': 'orb', 'pairing': 'orthogonal'}
    msim1, msim2 = reg.register_msims[0], reg.register_msims[1]

    overlap1, overlap2, sims_pixel_space = reg.select_pair_overlap(msim1, msim2, params=params)
    assert overlap1.ndim == 2
    assert overlap2.ndim == 2

    transform, quality, result = reg.register_overlap(overlap1, overlap2, sims_pixel_space, params=params)

    assert transform.shape[-2:] == (3, 3)  # 2D homogeneous affine
    assert not np.isnan(quality)

    # the raw pairwise_reg_func result must survive - feature-based methods report points/matches
    # used for a napari preview overlay
    assert 'affine_matrix' in result
    assert 'quality' in result
    assert 'fixed_points' in result
    assert 'moving_points' in result


def test_register_pairs_computes_without_linear_fusion(tmp_path):
    """dask's linear fusion can give two pairs' fused chains one key (a 115-char prefix plus 4 hash
    digits), handing a pair another pair's crop - so both computes of register_pairs() run without it.
    The metrics run on threads with OpenBLAS at one thread: its own pool, started from many threads,
    crashed the process."""
    import dask
    import multiview_stitcher.metrics
    from threadpoolctl import threadpool_info
    import muvis_align.MVSRegistration as mvs_registration_module

    seen = {}

    def recording(name, func):
        def wrapper(*args, **kwargs):
            seen[name] = {
                'fuse': dask.config.get('optimization.fuse.active', None),
                'scheduler': dask.config.get('scheduler', None),
                # only OpenBLAS crashed; macOS numpy uses Accelerate, which has no pool to limit
                'openblas_threads': {pool['num_threads'] for pool in threadpool_info()
                                     if pool['internal_api'] == 'openblas'},
            }
            return func(*args, **kwargs)
        return wrapper

    reg = prepared_registration(TWO_TILES, tmp_path)
    with patch.object(mvs_registration_module, 'compute_pairwise_registrations',
                      recording('pairs', mvs_registration_module.compute_pairwise_registrations)),             patch.object(multiview_stitcher.metrics, 'tile_pair_image_metrics',
                         recording('metrics', multiview_stitcher.metrics.tile_pair_image_metrics)):
        reg.register_pairs(reg.register_msims, params={'method': 'phase_correlation', 'pairing': 'orthogonal'})

    assert seen['pairs']['fuse'] is False
    # one synchronous compute a pair, on several threads
    assert seen['metrics']['fuse'] is False
    assert seen['metrics']['scheduler'] == 'synchronous'
    assert seen['metrics']['openblas_threads'] <= {1}


def test_register_pairs_defers_link_quality_without_changing_results(tmp_path):
    """Only the chosen candidate's spearman quality is computed; the result must match computing
    all of them, and multiview_stitcher's function is put back afterwards."""
    import contextlib
    import networkx as nx
    from multiview_stitcher import registration
    import muvis_align.MVSRegistration as mvs_registration_module

    reg = prepared_registration(TWO_TILES, tmp_path)
    params = {'method': 'phase_correlation', 'pairing': 'orthogonal'}
    original = registration.link_quality_metric_func

    def register():
        reg.register_pairs(reg.register_msims, params=params)
        return {edge: (float(np.asarray(quality).squeeze()),
                       np.asarray(nx.get_edge_attributes(reg.pairs_graph, 'transform')[edge]))
                for edge, quality in nx.get_edge_attributes(reg.pairs_graph, 'quality').items()}

    deferred = register()
    assert registration.link_quality_metric_func is original
    with patch.object(mvs_registration_module, 'deferred_link_quality', contextlib.nullcontext):
        eager = register()

    assert deferred.keys() == eager.keys()
    for edge, (quality, transform) in eager.items():
        assert deferred[edge][0] == pytest.approx(quality, nan_ok=True)
        np.testing.assert_array_equal(deferred[edge][1], transform)


def test_register_pairs_one_compute_a_pair_matches_across_thread_counts(tmp_path):
    """Each pair is its own compute on a thread (when a pair cannot go to a worker process); results must
    not depend on how many threads. A lone thread keeps the threads scheduler so a single pair still runs
    its tasks in parallel. Pairs go largest estimated cost first."""
    import dask
    import networkx as nx
    import muvis_align.MVSRegistration as mvs_registration_module

    reg = prepared_registration(S000_TILES, tmp_path)
    original = mvs_registration_module.compute_pairwise_registrations
    original_graph = mvs_registration_module.build_view_adjacency_graph

    def register(threads):
        seen = []

        def recording(msims, g_reg, **kwargs):
            seen.append((g_reg.number_of_edges(), dask.config.get('scheduler', None)))
            order.append(tuple(sorted(g_reg.edges))[0])
            return original(msims, g_reg, **kwargs)

        def unequal_overlaps(*args, **kwargs):
            # the four tiles overlap their neighbours equally: the last pair in source order made the largest
            graph = original_graph(*args, **kwargs)
            graph.edges[2, 3]['overlap'] *= 10
            graph.edges[0, 1]['overlap'] *= 0.1
            return graph

        with patch.object(mvs_registration_module, 'compute_pairwise_registrations', recording),                 patch.object(mvs_registration_module, 'build_view_adjacency_graph', unequal_overlaps),                 patch.object(mvs_registration_module, 'picklable', lambda value: False):
            reg.register_pairs(reg.register_msims, params={'method': 'phase_correlation', 'pairing': 'orthogonal',
                                                           'n_parallel_pairwise_regs': threads})
        results = {edge: (float(np.asarray(quality).squeeze()),
                          np.asarray(nx.get_edge_attributes(reg.pairs_graph, 'transform')[edge]))
                   for edge, quality in nx.get_edge_attributes(reg.pairs_graph, 'quality').items()}
        return results, seen

    order = []
    single, single_seen = register(1)
    assert order[0] == (2, 3) and order[-1] == (0, 1)
    threaded, threaded_seen = register(3)

    assert len(single) > 1
    assert set(single_seen) == {(1, 'threads')}
    assert set(threaded_seen) == {(1, 'synchronous')}
    assert threaded.keys() == single.keys()
    for edge, (quality, transform) in single.items():
        assert threaded[edge][0] == pytest.approx(quality, nan_ok=True)
        np.testing.assert_array_equal(threaded[edge][1], transform)


def test_register_pairs_default_pairing_hands_over_candidates_for_the_same_graph(tmp_path):
    """Left to search itself, multiview_stitcher pairs every source within the largest one's
    diameter - with overview images among tiles, nearly every pair. It gets the bounding-box
    candidates instead, and must end up with the graph its own search gives."""
    import dask
    from multiview_stitcher import mv_graph
    import muvis_align.MVSRegistration as mvs_registration_module

    reg = prepared_registration(S000_TILES, tmp_path)
    original = mvs_registration_module.build_view_adjacency_graph
    handed = []

    def recording(msims, transform_key, pairs, **kwargs):
        handed.append(pairs)
        return original(msims, transform_key, pairs, **kwargs)

    with patch.object(mvs_registration_module, 'build_view_adjacency_graph', recording):
        reg.register_pairs(reg.register_msims, params={'method': 'phase_correlation', 'pairing': 'default'})
    with dask.config.set(scheduler='threads'):
        own = mv_graph.build_view_adjacency_graph_from_msims(reg.pair_msims, transform_key=reg.source_transform_key,
                                                             overlap_tolerance=0)

    assert handed and handed[0] is not None
    assert {frozenset(edge) for edge in reg.pairs_graph.edges} == {frozenset(edge) for edge in own.edges}


def _channel_msim(labels):
    return make_msim(np.zeros((1, len(labels), 4, 4), dtype=np.uint8), dims='tcyx', c_coords=list(labels))


def test_registration_channel_by_index_by_label_or_the_only_one():
    from muvis_align.MVSRegistration import resolve_registration_channel

    assert resolve_registration_channel(_channel_msim(['#0', '#1']), 1) == '#1'
    assert resolve_registration_channel(_channel_msim(['#0', '#1']), '#1') == '#1'
    # a project set up for files that named their one channel differently
    assert resolve_registration_channel(_channel_msim(['#0']), 'channel 0') == '#0'
    with pytest.raises(ValueError, match="'#0', '#1'"):
        resolve_registration_channel(_channel_msim(['#0', '#1']), 'channel 0')


def test_registration_channel_is_chosen_once_per_set_of_labels(caplog):
    from muvis_align.MVSRegistration import resolve_registration_channel

    chosen = {}
    with caplog.at_level('WARNING'):
        labels = [resolve_registration_channel(_channel_msim([name]), 'channel 0', chosen)
                  for name in ('#0', '#0', 'other', '#0')]

    assert labels == ['#0', '#0', 'other', '#0']
    assert sum('not found' in message for message in caplog.messages) == 2


def test_register_pairs_registers_on_the_only_channel_whatever_it_is_called(caplog, tmp_path):
    import networkx as nx

    reg = prepared_registration(TWO_TILES, tmp_path)

    def register(channel):
        reg.register_pairs(reg.register_msims, params={'method': 'phase_correlation', 'pairing': 'orthogonal',
                                                       'channel': channel})
        return {edge: np.asarray(transform) for edge, transform in nx.get_edge_attributes(reg.pairs_graph, 'transform').items()}

    by_name = register('0')
    with caplog.at_level('WARNING'):
        by_wrong_name = register('channel 0')

    assert any("'channel 0' not found" in message for message in caplog.messages)
    assert by_wrong_name.keys() == by_name.keys()
    for edge, transform in by_name.items():
        np.testing.assert_array_equal(by_wrong_name[edge], transform)

    # sources of one project naming their channel differently: each registers on its own
    reg.register_msims[1] = reg.register_msims[1].map_over_datasets(
        lambda dataset: dataset.assign_coords(c=['other']) if 'c' in dataset.coords else dataset)
    for channel in ('0', 'channel 0'):
        caplog.clear()
        with caplog.at_level('WARNING'):
            mixed = register(channel)
        assert sum('not found' in message for message in caplog.messages) == (1 if channel == '0' else 2)
        assert mixed.keys() == by_name.keys()
        for edge, transform in by_name.items():
            np.testing.assert_array_equal(mixed[edge], transform)


def test_register_overlap_reuses_cached_overlap_across_param_changes(tmp_path):
    """The whole point of splitting select_pair_overlap()/register_overlap(): the same overlap
    crop can be registered again with different registration parameters, without recomputing the
    crop - the two calls below reuse the exact same overlap1/overlap2/sims_pixel_space."""
    reg = prepared_registration(TWO_TILES, tmp_path)

    msim1, msim2 = reg.register_msims[0], reg.register_msims[1]
    overlap1, overlap2, sims_pixel_space = reg.select_pair_overlap(
        msim1, msim2, params={'method': 'orb'}
    )

    transform1, quality1, result1 = reg.register_overlap(
        overlap1, overlap2, sims_pixel_space, params={'method': 'orb'}
    )
    transform2, quality2, result2 = reg.register_overlap(
        overlap1, overlap2, sims_pixel_space, params={'method': 'sift'}
    )

    # different methods, same crop - both must produce a valid result from the identical input
    assert transform1.shape == transform2.shape
    assert not np.isnan(quality1)
    assert not np.isnan(quality2)


def test_build_msims_is_parallel_but_keeps_source_order():
    """_build_msims does the per-file reading that sources defer out of init, so it runs
    threaded (as init_sources does). Futures complete in whatever order they finish, so the
    result must still be indexed back into source order - a shuffle here would silently pair
    every msim with the wrong position/transform.
    """
    from pathlib import Path

    from multiview_stitcher import spatial_image_utils as si_utils

    from muvis_align.MVSRegistration import MVSRegistration
    from muvis_align.image.source_helper import create_image_source
    from muvis_align.image.util import build_source_msim, get_msim_image0

    data_dir = Path(__file__).resolve().parent.parent / 'data' / 'S000'
    files = sorted(str(path) for path in data_dir.glob('*.tiff'))
    assert len(files) > 1, 'need several sources to exercise the thread pool'

    reg = MVSRegistration()
    reg.sources = [create_image_source(name) for name in files]
    # distinct positions, so a mis-paired msim is detectable from its origin alone
    reg.positions = [{'x': 100.0 * index, 'y': 10.0 * index} for index in range(len(files))]
    reg._msim_transforms = [None] * len(files)
    reg._msim_output_order = 'tcyx'
    reg._msim_z_scale = None
    reg.source_transform_key = 'source_metadata'
    reg.logging_time = False

    reg._build_msims()
    built = reg._msims

    assert len(built) == len(files)
    assert all(msim is not None for msim in built)

    expected = [build_source_msim(create_image_source(name), 'tcyx', position, None,
                                  'source_metadata')
                for name, position in zip(files, reg.positions)]
    for got, want in zip(built, expected):
        got_origin = si_utils.get_origin_from_sim(get_msim_image0(got))
        want_origin = si_utils.get_origin_from_sim(get_msim_image0(want))
        assert got_origin == pytest.approx(want_origin)
        assert get_msim_image0(got).shape == get_msim_image0(want).shape


@pytest.mark.parametrize(
    ("operation", "pairing", "expected"),
    [
        ("register", "stack", True),
        ("register", "orthogonal", False),
        ("register", "", False),
        # the operation no longer selects stacking - pairing does, and only pairing
        ("register stack", "", False),
        ("stack", "", False),
    ],
)
def test_is_stack_reads_pairing_only(operation, pairing, expected):
    reg = MVSRegistration.__new__(MVSRegistration)
    reg.operation = operation
    reg.pairing = pairing
    assert reg.is_stack is expected


def _scaled_source(factors=(1, 2, 4)):
    return SimpleNamespace(scale_factors=[{'y': factor, 'x': factor} for factor in factors],
                           get_pixel_size=lambda: {'y': 0.5, 'x': 0.5})


def test_ensure_msims_builds_each_scale_once_and_keeps_the_full_pyramid_shared():
    """Pre-processing at a coarse scale must not build the finer levels it is about to drop -
    but must also not turn scale 1 into a private second copy of the full-resolution build,
    which is the one self.msims caches for everything else.
    """
    registration = MVSRegistration()
    registration._msims = None
    registration.sources = [_scaled_source() for _ in range(3)]

    builds = []

    def record_build(progress_factory=None, from_levels=None, store=True, weight=1, **kwargs):
        builds.append((tuple(from_levels) if from_levels else None, store))
        msims = [f'msim{index}' for index in range(len(registration.sources))]
        if store:
            registration._msims = msims
        return msims

    registration._build_msims = record_build

    registration.ensure_msims()
    registration.ensure_msims()
    registration.ensure_msims(target_scale=1)
    registration.ensure_msims(target_scale=4)
    registration.ensure_msims(target_scale=4)
    registration.ensure_msims(target_scale=2)

    # scale 1 skips nothing, so it reuses the full build rather than adding a fourth
    assert builds == [(None, True), ((2, 2, 2), False), ((1, 1, 1), False)]
    assert sorted(registration._scaled_msims) == ['2', '4']


def test_register_pairs_in_worker_processes_matches_registering_in_this_process(tmp_path):
    """Pairs sent to worker processes (their sources pickled, reopened there) must register as they do
    here - compared with one native thread here too, as the workers run: threading changes BLAS sums."""
    import cv2
    from threadpoolctl import threadpool_limits
    import muvis_align.MVSRegistration as mvs_registration_module

    reg = prepared_registration(S000_TILES, tmp_path)

    def register(workers):
        reg.register_pairs(reg.register_msims, params={'method': 'phase_correlation', 'pairing': 'orthogonal',
                                                       'n_parallel_pairwise_regs': workers})
        return {edge: (float(np.asarray(reg.pairs_graph.edges[edge]['quality']).squeeze()),
                       np.asarray(reg.pairs_graph.edges[edge]['transform'])) for edge in reg.pairs_graph.edges}

    threads = cv2.getNumThreads()
    try:
        cv2.setNumThreads(1)
        with threadpool_limits(1):
            here = register(1)
    finally:
        cv2.setNumThreads(threads)
    with patch.object(mvs_registration_module, 'worker_process_pool',
                      wraps=mvs_registration_module.worker_process_pool) as pool:
        in_workers = register(2)

    assert pool.called
    assert len(here) > 2
    assert here.keys() == in_workers.keys()
    for edge in here:
        assert np.array_equal(here[edge][1], in_workers[edge][1])
        assert here[edge][0] == in_workers[edge][0] or (np.isnan(here[edge][0]) and np.isnan(in_workers[edge][0]))


def test_register_global_resolves_with_the_named_groupwise_method_and_reports_metrics_by_pair(monkeypatch, tmp_path):
    """The robust linear method is selectable by name: register_global hands it to multiview_stitcher's resolution
    and maps every tile - not how close its fit comes to another method's, which is not this code's to decide.
    Its metrics count the registered pairs rather than being one bar step."""
    import muvis_align.MVSRegistration as mvs_module
    reg = prepared_registration(S000_TILES, tmp_path)
    params = {'method': 'phase_correlation', 'pairing': 'orthogonal', 'transform_type': 'rigid',
              'groupwise_resolution_method': 'robust_linear'}
    reg.register_pairs(reg.register_msims, params=params)
    methods = []
    original = mvs_module.groupwise_resolution

    def recorded(graph, method=None, **kwargs):
        methods.append((method, kwargs.get('transform')))
        return original(graph, method=method, **kwargs)
    monkeypatch.setattr(mvs_module, 'groupwise_resolution', recorded)
    progress_factory, phases = recording_phase_factory()

    results = reg.register_global(reg.pair_msims, params=params, progress_factory=progress_factory)

    assert methods == [('robust_linear', 'rigid')]
    assert sorted(results['mappings']) == list(range(len(reg.pair_msims)))
    metrics_phases = [phase for phase in phases if 'metrics' in phase['desc'].lower()]
    assert metrics_phases == [{'desc': 'Global metrics', 'total': reg.pairs_graph.number_of_edges(),
                               'done': reg.pairs_graph.number_of_edges()}]


def test_a_cancel_stops_building_sources_without_building_the_rest():
    """The build used to submit every source at once, so a cancel waited for all of them (34k on the HPC)."""
    import threading
    import muvis_align.MVSRegistration as mvs_registration_module
    from muvis_align.util import OperationCancelled, cancellable, request_cancel

    reg = MVSRegistration()
    reg.sources = [SimpleNamespace() for _ in range(400)]
    reg.positions = [{}] * 400
    reg._msim_transforms = [None] * 400
    reg._msim_output_order, reg._msim_z_scale, reg.source_transform_key = 'yx', None, 'source'
    built = []

    def slow_build(*args, **kwargs):
        time.sleep(0.05)
        built.append(1)
        return 'msim'

    class CancellingBar:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, n=1):
            if len(built) >= 10:
                request_cancel()
            from muvis_align.util import raise_if_cancelled
            raise_if_cancelled()

    with cancellable(), patch.object(mvs_registration_module, 'build_source_msim', slow_build):
        with pytest.raises(OperationCancelled):
            reg._build_msims(progress_factory=lambda **kwargs: CancellingBar())
    assert len(built) < 100


def test_export_fusion_by_z_slabs_writes_what_one_fusion_of_all_sources_does(tmp_path):
    """Each z-slab of blocks fused from only the sources reaching it, every pyramid level identical."""
    import glob
    from multiview_stitcher import msi_utils
    from muvis_align.MVSRegistration import MVSRegistration

    reg = prepared_registration(ALL_TILES, tmp_path, preprocess=False)
    fused = {}
    for by_slabs in (False, True):
        reg.fuse_by_z_slabs = by_slabs
        fused[by_slabs], saved = reg.fuse(reg.msims, fusion_method='average', transform_key=reg.source_transform_key,
                                          output_spacing='mean', output_filename=f'fused_{by_slabs}', ome_version='0.5')
        assert saved

    for level in msi_utils.get_sorted_scale_keys(fused[False]):
        assert np.array_equal(np.asarray(msi_utils.get_sim_from_msim(fused[False], scale=level).data),
                              np.asarray(msi_utils.get_sim_from_msim(fused[True], scale=level).data))


def test_the_middle_section_is_the_middle_folders_files_registered_as_the_whole_labels_them(tmp_path):
    reg = MVSRegistration()
    reg.init(operation='register', input_path=sorted(glob.glob('data/S*/*.ome.zarr')),
             output_path=tmp_path.as_posix() + '/')

    indices = reg.middle_section_indices()
    section = reg.section_registration(indices)

    assert indices == [index for index, filename in enumerate(reg.filenames) if '/S000/' in filename]
    assert section.filenames == [reg.filenames[index] for index in indices]
    assert section.file_labels == [reg.file_labels[index] for index in indices]
    assert section.output == reg.output


@pytest.mark.parametrize('filenames, expected', [
    # SBEMimage: a folder per tile, the overviews of every slice in one folder
    (['ov/ov000/x_ov000_s00538.tif', 'ov/ov000/x_ov000_s00539.tif',
      'tiles/g0000/t0000/x_g0000_t0000_s00538.tif', 'tiles/g0000/t0000/x_g0000_t0000_s00539.tif',
      'tiles/g0000/t0001/x_g0000_t0001_s00538.tif', 'tiles/g0000/t0001/x_g0000_t0001_s00539.tif'], [0, 2, 4]),
    # the last number is a tile index here, the section is the labelled one
    (['S000/S000_000_000.ome.zarr', 'S000/S000_000_001.ome.zarr',
      'S001/S001_000_000.ome.zarr', 'S001/S001_000_001.ome.zarr'], [0, 1]),
    (['a/tile_0.tif', 'a/tile_1.tif', 'b/tile_0.tif'], [0, 1]),
    (['a/S000_0.tif', 'a/S000_1.tif'], None),
    (['S000/S000_000_000.ome.zarr', 'S000/S000_000_001.ome.zarr'], None),
    # the step napari's dims start on: sections sorted by number, int((n - 1) / 2)
    (['S002_0.tif', 'S000_0.tif', 'S001_0.tif', 'S001_1.tif'], [2, 3]),
    (['S010_0.tif', 'S003_0.tif', 'S007_0.tif', 'S004_0.tif'], [3]),
])
def test_the_middle_section_is_found_by_the_labelled_section_number_else_the_folder(filenames, expected):
    reg = MVSRegistration()
    reg.filenames = filenames

    assert reg.middle_section_indices() == expected


def test_ome_zarr_06_is_refused_before_fusing_other_than_natively(tmp_path):
    import pytest
    reg = prepared_registration(S000_TILES, tmp_path, preprocess=False)

    with pytest.raises(ValueError, match="'native' output spacing only"):
        reg.fuse(reg.msims, fusion_method='average', transform_key=reg.source_transform_key, output_spacing='mean',
                 output_filename='fused', ome_version='0.6')
    assert not glob.glob(tmp_path.as_posix() + '/fused*')


def test_a_stack_is_stored_2d_and_becomes_3d_only_when_fused(tmp_path):
    """A stack's sources carry no z of their own: fuse() promotes them, spacing the slices by index."""
    source_metadata = {'scale': {'y': '0.032', 'x': '0.032'}}
    reg = MVSRegistration()
    reg.init(operation='register', pairing='stack',
             input_path=['data/S000/000_000_0.tiff', 'data/S000/000_001_0.tiff'],
             output_path=tmp_path.as_posix() + '/', source_metadata=source_metadata)
    reg.init_data(source_metadata=source_metadata)

    assert reg.is_stack
    assert not any('z' in position for position in reg.positions)
    assert all('z' not in msim['scale0'].ds['image'].dims for msim in reg.msims)

    fused_msim, _ = reg.fuse(reg.msims, transform_key=reg.source_transform_key)

    # one plane per source, while the sources stay as stored
    assert fused_msim['scale0'].ds['image'].sizes['z'] == 2
    assert all('z' not in msim['scale0'].ds['image'].dims for msim in reg.msims)


# a zarr export runs its fusion blocks in batches; dask already parallelises the in-memory graph
@pytest.mark.parametrize('saving_zarr, workers, expected', [
    (False, None, None),    # the in-memory path builds a lazy graph dask already parallelises
    (True, 1, None),
    (True, 4, 4),
    (True, None, default_fusion_workers if default_fusion_workers > 1 else None),
])
def test_only_a_zarr_export_batches_its_blocks(saving_zarr, workers, expected):
    options = MVSRegistration._fusion_batch_options(saving_zarr, max_workers=workers)
    assert (options['n_batch'] if options else None) == expected


def test_a_batch_fuses_every_block_once_and_concurrently():
    """Sequentially these would deadlock on the barrier, so reaching the assert is the test."""
    workers = 4
    barrier = threading.Barrier(workers, timeout=10)
    fused, lock = [], threading.Lock()

    def fuse_chunk(block_id):
        barrier.wait()
        with lock:
            fused.append(block_id)

    blocks = [(0, index) for index in range(workers)]
    MVSRegistration._fusion_batch_options(True, max_workers=workers)['batch_func'](
        fuse_chunk, blocks)

    assert sorted(fused) == sorted(blocks)


def test_a_failing_block_is_not_swallowed():
    """A block that raises must fail the export, not leave a hole in the output."""
    def fuse_chunk(block_id):
        if block_id == 3:
            raise ValueError('block 3')

    options = MVSRegistration._fusion_batch_options(True, max_workers=4)
    with pytest.raises(ValueError, match='block 3'):
        options['batch_func'](fuse_chunk, list(range(8)))


# pre-processing's bar reserves a phase for the msim build only when that build will run
@pytest.mark.parametrize('msims, scaled, scale, coarser_level, expected', [
    (None, {}, None, False, True),
    (['msim'], {}, None, False, False),
    (None, {'2': ['msim']}, 2, False, False),
    (['msim'], {}, 2, True, True),
], ids=['nothing built', 'already built', 'that scale cached', 'a coarser level exists for that scale'])
def test_the_msims_build_is_pending_unless_built_or_cached(msims, scaled, scale, coarser_level, expected):
    registration = MVSRegistration()
    registration._msims = msims
    registration._scaled_msims = scaled
    registration.sources = [object()] if coarser_level else []

    with patch('muvis_align.MVSRegistration.get_level_from_scale', return_value=(1, None)):
        pending = registration.msims_build_pending(scale) if scale else registration.msims_build_pending()

    assert pending is expected


def test_eager_pre_processing_is_recognised_from_the_project_params():
    eager = MVSRegistration.has_eager_pre_processing
    assert eager({'scale': 8, 'gaussian_sigma': 2.0}) is False
    # as a project file stores "no normalisation" - plain truthiness reads it as on
    assert eager({'normalisation': 'none'}) is False
    assert eager({'normalisation': 'global'}) is True
    assert eager({'flatfield_quantiles': '0.05, 0.95'}) is True
    assert eager({'filter_foreground': True}) is True
