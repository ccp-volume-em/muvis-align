import networkx as nx
import numpy as np
import pytest
import xarray as xr
from multiview_stitcher import param_utils

from muvis_align.metrics import calc_pair_metrics, quality_to_scalar
from tests.data_builders import DATA_DIR, ZARR_FILES, prepared_registration


def test_quality_to_scalar_selects_t0_from_dataarray_with_t_dim():
    quality = xr.DataArray([0.75], dims=['t'], coords={'t': [0]})

    result = quality_to_scalar(quality)

    assert result == 0.75
    assert isinstance(result, float)


def test_quality_to_scalar_reduces_dataarray_without_t_dim():
    quality = xr.DataArray(0.5)

    result = quality_to_scalar(quality)

    assert result == 0.5
    assert isinstance(result, float)


def test_quality_to_scalar_passes_through_plain_scalar():
    assert quality_to_scalar(0.3) == 0.3
    assert quality_to_scalar(None) is None


@pytest.fixture(scope='module')
def register_msims(tmp_path_factory):
    reg = prepared_registration([(DATA_DIR / name).as_posix() for name in ZARR_FILES], tmp_path_factory.mktemp('metrics'))
    return reg.register_msims, reg.source_transform_key


def test_calc_msims_metrics_uses_real_pyramid_directly(register_msims):
    """calc_msims_metrics takes msims directly (no sim<->msim round trip) - the real, possibly
    multi-level pyramid is what gets fed to the underlying metrics computation."""
    from multiview_stitcher import msi_utils
    from muvis_align.metrics import calc_msims_metrics

    msims, _ = register_msims
    assert len(msi_utils.get_sorted_scale_keys(msims[0])) > 1  # sanity check: a real pyramid

    metrics = calc_msims_metrics(msims[:2], {(0, 1): param_utils.identity_transform(ndim=2)}, metric_methods=['ncc'])

    assert isinstance(metrics['pairs'][(0, 1)]['transform']['ncc'], float)


def test_pair_metrics_one_call_a_pair_match_one_call_over_all(register_msims):
    import multiview_stitcher.metrics
    from muvis_align.metrics import create_metric_methods

    msims, transform_key = register_msims
    graph = nx.Graph()
    for pair in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        graph.add_edge(*pair, transform=param_utils.identity_transform(ndim=2), quality=0.5)

    whole = multiview_stitcher.metrics.tile_pair_image_metrics(
        msims, base_transform_key=transform_key, pairs_graph=graph,
        metric_funcs=create_metric_methods(['ncc'], msims[0]))
    single = calc_pair_metrics(msims, graph, ['ncc'], transform_key, n_parallel_pairs=1)
    threaded = calc_pair_metrics(msims, graph, ['ncc'], transform_key, n_parallel_pairs=3)

    for result in (single, threaded):
        assert set(result['pairs']) == set(whole['pairs'])
        for pair, value in whole['pairs'].items():
            assert np.isclose(result['pairs'][pair]['transform']['ncc'], value['transform']['ncc'],
                              equal_nan=True)
        # axis-aligned tiles: each pair's bbox area equals the overlap area the summary weights by
        assert result['summary']['transform']['ncc'] == pytest.approx(whole['summary']['transform']['ncc'])
        assert result['summary']['transform']['quality'] == 0.5


def test_ssim_takes_the_crops_as_single_channel():
    """The registration channel is a channel label or index, never an axis of the 2D crops SSIM gets."""
    from skimage.metrics import structural_similarity
    from muvis_align.metrics import create_metric_methods

    msim = {'scale0/image': np.zeros((1,), dtype=np.uint16)}
    rng = np.random.default_rng(0)
    image1, image2 = rng.integers(0, 4000, (2, 50, 80)).astype(np.float32)

    ssim = create_metric_methods(['ssim'], msim)['ssim']

    assert ssim(image1, image2) == structural_similarity(image1, image2, data_range=np.iinfo(np.uint16).max)


def test_global_metrics_measure_only_the_registered_pairs_as_the_overlap_mode_does(register_msims):
    """Over all msims at once the overlap mode measures every overlapping pair; per registered pair, in
    worker processes, the same values must come out for those pairs and no others."""
    import multiview_stitcher.metrics
    from multiview_stitcher import msi_utils
    from unittest.mock import patch
    import muvis_align.metrics as metrics_module
    from muvis_align.metrics import calc_global_metrics, create_metric_methods

    msims, transform_key = register_msims
    for index, msim in enumerate(msims):
        shift = param_utils.affine_to_xaffine(param_utils.affine_from_translation([0.0, 0.5 * index]))
        msi_utils.set_affine_transform(msim, shift, transform_key='registered', base_transform_key=transform_key)
    graph = nx.Graph()
    for pair in [(0, 1), (0, 2), (2, 3)]:
        graph.add_edge(*pair, quality=0.5)
    reg_results = {'pairwise_registration': {'graph': graph, 'metrics': {'qualities': {edge: 0.5 for edge in graph.edges}}}}

    whole = multiview_stitcher.metrics.tile_pair_image_metrics(
        msims, base_transform_key=transform_key, query_transform_keys=[transform_key, 'registered'],
        metric_funcs=create_metric_methods(['ncc'], msims[0]))
    with patch.object(metrics_module, 'worker_process_pool', wraps=metrics_module.worker_process_pool) as pool:
        result = calc_global_metrics(msims, transform_key, 'registered', ['ncc'], reg_results=reg_results,
                                     n_parallel_pairs=2)

    assert pool.called
    whole_by_pair = {frozenset(pair): value for pair, value in whole['pairs'].items()}
    assert len(whole_by_pair) > graph.number_of_edges()
    assert {frozenset(pair) for pair in result['pairs']} == {frozenset(edge) for edge in graph.edges}
    for pair, value in result['pairs'].items():
        for key in (transform_key, 'registered'):
            # workers run BLAS at one thread: summation order can move the last digit
            assert np.isclose(value[key]['ncc'], whole_by_pair[frozenset(pair)][key]['ncc'], rtol=1e-9, atol=0)
