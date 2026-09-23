import networkx as nx
import numpy as np
import pytest
import xarray as xr
from multiview_stitcher import param_utils

from muvis_align.metrics import calc_pair_metrics, quality_to_scalar


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


def test_calc_msims_metrics_uses_real_pyramid_directly():
    """calc_msims_metrics takes msims directly (no sim<->msim round trip) - the real, possibly
    multi-level pyramid is what gets fed to the underlying metrics computation."""
    from multiview_stitcher import param_utils, msi_utils
    from muvis_align.MVSRegistration import MVSRegistration
    from muvis_align.metrics import calc_msims_metrics

    reg = MVSRegistration()
    reg.init(
        operation='register',
        input_path=[
            'data/S000/S000_000_000.ome.zarr',
            'data/S000/S000_000_001.ome.zarr',
        ],
        output_path='../../output/test_calc_msims_metrics/',
    )
    reg.init_data()
    reg.preprocess(reg.msims)

    msim1, msim2 = reg.register_msims[0], reg.register_msims[1]
    assert len(msi_utils.get_sorted_scale_keys(msim1)) > 1  # sanity check: a real pyramid

    transforms = {(0, 1): param_utils.identity_transform(ndim=2)}

    metrics = calc_msims_metrics([msim1, msim2], transforms, metric_methods=['ncc'])

    assert isinstance(metrics['pairs'][(0, 1)]['transform']['ncc'], float)



@pytest.fixture(scope='module')
def register_msims():
    from muvis_align.MVSRegistration import MVSRegistration

    reg = MVSRegistration()
    reg.init(
        operation='register',
        input_path=[f'data/S000/S000_00{y}_00{x}.ome.zarr' for y in range(2) for x in range(2)],
        output_path='../../output/test_pair_batches/',
    )
    reg.init_data()
    reg.preprocess(reg.msims)
    return reg.register_msims, reg.source_transform_key


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
