import numpy as np
import xarray as xr

from muvis_align.metrics import quality_to_scalar


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
