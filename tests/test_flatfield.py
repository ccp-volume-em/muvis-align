import numpy as np
import dask.array as da
import pytest
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image import flatfield
from muvis_align.image.util import get_msim_image0, map_msim_levels, int2float_image, float2int_image


def _make_msim(seed, size=16):
    rng = np.random.RandomState(seed)
    data = rng.randint(1000, 60000, (size, size)).astype(np.uint16)
    sim = si_utils.get_sim_from_array(
        data, dims=['y', 'x'], scale={'x': 1.0, 'y': 1.0}, translation={'x': 0, 'y': 0},
        transform_key='source_metadata')
    return msi_utils.get_msim_from_sim(sim, scale_factors=[])


def _make_multilevel_msim(msim, factor=2):
    sim0 = msi_utils.get_sim_from_msim(msim, scale='scale0')
    small = np.asarray(sim0.data)[..., ::factor, ::factor]
    sim1 = si_utils.get_sim_from_array(
        small, dims=list(sim0.dims), scale={'x': float(factor), 'y': float(factor)},
        translation={'x': 0, 'y': 0}, transform_key='source_metadata')
    return msi_utils.get_msim_from_sims([sim0, sim1])


@pytest.fixture
def quantile_images():
    # manually built quantile "images" matching a 16x16 (t,c,y,x) sim's shape, standing in for
    # calc_flatfield_images' real output - avoids depending on da.quantile, which the installed
    # dask/numpy combination in this environment doesn't support (a pre-existing, unrelated bug:
    # da.quantile forwards a now-removed 'interpolation' kwarg to numpy internally)
    rng = np.random.RandomState(42)
    shape = (1, 1, 16, 16)
    q_low = da.from_array((rng.random(shape) * 0.3 + 0.05).astype(np.float32))
    q_high = da.from_array((rng.random(shape) * 0.3 + 0.65).astype(np.float32))
    return [q_low, q_high]


def test_flatfield_model_resizes_correction_images_per_level(quantile_images):
    # apply_flatfield_model must resize the (single-resolution) dark/bright_dark_range images to
    # match whatever pyramid level it's given, rather than only ever working at the resolution
    # the model was computed at
    msim = _make_msim(0)
    sim0 = msi_utils.get_sim_from_msim(msim, scale='scale0')
    model = flatfield.calc_flatfield_model(sim0.dims, [0.1, 0.9], quantile_images)

    multi_msim = _make_multilevel_msim(msim, factor=2)

    def level_func(level_sim, scale_key, model=model):
        return flatfield.apply_flatfield_model(level_sim, 'source_metadata', model)

    result_msim = map_msim_levels(multi_msim, level_func)
    scale_keys = msi_utils.get_sorted_scale_keys(result_msim)
    assert scale_keys == ['scale0', 'scale1']

    result0 = msi_utils.get_sim_from_msim(result_msim, scale='scale0')
    result1 = msi_utils.get_sim_from_msim(result_msim, scale='scale1')
    assert result0.shape == (1, 1, 16, 16)
    assert result1.shape == (1, 1, 8, 8)

    # cross-check scale1 by hand: resize dark/bright_dark_range to scale1's own shape and apply
    # the correction formula directly, independently of apply_flatfield_model's own resize step
    from skimage.transform import resize
    orig_sim1 = msi_utils.get_sim_from_msim(multi_msim, scale='scale1')
    image0 = orig_sim1.transpose(..., 'c')
    dark_r = resize(np.asarray(model['dark']), image0.shape, preserve_range=True)
    range_r = resize(np.asarray(model['bright_dark_range']), image0.shape, preserve_range=True)
    manual = float2int_image(
        flatfield.image_flatfield_correction(int2float_image(image0), dark_r, range_r, model['mean_bright_dark']),
        image0.dtype)
    manual = manual.transpose(*model['dims0'])

    a = np.asarray(manual.data if hasattr(manual, 'data') else manual)
    b = np.asarray(result1.data.compute() if hasattr(result1.data, 'compute') else result1.data)
    np.testing.assert_array_equal(a, b)


def test_flatfield_correction_is_msims_in_msims_out(monkeypatch, quantile_images):
    # flatfield_correction() itself takes msims and returns msims - verify the public entry point
    # end to end (with calc_flatfield_images stubbed out, for the same pre-existing da.quantile
    # environment reason as above), and that every level of the result is corrected consistently
    msims = [_make_msim(i) for i in range(3)]
    multi_msims = [_make_multilevel_msim(m, factor=2) for m in msims]

    monkeypatch.setattr(flatfield, 'calc_flatfield_images', lambda sims, quantiles, foreground_map=None: quantile_images)

    result = flatfield.flatfield_correction(multi_msims, 'source_metadata', [0.1, 0.9])

    assert len(result) == len(multi_msims)
    for msim in result:
        scale_keys = msi_utils.get_sorted_scale_keys(msim)
        assert scale_keys == ['scale0', 'scale1']
        sim0 = msi_utils.get_sim_from_msim(msim, scale='scale0')
        sim1 = msi_utils.get_sim_from_msim(msim, scale='scale1')
        assert sim0.shape == (1, 1, 16, 16)
        assert sim1.shape == (1, 1, 8, 8)
