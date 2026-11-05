from types import SimpleNamespace

import numpy as np
import dask.array as da
import pytest
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image import flatfield
from muvis_align.image.util import map_msim_levels, int2float_image, float2int_image
from tests.data_builders import make_msim


def noise_msim(seed):
    """Two levels of uint16 noise, 16x16 then 8x8."""
    data = np.random.default_rng(seed).integers(1000, 60000, (16, 16)).astype(np.uint16)
    return make_msim(data, scale_factors=[2])


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
    multi_msim = noise_msim(0)
    sim0 = msi_utils.get_sim_from_msim(multi_msim, scale='scale0')
    model = flatfield.calc_flatfield_model(sim0.dims, [0.1, 0.9], quantile_images)

    def level_func(level_sim, scale_key, model=model):
        return flatfield.apply_flatfield_model(level_sim, model)

    result_msim = map_msim_levels(multi_msim, level_func)
    scale_keys = msi_utils.get_sorted_scale_keys(result_msim)
    assert scale_keys == ['scale0', 'scale1']

    result0 = msi_utils.get_sim_from_msim(result_msim, scale='scale0')
    result1 = msi_utils.get_sim_from_msim(result_msim, scale='scale1')
    assert result0.shape == (1, 1, 16, 16)
    assert result1.shape == (1, 1, 8, 8)

    # cross-check scale1 by hand, in plain numpy: the (c, y, x) model images resized to scale1's size
    from skimage.transform import resize
    dark, light = (np.asarray(image)[0] for image in quantile_images)
    image = int2float_image(np.asarray(msi_utils.get_sim_from_msim(multi_msim, scale='scale1').data)[0, 0])
    dark_r = resize(dark, (1, 8, 8), preserve_range=True)[0]
    range_r = resize(light - dark, (1, 8, 8), preserve_range=True)[0]
    mean = np.mean(light - dark, axis=(1, 2))
    expected = float2int_image(((image - dark_r) * mean / range_r).clip(0, 1), np.dtype(np.uint16))
    np.testing.assert_array_equal(np.asarray(result1.data)[0, 0], expected)


@pytest.mark.parametrize('quantile', [0.1, 0.9])
def test_single_quantile_divides_every_tile_by_one_shared_light_image(monkeypatch, quantile_images, quantile):
    # any single quantile is the light image (no dark subtraction), with no per-tile offset
    light = quantile_images[1]
    multi_msims = [noise_msim(seed) for seed in range(2)]
    monkeypatch.setattr(flatfield, 'calc_flatfield_images', lambda sims, quantiles, foreground_map=None: [light])

    model = flatfield.calc_msims_flatfield_model(multi_msims, [quantile])
    result = [flatfield.apply_msim_flatfield_model(msim, model) for msim in multi_msims]

    light_image = np.asarray(light)[0, 0]
    for msim, corrected_msim in zip(multi_msims, result):
        image = int2float_image(msi_utils.get_sim_from_msim(msim, scale='scale0').data[0, 0])
        expected_float = image * np.mean(light_image) / light_image
        expected = float2int_image(expected_float.clip(0, 1), np.dtype(np.uint16))
        corrected = msi_utils.get_sim_from_msim(corrected_msim, scale='scale0').data[0, 0]
        np.testing.assert_array_equal(np.asarray(corrected), np.asarray(expected))


def test_flatfield_correction_is_msims_in_msims_out(monkeypatch, quantile_images):
    # every level corrected, and every transform kept: fusion applies it after registration
    multi_msims = [noise_msim(seed) for seed in range(3)]
    registered = np.array([[1, 0, 0, 5], [0, 1, 0, 7], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)[1:, 1:]
    for msim in multi_msims:
        msi_utils.set_affine_transform(msim, registered[None], transform_key='registered')

    monkeypatch.setattr(flatfield, 'calc_flatfield_images', lambda sims, quantiles, foreground_map=None: quantile_images)

    model = flatfield.calc_msims_flatfield_model(multi_msims, [0.1, 0.9])
    result = [flatfield.apply_msim_flatfield_model(msim, model) for msim in multi_msims]

    assert len(result) == len(multi_msims)
    for msim in result:
        scale_keys = msi_utils.get_sorted_scale_keys(msim)
        assert scale_keys == ['scale0', 'scale1']
        sim0 = msi_utils.get_sim_from_msim(msim, scale='scale0')
        sim1 = msi_utils.get_sim_from_msim(msim, scale='scale1')
        assert sim0.shape == (1, 1, 16, 16)
        assert sim1.shape == (1, 1, 8, 8)
        for scale_key in scale_keys:
            level_sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
            np.testing.assert_array_equal(np.asarray(si_utils.get_affine_from_sim(level_sim, 'registered'))[0],
                                          registered)


def test_a_raw_level_without_t_or_c_is_corrected_as_its_msim_level(quantile_images):
    # the view reads a source's stored levels as they are, e.g. a tiff's (y, x)
    import xarray as xr
    from muvis_align.image.lazy_overview import FlatfieldLevels

    multi_msim = noise_msim(0)
    sim0 = msi_utils.get_sim_from_msim(multi_msim, scale='scale0')
    model = flatfield.calc_flatfield_model(sim0.dims, [0.1, 0.9], quantile_images)
    levels = [np.asarray(msi_utils.get_sim_from_msim(multi_msim, scale=key).data)[0, 0] for key in ('scale0', 'scale1')]
    reader = SimpleNamespace(pixel_sizes=[{}, {}], dtype=np.dtype(np.uint16),
                             level_data=lambda level: xr.DataArray(da.from_array(levels[level]), dims=['y', 'x']))
    corrected_msim = flatfield.apply_msim_flatfield_model(multi_msim, model)

    for level, key in enumerate(('scale0', 'scale1')):
        corrected = FlatfieldLevels(reader, lambda: model).level_data(level)
        assert corrected.dims == ('y', 'x') and corrected.dtype == np.uint16
        np.testing.assert_array_equal(np.asarray(corrected),
                                      np.asarray(msi_utils.get_sim_from_msim(corrected_msim, scale=key).data)[0, 0])
    # no model (yet): the level as read
    assert FlatfieldLevels(reader, lambda: None).level_data(1) is not None
    np.testing.assert_array_equal(np.asarray(FlatfieldLevels(reader, lambda: None).level_data(1)), levels[1])
