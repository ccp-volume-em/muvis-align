import dask.array as da
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
import os
from skimage.transform import resize

from muvis_align.image.ome_tiff_helper import load_tiff, save_tiff
from muvis_align.image.util import *


def flatfield_correction(msims, transform_key, quantiles, foreground_map=None, cache_location=None):
    """msims in, msims out - the flatfield correction model (quantile images) is computed once,
    from each source's own working (scale0) resolution - the cross-source quantile stacking
    genuinely needs concrete pixel data and can't be generalised per level - then resized to
    match every other pyramid level before being applied there, so the whole msim ends up
    consistently corrected instead of only ever its scale0 level.
    """
    sims = [get_msim_image0(msim) for msim in msims]

    quantile_images = []
    if cache_location is not None:
        for quantile in quantiles:
            filename = get_quantile_filename(cache_location, quantile)
            if os.path.exists(filename):
                quantile_images.append(load_tiff(filename))

    if len(quantile_images) < len(quantiles):
        quantile_images = calc_flatfield_images(sims, quantiles, foreground_map)
        if cache_location is not None:
            for quantile, quantile_image in zip(quantiles, quantile_images):
                filename = get_quantile_filename(cache_location, quantile)
                save_tiff(filename, quantile_image)

    model = calc_flatfield_model(sims[0].dims, quantiles, quantile_images)

    new_msims = []
    for msim in msims:
        def level_func(level_sim, scale_key, model=model):
            return apply_flatfield_model(level_sim, transform_key, model)
        new_msims.append(map_msim_levels(msim, level_func))
    return new_msims


def get_quantile_filename(cache_location, quantile):
    filename = os.path.join(cache_location, 'quantile_' + f'{quantile}'.replace('.', '_') + '.tiff')
    return filename


def calc_flatfield_images(sims, quantiles, foreground_map=None):
    if foreground_map is not None:
        back_sims = [sim for sim, is_foreground in zip(sims, foreground_map) if not is_foreground]
    else:
        back_sims = sims
    dtype = sims[0].dtype
    maxval = 2 ** (8 * dtype.itemsize) - 1
    flatfield_images = [image.astype(np.float32) / np.float32(maxval)
                        for image in da.quantile(da.asarray(back_sims), quantiles, axis=0)]
    return flatfield_images


def calc_flatfield_model(dims0, quantiles, quantile_images):
    """The flatfield correction model derived once at a reference resolution - dark/bright
    quantile images (moved to channel-last order, matching apply_flatfield_model's element-wise
    formula) plus the scalar per-channel mean_bright_dark - reused (dark/bright resized as needed)
    at every pyramid level, instead of being recomputed or only ever applied at one resolution.
    """
    has_c_dim = 'c' in dims0
    dark = 0
    bright = 1
    for quantile, quantile_image in zip(quantiles, quantile_images):
        if has_c_dim and dims0.index('c') != -1:
            quantile_image = da.moveaxis(quantile_image, dims0.index('c'), -1)
        if quantile <= 0.5:
            dark = quantile_image
        else:
            bright = quantile_image

    bright_dark_range = bright - dark
    if has_c_dim:
        axes = list(range(len(dims0) - 1))   # all except final 'c' axis
    else:
        axes = None
    mean_bright_dark = np.array(np.mean(bright - dark, axis=axes))
    return {
        'dims0': dims0,
        'has_c_dim': has_c_dim,
        'dark': dark,
        'bright_dark_range': bright_dark_range,
        'mean_bright_dark': mean_bright_dark,
    }


def _resize_correction_image(image, target_shape):
    # dark/bright_dark_range are only ever computed at one reference resolution - resize (rather
    # than recompute) to match a coarser/finer pyramid level's own shape; a no-op when already at
    # the reference resolution (e.g. the level the model was itself computed from)
    if tuple(np.shape(image)) == tuple(target_shape):
        return image
    return resize(np.asarray(image), target_shape, preserve_range=True)


def apply_flatfield_model(sim, transform_key, model):
    dims0, has_c_dim = model['dims0'], model['has_c_dim']
    dtype = sim.dtype
    if has_c_dim:
        image0 = sim.transpose(..., 'c')
    else:
        image0 = sim

    dark = _resize_correction_image(model['dark'], image0.shape)
    bright_dark_range = _resize_correction_image(model['bright_dark_range'], image0.shape)

    image = float2int_image(
        image_flatfield_correction(int2float_image(image0), dark, bright_dark_range, model['mean_bright_dark']),
        dtype)
    if has_c_dim:
        image = image.transpose(*dims0)     # revert to original order
    return si_utils.get_sim_from_array(
        image,
        dims=sim.dims,
        scale=si_utils.get_spacing_from_sim(sim),
        translation=si_utils.get_origin_from_sim(sim),
        transform_key=transform_key,
        affine=si_utils.get_affine_from_sim(sim, transform_key),
        c_coords=sim.c
    )


def image_flatfield_correction(image0, dark, bright_dark_range, mean_bright_dark, clip=False):
    # Input/output: float images
    # https://en.wikipedia.org/wiki/Flat-field_correction
    image = (image0 - dark) * mean_bright_dark / bright_dark_range
    if clip:
        image = image.clip(0, 1)    # np.clip(image) is not dask-compatible, use image.clip() instead
    else:
        image -= np.min(image)
    return image
