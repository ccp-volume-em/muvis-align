import dask.array as da
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
import os

from muvis_align.image.ome_tiff_helper import load_tiff, save_tiff
from muvis_align.image.util import *


def flatfield_correction(sims, transform_key, quantiles, foreground_map=None, cache_location=None):
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

    return apply_flatfield_correction(sims, transform_key, quantiles, quantile_images)


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


def apply_flatfield_correction(sims, transform_key, quantiles, quantile_images):
    new_sims = []
    sim0 = sims[0]
    dims0 = sim0.dims
    has_c_dim = 'c' in dims0
    dtype = sim0.dtype
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
        axes = list(range(len(dims0) - 1))   # all accept final 'c' axis
    else:
        axes = None
    mean_bright_dark = np.array(np.mean(bright - dark, axis=axes))

    for sim in sims:
        if has_c_dim:
            image0 = sim.transpose(..., 'c')
        else:
            image0 = sim
        image = float2int_image(image_flatfield_correction(int2float_image(image0), dark, bright_dark_range, mean_bright_dark), dtype)
        if has_c_dim:
            image = image.transpose(*dims0)     # revert to original order
        new_sim = si_utils.get_sim_from_array(
            image,
            dims=sim.dims,
            scale=si_utils.get_spacing_from_sim(sim),
            translation=si_utils.get_origin_from_sim(sim),
            transform_key=transform_key,
            affine=si_utils.get_affine_from_sim(sim, transform_key),
            c_coords=sim.c
        )
        new_sims.append(new_sim)
    return new_sims

def image_flatfield_correction(image0, dark, bright_dark_range, mean_bright_dark, clip=False):
    # Input/output: float images
    # https://en.wikipedia.org/wiki/Flat-field_correction
    image = (image0 - dark) * mean_bright_dark / bright_dark_range
    if clip:
        image = image.clip(0, 1)    # np.clip(image) is not dask-compatible, use image.clip() instead
    else:
        image -= np.min(image)
    return image
