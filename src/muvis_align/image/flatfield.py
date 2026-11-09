import dask.array as da
import numpy as np
from skimage.transform import resize
import uuid
import xarray as xr

from muvis_align.image.util import *


def calc_msims_flatfield_model(msims, quantiles, foreground_map=None):
    """The flat-field model of msims, from their scale0 quantile images."""
    sims = [get_msim_image0(msim) for msim in msims]
    quantile_images = calc_flatfield_images(sims, quantiles, foreground_map)
    return calc_flatfield_model(sims[0].dims, quantiles, quantile_images)


def apply_msim_flatfield_model(msim, model):
    """msim flat-field corrected at every pyramid level, the model resized to each."""
    def level_func(level_sim, scale_key):
        return apply_flatfield_model(level_sim, model)
    return map_msim_levels(msim, level_func)


def calc_flatfield_images(sims, quantiles, foreground_map=None):
    if foreground_map is not None:
        back_sims = [sim for sim, is_foreground in zip(sims, foreground_map) if not is_foreground]
    else:
        back_sims = sims
    dtype = sims[0].dtype
    maxval = 2 ** (8 * dtype.itemsize) - 1
    shapes = {sim.shape for sim in back_sims}
    if len(shapes) > 1:
        # one field at several pixel sizes: each image resampled to the finest (largest) before the quantiles
        shape = max(shapes, key=np.prod)
        stack = da.stack([sim.data if sim.shape == shape
                          else da.from_array(resize(np.asarray(sim), shape, preserve_range=True).astype(dtype))
                          for sim in back_sims])
    else:
        stack = da.asarray(back_sims)
    flatfield_images = [image.astype(np.float32) / np.float32(maxval)
                        for image in da.quantile(stack, quantiles, axis=0)]
    return flatfield_images


def calc_flatfield_model(dims0, quantiles, quantile_images):
    """The flat-field model at the quantile images' resolution, as named images (dims0, t dropped): the dark image,
    the light-dark range and its per-channel mean."""
    images = [xr.DataArray(np.asarray(image), dims=list(dims0)).isel(t=0, missing_dims='ignore')
              for _, image in sorted(zip(quantiles, quantile_images), key=lambda pair: pair[0])]
    # a single quantile is always the light (flat) image, with a zero dark stub; a pair is dark, light
    dark = images[0] if len(images) > 1 else 0
    bright_dark_range = images[-1] - dark
    spatial_dims = [dim for dim in bright_dark_range.dims if dim in 'zyx']
    # skipna off: np.mean's own summation, as for any other mean of the image
    mean_bright_dark = bright_dark_range.mean(spatial_dims, skipna=False)
    # 'resized': each image resized once per shape, shared by every tile the model corrects
    return {'dark': dark, 'bright_dark_range': bright_dark_range, 'mean_bright_dark': mean_bright_dark,
            'token': uuid.uuid4().hex, 'resized': {}}


def _correction_at(model, key, image):
    """A model image matched to `image` by dim name: a dim the image lacks dropped, spatial sizes resized to its."""
    correction = model[key]
    if np.ndim(correction) == 0:
        return correction
    correction = correction.isel({dim: 0 for dim in correction.dims if dim not in image.dims})
    shape = tuple(image.sizes[dim] for dim in correction.dims)
    if shape == correction.shape:
        return correction
    cache_key = (key, correction.dims, shape)
    resized = model['resized'].get(cache_key)
    if resized is None:
        data = resize(correction.values, shape, preserve_range=True)
        chunks = (tuple(image.data.chunks[image.dims.index(dim)] for dim in correction.dims)
                  if hasattr(image.data, 'chunks') else 'auto')
        # named, not tokenized: hashing the full-size array again for every tile is what would cost
        name = f'flatfield-{model["token"]}-{key}-{"x".join(map(str, shape))}'
        resized = xr.DataArray(da.from_array(data, chunks=chunks, name=name), dims=correction.dims)
        model['resized'][cache_key] = resized
    return resized


def apply_flatfield_model(image, model):
    """`image` (a sim or any DataArray with named dims) flat-field corrected, in its own dtype and dims."""
    dark = _correction_at(model, 'dark', image)
    bright_dark_range = _correction_at(model, 'bright_dark_range', image)
    mean_bright_dark = _correction_at(model, 'mean_bright_dark', image)
    corrected = float2int_image(
        image_flatfield_correction(int2float_image(image), dark, bright_dark_range, mean_bright_dark), image.dtype)
    # a copy keeps every transform (registered too, for fusion), not only the source one
    return image.copy(data=corrected.transpose(*image.dims).data)


def image_flatfield_correction(image0, dark, bright_dark_range, mean_bright_dark):
    # Input/output: float images
    # https://en.wikipedia.org/wiki/Flat-field_correction
    image = (image0 - dark) * mean_bright_dark / bright_dark_range
    # clip rather than offset per image, so all tiles keep the same intensity mapping
    return image.clip(0, 1)    # np.clip(image) is not dask-compatible, use image.clip() instead
