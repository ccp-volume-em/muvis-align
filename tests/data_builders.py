import numpy as np
from multiview_stitcher import spatial_image_utils as si_utils


def make_dummy_blob_spatial_image(shape, points, dims, seed=1234, noise_max=16, radius=None):
    ndim = len(shape)
    if len(dims) != ndim:
        raise ValueError('shape and dims must have the same length')
    if ndim not in (2, 3):
        raise ValueError('only 2D and 3D dummy blob data are supported')

    if radius is None:
        radius = 2.5 if ndim == 2 else 1.75

    rng = np.random.default_rng(seed)
    image = rng.integers(0, noise_max, size=shape, dtype=np.uint8)
    grids = np.ogrid[tuple(slice(0, size) for size in shape)]

    for point in np.asarray(points, dtype=float):
        if len(point) != ndim:
            raise ValueError('point dimensionality must match shape dimensionality')
        distance2 = np.zeros(shape, dtype=np.float32)
        for axis, grid in enumerate(grids):
            distance2 += (grid - point[axis]) ** 2
        blob = distance2 <= radius ** 2
        image = np.maximum(image, np.where(blob, 255, 0).astype(np.uint8))

    return si_utils.get_sim_from_array(image, dims=list(dims))


def make_dummy_blob_spatial_image_2d(shape, points, dims='yx', **kwargs):
    if len(shape) != 2 or len(dims) != 2:
        raise ValueError('2D helper requires 2D shape and dims')
    return make_dummy_blob_spatial_image(shape, points, dims, **kwargs)

