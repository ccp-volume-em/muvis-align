from abc import ABC, abstractmethod
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
from spatial_image import SpatialImage


class RegistrationMethod(ABC):
    def __init__(self, source, params, debug=False):
        self.source_type = source.dtype
        if hasattr(source, 'dims'):
            self.full_size = [size for size in list(si_utils.get_shape_from_sim(source).values())
                              if size > 1]
            self.ndims = len(si_utils.get_spatial_dims_from_sim(source))
            self.is_3d = ('z' in source.dims and source.sizes['z'] > 1)
        else:
            self.full_size = [size for dim, size in zip(source.dimension_order, source.shape)
                              if dim in 'xyz' and size > 1]
            self.ndims = len(self.full_size)
            if 'z' in source.dimension_order:
                z_index = source.dimension_order.index('z')
                self.is_3d = (source.shape[z_index] > 1)
            else:
                self.is_3d = False
        self.params = params
        self.debug = debug
        self.count = 0  # for debugging

    def convert_data_to_float(self, data, dtype=float):
        maxval = 2 ** (8 * self.source_type.itemsize) - 1
        return data / dtype(maxval)

    @abstractmethod
    def registration(self, fixed_data: SpatialImage, moving_data: SpatialImage, **kwargs) -> dict:
        # this returns the transform in pixel space, needs to be thread-safe!
        # reg_func_transform = linalg.inv(params_transform) / spacing
        # params_transform = linalg.inv(reg_func_transform * spacing)
        return {
            "affine_matrix": [],  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
            "quality": 1  # float between 0 and 1 (if not available, set to 1.0)
        }
