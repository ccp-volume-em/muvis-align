import logging
import numpy as np
from probreg import cpd
from spatial_image import SpatialImage

from src.muvis_align.metrics import calc_match_metrics
from src.muvis_align.registration_methods.RegistrationMethod import RegistrationMethod
from src.muvis_align.image.util import *
from src.muvis_align.util import points_to_3d


class RegistrationMethodCPD(RegistrationMethod):
    def detect_points(self, data0):
        data = data0.astype(self.source_type)
        area_points = detect_area_points(data)
        points = [point for point, area in area_points]
        return points

    def registration(self, fixed_data: SpatialImage, moving_data: SpatialImage, **kwargs) -> dict:
        max_iter = kwargs.get('max_iter', 1000)

        fixed_points = self.detect_points(fixed_data)
        moving_points = self.detect_points(moving_data)
        threshold = get_mean_nn_distance(fixed_points, moving_points)

        transform = None
        quality = 0
        if len(moving_points) > 1 and len(fixed_points) > 1:
            result_cpd = cpd.registration_cpd(points_to_3d(moving_points), points_to_3d(fixed_points),
                                              maxiter=max_iter)
            transformation = result_cpd.transformation
            S = transformation.scale * np.eye(3)
            R = transformation.rot
            T = np.eye(3) + np.hstack([np.zeros((3, 2)), transformation.t.reshape(-1, 1)])
            transform = T @ R @ S

            metrics = calc_match_metrics(fixed_points, moving_points, transform, threshold)
            quality = metrics['match_rate']

        if not validate_transform(transform):
            logging.error('Feature extraction: Unable to find CPD registration')

        return {
            "affine_matrix": transform,  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
            "quality": quality  # float between 0 and 1 (if not available, set to 1.0)
        }
