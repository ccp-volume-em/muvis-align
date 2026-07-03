import logging
import numpy as np
try:
    from probreg import cpd
except ImportError:
    cpd = None
from spatial_image import SpatialImage

from src.muvis_align.metrics import calc_match_metrics
from src.muvis_align.registration_methods.RegistrationMethod import RegistrationMethod
from src.muvis_align.image.util import *
from src.muvis_align.util import points_to_3d


class RegistrationMethodCPD(RegistrationMethod):
    def __init__(self, source, params, debug=False):
        super().__init__(source, params, debug=debug)
        self.full_size_gaussian_sigma = params.get('gaussian_sigma', params.get('sigma', 0))
        self.normalisation = params.get('normalisation')
        self.max_iter = params.get('max_iter', 1000)

    def detect_points(self, data0, gaussian_sigma=None):
        data = data0.astype(self.source_type)
        if gaussian_sigma:
            data = gaussian_filter_image(data, gaussian_sigma, is_3d=self.is_3d)
        if self.normalisation:
            data = normalise_values(data)
        if self.is_3d:
            points = detect_volume_points(data)
        else:
            area_points = detect_area_points(data)
            # OpenCV contour centers are returned in x,y order, while the
            # registration pipeline uses image-axis order y,x.
            points = [np.flip(point) for point, stat in area_points]
        return points

    def registration(self, fixed_data: SpatialImage, moving_data: SpatialImage, **kwargs) -> dict:
        if cpd is None:
            raise ImportError(
                "The 'probreg' package is required for CPD registration. "
                "Install it with: pip install muvis-align[cpd]"
            )
        ndim = 3 if self.is_3d else 2

        full_size_min = np.min(self.full_size)
        sizes = [[size for size in list(si_utils.get_shape_from_sim(data).values()) if size > 1]
                 for data in [fixed_data, moving_data]]
        mean_size_min = np.mean([np.min(size) for size in sizes])
        scale = mean_size_min / full_size_min
        gaussian_sigma = self.full_size_gaussian_sigma * (scale ** (1/2))

        fixed_points = self.detect_points(fixed_data, gaussian_sigma)
        moving_points = self.detect_points(moving_data, gaussian_sigma)
        threshold = get_mean_nn_distance(fixed_points, moving_points)

        transform = None
        quality = 0
        metrics = {}
        if len(moving_points) > 1 and len(fixed_points) > 1:
            moving_points_3d = points_to_3d(moving_points) if not self.is_3d else moving_points
            fixed_points_3d = points_to_3d(fixed_points) if not self.is_3d else fixed_points
            # Keep direction consistent with other methods: transform fixed -> moving.
            result_cpd = cpd.registration_cpd(fixed_points_3d, moving_points_3d, maxiter=self.max_iter)
            transformation = result_cpd.transformation

            transform = np.eye(ndim + 1)
            transform[:ndim, :ndim] = transformation.rot[:ndim, :ndim] * transformation.scale
            transform[:ndim, -1] = transformation.t[:ndim]

            metrics = calc_match_metrics(fixed_points, moving_points, transform, threshold)
            quality = metrics['match_rate']

        if not validate_transform(transform):
            logging.error('Feature extraction: Unable to find CPD registration')

        return {
            "affine_matrix": transform,  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
            "quality": quality,  # float between 0 and 1 (if not available, set to 1.0)
            "fixed_points": fixed_points,
            "moving_points": moving_points,
            "matches": metrics.get('matches'),
            "inliers": metrics.get('inliers')
        }
