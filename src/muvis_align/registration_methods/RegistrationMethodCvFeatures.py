import cv2 as cv
import logging
import numpy as np
from multiview_stitcher import param_utils
from spatial_image import SpatialImage

from src.muvis_align.image.util import *
from src.muvis_align.metrics import calc_match_metrics
from src.muvis_align.registration_methods.RegistrationMethod import RegistrationMethod
from src.muvis_align.util import get_mean_nn_distance


class RegistrationMethodCvFeatures(RegistrationMethod):
    def __init__(self, source, params, debug=False):
        super().__init__(source, params, debug)
        self.method = params.get('name', 'sift').lower()
        self.full_size_gaussian_sigma = params.get('gaussian_sigma', params.get('sigma', 1))
        self.gaussian_sigma = self.full_size_gaussian_sigma
        self.downscale_factor = params.get('downscale_factor', params.get('downscale', np.sqrt(2)))
        self.nkeypoints = params.get('max_keypoints', 5000)
        self.cross_check = params.get('cross_check', True)
        self.lowe_ratio = params.get('lowe_ratio', 0.92)
        self.inlier_threshold_factor = params.get('inlier_threshold_factor', 0.05)
        self.min_matches = params.get('min_matches', 10)
        self.max_trails = params.get('max_trials', 100)
        transform_type = params.get('transform_type', '').lower()

        if transform_type in ['translation', 'translate']:
            self.max_rotation = 10  # rotation should be ~0; check <10 degrees
        else:
            self.max_rotation = None


    def detect_features(self, data0, gaussian_sigma=None):
        if 't' in data0.dims:
            data0 = data0.isel(t=0)
        if 'z' in data0.dims:
            # make data 2D
            data0 = data0.max('z')
        if 'c' in data0.dims:
            if data0.sizes['c'] > 1:
                data0 = data0.max('c')
            else:
                data0 = data0.isel(c=0)
        data = self.convert_data_to_float(data0)
        data = np.array(norm_image_variance2(data))
        if gaussian_sigma is None:
            gaussian_sigma = self.full_size_gaussian_sigma
        if gaussian_sigma:
            data = cv.GaussianBlur(data, (0, 0), gaussian_sigma)

        #feature_model = cv.SIFT_create(contrastThreshold=0.1)
        feature_model = cv.ORB_create(nfeatures=self.nkeypoints, patchSize=8, edgeThreshold=7)
        keypoints, desc = feature_model.detectAndCompute(uint8_image(data), None)
        points = np.array([np.flip(keypoint.pt) for keypoint in keypoints])
        return points, desc, data

    def match(self, fixed_points, fixed_desc, moving_points, moving_desc,
              min_matches, cross_check, lowe_ratio, inlier_threshold, mean_size_dist):
        transform = None
        quality = 0
        inliers = []

        matcher = cv.BFMatcher()
        #matches0 = matcher.match(fixed_desc, moving_desc)
        matches0 = matcher.knnMatch(fixed_desc, moving_desc, k=2)

        matches = []
        for m, n in matches0:
            if m.distance < 0.92 * n.distance:
                matches.append(m)

        if len(matches) >= min_matches:
            fixed_points2 = np.float32([fixed_points[match.queryIdx] for match in matches])
            moving_points2 = np.float32([moving_points[match.trainIdx] for match in matches])
            transform, inliers = cv.findHomography(fixed_points2, moving_points2,
                                                   method=cv.USAC_MAGSAC, ransacReprojThreshold=inlier_threshold)
            if inliers is None:
                inliers = []
            if len(inliers) > 0 and validate_transform(transform, max_rotation=self.max_rotation):
                quality = (np.sum(inliers) / self.nkeypoints) ** (1/3) # ^1/3 to decrease sensitivity

            if self.debug:
                print('%inliers', np.mean(inliers))

        return transform, quality, matches, inliers

    def registration(self, fixed_data: SpatialImage, moving_data: SpatialImage, **kwargs) -> dict:
        transform = None
        quality = 0
        full_size_dist = np.linalg.norm(self.full_size)
        mean_size_dist = np.mean([np.linalg.norm(data.shape) for data in [fixed_data, moving_data]])
        scale = mean_size_dist / full_size_dist
        gaussian_sigma = self.full_size_gaussian_sigma * (scale ** (1/3))
        mean_size = np.mean(
            [np.linalg.norm(data.shape) / np.sqrt(self.ndims) for data in [fixed_data, moving_data]])
        inlier_threshold = mean_size * self.inlier_threshold_factor

        fixed_points, fixed_desc, fixed_data2 = self.detect_features(fixed_data, gaussian_sigma)
        moving_points, moving_desc, moving_data2 = self.detect_features(moving_data, gaussian_sigma)

        if len(fixed_desc) > 0 and len(moving_desc) > 0:
            transform, quality, matches, inliers = self.match(fixed_points, fixed_desc, moving_points, moving_desc,
                                                              min_matches=self.min_matches, cross_check=self.cross_check,
                                                              lowe_ratio=self.lowe_ratio, inlier_threshold=inlier_threshold,
                                                              mean_size_dist=mean_size_dist)
            if transform is not None:
                if self.debug:
                    print(f'#keypoints: {len(fixed_desc)},{len(moving_desc)}'
                          f' #matches: {len(matches)} #inliers: {np.sum(inliers):.0f} quality: {quality:.3f}')
                    matches2 = [(match.queryIdx, match.trainIdx) for match in matches]
                    draw_keypoints_matches(fixed_data2, fixed_points,
                                           moving_data2, moving_points,
                                           matches2, inliers,
                                           show_plot=True)

        if not validate_transform(transform):
            logging.warning('Feature extraction: Unable to find feature-based registration')
            transform = np.eye(3)

        return {
            "affine_matrix": param_utils.invert_coordinate_order(transform),  # homogenous matrix of shape (ndim + 1, ndim + 1), axis order (z, y, x)
            "quality": quality  # float between 0 and 1 (if not available, set to 1.0)
        }
