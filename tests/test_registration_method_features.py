import numpy as np
import pytest
from skimage.transform import EuclideanTransform

from muvis_align.registration_methods.RegistrationMethodCvFeatures import RegistrationMethodCvFeatures
from muvis_align.registration_methods.RegistrationMethodSkFeatures import RegistrationMethodSkFeatures
from tests.data_builders import make_dummy_blob_spatial_image_2d


@pytest.mark.parametrize('method_class', [RegistrationMethodSkFeatures, RegistrationMethodCvFeatures])
def test_registration_returns_identity_when_matching_fails(monkeypatch, method_class):
    fixed_points = np.array([[0.0, 0.0], [0.0, 10.0]], dtype=np.float32)
    fixed_data = make_dummy_blob_spatial_image_2d((32, 48), fixed_points, 'yx')
    moving_data = make_dummy_blob_spatial_image_2d((32, 48), fixed_points, 'yx')
    method = method_class(fixed_data, params={'method': 'orb', 'gaussian_sigma': 0, 'min_matches': 2,
                                              'ransac_iterations': 1})

    monkeypatch.setattr(method_class, 'detect_features', lambda self, data, gaussian_sigma=None: (
        fixed_points, np.array([[1], [2]], dtype=np.uint8), np.asarray(data)))
    monkeypatch.setattr(method_class, 'match', lambda self, *args, **kwargs: (None, 0, [], []))

    result = method.registration(fixed_data, moving_data)

    assert np.allclose(result['affine_matrix'], np.eye(3))
    assert result['quality'] == 0


def test_sk_detect_features_returns_skimage_keypoints_in_yx_order(monkeypatch):
    data = make_dummy_blob_spatial_image_2d((32, 48), [(7, 11), (2, 6)], 'yx')
    method = RegistrationMethodSkFeatures(data, params={'method': 'orb', 'gaussian_sigma': 0, 'max_keypoints': 10})

    class DummyFeatureModel:
        def __init__(self):
            self.keypoints = np.array([[7.0, 11.0], [2.0, 6.0]], dtype=np.float32)
            self.descriptors = np.array([[1], [2]], dtype=np.uint8)

        def detect_and_extract(self, image):
            assert image.shape == (32, 48)

    monkeypatch.setattr(
        'muvis_align.registration_methods.RegistrationMethodSkFeatures.ORB',
        lambda **kwargs: DummyFeatureModel(),
    )

    points, desc, processed = method.detect_features(data)

    assert processed.shape == (32, 48)
    assert np.allclose(points, [[7.0, 11.0], [2.0, 6.0]])
    assert np.array_equal(desc, [[1], [2]])


def test_sk_match_returns_best_ransac_transform_and_inliers(monkeypatch):
    data = make_dummy_blob_spatial_image_2d((32, 48), [(0, 0)], 'yx')
    method = RegistrationMethodSkFeatures(
        data,
        params={'method': 'orb', 'min_matches': 2, 'max_keypoints': 8, 'ransac_iterations': 2},
    )

    fixed_points = np.array([
        [0.0, 0.0],
        [0.0, 10.0],
        [10.0, 0.0],
    ], dtype=np.float32)
    moving_points = fixed_points + np.array([1.0, 2.0], dtype=np.float32)
    fixed_desc = np.array([[1], [2], [3]], dtype=np.uint8)
    moving_desc = np.array([[4], [5], [6]], dtype=np.uint8)
    transform1 = EuclideanTransform(translation=[2.0, 1.0])
    transform2 = EuclideanTransform(translation=[2.2, 1.2])
    inliers1 = np.array([True, True, True])
    inliers2 = np.array([True, True, False])
    ransac_results = iter([
        (transform1, inliers1),
        (transform2, inliers2),
    ])

    monkeypatch.setattr(
        'muvis_align.registration_methods.RegistrationMethodSkFeatures.match_descriptors',
        lambda fixed_desc_arg, moving_desc_arg, cross_check, max_ratio: np.array([[0, 0], [1, 1], [2, 2]]),
    )
    monkeypatch.setattr(
        'muvis_align.registration_methods.RegistrationMethodSkFeatures.ransac',
        lambda data, transform_type, min_samples, residual_threshold, max_trials: next(ransac_results),
    )

    transform, quality, matches, inliers = method.match(
        fixed_points,
        fixed_desc,
        moving_points,
        moving_desc,
        min_matches=2,
        cross_check=True,
        lowe_ratio=0.92,
        inlier_threshold=2.0,
        mean_size_dist=10.0,
    )

    assert np.allclose(transform.translation, [2.0, 1.0])
    assert np.array_equal(matches, [[0, 0], [1, 1], [2, 2]])
    assert np.array_equal(inliers, inliers1)
    assert quality > 0


def test_sk_registration_returns_affine_matrix_and_match_metadata(monkeypatch):
    fixed_points = np.array([
        [0.0, 0.0],
        [0.0, 10.0],
        [10.0, 0.0],
    ], dtype=np.float32)
    moving_points = fixed_points + np.array([1.0, 2.0], dtype=np.float32)

    fixed_data = make_dummy_blob_spatial_image_2d((32, 48), fixed_points, 'yx')
    moving_data = make_dummy_blob_spatial_image_2d((32, 48), moving_points, 'yx')
    method = RegistrationMethodSkFeatures(
        fixed_data,
        params={'method': 'orb', 'gaussian_sigma': 0, 'min_matches': 2, 'ransac_iterations': 1},
    )

    def _fake_detect_features(self, data, gaussian_sigma=None):
        points = fixed_points if data is fixed_data else moving_points
        desc = np.array([[1], [2], [3]], dtype=np.uint8)
        return points, desc, np.asarray(data)

    monkeypatch.setattr(RegistrationMethodSkFeatures, 'detect_features', _fake_detect_features)
    monkeypatch.setattr(
        RegistrationMethodSkFeatures,
        'match',
        lambda self, fixed_points, fixed_desc, moving_points, moving_desc, min_matches, cross_check, lowe_ratio,
               inlier_threshold, mean_size_dist: (
            EuclideanTransform(translation=[2.0, 1.0]),
            0.75,
            np.array([[0, 0], [1, 1]]),
            np.array([True, True]),
        ),
    )

    result = method.registration(fixed_data, moving_data)

    assert np.allclose(result['affine_matrix'][:2, 2], [2.0, 1.0])
    assert result['quality'] == 0.75
    assert np.array_equal(result['fixed_points'], fixed_points)
    assert np.array_equal(result['moving_points'], moving_points)
    assert np.array_equal(result['matches'], [[0, 0], [1, 1]])
    assert np.array_equal(result['inliers'], [True, True])


def test_sk_sift_extracts_only_the_sampled_keypoints_as_extracting_all_would():
    from skimage.feature import SIFT

    from scipy.ndimage import gaussian_filter

    # blurred noise wrapped to full uint8 range: hundreds of keypoints in a small image
    noise = gaussian_filter(np.random.default_rng(0).random((96, 128)), 2)
    data = make_dummy_blob_spatial_image_2d((96, 128), [(20, 30)], 'yx')
    data = data.copy(data=(noise * 255 * 8 % 255).astype(np.uint8).reshape(data.shape))
    method = RegistrationMethodSkFeatures(data, params={'method': 'sift', 'gaussian_sigma': 0, 'max_keypoints': 20,
                                                        'normalisation': False})

    np.random.seed(1)
    points, desc, processed = method.detect_features(data)

    reference = SIFT(upsampling=1)
    reference.detect_and_extract(processed)
    assert len(reference.keypoints) > 20
    np.random.seed(1)
    indices = np.random.choice(len(reference.keypoints), 20, replace=False)
    np.testing.assert_array_equal(points, reference.keypoints[indices])
    np.testing.assert_array_equal(desc, reference.descriptors[indices])


class DummyKeyPoint:
    def __init__(self, x, y):
        self.pt = (x, y)


class DummyMatch:
    def __init__(self, query_idx, train_idx, distance):
        self.queryIdx = query_idx
        self.trainIdx = train_idx
        self.distance = distance

def test_cv_detect_features_flips_opencv_keypoints_to_yx_order(monkeypatch):
    data = make_dummy_blob_spatial_image_2d((32, 48), [(7, 11), (2, 6)], 'yx')
    method = RegistrationMethodCvFeatures(data, params={'gaussian_sigma': 0})

    class DummyORB:
        def detectAndCompute(self, image, mask):
            return [DummyKeyPoint(11.0, 7.0), DummyKeyPoint(2.0, 5.5)], np.array([[1], [2]], dtype=np.uint8)

    monkeypatch.setattr(
        'muvis_align.registration_methods.RegistrationMethodCvFeatures.cv.ORB_create',
        lambda **kwargs: DummyORB(),
    )

    points, desc, processed = method.detect_features(data)

    assert desc.shape == (2, 1)
    assert processed.shape == (32, 48)
    assert np.allclose(points[0], [7.0, 11.0])
    assert np.allclose(points[1], [5.5, 2.0])

def test_cv_match_returns_translation_and_inliers_for_consistent_descriptor_pairs(monkeypatch):
    data = make_dummy_blob_spatial_image_2d((32, 48), [(0, 0)], 'yx')
    method = RegistrationMethodCvFeatures(data, params={'max_keypoints': 8, 'min_matches': 2})

    fixed_points = np.array([
        [0.0, 0.0],
        [0.0, 10.0],
        [10.0, 0.0],
    ], dtype=np.float32)
    moving_points = fixed_points + np.array([1.0, 2.0], dtype=np.float32)
    fixed_desc = np.array([[1], [2], [3]], dtype=np.uint8)
    moving_desc = np.array([[4], [5], [6]], dtype=np.uint8)

    class DummyMatcher:
        def knnMatch(self, fixed_desc_arg, moving_desc_arg, k):
            assert k == 2
            assert np.array_equal(fixed_desc_arg, fixed_desc)
            assert np.array_equal(moving_desc_arg, moving_desc)
            return [
                [DummyMatch(0, 0, 0.1), DummyMatch(0, 1, 1.0)],
                [DummyMatch(1, 1, 0.1), DummyMatch(1, 0, 1.0)],
                [DummyMatch(2, 2, 0.1), DummyMatch(2, 1, 1.0)],
            ]

    monkeypatch.setattr(
        'muvis_align.registration_methods.RegistrationMethodCvFeatures.cv.BFMatcher',
        lambda: DummyMatcher(),
    )
    monkeypatch.setattr(
        'muvis_align.registration_methods.RegistrationMethodCvFeatures.cv.findHomography',
        lambda src, dst, method, ransacReprojThreshold: (
            np.array([
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32),
            np.array([[1], [1], [1]], dtype=np.uint8),
        ),
    )

    transform, quality, matches, inliers = method.match(
        fixed_points,
        fixed_desc,
        moving_points,
        moving_desc,
        min_matches=2,
        cross_check=True,
        lowe_ratio=0.92,
        inlier_threshold=2.0,
        mean_size_dist=10.0,
    )

    assert np.allclose(transform[:2, 2], [2.0, 1.0])
    assert [(match.queryIdx, match.trainIdx) for match in matches] == [(0, 0), (1, 1), (2, 2)]
    assert inliers.shape == (3, 1)
    assert np.all(inliers)
    assert quality > 0


def test_cv_registration_returns_inverted_coordinate_order_transform(monkeypatch):
    fixed_points = np.array([
        [0.0, 0.0],
        [0.0, 10.0],
        [10.0, 0.0],
    ], dtype=np.float32)
    moving_points = fixed_points + np.array([1.0, 2.0], dtype=np.float32)

    fixed_data = make_dummy_blob_spatial_image_2d((32, 48), fixed_points, 'yx')
    moving_data = make_dummy_blob_spatial_image_2d((32, 48), moving_points, 'yx')
    method = RegistrationMethodCvFeatures(fixed_data, params={'gaussian_sigma': 0, 'min_matches': 2})

    def _fake_detect_features(self, data, gaussian_sigma=None):
        points = fixed_points if data is fixed_data else moving_points
        desc = np.array([[1], [2], [3]], dtype=np.uint8)
        return points, desc, np.asarray(data)

    monkeypatch.setattr(RegistrationMethodCvFeatures, 'detect_features', _fake_detect_features)
    monkeypatch.setattr(
        RegistrationMethodCvFeatures,
        'match',
        lambda self, fixed_points, fixed_desc, moving_points, moving_desc, min_matches, cross_check, lowe_ratio,
               inlier_threshold, mean_size_dist: (
            np.array([
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32),
            0.75,
            [DummyMatch(0, 0, 0.1), DummyMatch(1, 1, 0.1)],
            np.array([[1], [1]], dtype=np.uint8),
        ),
    )

    result = method.registration(fixed_data, moving_data)

    assert np.allclose(result['affine_matrix'][:2, 2], [1.0, 2.0])
    assert result['quality'] == 0.75
