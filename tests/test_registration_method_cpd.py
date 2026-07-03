import numpy as np
import pytest

pytest.importorskip("probreg")

from src.muvis_align.registration_methods.RegistrationMethodCPD import (
    RegistrationMethodCPD,
)
from tests.data_builders import make_dummy_blob_spatial_image

def test_detect_points_flips_2d_contour_centers_to_yx_order(monkeypatch):
    data = make_dummy_blob_spatial_image((32, 48), [(7, 11), (2, 6)], 'yx')
    method = RegistrationMethodCPD(data, params={})

    monkeypatch.setattr(
        'src.muvis_align.registration_methods.RegistrationMethodCPD.detect_area_points',
        lambda data: [
            (np.array([11.0, 7.0], dtype=np.float32), 10.0),
            (np.array([5.5, 2.0], dtype=np.float32), 4.0),
        ],
    )

    points = method.detect_points(data)

    assert len(points) == 2
    assert np.allclose(points[0], [7.0, 11.0])
    assert np.allclose(points[1], [2.0, 5.5])


def test_detect_points_keeps_3d_blob_coordinates_unchanged(monkeypatch):
    data = make_dummy_blob_spatial_image((8, 32, 48), [(1, 6, 11), (3, 8, 5)], 'zyx')
    method = RegistrationMethodCPD(data, params={})

    monkeypatch.setattr(
        'src.muvis_align.registration_methods.RegistrationMethodCPD.detect_volume_points',
        lambda data: np.array([
            [1.0, 6.0, 11.0],
            [3.0, 8.0, 5.0],
        ], dtype=np.float32),
    )

    points = method.detect_points(data)

    assert np.allclose(points, [[1.0, 6.0, 11.0], [3.0, 8.0, 5.0]])


def test_detect_points_returns_empty_list_when_no_2d_areas_are_found(monkeypatch):
    data = make_dummy_blob_spatial_image((32, 48), [(7, 11)], 'yx')
    method = RegistrationMethodCPD(data, params={})

    monkeypatch.setattr(
        'src.muvis_align.registration_methods.RegistrationMethodCPD.detect_area_points',
        lambda data: [],
    )

    points = method.detect_points(data)

    assert points == []


def test_registration_reports_consistent_matches_and_inliers_for_translated_points(monkeypatch):
    fixed_points = np.array([
        [0.0, 0.0],
        [0.0, 10.0],
        [10.0, 0.0],
    ], dtype=np.float32)
    moving_points = np.array([
        [11.0, 2.0],
        [1.0, 2.0],
        [1.0, 12.0],
    ], dtype=np.float32)

    fixed_data = make_dummy_blob_spatial_image((32, 48), fixed_points, 'yx')
    moving_data = make_dummy_blob_spatial_image((32, 48), moving_points, 'yx')
    method = RegistrationMethodCPD(fixed_data, params={'max_iter': 5})

    def _fake_detect_points(self, data, gaussian_sigma=None):
        return fixed_points if data is fixed_data else moving_points

    monkeypatch.setattr(RegistrationMethodCPD, 'detect_points', _fake_detect_points)
    monkeypatch.setattr(
        'src.muvis_align.registration_methods.RegistrationMethodCPD.get_mean_nn_distance',
        lambda points1, points2: 5.0,
    )

    captured = {}

    def _fake_registration_cpd(source, target, maxiter):
        captured['source'] = np.asarray(source)
        captured['target'] = np.asarray(target)

        class _Transformation:
            rot = np.eye(3, dtype=np.float32)
            scale = 1.0
            t = np.array([1.0, 2.0, 0.0], dtype=np.float32)

        class _Result:
            transformation = _Transformation()

        return _Result()

    monkeypatch.setattr(
        'src.muvis_align.registration_methods.RegistrationMethodCPD.cpd.registration_cpd',
        _fake_registration_cpd,
    )

    result = method.registration(fixed_data, moving_data)

    assert np.allclose(captured['source'][:, :2], fixed_points)
    assert np.allclose(captured['target'][:, :2], moving_points)
    assert set(map(tuple, result['matches'])) == {(0, 1), (1, 2), (2, 0)}
    assert all(result['inliers'])
    assert result['quality'] > 0


def test_registration_returns_none_matches_when_too_few_points(monkeypatch):
    fixed_data = make_dummy_blob_spatial_image((32, 48), [(1, 2)], 'yx')
    moving_data = make_dummy_blob_spatial_image((32, 48), [(1, 2)], 'yx')
    method = RegistrationMethodCPD(fixed_data, params={})

    def _fake_detect_points(self, data, gaussian_sigma=None):
        return [np.array([1.0, 2.0], dtype=np.float32)]

    monkeypatch.setattr(RegistrationMethodCPD, 'detect_points', _fake_detect_points)
    monkeypatch.setattr(
        'src.muvis_align.registration_methods.RegistrationMethodCPD.get_mean_nn_distance',
        lambda points1, points2: 5.0,
    )

    result = method.registration(fixed_data, moving_data)

    assert result['quality'] == 0
    assert result['matches'] is None
    assert result['inliers'] is None


