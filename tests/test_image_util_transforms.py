"""
Unit tests for image utility functions, particularly transform dimension handling.

Tests focus on the fix for dimension mismatch when 2D images have 3D transforms.
"""

import numpy as np
import pytest
import xarray as xr
from unittest.mock import MagicMock, patch

from muvis_align.image.util import (
    _adapt_transform_to_image_dims,
    draw_keypoints_matches_napari,
    gaussian_filter_sim,
    get_overlap_images,
)


def create_mock_sim_2d(shape=(400, 400), origin=(0.048, 0.048), spacing=(0.064, 0.064)):
    """Create a mock 2D image (xarray DataArray) with spatial coordinates."""
    y_coords = np.arange(shape[0]) * spacing[0] + origin[0]
    x_coords = np.arange(shape[1]) * spacing[1] + origin[1]
    
    data = np.random.randint(0, 100, size=(1, 1, *shape), dtype=np.uint16)
    
    sim = xr.DataArray(
        data,
        dims=['t', 'c', 'y', 'x'],
        coords={
            't': [0],
            'c': [''],
            'y': y_coords,
            'x': x_coords,
        }
    )
    
    return sim


def create_3d_transform():
    """Create a 4x4 3D affine transform matrix (for z, y, x, homogeneous)."""
    transform_data = np.eye(4)
    transform = xr.DataArray(
        transform_data,
        dims=['x_in', 'x_out'],
        coords={
            'x_in': ['z', 'y', 'x', '1'],
            'x_out': ['z', 'y', 'x', '1'],
        }
    )
    return transform


def create_2d_transform():
    """Create a 3x3 2D affine transform matrix (for y, x, homogeneous)."""
    transform_data = np.eye(3)
    transform = xr.DataArray(
        transform_data,
        dims=['x_in', 'x_out'],
        coords={
            'x_in': ['y', 'x', '1'],
            'x_out': ['y', 'x', '1'],
        }
    )
    return transform


def test_adapt_transform_2d_image_with_3d_transform():
    """Test that 3D transform is reduced to 3x3 for 2D images."""
    sim = create_mock_sim_2d()
    transform_3d = create_3d_transform()
    
    # Add 3D transform to sim
    sim.attrs['transforms'] = {'source_metadata': transform_3d}
    
    # Adapt transform
    adapted = _adapt_transform_to_image_dims(sim, transform_3d, 'source_metadata')
    
    # Should extract y, x, 1 from z, y, x, 1
    assert adapted.shape == (3, 3)
    assert list(adapted.coords['x_in'].values) == ['y', 'x', '1']
    assert list(adapted.coords['x_out'].values) == ['y', 'x', '1']


def test_adapt_transform_2d_image_with_2d_transform():
    """Test that 2D transform stays 3x3 for 2D images (no-op)."""
    sim = create_mock_sim_2d()
    transform_2d = create_2d_transform()
    
    # Add 2D transform to sim
    sim.attrs['transforms'] = {'source_metadata': transform_2d}
    
    # Adapt transform
    adapted = _adapt_transform_to_image_dims(sim, transform_2d, 'source_metadata')
    
    # Should remain 3x3
    assert adapted.shape == (3, 3)
    assert list(adapted.coords['x_in'].values) == ['y', 'x', '1']
    assert np.allclose(adapted.values, np.eye(3))


def test_adapt_transform_3d_image_with_3d_transform():
    """Test that 3D transform stays 4x4 for 3D images (no-op)."""
    # Create 3D sim
    sim = create_mock_sim_2d()
    sim = sim.expand_dims(z=10)
    
    transform_3d = create_3d_transform()
    
    # Add 3D transform to sim
    sim.attrs['transforms'] = {'source_metadata': transform_3d}
    
    # Adapt transform
    adapted = _adapt_transform_to_image_dims(sim, transform_3d, 'source_metadata')
    
    # Should remain 4x4
    assert adapted.shape == (4, 4)
    assert list(adapted.coords['x_in'].values) == ['z', 'y', 'x', '1']


def test_get_overlap_images_adapts_transform_for_2d():
    """Test that get_overlap_images adapts 3D transforms for 2D images."""
    sim1 = create_mock_sim_2d(shape=(400, 400), origin=(0.048, 0.048))
    sim2 = create_mock_sim_2d(shape=(400, 400), origin=(24.048, 0.048))
    
    transform_3d = create_3d_transform()
    
    # Add transforms to both sims
    sim1.attrs['transforms'] = {'source_metadata': transform_3d}
    sim2.attrs['transforms'] = {'source_metadata': transform_3d}
    
    # Mock the multiview_stitcher functions
    with patch('muvis_align.image.util._get_overlap_bboxes') as mock_overlap:
        with patch('muvis_align.image.util.si_utils.get_spatial_dims_from_sim') as mock_dims:
            with patch('muvis_align.image.util.si_utils.get_spacing_from_sim') as mock_spacing:
                mock_dims.return_value = ['y', 'x']
                mock_spacing.return_value = {'y': 0.064, 'x': 0.064}
                mock_overlap.return_value = {
                    'lowers': np.array([0, 0]),
                    'uppers': np.array([25.584, 25.584]),
                }
                
                # This should not raise an error
                try:
                    get_overlap_images(sim1, sim2, 'source_metadata')
                except Exception as e:
                    # If we get a shape mismatch error from np.dot, the fix didn't work
                    if "shapes" in str(e) and "not aligned" in str(e):
                        raise AssertionError(
                            f"Transform dimension mismatch still occurring: {e}"
                        )
                    # Other errors are OK for this mock test (we're only testing transform adaptation)
                    pass


def test_adapted_transform_values_are_identity_submatrix():
    """Test that adapted transform preserves correct values from original."""
    sim = create_mock_sim_2d()
    
    # Create 3D transform with non-trivial values
    transform_3d_data = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 5.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    transform_3d = xr.DataArray(
        transform_3d_data,
        dims=['x_in', 'x_out'],
        coords={
            'x_in': ['z', 'y', 'x', '1'],
            'x_out': ['z', 'y', 'x', '1'],
        }
    )
    
    sim.attrs['transforms'] = {'source_metadata': transform_3d}
    
    # Adapt - should extract rows/cols for y, x, 1
    adapted = _adapt_transform_to_image_dims(sim, transform_3d, 'source_metadata')
    
    # Check that we got the right 3x3 submatrix
    # Should be rows [y, x, 1] and cols [y, x, 1] from original
    # y is index 1, x is index 2, 1 is index 3
    expected = np.array([
        [1.0, 0.0, 5.0],   # y row
        [0.0, 1.0, 0.0],   # x row
        [0.0, 0.0, 1.0],   # 1 row
    ])
    
    assert np.allclose(adapted.values, expected)


def test_gaussian_filter_sim_preserves_uint_range_for_multichannel_sim():
    sim = create_mock_sim_2d(shape=(32, 32))
    sim = xr.concat([sim, sim * 2], dim="c")
    sim = sim.assign_coords(c=["ch0", "ch1"])
    sim.attrs["transforms"] = {"source_metadata": create_2d_transform()}

    filtered = gaussian_filter_sim(sim, "source_metadata", sigma=2.0)

    assert filtered.dtype == sim.dtype
    assert np.max(np.asarray(filtered)) > 1


def test_draw_keypoints_matches_napari_splits_match_layers_by_inlier_state():
    image = np.zeros((16, 16), dtype=np.uint16)
    points1 = np.array([[2, 2], [4, 4]], dtype=float)
    points2 = np.array([[2, 2], [4, 4]], dtype=float)
    matches = np.array([[0, 0], [1, 1]], dtype=int)
    inliers = np.array([False, True], dtype=bool)

    layers = draw_keypoints_matches_napari(
        image, points1, image, points2, matches=matches, inliers=inliers
    )
    layer_map = {kwargs["name"]: (data, kwargs, layer_type) for data, kwargs, layer_type in layers}

    points_data, points_kwargs, points_type = layer_map["keypoints"]
    non_inliers_data, non_inliers_kwargs, non_inliers_type = layer_map["matches"]
    inliers_data, inliers_kwargs, inliers_type = layer_map["matches_inliers"]

    assert points_type == "points"
    assert non_inliers_type == "shapes"
    assert inliers_type == "shapes"

    assert len(non_inliers_data) == 1
    assert len(inliers_data) == 1


def test_restoring_a_transform_snapshot_puts_back_the_previous_value_or_removes_a_new_one():
    import glob
    from multiview_stitcher import msi_utils, param_utils
    from muvis_align.MVSRegistration import MVSRegistration
    from muvis_align.image.util import get_msim_transform_keys, restore_msims_transform, snapshot_msims_transform

    reg = MVSRegistration()
    reg.init(operation='register', input_path=sorted(glob.glob('data/S000/*.ome.zarr'))[:2],
             output_path='../../output/test_transform_snapshot/')
    reg.init_data()
    msims = reg.msims

    def shifted(shift):
        return param_utils.affine_to_xaffine(param_utils.affine_from_translation([0.0, shift]))

    def translation(msim):
        return float(np.asarray(msi_utils.get_transform_from_msim(msim, transform_key='registered')).squeeze()[1, 2])

    msi_utils.set_affine_transform(msims[0], shifted(1.0), transform_key='registered',
                                   base_transform_key=reg.source_transform_key)
    snapshot = snapshot_msims_transform(msims, 'registered')
    for msim in msims:
        msi_utils.set_affine_transform(msim, shifted(5.0), transform_key='registered',
                                       base_transform_key=reg.source_transform_key)

    restore_msims_transform(msims, 'registered', snapshot)

    assert translation(msims[0]) == pytest.approx(1.0)
    assert 'registered' in get_msim_transform_keys(msims[0])
    assert 'registered' not in get_msim_transform_keys(msims[1])


def test_grid_point_pairs_follow_the_transform_from_moving_to_fixed():
    from muvis_align.image.util import grid_point_pairs

    # moving (y, x) -> fixed (y - 5, x - 10): content at fixed p lies at p + (5, 10) in the moving image
    matrix = np.array([[1, 0, -5], [0, 1, -10], [0, 0, 1]], dtype=float)

    fixed, moving, matches, inliers = grid_point_pairs((60, 120), (60, 120), matrix, points_per_short_axis=6)

    np.testing.assert_array_equal(moving, fixed + [5, 10])
    assert fixed[:, 0].min() == 5 and fixed[:, 1].min() == 5
    assert np.all(moving <= [59, 119])
    np.testing.assert_array_equal(matches, np.column_stack([np.arange(len(fixed))] * 2))
    assert inliers.dtype == bool and inliers.all()


def test_grid_point_pairs_take_the_spatial_block_of_a_larger_matrix_and_drop_points_off_the_moving_image():
    from muvis_align.image.util import grid_point_pairs

    # (t, c, y, x) as multiview-stitcher gives it: spatial dims last
    matrix = np.eye(5)
    matrix[2:4, 4] = [-30, 0]

    fixed, moving, _, _ = grid_point_pairs((60, 60), (40, 60), matrix, points_per_short_axis=6)

    np.testing.assert_array_equal(moving, fixed + [30, 0])
    assert np.all(moving[:, 0] <= 39)
    assert len(fixed) == 6 * 1
