"""Test copy_transforms handles 't' dimension and transform size mismatches.

This test covers the fix for ValueError that occurred when copy_transforms tried to
assign a 3D transform (with 't' dimension) to a 4×4 target transform during
global registration results update.
"""

import numpy as np
import xarray as xr
import dask.array as da
from multiview_stitcher import spatial_image_utils as si_utils, param_utils

from muvis_align.image.util import copy_transforms


def create_2d_source_image_with_transform():
    """Create a 2D source image with 3×3 transform."""
    # Create 2D image
    data = da.from_array(
        np.random.randint(0, 255, (1, 1, 100, 100), dtype=np.uint16),
        chunks=(1, 1, 100, 100)
    )
    image = xr.DataArray(
        data,
        dims=('t', 'c', 'y', 'x'),
        coords={
            't': [0],
            'c': [''],
            'y': np.arange(0, 100 * 0.025, 0.025),
            'x': np.arange(0, 100 * 0.025, 0.025),
        }
    )
    
    # Add 3×3 transform with 't' dimension
    transform_3x3 = xr.DataArray(
        np.eye(3)[np.newaxis, :, :],  # Add 't' dimension
        dims=('t', 'x_in', 'x_out'),
        coords={
            't': [0],
            'x_in': np.array(['y', 'x', '1'], dtype='<U1'),
            'x_out': np.array(['y', 'x', '1'], dtype='<U1'),
        },
        name='registered'
    )
    
    image.attrs['transforms'] = {'registered': transform_3x3}
    return image


def create_3d_target_image():
    """Create a 3D target image with 4×4 transform."""
    # Create 3D image
    data = da.from_array(
        np.random.randint(0, 255, (1, 1, 1, 100, 100), dtype=np.uint16),
        chunks=(1, 1, 1, 100, 100)
    )
    image = xr.DataArray(
        data,
        dims=('t', 'c', 'z', 'y', 'x'),
        coords={
            't': [0],
            'c': [''],
            'z': [0],
            'y': np.arange(0, 100 * 0.025, 0.025),
            'x': np.arange(0, 100 * 0.025, 0.025),
        }
    )
    
    # Add 4×4 transform (no 't' dimension in target)
    transform_4x4 = xr.DataArray(
        np.eye(4),
        dims=('x_in', 'x_out'),
        coords={
            'x_in': np.array(['z', 'y', 'x', '1'], dtype='<U1'),
            'x_out': np.array(['z', 'y', 'x', '1'], dtype='<U1'),
        },
        name='registered'
    )
    
    image.attrs['transforms'] = {'registered': transform_4x4}
    return image


def test_copy_transforms_handles_t_dimension():
    """Test that copy_transforms handles 't' dimension in source transforms.
    
    This reproduces the error scenario from the global registration update where
    source transforms with 't' dimension couldn't be assigned to target transforms
    without 't' dimension.
    """
    # Create source and target images
    source_image = create_2d_source_image_with_transform()
    target_image = create_3d_target_image()
    
    # Get the transform before copy
    original_transform = si_utils.get_affine_from_sim(source_image, 'registered')
    assert 't' in original_transform.dims, "Source transform should have 't' dimension"
    assert original_transform.shape == (1, 3, 3), f"Expected (1,3,3), got {original_transform.shape}"
    
    # This should not raise an error
    copy_transforms([source_image], [target_image], 'registered')
    
    # Verify target transform was updated
    target_transform = si_utils.get_affine_from_sim(target_image, 'registered')
    assert target_transform.shape == (4, 4), f"Expected (4,4), got {target_transform.shape}"
    assert 't' not in target_transform.dims, "Target transform should not have 't' dimension"


def test_copy_transforms_3x3_to_4x4_identity():
    """Test that 3×3 identity gets expanded to 4×4 identity correctly."""
    # Create source with 3×3 identity
    source_image = create_2d_source_image_with_transform()
    target_image = create_3d_target_image()
    
    # Perform copy
    copy_transforms([source_image], [target_image], 'registered')
    
    # Get target transform and verify it's identity
    target_transform = si_utils.get_affine_from_sim(target_image, 'registered')
    expected_4x4_identity = np.eye(4)
    np.testing.assert_array_almost_equal(target_transform.values, expected_4x4_identity)


def test_copy_transforms_multiple_images():
    """Test copy_transforms with multiple source/target pairs."""
    # Create multiple image pairs
    source_images = [create_2d_source_image_with_transform() for _ in range(3)]
    target_images = [create_3d_target_image() for _ in range(3)]
    
    # This should handle multiple images
    copy_transforms(source_images, target_images, 'registered')
    
    # Verify all targets were updated
    for target_image in target_images:
        target_transform = si_utils.get_affine_from_sim(target_image, 'registered')
        assert target_transform.shape == (4, 4)
        assert 't' not in target_transform.dims


if __name__ == '__main__':
    test_copy_transforms_handles_t_dimension()
    test_copy_transforms_3x3_to_4x4_identity()
    test_copy_transforms_multiple_images()
    print("All tests passed!")
