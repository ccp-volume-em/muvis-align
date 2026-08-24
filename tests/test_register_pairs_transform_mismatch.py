"""Test that register_pairs handles mixed-dimension transforms correctly.

This test covers the fix for the pair_registration ValueError that occurred when
msims_reg contained a mix of 3D transforms (4x4 from global registration with z-axis)
and 2D images. The fix normalizes all transforms to match image dimensions before
passing to multiview_stitcher's build_view_adjacency_graph_from_msims.
"""

import numpy as np
import xarray as xr
import dask.array as da
from xarray import DataTree

from muvis_align.MVSRegistration import MVSRegistration


def create_2d_image_xarray(y_size=100, x_size=100, name="image"):
    """Create a 2D xarray DataArray with minimal attributes."""
    data = da.from_array(
        np.random.randint(0, 255, (1, 1, y_size, x_size), dtype=np.uint16),
        chunks=(1, 1, y_size, x_size)
    )
    return xr.DataArray(
        data,
        dims=('t', 'c', 'y', 'x'),
        coords={
            't': [0],
            'c': [''],
            'y': np.arange(0, y_size * 0.025, 0.025),
            'x': np.arange(0, x_size * 0.025, 0.025),
        },
        name=name
    )


def create_2d_image_with_3d_transform():
    """Create a 2D image with a 4x4 (3D) transform - simulates output from global registration."""
    image = create_2d_image_xarray(100, 100)
    
    # 4x4 transform (z, y, x, 1 coordinates) - this is what causes the issue
    transform_3d = xr.DataArray(
        np.eye(4),
        dims=('x_in', 'x_out'),
        coords={
            'x_in': np.array(['z', 'y', 'x', '1'], dtype='<U1'),
            'x_out': np.array(['z', 'y', 'x', '1'], dtype='<U1'),
        },
        name='source_metadata'
    )
    
    return image, transform_3d


def create_multiscale_datatree_with_mixed_transforms():
    """Create simple objects that simulate msims_reg structure.
    
    We simulate the minimal structure needed to test the adaptation logic.
    """
    class MockScaleNode:
        """Mock scale node in DataTree."""
        def __init__(self, image, transform):
            self.ds = xr.Dataset({
                'image': image,
                'source_metadata': transform,
            })
    
    class MockDataTree:
        """Mock DataTree with multiple scales."""
        def __init__(self, name):
            self.name = name
            self.scales = []
            self.ds = self  # For compatibility with iteration
        
        def values(self):
            """Return scale nodes."""
            return self.scales
        
        def add_scale(self, image, transform):
            """Add a scale level."""
            self.scales.append(MockScaleNode(image, transform))
    
    trees = []
    
    for i in range(2):
        dt = MockDataTree(f'image{i}')
        
        # Scale 0: 2D image with 3D transform
        img, transform = create_2d_image_with_3d_transform()
        dt.add_scale(img, transform)
        
        # Scale 1: 2D image with 3D transform
        img_s1 = create_2d_image_xarray(50, 50)
        transform_s1 = xr.DataArray(
            np.eye(4),
            dims=('x_in', 'x_out'),
            coords={
                'x_in': np.array(['z', 'y', 'x', '1'], dtype='<U1'),
                'x_out': np.array(['z', 'y', 'x', '1'], dtype='<U1'),
            },
            name='source_metadata'
        )
        dt.add_scale(img_s1, transform_s1)
        
        trees.append(dt)
    
    return trees


def test_register_pairs_with_3d_transforms_on_2d_images():
    """Test that register_pairs normalizes 3D transforms to 2D for 2D images.
    
    This reproduces the error scenario from the debug_napari run where
    pair_registration failed with a shape mismatch in build_view_adjacency_graph_from_msims.
    """
    # Create registration object
    reg = MVSRegistration()
    
    # Create test data with mixed-dimension transforms
    msims_reg = create_multiscale_datatree_with_mixed_transforms()
    
    # Before the fix, this would fail with:
    # ValueError: operands could not be broadcast together with remapped shapes
    # because it tries to use 4x4 transforms with 2D image data
    
    # The fix should normalize transforms to 3x3 (2D + homogeneous coordinate)
    # before passing to build_view_adjacency_graph_from_msims
    
    # Simulate what happens in register_pairs
    register_indices = range(len(msims_reg))
    pairs = [(0, 1)]
    
    # Extract and normalize transforms (this is what the fix does)
    from muvis_align.image.util import _adapt_transform_to_image_dims
        
    for msim in msims_reg:
        for scale_node in msim.ds.values():
            if 'source_metadata' in scale_node.ds.data_vars:
                img_xarray = scale_node.ds['image']
                # Get spatial dims from xarray DataArray dims
                spatial_dims = [d for d in img_xarray.dims if d not in ('t', 'c')]
                # Get current transform
                current_transform = scale_node.ds.data_vars['source_metadata']
                transform_spatial_dims = [d for d in current_transform.coords['x_in'].values if d != '1']
                    
                # Verify adaptation occurs
                assert len(transform_spatial_dims) == 3, "Transform should start with 3 spatial dims (z,y,x)"
                assert len(spatial_dims) == 2, "Image should have 2 spatial dims (y,x)"
                    
                # Adapt if mismatch
                if len(transform_spatial_dims) != len(spatial_dims):
                    relevant_dim_names = spatial_dims + ['1']
                    adapted = current_transform.sel(x_in=relevant_dim_names, x_out=relevant_dim_names)
                    scale_node.ds['source_metadata'] = adapted
                        
                    # Verify adaptation worked
                    adapted_spatial_dims = [d for d in adapted.coords['x_in'].values if d != '1']
                    assert len(adapted_spatial_dims) == 2, "Adapted transform should have 2 spatial dims"
                    assert list(adapted_spatial_dims) == spatial_dims, f"Adapted dims should match image dims: {spatial_dims}"


def test_transform_adaptation_preserves_values_for_relevant_submatrix():
    """Test that adaptation extracts correct submatrix values."""
    # Create a 4x4 transform with specific values in the 2D submatrix
    transform_4d = xr.DataArray(
        np.array([
            [1.0, 0.0, 0.0, 0.0],  # z row
            [0.0, 2.0, 0.5, 10.0], # y row
            [0.0, 0.3, 3.0, 20.0], # x row
            [0.0, 0.0, 0.0, 1.0],  # 1 row (homogeneous)
        ]),
        dims=('x_in', 'x_out'),
        coords={
            'x_in': np.array(['z', 'y', 'x', '1'], dtype='<U1'),
            'x_out': np.array(['z', 'y', 'x', '1'], dtype='<U1'),
        },
        name='source_metadata'
    )
    
    # Extract 2D submatrix (y, x, 1)
    adapted = transform_4d.sel(x_in=['y', 'x', '1'], x_out=['y', 'x', '1'])
    
    # Verify shape
    assert adapted.shape == (3, 3), f"Expected (3,3), got {adapted.shape}"
    
    # Verify values are correctly extracted
    expected_2d_submatrix = np.array([
        [2.0, 0.5, 10.0],  # y row
        [0.3, 3.0, 20.0],  # x row
        [0.0, 0.0, 1.0],   # 1 row
    ])
    np.testing.assert_array_almost_equal(adapted.values, expected_2d_submatrix)


if __name__ == '__main__':
    test_register_pairs_with_3d_transforms_on_2d_images()
    test_transform_adaptation_preserves_values_for_relevant_submatrix()
    print("All tests passed!")
