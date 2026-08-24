import numpy as np
import pytest

from muvis_align.util import calculate_rigid_difference, create_transform


@pytest.mark.parametrize(
    'transform1, transform2, expected',
    [
        (
            np.eye(3),
            create_transform((0, 0), 0, translation=[1, 2], matrix_size=3),
            create_transform((0, 0), 0, translation=[1, 2], matrix_size=3),
        ),
        (
            create_transform((0, 0), 0, translation=[-1, -2], matrix_size=3),
            np.eye(3),
            create_transform((0, 0), 0, translation=[1, 2], matrix_size=3),
        ),
        (
            np.eye(3),
            create_transform((0, 0), 10, translation=[0, 0], matrix_size=3),
            create_transform((0, 0), 10, translation=[0, 0], matrix_size=3),
        ),
        (
            create_transform((0, 0), -10, translation=[0, 0], matrix_size=3),
            np.eye(3),
            create_transform((0, 0), 10, translation=[0, 0], matrix_size=3),
        ),
        (
            create_transform((0, 0), 10, translation=[1, 2], matrix_size=3),
            create_transform((0, 0), 35, translation=[4, 6], matrix_size=3),
            create_transform((0, 0), 25, translation=[2.25983055, 4.46017555], matrix_size=3),
        ),
        (
            create_transform((0, 0), 15, translation=[1, 2, 3], matrix_size=4),
            create_transform((0, 0), 40, translation=[5, 7, 11], matrix_size=4),
            create_transform((0, 0), 25, translation=[2.56960808, 5.86490531, 8], matrix_size=4),
        ),
    ],
)
def test_calculate_rigid_difference(transform1, transform2, expected):
    """Calculate rigid differences for 2D and 3D affine transforms."""
    np.testing.assert_allclose(calculate_rigid_difference(transform1, transform2), expected)
