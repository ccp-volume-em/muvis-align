import dask.array as da
import numpy as np
import pytest
import xarray as xr
from multiview_stitcher import msi_utils, param_utils

from muvis_align.image.util import (_adapt_transform_to_image_dims, copy_transforms_to_msims,
                                    draw_keypoints_matches_napari, gaussian_filter_sim, get_msim_transform_keys,
                                    get_overlap_images, grid_point_pairs, make_msims_3d, NoOverlapError,
                                    restore_msims_transform, set_msim_affine, snapshot_msims_transform,
                                    widen_xaffine_to_3d, widened_affine_matrix)
from muvis_align.util import create_transform
from tests.data_builders import make_msim, make_sim


def labelled_affine(matrix):
    dims = ['z', 'y', 'x', '1'][-len(matrix):]
    return xr.DataArray(np.asarray(matrix, dtype=float), dims=['x_in', 'x_out'], coords={'x_in': dims, 'x_out': dims})


TRANSLATED_3D = np.eye(4)
TRANSLATED_3D[1, 3] = 5.0


@pytest.mark.parametrize('image_dims, transform, expected_dims, expected', [
    # a 3D transform on a 2D image keeps its y/x block, translation included
    ('tcyx', TRANSLATED_3D, ['y', 'x', '1'], [[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ('tcyx', np.eye(3), ['y', 'x', '1'], np.eye(3)),
    ('tczyx', np.eye(4), ['z', 'y', 'x', '1'], np.eye(4)),
])
def test_a_transform_is_adapted_to_the_images_own_dims(image_dims, transform, expected_dims, expected):
    sim = make_sim(np.zeros((1, 1) + (4, 32, 32)[5 - len(image_dims):], dtype=np.uint16), image_dims)
    sim.attrs['transforms'] = {'source_metadata': labelled_affine(transform)}

    adapted = _adapt_transform_to_image_dims(sim, labelled_affine(transform), 'source_metadata')

    assert list(adapted.coords['x_in'].values) == list(adapted.coords['x_out'].values) == expected_dims
    np.testing.assert_allclose(adapted.values, expected)


def test_gaussian_filter_sim_preserves_uint_range_for_multichannel_sim():
    base = np.random.default_rng(0).integers(0, 100, (32, 32), dtype=np.uint16)
    sim = make_sim(np.stack([base, base * 2])[np.newaxis], 'tcyx', c_coords=['ch0', 'ch1'])

    filtered = gaussian_filter_sim(sim, 'source_metadata', sigma=2.0)

    assert filtered.dtype == sim.dtype
    assert np.max(np.asarray(filtered)) > 1


@pytest.mark.parametrize('side, point_size, line_width', [(60, 6, 1), (1600, 20, 4)])
def test_preview_matches_split_by_inlier_state_and_scale_with_the_image_shown(side, point_size, line_width):
    image = np.zeros((side, side // 2), np.float32)
    points = np.array([[10.0, 10.0], [20.0, 20.0]])

    layers = {kwargs['name']: (data, kwargs, layer_type) for data, kwargs, layer_type in
              draw_keypoints_matches_napari(image, points, image, points, matches=np.array([[0, 0], [1, 1]]),
                                            inliers=np.array([False, True]))}

    assert [layers[name][2] for name in ('keypoints', 'matches', 'matches_inliers')] == ['points', 'shapes', 'shapes']
    assert len(layers['matches'][0]) == len(layers['matches_inliers'][0]) == 1
    assert layers['keypoints'][1]['size'] == point_size
    assert layers['matches_inliers'][1]['edge_width'] == line_width


def pyramid_msim(index=0):
    return make_msim(da.zeros((1, 1, 64, 64), dtype=np.uint8, chunks=32), 'tcyx', scale_factors=[2, 4],
                     translation={'y': 0.0, 'x': 10.0 * index})


def test_restoring_a_transform_snapshot_puts_back_the_previous_value_or_removes_a_new_one():
    msims = [pyramid_msim(0), pyramid_msim(1)]

    def shifted(shift):
        return param_utils.affine_to_xaffine(param_utils.affine_from_translation([0.0, shift]))

    def translation(msim):
        return float(np.asarray(msi_utils.get_transform_from_msim(msim, transform_key='registered')).squeeze()[1, 2])

    msi_utils.set_affine_transform(msims[0], shifted(1.0), transform_key='registered',
                                   base_transform_key='source_metadata')
    snapshot = snapshot_msims_transform(msims, 'registered')
    for msim in msims:
        msi_utils.set_affine_transform(msim, shifted(5.0), transform_key='registered',
                                       base_transform_key='source_metadata')

    restore_msims_transform(msims, 'registered', snapshot)

    assert translation(msims[0]) == pytest.approx(1.0)
    assert 'registered' in get_msim_transform_keys(msims[0])
    assert 'registered' not in get_msim_transform_keys(msims[1])


def test_grid_point_pairs_follow_the_transform_from_fixed_to_moving():
    # fixed p lies at p + (5, 10) in the moving image
    matrix = np.array([[1, 0, 5], [0, 1, 10], [0, 0, 1]], dtype=float)

    fixed, moving, matches, inliers = grid_point_pairs((60, 120), (60, 120), matrix)

    np.testing.assert_array_equal(moving, fixed + [5, 10])
    step = np.diff(np.unique(fixed[:, 0]))[0]
    np.testing.assert_allclose(fixed.min(axis=0), step / 2)
    np.testing.assert_array_equal(matches, np.column_stack([np.arange(len(fixed))] * 2))
    assert inliers.dtype == bool and inliers.all()


def test_grid_point_pairs_take_the_spatial_block_of_a_larger_matrix_and_keep_points_off_the_moving_image():
    # (t, c, y, x) as multiview-stitcher gives it: spatial dims last
    matrix = np.eye(5)
    matrix[2:4, 4] = [20, 0]

    fixed, moving, _, _ = grid_point_pairs((60, 60), (40, 60), matrix)

    np.testing.assert_array_equal(moving, fixed + [20, 0])
    # the whole fixed image at one spacing, also rows whose partner lies past the moving image's 40 rows
    rows = np.unique(fixed[:, 0])
    assert rows.max() > 60 - np.diff(rows)[0] and moving[:, 0].max() > 39


@pytest.mark.parametrize('shape, expected', [((800, 50), (30, 3)), ((1407, 422), (30, 9)), ((62, 83), (4, 6))])
def test_grid_point_pairs_put_30_along_the_longest_side_at_least_3_along_any_and_rings_apart(shape, expected):
    fixed, _, _, _ = grid_point_pairs(shape, shape, np.eye(3))

    assert (len(np.unique(fixed[:, 0])), len(np.unique(fixed[:, 1]))) == expected


@pytest.mark.parametrize('offset, overlaps, widen', [(1000.0, False, False), (60.0, True, False), (60.0, True, True)])
def test_get_overlap_images_says_plainly_when_two_images_do_not_overlap(offset, overlaps, widen):
    sims = [make_sim(np.ones((100, 100), np.float32), translation={'y': 0.0, 'x': x_offset}, transform_key='source')
            for x_offset in (0.0, offset)]
    if widen:
        # a 3D transform on a 2D image, as a promoted msim leaves it
        for sim in sims:
            sim.attrs['transforms']['source'] = widen_xaffine_to_3d(sim.attrs['transforms']['source'])

    if overlaps:
        overlap1, overlap2, _ = get_overlap_images(sims[0], sims[1], 'source')
        assert overlap1.sizes['x'] >= 40 and overlap2.sizes['x'] >= 40
    else:
        with pytest.raises(NoOverlapError):
            get_overlap_images(sims[0], sims[1], 'source')


@pytest.mark.parametrize('base_transform_key', [None, 'source_metadata'])
def test_setting_a_msim_affine_matches_multiview_stitchers(base_transform_key):
    matrix = np.array([[0.9, -0.1, 5.0], [0.1, 0.9, -3.0], [0.0, 0.0, 1.0]])
    transform = param_utils.affine_to_xaffine(matrix, t_coords=[0])
    expected, msim = pyramid_msim(), pyramid_msim()
    msi_utils.set_affine_transform(expected, transform, transform_key='registered', base_transform_key=base_transform_key)

    set_msim_affine(msim, transform, transform_key='registered', base_transform_key=base_transform_key)

    for scale_key in msi_utils.get_sorted_scale_keys(expected):
        xr.testing.assert_identical(msim[scale_key].to_dataset(), expected[scale_key].to_dataset())


def test_a_2d_transform_copied_onto_3d_msims_is_widened_as_before():
    matrix = np.array([[0.9, -0.1, 5.0], [0.1, 0.9, -3.0], [0.0, 0.0, 1.0]])
    source = pyramid_msim()
    msi_utils.set_affine_transform(source, param_utils.affine_to_xaffine(matrix, t_coords=[0]), transform_key='registered')
    target = make_msims_3d([pyramid_msim()], z_scale=1.0)[0]
    widened = param_utils.identity_transform(ndim=3)
    widened.loc[{'x_in': ['y', 'x', '1'], 'x_out': ['y', 'x', '1']}] = matrix

    copy_transforms_to_msims([source], [target], 'registered')

    xr.testing.assert_identical(msi_utils.get_transform_from_msim(target, 'registered').rename(None), widened)


def reference_widen(transform):
    """The label-based implementation the numpy one replaces."""
    if 4 in transform.shape:
        return transform
    transform_3d = param_utils.identity_transform(ndim=3)
    if 't' in transform.dims:
        transform = transform.sel(t=0)
    transform_3d.loc[{dim: transform.coords[dim] for dim in transform.dims}] = transform
    return transform_3d


@pytest.mark.parametrize('transform', [
    param_utils.identity_transform(ndim=2),
    param_utils.affine_to_xaffine(create_transform({'x': 5.0, 'y': 7.0}, 0, matrix_size=3)),
    param_utils.affine_to_xaffine(create_transform({'x': -3.5, 'y': 11.25}, 37, matrix_size=3)),
    param_utils.identity_transform(ndim=3),
], ids=['identity', 'translation', 'rotation + translation', 'already 3d'])
def test_widening_matches_the_label_based_original(transform):
    np.testing.assert_allclose(np.asarray(widen_xaffine_to_3d(transform)), np.asarray(reference_widen(transform)))
    np.testing.assert_allclose(widened_affine_matrix(transform), np.asarray(reference_widen(transform)))
    # an already 3D transform is returned as is
    assert (widen_xaffine_to_3d(transform) is transform) == (4 in transform.shape)
