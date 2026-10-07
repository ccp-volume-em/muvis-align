"""Drawing shapes must not read, create or allocate image data.

build_source_stack_props() produces shapes' geometry directly; build_source_shape_sim() wraps it in a sim
for multiview_stitcher's exact overlap test. Both must agree with each other and with the real msim.
"""
import numpy as np
import pytest
from multiview_stitcher import msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.TiffImageSource import TiffImageSource
from muvis_align.image.util import (build_source_msim, build_source_shape_sim, build_source_stack_props,
                                    create_image_shapes, create_overlap_shapes, make_msims_3d)
from muvis_align.util import create_transform
from tests.data_builders import DATA_DIR, TIFF_FILES

TRANSLATION = {'x': 3.0, 'y': 4.0, 'z': 9.0}

# (output_order, z_scale, promote_z, transform matrix size or None)
CASES = [
    ('yx', None, False, None),
    ('yx', None, False, 3),
    ('yx', None, True, None),
    ('yx', None, True, 3),
    ('zyx', 2.5, False, None),
    ('zyx', 2.5, False, 4),
]


def make_transform(matrix_size):
    if matrix_size is None:
        return None
    return create_transform({'x': 5.0, 'y': 7.0}, 10, matrix_size=matrix_size)


def tiff_sources(count):
    return [TiffImageSource(str(DATA_DIR / name)) for name in TIFF_FILES[:count]]


def assert_same_shapes(got_shapes, expected_shapes):
    assert len(got_shapes) == len(expected_shapes)
    for got, expected in zip(got_shapes, expected_shapes):
        np.testing.assert_allclose(np.asarray(got, dtype=float), np.asarray(expected, dtype=float))


@pytest.mark.parametrize('output_order, z_scale, promote_z, matrix_size', CASES)
def test_stack_props_match_the_sim_they_replace(output_order, z_scale, promote_z, matrix_size):
    source = tiff_sources(1)[0]
    args = (source, output_order, TRANSLATION, make_transform(matrix_size), 'source_metadata')
    kwargs = dict(z_scale=z_scale, promote_z=promote_z)

    reference = si_utils.get_stack_properties_from_sim(build_source_shape_sim(*args, **kwargs),
                                                       transform_key='source_metadata')
    props = build_source_stack_props(*args, **kwargs)

    assert props['shape'] == reference['shape']
    assert props['spacing'] == reference['spacing']
    assert props['origin'] == reference['origin']
    np.testing.assert_allclose(np.asarray(props['transform']), np.asarray(reference['transform']))


@pytest.mark.parametrize('builder', [build_source_stack_props, build_source_shape_sim])
def test_building_geometry_opens_no_file_and_builds_no_msim(builder):
    sources = tiff_sources(2)
    assert not any(source._data_loaded for source in sources)

    geometries = [builder(source, 'tcyx', {'x': 0.0, 'y': 0.0}, None, 'source_metadata') for source in sources]

    for source, geometry in zip(sources, geometries):
        assert source._data_loaded is False, 'shape geometry must not load the source arrays'
        assert source._msim is None, 'shape geometry must not build the source msim'
        shape = geometry['shape'] if isinstance(geometry, dict) else dict(geometry.sizes)
        # it still describes the real image
        assert shape['y'] == source.get_shape(0)[source.dimension_order.index('y')]
        assert shape['x'] == source.get_shape(0)[source.dimension_order.index('x')]


@pytest.mark.parametrize('output_order, z_scale', [('yx', None), ('zyx', 2.5)])
def test_shape_sims_match_the_real_msims_scale0(output_order, z_scale):
    """Including the z-padding a 2D source gets when output_order forces a z it does not have."""
    matrix_size = len([dim for dim in output_order if dim in 'xyz']) + 1
    real_sims, shape_sims = [], []
    for source, translation, rotation in zip(tiff_sources(2), [{'x': 0.0, 'y': 0.0}, {'x': 50.0, 'y': 30.0}], [0, 15]):
        transform = param_utils.invert_coordinate_order(create_transform(translation, rotation, matrix_size=matrix_size))
        real_msim = build_source_msim(source, output_order, translation, transform, 'source_metadata', z_scale=z_scale)
        real_sims.append(msi_utils.get_sim_from_msim(real_msim, scale='scale0'))
        shape_sims.append(build_source_shape_sim(source, output_order, translation, transform, 'source_metadata',
                                                 z_scale=z_scale))

    for real_sim, shape_sim in zip(real_sims, shape_sims):
        real_props = si_utils.get_stack_properties_from_sim(real_sim, transform_key='source_metadata')
        shape_props = si_utils.get_stack_properties_from_sim(shape_sim, transform_key='source_metadata')
        for key in ('shape', 'spacing', 'origin'):
            assert shape_props[key] == pytest.approx(real_props[key])
        np.testing.assert_allclose(np.asarray(real_props['transform']), np.asarray(shape_props['transform']))
    assert_same_shapes(create_image_shapes(shape_sims, transform_key='source_metadata'),
                       create_image_shapes(real_sims, transform_key='source_metadata'))


def test_promoted_shape_sims_keep_each_sources_own_z_as_make_msims_3d_does():
    """2D sources at different heights (a z-stack of tiles) must reach the shapes at their own z."""
    real_sims, shape_sims = [], []
    for source, translation in zip(tiff_sources(2), [{'x': 0.0, 'y': 0.0, 'z': 0.0}, {'x': 50.0, 'y': 30.0, 'z': 10.0}]):
        transform = param_utils.invert_coordinate_order(create_transform(translation, 0, matrix_size=3))
        real_msim = build_source_msim(source, 'yx', translation, transform, 'source_metadata')
        promoted_msim = make_msims_3d([real_msim], positions=[translation])[0]
        real_sims.append(msi_utils.get_sim_from_msim(promoted_msim, scale='scale0'))
        shape_sims.append(build_source_shape_sim(source, 'yx', translation, transform, 'source_metadata',
                                                 promote_z=True))

    for sims in (real_sims, shape_sims):
        assert [si_utils.get_origin_from_sim(sim)['z'] for sim in sims] == pytest.approx([0.0, 10.0])
    shapes = create_image_shapes(shape_sims, transform_key='source_metadata')
    assert_same_shapes(shapes, create_image_shapes(real_sims, transform_key='source_metadata'))
    assert all(np.asarray(shape).shape[1] == 3 for shape in shapes)
    real_overlaps, real_pairs = create_overlap_shapes(real_sims, transform_key='source_metadata')
    shape_overlaps, shape_pairs = create_overlap_shapes(shape_sims, transform_key='source_metadata')
    assert [tuple(pair) for pair in shape_pairs] == [tuple(pair) for pair in real_pairs]
    assert_same_shapes(shape_overlaps, real_overlaps)


@pytest.mark.parametrize('output_order, z_scale, promote_z, matrix_size', CASES)
def test_image_shapes_identical_from_props_and_from_sims(output_order, z_scale, promote_z, matrix_size):
    translations = [{'x': 0.0, 'y': 0.0}, {'x': 50.0, 'y': 30.0}, {'x': 25.0, 'y': 60.0}]
    transform = make_transform(matrix_size)
    kwargs = dict(z_scale=z_scale, promote_z=promote_z)
    sources = tiff_sources(3)

    sims = [build_source_shape_sim(source, output_order, translation, transform, 'source_metadata', **kwargs)
            for source, translation in zip(sources, translations)]
    props = [build_source_stack_props(source, output_order, translation, transform, 'source_metadata', **kwargs)
             for source, translation in zip(sources, translations)]

    for force_2d in (False, True):
        assert_same_shapes(create_image_shapes(props, transform_key='source_metadata', force_2d=force_2d),
                           create_image_shapes(sims, transform_key='source_metadata', force_2d=force_2d))


@pytest.mark.parametrize('force_2d, pairs', [(False, None), (True, None), (False, [(0, 1)])],
                         ids=['broad phase', 'broad phase 2d', 'explicit pairs'])
def test_overlap_shapes_identical_from_props_and_from_sims(force_2d, pairs):
    """Without pairs (initial load) the broad-phase path runs; given pairs (post-registration), the exact one."""
    # deliberately overlapping, so pairs survive the broad phase
    translations = [{'x': 0.0, 'y': 0.0}, {'x': 5.0, 'y': 3.0}, {'x': 2.0, 'y': 6.0}]
    sources = tiff_sources(3)

    sims = [build_source_shape_sim(source, 'yx', translation, None, 'source_metadata')
            for source, translation in zip(sources, translations)]
    props = [build_source_stack_props(source, 'yx', translation, None, 'source_metadata')
             for source, translation in zip(sources, translations)]

    shapes_sims, pairs_sims = create_overlap_shapes(sims, 'source_metadata', pairs=pairs, force_2d=force_2d)
    shapes_props, pairs_props = create_overlap_shapes(props, 'source_metadata', pairs=pairs, force_2d=force_2d)

    assert [tuple(pair) for pair in pairs_props] == [tuple(pair) for pair in pairs_sims]
    assert_same_shapes(shapes_props, shapes_sims)
