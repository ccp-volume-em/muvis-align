import numpy as np
import pytest

from muvis_align.fusion_methods.FusionMethodExclusive import FusionMethodExclusive
from tests.data_builders import make_sim

nan = np.nan


def exclusive(views):
    return FusionMethodExclusive(np.zeros(1, np.uint16)).fusion(np.asarray(views, dtype=np.float32))


def _views_3d():
    views = np.full((2, 2, 2, 3), nan)
    views[0, 0, :, :2] = 5
    views[1, :, :, 1:] = 7
    return views


@pytest.mark.parametrize('views, expected', [
    # each pixel from the first view covering it
    ([[[nan, 1, 1], [nan, nan, 1]], [[2, 2, 2], [2, nan, nan]], [[3, 3, 3], [3, 3, nan]]], [[2, 1, 1], [2, 3, 1]]),
    # 3D chunks, uncovered pixels 0
    (_views_3d(), [[[5, 5, 7], [5, 5, 7]], [[0, 7, 7], [0, 7, 7]]]),
    # one view is kept
    ([[[nan, 4], [6, 8]]], [[0, 4], [6, 8]]),
], ids=['first covering view', '3d', 'one view'])
def test_exclusive_takes_each_pixel_from_the_first_view_covering_it(views, expected):
    fused = exclusive(views)

    np.testing.assert_array_equal(fused, expected)
    assert fused.dtype == np.float32


@pytest.mark.parametrize('order', [(0, 1), (1, 0)])
def test_exclusive_fuse_keeps_the_earlier_source_where_tiles_overlap(order):
    from multiview_stitcher import fusion, msi_utils

    values, origins = (10, 20), (0.0, 6.0)
    sims = [make_sim(np.full((8, 8), values[index], np.uint16), translation={'y': 0.0, 'x': origins[index]},
                     transform_key='source')
            for index in order]
    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]
    fuse = FusionMethodExclusive(sims[0]).fusion

    fused = fusion.fuse(msims, transform_key='source', fusion_func=fuse, output_chunksize=4)
    fused = np.asarray(msi_utils.get_sim_from_msim(fused).data).squeeze()

    first = values[order[0]]
    # tiles overlap in x 6-7: the first source given wins there, whichever it is
    np.testing.assert_array_equal(fused[:, 6:8], first)
    np.testing.assert_array_equal(fused[:, :6], 10)
    np.testing.assert_array_equal(fused[:, 8:], 20)
