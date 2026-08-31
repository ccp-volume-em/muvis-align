from types import SimpleNamespace

import dask.array as da
import numpy as np
import xarray as xr
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.util import rechunk_if_monolithic, build_source_redimensioned_msim


def test_rechunk_if_monolithic_splits_single_chunk_data():
    data = da.from_array(np.zeros((2000, 2000), dtype=np.uint8), chunks=(2000, 2000))
    image = xr.DataArray(data, dims=['y', 'x'])

    rechunked = rechunk_if_monolithic(image, 1024)

    assert rechunked.chunksizes['y'] == (1024, 976)
    assert rechunked.chunksizes['x'] == (1024, 976)


def test_rechunk_if_monolithic_leaves_already_chunked_data_alone():
    data = da.from_array(np.zeros((2000, 2000), dtype=np.uint8), chunks=(500, 500))
    image = xr.DataArray(data, dims=['y', 'x'])

    rechunked = rechunk_if_monolithic(image, 1024)

    assert rechunked.chunksizes == image.chunksizes


def test_rechunk_if_monolithic_noop_when_chunk_size_falsy():
    data = da.from_array(np.zeros((2000, 2000), dtype=np.uint8), chunks=(2000, 2000))
    image = xr.DataArray(data, dims=['y', 'x'])

    rechunked = rechunk_if_monolithic(image, None)

    assert rechunked.chunksizes == image.chunksizes


def test_build_source_redimensioned_msim_rechunks_monolithic_levels():
    """A badly-chunked source (e.g. a single-chunk TIFF) must still get split into smaller dask
    chunks - here, once, at msim-creation time, rather than downstream every time a sim is
    extracted from it."""
    data = da.from_array(np.zeros((2000, 2000), dtype=np.uint8), chunks=(2000, 2000))
    sim = si_utils.get_sim_from_array(data, dims=['y', 'x'], transform_key='source_metadata')
    msim = msi_utils.get_msim_from_sims([sim])
    source = SimpleNamespace(msim=msim, dimension_order='yx', get_channels=lambda: [])

    redimensioned = build_source_redimensioned_msim(source, 'yx', chunk_size=1024)

    image0 = msi_utils.get_sim_from_msim(redimensioned, scale='scale0')
    assert image0.chunksizes['y'] == (1024, 976)
    assert image0.chunksizes['x'] == (1024, 976)
