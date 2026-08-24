# https://blog.dask.org/2019/06/20/load-image-data
# https://docs.dask.org/en/stable/best-practices.html#load-data-with-dask
# https://dask.discourse.group/t/could-not-serialize-object-of-type-highlevelgraph-with-client-ome-tiff/3959

import dask
import dask.array as da
from dask.diagnostics import Profiler, ResourceProfiler
from dask.distributed import Client
from distributed import performance_report
import glob
from multiview_stitcher import spatial_image_utils as si_utils
import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import os
from skimage.transform import resize
import tifffile
from tifffile import TiffFile, TiffWriter, imread
import zarr

from muvis_align.Timer import Timer
from muvis_align.image.source_helper import create_dask_source
from muvis_align.image.util import normalise_sims
from muvis_align.util import xyz_to_dict


def save_ome_tiff(filename, data, npyramid_add=0):
    size = np.array(data.shape[:2])
    with TiffWriter(filename, ome=True) as writer:
        for i in range(npyramid_add + 1):
            if i == 0:
                subifds = npyramid_add
                subfiletype = None
            else:
                subifds = None
                subfiletype = 1
                size //= 2
                data = resize(data, size)
            writer.write(data, subifds=subifds, subfiletype=subfiletype)


def ome_tiff(filename, level=0):
    store = imread(filename, aszarr=True)
    group = zarr.open(store=store, mode='r')
    paths = group.attrs['multiscales'][0]['datasets']
    path = paths[level]['path']
    data = group[path]
    return data


def dask_ome_tiff(filename):
    # incompatible with Client/Cluster
    store = TiffFile(filename).aszarr(level=0)
    dask_data = da.from_zarr(store)
    return dask_data


def dask_ome_tiff_array(filename):
    data = TiffFile(filename).asarray(level=0)
    dask_data = da.from_array(data)
    return dask_data


def dask_ome_tiff_lazy(filename, level=0):
    # use this for tiff files
    with TiffFile(filename) as tif:
        series0 = tif.series[0]
        shape = series0.shape
        dtype = series0.dtype
    lazy_array = dask.delayed(tifffile.imread)(filename, level=level)
    dask_data = da.from_delayed(lazy_array, shape=shape, dtype=dtype)
    return dask_data


def dask_ome_tiff_lazy_array(filename):
    # incompatible with Client/Cluster
    with TiffFile(filename) as tif:
        series0 = tif.series[0]
        shape = series0.shape
        dtype = series0.dtype
    lazy_array = dask.delayed(TiffFile(filename).asarray)()
    dask_data = da.from_delayed(lazy_array, shape=shape, dtype=dtype)
    return dask_data


def ome_zarr(filename, level=0):
    group = zarr.open_group(filename, mode='r')
    # using group.attrs to get multiscales is recommended by cgohlke
    paths = group.attrs['multiscales'][0]['datasets']
    path = paths[level]['path']
    data = group[path]
    return data


def dask_ome_zarr(filename, level=0):
    # use this for zarr files
    group = zarr.open_group(filename, mode='r')
    # using group.attrs to get multiscales is recommended by cgohlke
    paths = group.attrs['multiscales'][0]['datasets']
    path = paths[level]['path']
    return da.from_zarr(os.path.join(filename, path))


def dask_ome_zarr_py(filename, level=0):
    reader = Reader(parse_url(filename))
    # nodes may include images, labels etc
    nodes = list(reader())
    # first node will be the image pixel data
    image_node = nodes[0]
    dask_data = image_node.data[level]
    return dask_data


def dask_ome_zarr_source(filename):
    source = create_dask_source(filename)
    return source.get_data(0)


def load_dask0(filename):
    if filename.endswith('.tiff'):
        return dask_ome_tiff_lazy(filename)
    elif filename.endswith('.zarr'):
        return dask_ome_zarr(filename)
    return None


def load_dask(filename, level=0):
    dask_source = create_dask_source(filename)
    return dask_source.get_data(level=level)


def task(filenames):
    if isinstance(filenames, str):
        filenames = glob.glob(filenames)

    chunk_size = [1024, 1024]

    sims = []
    transform_key = 'transform_key'
    with Timer('reading', auto_unit=False):
        for filename in filenames:
            print(f'reading {filename}')
            dask_data = load_dask(filename)
            sim = si_utils.get_sim_from_array(dask_data, transform_key=transform_key)
            sim = sim.chunk(xyz_to_dict(chunk_size))
            sims.append(sim)

    with Timer('normalise', auto_unit=False):
        #value = np.mean(dask_data).compute()
        new_sims = normalise_sims(sims, transform_key=transform_key)
    with Timer('computing', auto_unit=False):
        for new_sim in new_sims:
            new_sim.compute()


if __name__ == "__main__":
    #data = np.ones(shape=(512, 1024)).astype(np.float32)
    #filename = 'test.ome.tiff'
    #save_ome_tiff(filename, data, npyramid_add=4)

    #base_folder = '/nemo/project/proj-ccp-vem/datasets/12193/stitched/'
    base_folder = 'D:/slides/12193/stitched/'

    filesets = [
        [base_folder + 'S000/registered.ome.zarr', base_folder + 'S001/registered.ome.zarr', base_folder + 'S002/registered.ome.zarr'],
        #[base_folder + 'S000/registered.ome.tiff', base_folder + 'S001/registered.ome.tiff', base_folder + 'S002/registered.ome.tiff'],
    ]

    print('Without client')
    for filenames in filesets:
        print('Fileset:', filenames)
        task(filenames)
        print()

    print('Without client & profiling')
    with Profiler() as prof, ResourceProfiler(dt=1) as rprof:
        for filenames in filesets:
            print('Fileset:', filenames)
            task(filenames)
            print()
    prof.visualize(filename='profile.html', save=True, show=False)
    rprof.visualize(filename='resource_profile.html', save=True, show=False)

    print('With client & performance profiling')
    with Client(processes=False) as client:
        print(client)
        with performance_report(filename="report.html"):
            for filenames in filesets:
                print('Fileset:', filenames)
                task(filenames)
                print()
    print('Done')
