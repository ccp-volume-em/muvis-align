from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from ome_zarr.writer import write_image
from skimage.transform import resize
import zarr

from muvis_align.image.source_helper import create_dask_source
from muvis_align.image.util import get_level_from_scale


def zarr_test(url):
    # read the image data
    reader = Reader(parse_url(url))
    # nodes may include images, labels etc
    nodes = list(reader())
    # first node will be the image pixel data
    image_node = nodes[0]

    # list of dask arrays at different pyramid size levels
    data = image_node.data
    # dictionary of OME-Zarr metadata
    metadata = image_node.metadata
    axes = ''.join([axis['name'] for axis in metadata['axes']])

    full_size_image_data = data[0]  # access the image data array at full size


    path = "path/to/output_image.zarr"

    root = zarr.open_group(store=path)
    # supports dask data, by default written out at various pyramid sizes
    write_image(image=full_size_image_data, group=root, axes=axes)


def read_dask(url, target_scale=16):
    source = create_dask_source(url)
    level, rescale, scale = get_level_from_scale(source, target_scale=target_scale)
    print('level', level)
    print('rescale', rescale)
    print('scale', scale)
    data = source.get_data()
    if any(value != 1 for value in rescale.values()):
        new_shape = [int(size / rescale.get(dim, 1))
                     for dim, size in zip(source.dimension_order, source.shapes[level])]
        print(source.shapes[level], ' -> ', new_shape)
        data = resize(data, new_shape, preserve_range=True).astype(data.dtype)
    return source


if __name__ == '__main__':
    # Example URL of remote data
    #url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"
    #url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr"
    #url = 'D:/slides/Emb3_2x2_Mosaic_2CAMs/RL00--X00--Y00--C00.ome.zarr'
    url = r'D:\slides\EM04768_01_substrate_04\Stitched\registered.ome.tiff'
    #zarr_test(url)
    read_dask(url)
