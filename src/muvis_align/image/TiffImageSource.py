import tifffile
from ngff_zarr import tiff_file_to_ngff_images, NgffMultiscales

from muvis_align.image.ome_tiff_helper import read_tiff_level_arrays, read_tiff_source_metadata, read_tiff_creator
from muvis_align.util import convert_to_um
from muvis_align.image.ImageSource import ImageSource
from muvis_align.image.tiff_metadata import get_extra_metadata
from muvis_align.image.color_conversion import hexrgb_to_rgba


class TiffImageSource(ImageSource):
    def init_metadata(self):
        # metadata only, straight off the tags/OME-XML: tiff_file_to_ngff_images would also wrap every level
        # in a dask array; the arrays wait for first access to self.data
        metadata = read_tiff_source_metadata(self.filename)
        if metadata is None:
            # ngff_zarr's private axis mapping is unavailable: let it read arrays and all
            self._init_metadata_from_ngff_zarr()
            return

        self.dimension_order = metadata['dimension_order']
        self.shapes = metadata['shapes']
        self.shape = self.shapes[0]
        self.dtype = metadata['dtype']
        self.pixel_sizes = metadata['pixel_sizes']
        self.pixel_size = self.pixel_sizes[0]
        self.position = metadata['position']
        self.rotation = metadata['rotation']
        self.channels = metadata['channels']
        self.creator = metadata['creator']
        # TODO: check with RGB image if better approach is possible
        self.is_rgb = (self.get_nchannels() in (3, 4))

    def _read_metadata(self):
        # full OME-XML root dict, else every tag but the pixel layout
        with tifffile.TiffFile(self.filename) as tif:
            if tif.is_ome and tif.ome_metadata is not None:
                return tifffile.xml2dict(tif.ome_metadata).get('OME', {})
            return get_extra_metadata(tif)

    def _load_data(self):
        # tifffile's zarr store, as ngff_zarr reads it but without rebuilding metadata; ngff_zarr as fallback
        datas = read_tiff_level_arrays(self.filename)
        if datas is None:
            datas = self._read_ngff_images()[1]
        self._data = datas

    def _read_ngff_images(self):
        """(ngff_images, per-level arrays) from ngff_zarr, the fallback reader. Only the first series is the
        image; later ones are auxiliary (e.g. a thumbnail), not pyramid levels.
        """
        ngff_image_data1 = tiff_file_to_ngff_images(self.filename, reuse_existing_pyramids=True)[0][1]
        if isinstance(ngff_image_data1, NgffMultiscales):
            ngff_images = ngff_image_data1.images
        else:
            ngff_images = [ngff_image_data1]
        return ngff_images, [ngff_image.data for ngff_image in ngff_images]

    def _init_metadata_from_ngff_zarr(self):
        ngff_images, datas = self._read_ngff_images()
        # ngff_zarr's NgffImage carries no Creator - read it separately
        self.creator = read_tiff_creator(self.filename)
        self.dimension_order = ''.join(ngff_images[0].dims)
        for index, ngff_image in enumerate(ngff_images):
            axes_units = ngff_image.axes_units or {}
            if index == 0:
                self.position = {dim: convert_to_um(value, axes_units.get(dim, 'um'))
                                  for dim, value in ngff_image.translation.items() if dim in 'xyz'}
            if index == 0 and ngff_image.channel_names:
                for channel_index, channel_name in enumerate(ngff_image.channel_names):
                    channel = {'label': channel_name or f'channel {channel_index}'}
                    if ngff_image.channel_colors:
                        channel['color'] = hexrgb_to_rgba(ngff_image.channel_colors[channel_index])
                    self.channels.append(channel)
            self.pixel_sizes.append({dim: convert_to_um(value, axes_units.get(dim, 'um'))
                                     for dim, value in ngff_image.scale.items() if dim in 'xyz'})
        self.data = datas
        self.dtype = datas[0].dtype
        self.shapes = [data.shape for data in datas]
        self.shape = self.shapes[0]
        self.is_rgb = (self.get_nchannels() in (3, 4))
        self.pixel_size = self.pixel_sizes[0]


