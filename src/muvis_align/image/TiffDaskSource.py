import dask.array as da
import dask.array
import numpy as np
import os.path
import tifffile
from tifffile import PHOTOMETRIC

from src.muvis_align.image.DaskSource import DaskSource
from src.muvis_align.image.color_conversion import int_to_rgba
from src.muvis_align.util import ensure_list


class TiffDaskSource(DaskSource):
    def init_metadata(self):
        tiff = tifffile.TiffFile(self.filename)
        if tiff.series:
            pages = tiff.series
            page = pages[0]
        else:
            pages = tiff.pages
            page = tiff.pages.first
        if hasattr(page, 'levels') and len(page.levels) >= len(pages):
            pages = page.levels
        self.dimension_order = page.axes.lower()
        self.dtype = page.dtype.type
        self.pages = pages
        self.shapes = [page.shape for page in pages]
        self.shape = self.shapes[0]
        self.scale_factors = [{dim: value0 / value for dim, value, value0 in zip(self.dimension_order, shape, self.shape)} for shape in self.shapes]
        photometric = page.keyframe.photometric
        nchannels = self.get_nchannels()
        self.is_rgb = (photometric in (PHOTOMETRIC.RGB, PHOTOMETRIC.PALETTE) and nchannels in (3, 4))

        pixel_size = {}
        position = {}
        rotation = None
        channels = []
        if tiff.is_ome and tiff.ome_metadata is not None:
            xml_metadata = tiff.ome_metadata
            metadata = tifffile.xml2dict(xml_metadata)
            if 'OME' in metadata:
                metadata = metadata['OME']
            use_image_ref = ('BinaryOnly' in metadata)
            if use_image_ref:
                metadata0 = metadata
                path = os.path.join(os.path.dirname(self.filename), metadata['BinaryOnly']['MetadataFile'])
                metadata = tifffile.xml2dict(open(path, encoding='utf-8').read())
                if 'OME' in metadata:
                    metadata = metadata['OME']

            images = ensure_list(metadata.get('Image', {}))
            if use_image_ref:
                matching_image = None
                for image in images:
                    if image.get('Pixels', {}).get('TiffData', {}).get('UUID', {}).get('value') == metadata0.get('UUID'):
                        matching_image = image
                image = matching_image
            else:
                image = images[0]
            pixels = image.get('Pixels', {})
            size = float(pixels.get('PhysicalSizeX', 0))
            if size:
                pixel_size['x'] = (size, pixels.get('PhysicalSizeXUnit', self.default_physical_unit))
            size = float(pixels.get('PhysicalSizeY', 0))
            if size:
                pixel_size['y'] = (size, pixels.get('PhysicalSizeYUnit', self.default_physical_unit))
            size = float(pixels.get('PhysicalSizeZ', 0))
            if size:
                pixel_size['z'] =  (size, pixels.get('PhysicalSizeZUnit', self.default_physical_unit))

            for plane in ensure_list(pixels.get('Plane', [])):
                if 'PositionX' in plane:
                    position['x'] = (float(plane.get('PositionX')), plane.get('PositionXUnit', self.default_physical_unit))
                if 'PositionY' in plane:
                    position['y'] = (float(plane.get('PositionY')), plane.get('PositionYUnit', self.default_physical_unit))
                if 'PositionZ' in plane:
                    position['z'] = (float(plane.get('PositionZ')), plane.get('PositionZUnit', self.default_physical_unit))
                # c, z, t = plane.get('TheC'), plane.get('TheZ'), plane.get('TheT')

            annotations = metadata.get('StructuredAnnotations')
            if annotations is not None:
                if not isinstance(annotations, (list, tuple)):
                    annotations = [annotations]
                for annotation_item in annotations:
                    for annotations2 in annotation_item.values():
                        if not isinstance(annotations2, (list, tuple)):
                            annotations2 = [annotations2]
                        for annotation in annotations2:
                            value = annotation.get('Value')
                            unit = None
                            if isinstance(value, dict) and 'Modulo' in value:
                                modulo = value.get('Modulo', {}).get('ModuloAlongZ', {})
                                unit = modulo.get('Unit')
                                value = modulo.get('Label')
                            elif isinstance(value, str) and value.lower().startswith('angle'):
                                if ':' in value:
                                    value = value.split(':')[1].split()
                                elif '=' in value:
                                    value = value.split('=')[1].split()
                                else:
                                    value = value.split()[1:]
                                if len(value) >= 2:
                                    unit = value[1]
                                value = value[0]
                            else:
                                value = None
                            if value is not None:
                                rotation = float(value)
                                if 'rad' in unit.lower():
                                    rotation = np.rad2deg(rotation)

            for channel0 in ensure_list(pixels.get('Channel', [])):
                channel = {'label': channel0.get('Name', '')}
                color = channel0.get('Color')
                if color:
                    channel['color'] = int_to_rgba(int(color))
                channels.append(channel)
        else:
            metadata = tags_to_dict(tiff.pages.first.tags)
        self.metadata = metadata
        self.pixel_size = pixel_size
        self.position = position
        self.rotation = rotation
        self.channels = channels

    def get_data(self, level=0):
        if level < 0:
            dask_data = []
            for level in range(len(self.shapes)):
                lazy_array = dask.delayed(tifffile.imread)(self.filename, level=level)
                #lazy_array = dask.delayed(self.pages[level].asarray)()
                data = dask.array.from_delayed(lazy_array, shape=self.shapes[level], dtype=self.dtype)
                dask_data.append(data)
        else:
            lazy_array = dask.delayed(tifffile.imread)(self.filename, level=level)
            #lazy_array = dask.delayed(self.pages[level].asarray)()
            dask_data = dask.array.from_delayed(lazy_array, shape=self.shapes[level], dtype=self.dtype)
        return dask_data


def tags_to_dict(tags: tifffile.TiffTags) -> dict:
    tag_dict = {}
    for tag in tags.values():
        tag_dict[tag.name] = tag.value
    return tag_dict
