import logging

import dask.array as da
from multiview_stitcher import ngff_utils

from muvis_align.image.ImageSource import ImageSource
from muvis_align.image.color_conversion import hexrgb_to_rgba
from muvis_align.image.ome_zarr_util import read_ome_zarr_source_metadata


class ZarrImageSource(ImageSource):
    def init_metadata(self):
        # metadata only, off consolidated metadata: the arrays wait for first access to self.data
        metadata = read_ome_zarr_source_metadata(self.filename)

        self.dimension_order = metadata['dimension_order']
        self.shapes = metadata['shapes']
        self.shape = self.shapes[0]
        self.dtype = metadata['dtype']

        self.pixel_sizes = metadata['pixel_sizes']
        self.pixel_size = self.pixel_sizes[0]
        self.position = metadata.get('position')
        self.rotation = metadata.get('rotation')

        # None when unknown, making _load_data() fall back to the msim reader
        self._level_paths = metadata.get('paths')
        # before any are synthesized on top
        self._native_nlevels = len(self.shapes)

        # a 'c' dim is always present, so there is at least one channel; labels match the sim's 'c' coords
        self.channels = [{'label': str(index)} for index in range(metadata['nchannels'])]
        omero = metadata['omero']
        if omero:
            # from the store's OME attributes: read_msim_from_ome_zarr leaves msim.attrs empty
            for channel, channel_metadata in zip(self.channels, omero.get('channels', [])):
                if channel_metadata.get('color'):
                    channel['color'] = hexrgb_to_rgba(channel_metadata['color'])

    def _load_data(self):
        """One dask array per level, opened straight off the store.

        Raw arrays (rather than the native msim reader) let the base class synthesize the coarse levels a
        single-resolution store lacks, so a reduced-resolution fusion need not run at native resolution.
        """
        if not self._level_paths:
            self._build_msim_natively()
            return
        try:
            datas = []
            for path, sim_shape in zip(self._level_paths, self.shapes):
                data = da.from_zarr(self.filename, component=path)
                if tuple(data.shape) != tuple(sim_shape):
                    # the sim only inserts size-1 t/c axes in the file's dim order, so reshape moves no data
                    data = data.reshape(sim_shape)
                datas.append(data)
        except Exception as e:
            # e.g. a remote URL or unusual layout: the msim reader copes, without synthesized levels
            logging.warning(f'{self.filename}: could not open pyramid levels directly'
                            f' ({e}) - falling back to reading the msim')
            self._level_paths = None
            # drop levels settled as synthesized at init: the native msim has only the store's own
            self.shapes = self.shapes[:self._native_nlevels]
            self.pixel_sizes = self.pixel_sizes[:self._native_nlevels]
            self.scale_factors = self.scale_factors[:self._native_nlevels]
            self._build_msim_natively()
            return
        self._data = datas

    def _build_msim_natively(self):
        # self._data stays empty, so no coarse levels are synthesized
        self._msim = ngff_utils.read_msim_from_ome_zarr(self.filename, array_backend='dask',
                                                        transform_key=self.transform_key)
        self._restamp_msim()

    def _build_msim(self):
        # _load_data() may already have built the msim natively
        data = self.data
        if self._msim is not None:
            return
        if not data:
            raise ValueError(f'No image data available for {self.filename}')
        super()._build_msim()

    def _synthesized_level_factors(self):
        if not self._level_paths:
            # the native msim has only the store's own levels and no raw arrays to synthesize from
            return []
        factors = super()._synthesized_level_factors()
        if factors:
            _warn_single_resolution_store(self.filename, len(factors))
        return factors


_warned_single_resolution = False


def _warn_single_resolution_store(filename, nlevels):
    """Warn once per run rather than per source, which would bury the log.

    Synthesized levels still read every full-resolution chunk they stride across; a real pyramid avoids that.
    """
    global _warned_single_resolution
    if _warned_single_resolution:
        return
    _warned_single_resolution = True
    logging.warning(f'Single-resolution OME-Zarr source(s), e.g. {filename}: synthesizing'
                    f' {nlevels} coarse level(s) so a reduced-resolution preview/pre-processing'
                    f' does not have to fuse at the store\'s native resolution. Building them'
                    f' re-reads full-resolution chunks - re-converting these sources with a full'
                    f' pyramid avoids that.')
