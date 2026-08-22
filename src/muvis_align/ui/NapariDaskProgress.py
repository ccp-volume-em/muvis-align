from dask.callbacks import Callback


class NapariDaskProgress(Callback):
    def __init__(self, progress_class=None, desc="Dask computation", **progress_kwargs):
        from napari.utils import progress

        self.progress_class = progress_class or progress
        self.desc = desc
        self.progress_kwargs = progress_kwargs
        self._pbar = None
        self._started = False

    def _start(self, dsk):
        total = len(dsk)
        self._pbar = self.progress_class(
            total=total,
            desc=self.desc,
            **self.progress_kwargs,
        )
        self._pbar.__enter__()
        self._started = True

    def _posttask(self, key, result, dsk, state, id):
        if self._pbar is not None:
            self._pbar.update(1)

    def _finish(self, dsk, state, errored):
        if self._pbar is not None and self._started:
            self._pbar.__exit__(None, None, None)
        self._pbar = None
        self._started = False
