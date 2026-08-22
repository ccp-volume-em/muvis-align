import time


class _QtNapariTqdm:
    def __init__(self, *args, tqdm_class=None, min_interval=0.1, **kwargs):
        from napari.utils import progress
        from qtpy.QtWidgets import QApplication

        self.tqdm_class = tqdm_class or progress
        self.min_interval = min_interval
        self._last_update = 0.0
        self._pending_update = 0
        self._pbar = None
        self._app = QApplication.instance()

        self._progress = self.tqdm_class(*args, **kwargs)

    def __iter__(self):
        with self as pbar:
            for item in self._progress:
                yield item
                pbar._process_events()

    def __enter__(self):
        self._pbar = self._progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._flush()
        return self._progress.__exit__(exc_type, exc_value, traceback)

    def update(self, n=1):
        self._pending_update += n
        now = time.monotonic()
        if now - self._last_update >= self.min_interval:
            self._flush()
            self._process_events()
            self._last_update = now

    def close(self):
        self._flush()
        if hasattr(self._progress, "close"):
            self._progress.close()

    def refresh(self, *args, **kwargs):
        if hasattr(self._progress, "refresh"):
            return self._progress.refresh(*args, **kwargs)
        return None

    def set_description(self, *args, **kwargs):
        if hasattr(self._progress, "set_description"):
            return self._progress.set_description(*args, **kwargs)
        return None

    def set_postfix(self, *args, **kwargs):
        if hasattr(self._progress, "set_postfix"):
            return self._progress.set_postfix(*args, **kwargs)
        return None

    def _flush(self):
        if self._pending_update:
            self._progress.update(self._pending_update)
            self._pending_update = 0

    def _process_events(self):
        if self._app is not None:
            self._app.processEvents()
