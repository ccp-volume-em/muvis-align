class NapariPreprocessProgress:
    def __init__(self, progress_class=None, min_duration=0.0, **progress_kwargs):
        from napari.utils import progress

        self.progress_class = progress_class or progress
        self.min_duration = max(float(min_duration), 0.0)
        self.progress_kwargs = progress_kwargs
        self._started_at = None

    def __enter__(self):
        import time

        self._started_at = time.monotonic()
        return self

    def __call__(self, total, desc=None):
        kwargs = dict(self.progress_kwargs)
        kwargs["total"] = total
        if desc is not None:
            kwargs.setdefault("desc", desc)
        return self.progress_class(**kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._started_at is not None and self.min_duration > 0:
            import time

            elapsed = time.monotonic() - self._started_at
            wait_s = self.min_duration - elapsed
            if wait_s > 0:
                time.sleep(wait_s)
        self._started_at = None
        return False
