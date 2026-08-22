class NapariPreprocessProgress:
    def __init__(self, progress_class=None, **progress_kwargs):
        from napari.utils import progress

        self.progress_class = progress_class or progress
        self.progress_kwargs = progress_kwargs

    def __enter__(self):
        return self

    def __call__(self, total, desc=None):
        kwargs = dict(self.progress_kwargs)
        kwargs["total"] = total
        if desc is not None:
            kwargs.setdefault("desc", desc)
        return self.progress_class(**kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        return False
