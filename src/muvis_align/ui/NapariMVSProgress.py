from muvis_align.ui.QtNapariTqdm import _QtNapariTqdm


class NapariMVSProgress:
    def __init__(
            self,
            tqdm_class=None,
            patch_fusion=True,
            patch_registration=False,
            patch_param_resolution=False,
            min_interval=0.1,
            **tqdm_kwargs,
    ):
        from napari.utils import progress

        self.tqdm_class = tqdm_class or progress
        self.patch_fusion = patch_fusion
        self.patch_registration = patch_registration
        self.patch_param_resolution = patch_param_resolution
        self.min_interval = min_interval
        self.tqdm_kwargs = tqdm_kwargs
        self._patched = []

    def __enter__(self):
        for module, replacement in self._get_patch_targets():
            if hasattr(module, "tqdm"):
                self._patched.append((module, module.tqdm))
                module.tqdm = replacement

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, original_tqdm in reversed(self._patched):
            module.tqdm = original_tqdm
        self._patched.clear()
        return False

    def _get_patch_targets(self):
        targets = []

        if self.patch_fusion:
            try:
                import multiview_stitcher.fusion._core as mvs_fusion_core

                targets.append((mvs_fusion_core, self._create_progress_replacement()))
            except ImportError:
                pass

        registration_replacement = self._create_qt_friendly_replacement()

        if self.patch_registration:
            try:
                import multiview_stitcher.registration as mvs_registration

                targets.append((mvs_registration, registration_replacement))
            except ImportError:
                pass

        if self.patch_param_resolution:
            try:
                import multiview_stitcher.param_resolution as mvs_param_resolution

                targets.append((mvs_param_resolution, registration_replacement))
            except ImportError:
                pass

        return targets

    def _create_progress_replacement(self):
        if not self.tqdm_kwargs:
            return self.tqdm_class

        tqdm_class = self.tqdm_class
        tqdm_kwargs = self.tqdm_kwargs

        class _ConfiguredTqdm(tqdm_class):
            def __init__(self, *args, **kwargs):
                merged_kwargs = {**tqdm_kwargs, **kwargs}
                super().__init__(*args, **merged_kwargs)

        return _ConfiguredTqdm

    def _create_qt_friendly_replacement(self):
        tqdm_class = self.tqdm_class
        tqdm_kwargs = self.tqdm_kwargs
        min_interval = self.min_interval

        class _ConfiguredQtNapariTqdm(_QtNapariTqdm):
            def __init__(self, *args, **kwargs):
                merged_kwargs = {**tqdm_kwargs, **kwargs}
                super().__init__(
                    *args,
                    tqdm_class=tqdm_class,
                    min_interval=min_interval,
                    **merged_kwargs,
                )

        return _ConfiguredQtNapariTqdm
