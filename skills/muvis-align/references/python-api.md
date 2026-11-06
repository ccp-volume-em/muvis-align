# muvis-align from Python

Two levels. Prefer the first; use the second only to inspect or change intermediate results.

## Whole operations: `Pipeline`

`Pipeline` takes the parameter file's content as a dict (`general` + `operations`) and runs each operation as
the YAML pipeline does, including `match` groups, the RO-Crate and all result files.

```python
from muvis_align.Pipeline import Pipeline

params = {
    'general': {'logging': {'filename': 'log/stitch.log', 'verbose': True}},
    'operations': [{
        'input_output': {'input_path': '/data/run1/*.ome.tif', 'output_path': 'stitched/', 'overwrite': True},
        'registration': {'operation': 'register', 'method': 'phase_correlation',
                         'transform_type': 'translation', 'pairing': 'orthogonal'},
        'fusion': {'format': 'ome.zarr'},
    }],
}
ok = Pipeline(params).run()     # blocks; False if any operation raised (details in the log)
```

`Pipeline` is a `threading.Thread`: `.run()` runs in the calling thread, `.start()` + `.join()` in a new one.
It sets up logging to `general.logging.filename`.

## Step by step: `MVSRegistration`

The steps the pipeline chains, for a notebook or a custom loop. The repository's
`notebooks/stitching2d.ipynb` runs exactly this on the sample data.

```python
from muvis_align.MVSRegistration import MVSRegistration
from muvis_align.image.util import extract_sims_from_fused

reg = MVSRegistration(operation='register', input_path='data/S000/*.zarr', output_path='../output/')
reg.init_data()                                  # read metadata: positions, scales, rotations
msims = reg.msims                                # multiscale spatial images, one per source (lazy)
register_msims, reg_indices, _ = reg.preprocess(msims)   # kwargs: scale, normalisation, gaussian_sigma, ...

results = reg.register(register_msims=register_msims, register_indices=reg_indices, params={
    'pairing': 'orthogonal', 'transform_type': 'rigid', 'method': 'sift',
    'gaussian_sigma': 2, 'normalisation': True, 'max_keypoints': 5000,
})
results['mappings']                  # {source index: affine transform (xarray)}
results['registration_qualities']    # {pair: quality}

fused_msim, _ = reg.fuse(msims)
fused_sim = extract_sims_from_fused(fused_msim)
```

- `reg.file_labels` names the sources in the order of `reg.msims` and the mapping keys.
- `register_params` take the names of the parameter file's `registration` section.
- The registered transform is stored on each msim under `reg.reg_transform_key` (`'registered'`); the
  metadata positions under `reg.source_transform_key` (`'source_metadata'`).
- Spatial images are [multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher) /
  `spatial_image` objects: use `multiview_stitcher.spatial_image_utils` and `msi_utils` to work with them.

These step methods are less stable than the parameter file: when in doubt, check their signatures in
`muvis_align/MVSRegistration.py` of the installed version.
