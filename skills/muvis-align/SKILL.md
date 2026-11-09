---
name: muvis-align
description: Register (stitch/align) and fuse microscopy image tiles, z-sections or channels with muvis-align, from a pipeline YAML file or from Python. Use when asked to stitch tiles, align a stack, register channels, fuse to OME-Zarr/OME-TIFF, write or fix a muvis-align parameter file, or script muvis-align from Python.
---

# Using muvis-align

muvis-align registers sets of images (tiles of a plane, sections of a stack, channels) using
[multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher), then fuses them into one
OME-Zarr or OME-TIFF. It reads OME-Zarr and (OME-)TIFF, taking positions and pixel sizes from their metadata.

Full documentation: https://ccp-volume-em.github.io/muvis-align/ - the parameter reference is the
[pipeline page](https://ccp-volume-em.github.io/muvis-align/pipeline/). This skill covers how to choose and
avoid the common mistakes; look settings up there rather than guessing names.

## Choosing an entry point

| Situation | Use |
|---|---|
| Batch / cluster run, reproducible settings | Pipeline YAML, run with `python run.py params.yml` from a checkout, or `Pipeline` from Python |
| Calling from your own Python code with your settings | `Pipeline(params_dict).run()` |
| Inspecting intermediate results (pairs, mappings) step by step | `MVSRegistration` - see [python-api.md](references/python-api.md) |
| A person wants to look, preview pairs, fix a pair by hand | The napari plugin (*Plugins > muvis-align*) - same settings as a project file |

Install: for the plugin, from the [napari hub](https://napari-hub.org/plugins/muvis-align) (in napari:
*Plugins > Install/Uninstall Plugins...*, search `muvis-align`). For scripts: `pip install muvis-align`
(`muvis-align[all]` adds napari and the elastix/ANTs/CPD methods), or the latest from GitHub:
`pip install "muvis-align @ git+https://github.com/ccp-volume-em/muvis-align"`. `run.py` is in the repository, not the
installed package: from an installed package, use the Python form below.

## Parameter file shape

```yaml
general:
  logging:
    verbose: true            # log to the console as well as log/muvis-align.log
  break_on_error: false
operations:                  # run in order; each has the napari project file's sections
- input_output:
    input_path: data/S000/*.zarr        # glob, folder (its files; a .zarr folder is one image), or a list of these
    output_path: ../stitched/           # RELATIVE TO THE INPUT DIRECTORY (see pitfalls)
    overwrite: true
  pre_processing: {}                    # optional: scale, normalisation, gaussian_sigma, flatfield_quantiles
  registration:
    operation: register                 # register | register match <LABEL> | fuse/merge (no registration)
    method: phase_correlation           # phase_correlation | sift | orb | cpd | elastix | ants
    transform_type: translation         # translation | rigid | similarity | affine
    pairing: orthogonal                 # orthogonal (tile grid) | stack (consecutive sections)
  fusion:
    format: ome.zarr                    # ome.zarr | ome.tiff | "ome.zarr, ome.tiff"
```

With `input_output: registration_dimension: c`, the images are channels on one grid: each `input_path` list item
is one channel (its tiles fused together); a single item is split by the files' metadata channel label, or one
channel per file if they share it.

Settings sit in the four sections `input_output`, `pre_processing`, `registration`, `fusion`. Keys at the
top level of an operation (`operation`, `input`, `output`) are an old layout and raise an error.

Worked examples in [examples/](examples/): `stitch_2d.yml` (tile grid), `match_sets.yml` (one registration
per section/folder), `stack.yml` (align a z-stack), `merge_no_registration.yml` (fuse at stage positions).

From Python, the same settings as a dict:

```python
import yaml
from muvis_align.Pipeline import Pipeline

with open('params.yml', encoding='utf8') as file:
    params = yaml.safe_load(file)
ok = Pipeline(params).run()     # True if no operation raised; runs in the calling thread
```

## Choosing settings

- **Start with stage positions only**: `operation: merge` fuses at the metadata positions without registering.
  If that already looks wrong (tiles mirrored, transposed, far apart), fix the metadata first - no registration
  method will recover a wrong layout.
- **Metadata axes/positions**: `source_position_x/y/z`, `source_scale_x/y/z`, `source_rotation` take `source`
  (the file's value), `source invert`, `source y` (another axis's value, e.g. for a camera with x and y swapped),
  a number, or a filename expression such as `fn[-3]*24`. Tiles without stage metadata need an expression.
- **Method**: `phase_correlation` for translation-only tiles with good overlap; `sift`/`orb` (features) when
  there is rotation or little overlap; `elastix`/`ants` for intensity-based registration where features are unreliable (e.g. different channels).
- **Transform**: use the simplest that fits - `translation` for stage tiles, `rigid` if tiles rotate.
  `affine` on a feature method with few matches can shear tiles.
- **Pairing**: keep `orthogonal` for a tile grid - default pairing adds corner-only diagonal pairs where a
  feature method can fit a wild rotation. Use `pairing: stack` (not an operation) for z-sections.
- **Large data**: `pre_processing.scale` (e.g. `4` or `1um`) registers on a downscaled copy; fusion stays at
  full resolution. Leave `fusion.block_size` unset - it is sized against the available memory. In the plugin
  the setting is also on the Registration tab, so its preview registration can be tried at different downscales.
- **Several sections/folders in one run**: `operation: register match S` groups files by a path part made of `S` and a
  number (e.g. a folder `S001`), and `{S}` in `output_path` names each group's output.

## Pitfalls

- `output_path` is joined to the input directory: `input_path: /data/run1/*.tif` with `output_path: out/`
  writes to `/data/run1/out/`. Give an absolute path to write elsewhere. Keep the trailing `/`.
- The output is named after the operation in the past tense: `register` writes `registered.ome.zarr`.
- `overwrite: false` (the pipeline default) skips work whose output exists; set `true` when re-running.
- `NotEnoughOverlapError` ("Not enough overlap between views") with `sift`/`orb` usually means too few
  pairs found matches, not that tiles do not overlap: set `registration.gaussian_sigma` (e.g. 2) and
  `normalisation: true`, or check the positions with `merge` first.
- `operation: convert` only works in the napari plugin; the pipeline does not convert.
- `cpd` needs `probreg` (not on Windows or Python 3.13+); `elastix`/`ants` need `muvis-align[all]`.
- Try settings on a subset first: a list of 2-4 neighbouring tiles as `input_path`, with
  `pre_processing.scale` set, before running a full dataset.

## Checking a result

The output folder holds, next to the image:
- `metrics.csv` - per registered pair: quality, ncc, ssim, onmi over the overlap. Low values flag bad pairs.
  With split pairing the group pairs (planes/channels) follow, without filenames.
- `residuals.csv` - each pair's disagreement with the global solution; a large one is a pair to distrust.
- `positions_*.pdf` - the tile layout, to spot a source that moved far from its stage position.
- `mappings.json`/`.csv` - the final transform per source; `parameters.yaml` + `ro-crate-metadata.json` -
  every setting used, for reproducing the run.
- The log (`general.logging.filename`, default `log/muvis-align.log`) reports skipped operations, matched file
  counts and errors. `Pipeline.run()` returning False means an operation raised.
