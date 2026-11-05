# Running the muvis-align Pipeline

## Overview

The muvis-align pipeline is command-line driven and uses YAML parameter files to configure all operations. The main entry point is `run.py`.

For interactive work - loading a dataset, previewing pairs and fused results, and
watching progress as it runs - see the [napari plugin](napari_guide.md). Both use the
same settings: each operation of a pipeline file has the sections of a plugin project
file (`input_output`, `pre_processing`, `registration`, `fusion`), with the names the
project template gives them. The pipeline file adds the run's `general` settings and
can list several operations.

## Basic Usage

```bash
python run.py resources/params_test_2d.yml
```

Where `resources/params_test_2d.yml` is your parameter configuration file.

## Parameter File Structure

The parameter file uses YAML format with two main sections:

### 1. General Configuration (`general`)

Settings of the run itself: logging and error handling.

```yaml
general:
  logging:
    verbose: True
    debug: False
    filename: log/muvis-align.log
    format: '%(asctime)s %(levelname)s: %(message)s'
  break_on_error: False
  metadata_summary: False
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `break_on_error` | bool | `False` | Stop on first error (vs. continue) |
| `metadata_summary` | bool | `False` | Print metadata summary for each fileset |

#### Logging options:

| Parameter | Type | Description |
|-----------|------|-------------|
| `filename` | str | Log file path (default: `log/muvis-align.log`) |
| `format` | str | Log message format string |
| `verbose` | bool | muvis-align's own logging, echoed to the console as well as the log file. Also turns on the dask progress bar and per-phase timing |
| `debug` | bool | External libraries (`multiview_stitcher`, `zarr`) at `DEBUG` level |

### 2. Operations (`operations`)

List of processing operations to run sequentially, each in the project template's sections.

```yaml
operations:
  - input_output:
      input_path: data/S000/*.zarr
      output_path: ../../output/stitched/
      overwrite: True
    registration:
      operation: register
      pairing: orthogonal
      transform_type: rigid
      method: sift
      gaussian_sigma: 1
      normalisation: True
      max_keypoints: 5000
      inlier_threshold_factor: 0.05
      max_trials: 1000
      ransac_iterations: 3
      metrics: [ncc, ssim, onmi]
      n_parallel_pairwise_regs: 1
    fusion:
      format: ome.zarr
```

An operation in the former layout (`operation`, `input` or `output` at its top level) is
refused with an error naming the sections to use instead.

A few settings have no field in the plugin. They go in the section they belong to, and
the run's RO-Crate keeps them, with a warning that the project template lacks them:
`input_output.labels`, `extra_metadata`, `source_metadata` and `n_set_workers`;
`registration.metrics`, `overlap_threshold` and `post_registration_quality_threshold`;
`fusion.format`, `compression`, `npyramid_add`, `pyramid_downsample`, `preview`,
`save_images` and `transition`.

## Operation Types

The operation is `registration.operation`.

### `register` - Image Registration

Registers (aligns) images using feature matching or phase correlation.

**Variants:**
- `register` - Basic registration
- `register match LABEL` - Group files by matching a label in filenames, then register each group
- `transition` - Register using a transition transform between filesets
- `fuse` / `merge` - Fuse without registering, using the source metadata positions as they are

The operation name also names the output: the verb is written in the past tense, so
`register` writes `registered.ome.zarr` and `fuse` writes `fused.ome.zarr`.

3D registration is not a separate variant - it follows from the data. Sources with a
real z extent are registered in 3D.

Registering a z-stack is asked for with `pairing: stack`, not by the operation - see
the pairing strategies under [Registration Configuration](#registration-configuration).

!!! warning "Convert is implemented in the napari plugin only"
    Converting each source individually to OME-Zarr, keeping its native pyramid levels,
    is currently implemented only in the plugin - see [Registration](napari_guide.md#4-registration).
    `operation: convert` is accepted by the YAML pipeline (and
    `resources/params_test_convert.yml` uses it), but the pipeline has no convert step:
    it neither registers nor writes converted output. Use the plugin for this.

#### Input and Output Configuration

```yaml
input_output:
  input_path: data/S000/*.zarr        # File path pattern (supports wildcards), or a list of paths
  output_path: ../../output/stitched/ # Output directory
  overwrite: True                     # Overwrite existing output (default False)
  clear: False                        # Clear the output directory before processing
  source_position_x: source           # Where each position, scale and rotation comes from
  source_position_y: source
  source_scale_x: '0.004'
  source_rotation: source
  normalise_rotated_positions: False
  preview_scale: 16                   # Preview downscale factor, or a physical size such as 1um
  channels_table: {label: [DAPI, GFP], color: ['(0, 0, 1)', '(0, 1, 0)']}
  labels: [tile_00_00, tile_00_01]    # Optional: custom labels for input files
  n_set_workers: 1                    # File sets of a `match` operation run at once
```

**Source position, scale and rotation** (`source_position_x/y/z`, `source_scale_x/y/z`,
`source_rotation`), as in the plugin:
- `source` - the source file's own value
- `source invert` - the source's own value, inverted
- `source y` - the source's value of another axis
- a value or expression, e.g. `fn[-3]*24` (see [Parameter Resolution](#parameter-resolution))

`normalise_rotated_positions` straightens a rotated tile grid.

What no field holds goes in `input_output.source_metadata`: flags such as `global center`
(the centre over all file sets of a `match` operation; `global` also needs
`metadata_summary: True`), per-file values, or a metadata file (`.json`/`.csv`).
`input_output.extra_metadata` holds further metadata such as a z `scale`, or a metadata file.

#### Registration Configuration

```yaml
registration:
  operation: register

  # Registration method
  method: sift              # phase_correlation, sift, orb, feature, cpd, elastix, ants

  # Overlap filtering (sift, orb and cpd only; pre-processing's apply to every method)
  gaussian_sigma: 1         # Gaussian filter sigma before detection
  normalisation: True

  max_keypoints: 5000       # Maximum number of keypoints to detect (sift, orb and cpd only)

  # Feature matching (sift and orb only; other methods ignore these)
  inlier_threshold_factor: 0.05
  max_trials: 1000
  ransac_iterations: 3

  # Transform type (rigid, translation, affine, similarity)
  transform_type: rigid

  # Pairing strategy
  pairing: orthogonal       # orthogonal, stack; unset = multiview-stitcher's own pairing

  # Global resolution method
  groupwise_resolution_method: global_optimization  # or other methods

  # Quality filtering
  post_registration_quality_threshold: 0.5
  overlap_threshold: 0.5    # Minimum overall overlap required to proceed

  # Parallel processing
  n_parallel_pairwise_regs: 1

  # Output metrics
  metrics: [ncc, ssim, onmi]  # Metrics to compute
```

**Registration Methods:**
- `phase_correlation` - Phase correlation (default)
- `sift` - Scale-Invariant Feature Transform (scikit-image)
- `orb` - Oriented FAST and Rotated BRIEF (OpenCV)
- `feature` - Generic feature-based registration
- `cpd` - Coherent Point Drift (requires `probreg`, not available for Windows or Python 3.13+)
- `elastix` - Elastix registration (requires `itk-elastix`)
- `ants` - Advanced Normalization Tools (requires `antspyx`)

The packages of these last three are installed with `muvis-align[all]`.

**Transform Types:**
- `translation` - Only translation
- `rigid` - Translation + rotation
- `affine` - Full affine transformation
- `similarity` - Rigid + uniform scaling

**Pairing Strategies:**
- `orthogonal` - Pair orthogonal tiles (X-Y grid), avoiding diagonal / very small overlaps
- `stack` - Treat the sources as a z-stack: pair consecutive views, and promote them to
  3D where the rest of the pipeline needs it
- unset - Use multiview-stitcher's own default pairing

The plugin preselects `orthogonal`: multiview-stitcher's own pairing adds corner-only diagonal pairs, on which a
feature method such as SIFT can fit an arbitrary rotation that throws tiles far off in the global registration.

`stack` pairs consecutive views - N to N+1, N+1 to N+2, and so on - so it is an
alternative to `orthogonal`, not a modifier on it: a stack is never paired
orthogonally. It max-projects z to register the pairs in 2D.

**Metrics:**
- `ncc` - Normalized Cross Correlation
- `ssim` - Structural Similarity Index
- `onmi` - Overlapping Normalized Mutual Information

#### Preprocessing Options

```yaml
pre_processing:
  flatfield_quantiles: [0.5, 0.95]  # Flat-field quantiles: one = light image, two = dark, light image
  gaussian_sigma: 1                  # Gaussian blur sigma (optional)
  normalisation: global              # Image normalization: single, global or none
  filter_foreground: True            # Filter out empty/background images
  scale: 4                           # Downsampling factor for processing, or a pixel size such as 1um
```

#### Fusion Options

Combine registered images into a single output, and write it.

```yaml
fusion:
  method: average          # average, exclusive, additive, compose
  spacing: native          # native, mean, min, max
  format: ome.zarr
  block_size: [4096, 4096]
  ome_version: '0.5'
  compression: null
  npyramid_add: 4
  pyramid_downsample: 2
  preview: ome.zarr
```

**Fusion Methods:**
- `average` (default) - Average overlapping pixels
- `exclusive` - Show only single tile data where overlapping
- `additive` - Sum overlapping regions
- `compose` - No fusion: each source keeps its own pyramid, written as one OME-Zarr 0.6 scene with a
  `registered` and a `source_metadata` coordinate system (`registered` is the default view). Always OME-Zarr 0.6, whatever `ome_version` is set

**Fusion Options:**

| Parameter            | Type | Default | Description |
|----------------------|------|---------|-------------|
| `method`             | str | `average` | Fusion method, as above |
| `spacing`            | str | `native` | Output pixel size: `native` (each level at the sources' own pixel sizes, never upsampled; written to zarr only, else `mean`), `mean`, `min` (highest resolution) or `max` (lowest resolution) |
| `format`             | str | `ome.zarr` | Output format: `ome.zarr` or `ome.tiff` (or both: `ome.zarr, ome.tiff`) |
| `block_size`         | list | | Block size for the output `[x, y]` or `[x, y, z]`. Leave unset to size it automatically - see [Block size](#block-size) |
| `ome_version`        | str | `'0.5'` | OME-Zarr version to write: `'0.4'` or `'0.5'` |
| `compression`        | str/list | | Compression method(s) |
| `npyramid_add`       | int | `0` | Number of pyramid levels to add |
| `pyramid_downsample` | int | `2` | Downsampling factor for pyramid |
| `preview`            | str | | Preview format; its scale is `input_output.preview_scale` |
| `save_images`        | bool | `True` | Write the fused image |
| `transition`         | dict | | A `transition` operation's video: `frames`, `fps`, `spacing`, `scale` |

### Block size

`fusion.block_size` does double duty for a zarr export: it is the on-disk chunk shape
*and* the block size the fusion is computed in. A small value therefore costs real time,
because the fixed per-block overhead is paid once per block. For a TIFF export it is the
TIFF tile size.

Left unset, it is 4096. Every CPU fuses blocks, and the blocks share a quarter of the
memory available to the process - the Slurm allocation or cgroup limit where there is
one, not the size of the machine: each block reserves what it needs before it runs, so
a block meeting many sources waits for room instead of every block being made smaller.
A level shrinks its blocks only where one alone would not fit. This is the recommended
setting; give an explicit value only when the on-disk layout matters more than the
export time. The napari plugin offers 1024, 2048, 4096 and 8192.

The size actually used is written to the log, so a run can be checked after the fact.

## Complete Example: 2D Stitching

```yaml
general:
  logging:
    verbose: True

operations:
  - input_output:
      input_path: data/S000/*.zarr
      output_path: ../../output/stitched/
      overwrite: True
    registration:
      operation: register
      pairing: orthogonal
      transform_type: rigid
      method: sift
      max_keypoints: 5000
      gaussian_sigma: 1
      normalisation: True
      metrics: [ncc, ssim]
    fusion:
      format: ome.zarr
```

## Advanced Examples

### Multi-Dataset Registration with Matching

```yaml
operations:
  - input_output:
      input_path: /data/S???/*.tiff
      output_path: ../../stitched/S{S}/
      source_scale_x: 0.004
      source_scale_y: 0.004
      source_position_x: fn[-2]*24
      source_position_y: fn[-3]*24
    registration:
      operation: register match S
      method: orb
      pairing: orthogonal
      transform_type: translation
```

Registers files from multiple directories (S001, S002, etc.) separately, using `{S}` placeholder in output path.

### Stack Registration

```yaml
operations:
  - input_output:
      input_path: ./registered.ome.zarr
      output_path: ../../aligned/
    pre_processing:
      scale: 4
    registration:
      operation: register
      pairing: stack
      method: orb
      transform_type: rigid
```

Registers slices as a z-stack, pairing them consecutively (N to N+1, N+1 to N+2, ...).

### Multi-Channel Registration

```yaml
operations:
  - input_output:
      input_path:
        - /data/channel0_registered.ome.zarr
        - /data/channel1_registered.ome.zarr
      output_path: /output/fused/
    pre_processing:
      normalisation: single
    registration:
      operation: register
      method: ants
      channel: 0  # Register using first channel
```

Combines multiple channels using registration from a specific channel.

## Running the Pipeline

1. **Create a parameter file** (e.g., `params.yml`)
2. **Run the pipeline:**
   ```bash
   python run.py params.yml
   ```

3. **Check the log file** for progress and errors:
   ```bash
   tail -f log/muvis-align.log
   ```

On a Slurm cluster, [`run_pipeline.sh`](https://github.com/ccp-volume-em/muvis-align/blob/main/run_pipeline.sh)
runs the same command as a batch job:

```bash
sbatch run_pipeline.sh params.yml
```

The job's CPUs, memory and time limit are the `#SBATCH` lines at the top of the script; its `ml` and `conda` lines
load one cluster's Python environment, so adapt them to yours. Fusion runs on every CPU of the job, its blocks
sharing a quarter of the job's memory (`--mem`), so a smaller allocation makes busy blocks wait rather than run out.
The script's 64G is enough for a 34,000-tile project.

## Output Structure

After successful registration:
- `registered.ome.zarr` - Registered & fused output image (named after the operation)
- `mappings.json` / `mappings.csv` - Final registration mappings
- `pair_mappings.json` - Per-pair registration mappings
- `prereg_mappings.csv` - Pre-registration mappings
- `metrics.json` - Calculated metrics
- `positions_original.pdf`, `positions_registered.pdf` - Position visualizations
- `ro-crate-metadata.json`, `parameters.yaml` - The run's RO-Crate, and every setting it used

## Troubleshooting

- **No files matched**: Check your `input_output.input_path` pattern and file extensions
- **No overlap found**: Verify source metadata (position/scale) or adjust pairing strategy
- **Out of memory**: Leave `block_size` unset so blocks are sized against available memory, or use `scale` to downsample
- **Poor registration**: Try different `method`, increase `max_keypoints`, or adjust `normalisation`

## Parameter Resolution

Parameters can use placeholders based on filename:
- `{S}` - Matches numeric values from filename (e.g., from S001)
- `fn[-3]` - Access filename elements by position
- Custom metadata patterns for position/scale resolution
