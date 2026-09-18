# Running the muvis-align Pipeline

## Overview

The muvis-align pipeline is command-line driven and uses YAML parameter files to configure all operations. The main entry point is `run.py`.

For interactive work - loading a dataset, previewing pairs and fused results, and
watching progress as it runs - see the [napari plugin](napari.md). The two use
separate configuration files: the pipeline takes the `general`/`operations` file
described here, while the plugin reads and writes its own project file.

## Basic Usage

```bash
python run.py resources/params_test_2d.yml
```

Where `resources/params_test_2d.yml` is your parameter configuration file.

## Parameter File Structure

The parameter file uses YAML format with two main sections:

### 1. General Configuration (`general`)

Global settings for logging, output, and error handling.

```yaml
general:
  overwrite: True
  clear: False
  logging:
    verbose: True
    debug: False
    filename: log/muvis-align.log
    format: '%(asctime)s %(levelname)s: %(message)s'
  output:
    format: ome.zarr
    tile_size: [4096, 4096]
    compression: null
    npyramid_add: 4
    pyramid_downsample: 2
    ome_version: '0.5'
    preview: ome.zarr
    preview_scale: 32
  break_on_error: False
  metadata_summary: False
```

#### General options explained:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `overwrite` | bool | `False` | Overwrite existing output files |
| `clear` | bool | `False` | Clear output directory before processing |
| `break_on_error` | bool | `False` | Stop on first error (vs. continue) |
| `metadata_summary` | bool | `False` | Print metadata summary for each fileset |

#### Logging options:

| Parameter | Type | Description |
|-----------|------|-------------|
| `filename` | str | Log file path (default: `log/muvis-align.log`) |
| `format` | str | Log message format string |
| `verbose` | bool | muvis-align's own logging, echoed to the console as well as the log file. Also turns on the dask progress bar and per-phase timing |
| `debug` | bool | External libraries (`multiview_stitcher`, `zarr`) at `DEBUG` level |

#### Output options:

| Parameter            | Type | Description                                                             |
|----------------------|------|-------------------------------------------------------------------------|
| `format`             | str | Output format: `ome.zarr` or `ome.tiff` (or both: `ome.zarr, ome.tiff`) |
| `tile_size`          | list | Tile size for zarr output `[x, y]` or `[x, y, z]`. Leave unset to size it automatically - see [Tile size](#tile-size) |
| `compression`        | str/list | Compression method(s)                                                   |
| `npyramid_add`       | int | Number of pyramid levels to add (default `0`)                           |
| `pyramid_downsample` | int | Downsampling factor for pyramid (default `2`)                           |
| `ome_version`        | str | OME-Zarr version to write: `'0.4'` or `'0.5'` (default `'0.5'`)         |
| `preview`            | str | Preview format                                                          |
| `preview_scale`      | int/str | Preview downscale factor (default `16`), or a physical size such as `1um` |

These may also be set per operation, under that operation's own `output:` - the
operation's value wins, falling back to `general.output`.

`clear` and `overwrite` are read at the `general` level, not under `output`.

#### Other options:

| Parameter | Type | Description |
|-----------|------|-------------|
| `overlap_threshold` | float | Minimum overall overlap required to proceed (default `0.5`). Set per operation |

### 2. Operations (`operations`)

List of processing operations to run sequentially.

```yaml
operations:
  - operation: register
    input:
      path: data/S000/*.zarr
      source_metadata: source
    registration:
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
    output:
      path: ../../output/stitched/
```

## Operation Types

### `register` - Image Registration

Registers (aligns) images using feature matching or phase correlation.

**Variants:**
- `register` - Basic registration
- `register match LABEL` - Group files by matching a label in filenames, then register each group
- `register stack` - Register images as a z-stack (consecutive 2D registrations)
- `transition` - Register using a transition transform between filesets
- `fuse` - Fuse without registering, using the source metadata positions as they are

The operation name also names the output: the verb is written in the past tense, so
`register` writes `registered.ome.zarr` and `fuse` writes `fused.ome.zarr`.

3D registration is not a separate variant - it follows from the data. Sources with a
real z extent are registered in 3D; `register stack` max-projects z and registers
consecutive 2D pairs instead.

!!! warning "Convert is implemented in the napari plugin only"
    Converting each source individually to OME-Zarr, keeping its native pyramid levels,
    is currently implemented only in the plugin - see [Convert](napari.md#convert).
    `operation: convert` is accepted by the YAML pipeline (and
    `resources/params_test_convert.yml` uses it), but the pipeline has no convert step:
    it neither registers nor writes converted output. Use the plugin for this.

#### Input Configuration

```yaml
input:
  path: data/S000/*.zarr            # File path pattern (supports wildcards)
  source_metadata: source           # How to interpret source file metadata
  labels: [tile_00_00, tile_00_01]  # Optional: custom labels for input files
  extra_metadata: {}                # Extra metadata to apply
```

**source_metadata options:**
- `source` - Use metadata from source image files
- `source invert` - Use source metadata but invert x/y coordinates
- `global` - Use global metadata (requires `metadata_summary: True`)
- `source_metadata: {'scale': {'x': 0.004, 'y': 0.004}, ...}` - Custom metadata dict

#### Registration Configuration

```yaml
registration:
  # Registration method
  method: sift              # phase_correlation, sift, orb, feature, cpd, elastix, ants
  name: orb                 # Alternative to 'method'
  
  # Feature detection (for sift, orb, feature methods)
  max_keypoints: 5000       # Maximum number of keypoints to detect
  gaussian_sigma: 1         # Gaussian filter sigma before detection
  
  # Feature matching parameters
  inlier_threshold_factor: 0.05
  max_trials: 1000
  ransac_iterations: 3
  
  # Transform type (rigid, translation, affine, similarity)
  transform_type: rigid
  
  # Pairing strategy
  pairing: orthogonal       # orthogonal, overlay, stack; unset = multiview-stitcher's own pairing
  
  # Normalization
  normalisation: True       # True, False, 'global', 'individual'
  
  # Global resolution method
  groupwise_resolution_method: global_optimization  # or other methods
  
  # Quality filtering
  post_registration_quality_threshold: 0.5
  
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
- `cpd` - Coherent Point Drift
- `elastix` - Elastix registration
- `ants` - Advanced Normalization Tools (requires the `antspyx` package)

**Transform Types:**
- `translation` - Only translation
- `rigid` - Translation + rotation
- `affine` - Full affine transformation
- `similarity` - Rigid + uniform scaling

**Pairing Strategies:**
- `orthogonal` - Pair orthogonal tiles (X-Y grid), avoiding diagonal / very small overlaps
- `overlay` - Pair tiles based on overlap, for stack-like overlaps
- `stack` - Pair consecutive views, as the `register stack` operation does
- unset - Use multiview-stitcher's own default pairing

`pairing: stack` and the `register stack` operation are equivalent: either one
max-projects z and registers consecutive pairs. Note that this applies to 2D sources -
sources with a real z extent are registered in 3D, and fall back to the default pairing.

**Metrics:**
- `ncc` - Normalized Cross Correlation
- `ssim` - Structural Similarity Index
- `onmi` - Overlapping Normalized Mutual Information

#### Preprocessing Options

```yaml
preprocessing:
  flatfield_quantiles: [0.5, 0.95]  # Flat-field correction quantiles
  gaussian_sigma: 1                  # Gaussian blur sigma (optional)
  normalisation: global              # Image normalization
  filter_foreground: True            # Filter out empty/background images
  scale: 4                           # Downsampling factor for processing
```

#### Fusion Options

Combine registered images into a single output.

```yaml
fusion:
  method: average          # average, exclusive, additive, compose
  output_spacing: mean     # mean, min, max
```

**Fusion Methods:**
- `average` (default) - Average overlapping pixels
- `exclusive` - Show only single tile data where overlapping
- `additive` - Sum overlapping regions
- `compose` - Compositional blending (*experimental*)

**Fusion Options:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `average` | Fusion method, as above |
| `output_spacing` | str | `mean` | Output pixel size across sources: `mean`, `min` (highest resolution) or `max` (lowest resolution) |

### Tile size

`general.output.tile_size` (or an operation's own `output.tile_size`) does double duty
for a zarr export: it is the on-disk tile shape *and* the block size the fusion is
computed in. A small value therefore costs real time, because the fixed per-block
overhead is paid once per tile.

Left unset, the tile size is chosen automatically from the memory one fused block
actually needs, budgeted against the memory available to the process - the SLURM
allocation or cgroup limit where there is one, not the size of the machine. This is
the recommended setting; give an explicit value only when the on-disk layout matters
more than the export time.

The size actually used is written to the log, so a run can be checked after the fact.

#### Output Configuration

```yaml
output:
  path: ../../output/stitched/    # Output directory
```

## Complete Example: 2D Stitching

```yaml
general:
  overwrite: True
  logging:
    verbose: True
  output:
    format: ome.zarr

operations:
  - operation: register
    input:
      path: data/S000/*.zarr
      source_metadata: source
    registration:
      pairing: orthogonal
      transform_type: rigid
      method: sift
      max_keypoints: 5000
      gaussian_sigma: 1
      normalisation: True
      metrics: [ncc, ssim]
    output:
      path: ../../output/stitched/
```

## Advanced Examples

### Multi-Dataset Registration with Matching

```yaml
operations:
  - operation: register match S
    input: /data/S???/*.tiff
    source_metadata: {'scale': {'x': 0.004, 'y': 0.004}, 'position': {'y':'fn[-3]*24', 'x':'fn[-2]*24'}}
    registration:
      name: orb
      pairing: orthogonal
      transform_type: translation
    output: ../../stitched/S{S}/
```

Registers files from multiple directories (S001, S002, etc.) separately, using `{S}` placeholder in output path.

### Stack Registration

```yaml
operations:
  - operation: register stack
    input: ./registered.ome.zarr
    source_metadata: source
    registration:
      name: orb
      transform_type: rigid
      scale: 4
    output: ../../aligned/
```

Registers slices as a z-stack, applying consecutive 2D registrations.

### Multi-Channel Registration

```yaml
operations:
  - operation: register
    input:
      - /data/channel0_registered.ome.zarr
      - /data/channel1_registered.ome.zarr
    normalisation: individual
    registration: ants
    channel: 0  # Register using first channel
    output: /output/fused/
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

## Output Structure

After successful registration:
- `registered.ome.zarr` - Registered & fused output image (named after the operation)
- `mappings.json` / `mappings.csv` - Final registration mappings
- `pair_mappings.json` - Per-pair registration mappings
- `prereg_mappings.csv` - Pre-registration mappings
- `metrics.json` - Calculated metrics
- `positions_original.pdf`, `positions_registered.pdf` - Position visualizations

## Troubleshooting

- **No files matched**: Check your `input.path` pattern and file extensions
- **No overlap found**: Verify source metadata (position/scale) or adjust pairing strategy
- **Out of memory**: Leave `tile_size` unset so blocks are sized against available memory, or use `scale` to downsample
- **Poor registration**: Try different `method`, increase `max_keypoints`, or adjust `normalisation`

## Parameter Resolution

Parameters can use placeholders based on filename:
- `{S}` - Matches numeric values from filename (e.g., from S001)
- `fn[-3]` - Access filename elements by position
- Custom metadata patterns for position/scale resolution
