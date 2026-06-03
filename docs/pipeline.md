# Running the muvis-align Pipeline

## Overview

The muvis-align pipeline is command-line driven and uses YAML parameter files to configure all operations. The main entry point is `run.py`.

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
  logging:
    verbose: True
    filename: log/muvis-align.log
    format: '%(asctime)s %(levelname)s: %(message)s'
    dask: False
    time: False
  output:
    format: ome.zarr
    clear: False
    tile_size: [4096, 4096]
    compression: null
    npyramid_add: 4
    pyramid_downsample: 2
    thumbnail: ome.zarr
    thumbnail_scale: 32
  break_on_error: False
  metadata_summary: False
  chunk_size: [1024, 1024]
  show_original: False
```

#### General options explained:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `overwrite` | bool | `False` | Overwrite existing output files |
| `clear` | bool | `False` | Clear output directory before processing |
| `break_on_error` | bool | `False` | Stop on first error (vs. continue) |
| `metadata_summary` | bool | `False` | Print metadata summary for each fileset |
| `show_original` | bool | `False` | Whether to show original images in UI |

#### Logging options:

| Parameter | Type | Description |
|-----------|------|-------------|
| `filename` | str | Log file path (default: `log/muvis-align.log`) |
| `format` | str | Log message format string |
| `verbose` | bool | Enable verbose logging |
| `dask` | bool | Enable dask progress bar |
| `time` | bool | Log timing information |

#### Output options:

| Parameter | Type | Description |
|-----------|------|-------------|
| `format` | str | Output format: `ome.zarr` or `ome.tiff` (or both: `ome.zarr, ome.tiff`) |
| `clear` | bool | Delete output directory before processing |
| `tile_size` | list | Tile size for zarr output `[x, y]` or `[x, y, z]` |
| `compression` | str/list | Compression method(s) |
| `npyramid_add` | int | Number of pyramid levels to add |
| `pyramid_downsample` | int | Downsampling factor for pyramid |
| `thumbnail` | str | Thumbnail format |
| `thumbnail_scale` | int | Thumbnail scale factor |

#### Other options:

| Parameter | Type | Description |
|-----------|------|-------------|
| `chunk_size` | list | Default chunk size `[x, y]` or `[x, y, z]` |

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
- `register 3d` - Full 3D registration

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
  method: sift              # sift, orb, feature, cpd, ANTsPy, phase_correlation
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
  pairing: orthogonal       # orthogonal, overlap, stack (default)
  
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
- `sift` - Scale-Invariant Feature Transform (scikit-image)
- `orb` - Oriented FAST and Rotated BRIEF (OpenCV)
- `feature` - Generic feature-based registration
- `cpd` - Coherent Point Drift
- `ANTsPy` - Advanced Normalization Tools
- `phase_correlation` - Phase correlation (default)

**Transform Types:**
- `translation` - Only translation
- `rigid` - Translation + rotation
- `affine` - Full affine transformation
- `similarity` - Rigid + uniform scaling

**Pairing Strategies:**
- `orthogonal` - Pair orthogonal tiles (X-Y grid)
- `overlap` - Pair tiles based on overlap
- `stack` - Pair consecutive slices (z-stacks)

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
  method: average          # average, composite, exclusive, additive
  blend_edges: True        # Blend overlapping regions
```

**Fusion Methods:**
- `simple_average_fusion` (default) - Average overlapping pixels
- `exclusive` - Showing only single tile data where overlapping
- `additive` - Sum overlapping regions
- `composite` - Compositional blending (*experimental*)

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
    registration: ANTsPy
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
- `registered.ome.zarr` - Registered & fused output image
- `mappings.json` - Registration mapping information
- `metrics.json` - Calculated metrics
- `*.pdf` - Position visualizations

## Troubleshooting

- **No files matched**: Check your `input.path` pattern and file extensions
- **No overlap found**: Verify source metadata (position/scale) or adjust pairing strategy
- **Out of memory**: Reduce `chunk_size`, `tile_size`, or use `scale` to downsample
- **Poor registration**: Try different `method`, increase `max_keypoints`, or adjust `normalisation`

## Parameter Resolution

Parameters can use placeholders based on filename:
- `{S}` - Matches numeric values from filename (e.g., from S001)
- `fn[-3]` - Access filename elements by position
- Custom metadata patterns for position/scale resolution
