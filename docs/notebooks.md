# Jupyter Notebooks Guide

This guide provides an overview of the Jupyter notebooks included in the muvis-align project. These notebooks demonstrate various aspects of the image registration and stitching pipeline, from basic workflows to interactive debugging and visualization.

## Quick Navigation

| Notebook | Purpose | Audience |
|----------|---------|----------|
| [pipeline.ipynb](#pipelineipynb) | Complete workflow example | Beginners |
| [stitching2d.ipynb](#stitching2dipynb) | Streamlined 2D stitching | General users |
| [feature_matching.ipynb](#feature_matchingipynb) | Interactive feature visualization | Method developers |
| [feature_matching_metrics.ipynb](#feature_matching_metricsipynb) | Feature matching with metrics | Advanced users |
| [feature_matching_overlay.ipynb](#feature_matching_overlayipynb) | Tile overlap visualization | Debugging |
| [stitching2d_metrics.ipynb](#stitching2d_metricsipynb) | 2D stitching with metrics | Quality analysis |
| [stitching2d_split_registration.ipynb](#stitching2d_split_registrationipynb) | Advanced registration control | Parameter optimization |
| [tile_staging_napari.ipynb](#tile_staging_napariipynb) | Interactive napari visualization | Results inspection |

---

## pipeline.ipynb

**Purpose:** Complete registration pipeline example showing all major steps

**Best for:** Understanding the full workflow for 2D x/y stitching

**Key Steps:**

1. **Initialization** - Set up the MVSRegistration object with input/output paths
2. **Load & Preprocess** - Initialize spatial images and apply preprocessing (normalization)
3. **Feature Detection** - Configure registration method parameters
4. **Registration** - Perform pairwise and global registration
5. **Fusion** - Combine registered images into a single output

**Main Operations:**

```python
# Initialize
reg = MVSRegistration(operation='register', 
                      input_path='../data/S000/*.zarr', 
                      output_path='../../output/', 
                      ui='mpl')

# Load and preprocess
sims = reg.init_sims()
reg_sims, reg_indices, _ = reg.preprocess(sims, normalisation='global')

# Configure and run registration
register_params = {
    'pairing': 'orthogonal',
    'transform_type': 'rigid',
    'method': 'sift',
    'max_keypoints': 5000,
    # ... more parameters
}

results = reg.register(sims, reg_sims, reg_indices, register_params)

# Fuse registered images
fusion_params = {'method': 'average'}
fused_sim = reg.fuse(sims, fusion_params)
```

**Output:** Registered and fused image with transformation matrices

**Customization:** All parameters can be modified to experiment with different registration methods and settings

---

## stitching2d.ipynb

**Purpose:** Streamlined example focusing on the essential 2D stitching workflow

**Best for:** Quick prototyping and understanding basic registration

**Key Differences from pipeline.ipynb:**
- Combines operations into fewer, more focused cells
- Emphasizes the core registration workflow
- Includes quality reporting

**Main Operations:**

```python
# One-step initialization, preprocessing, and registration
reg = MVSRegistration(operation='register', 
                      input_path='../data/S000/*.zarr', 
                      output_path='../../output/', 
                      ui='mpl')
sims = reg.init_sims()
reg_sims, reg_indices, _ = reg.preprocess(sims)

register_params = {
    'pairing': 'orthogonal',
    'transform_type': 'rigid',
    'method': 'sift',
    'gaussian_sigma': 2,
    'normalisation': True,
    'max_keypoints': 5000,
    'inlier_threshold_factor': 0.05,
    'max_trials': 1000,
    'ransac_iterations': 3,
}

results = reg.register(sims, reg_sims, reg_indices, register_params)

# Report quality metrics
qualities = {key: value.item() for key, value in results['registration_qualities'].items()}
print_dict_simple(qualities)
```

**Output:** Registration quality metrics for pairwise registrations

**Tip:** Use this as a starting template for your own registration pipelines

---

## feature_matching.ipynb

**Purpose:** Interactive exploration of feature detection and matching between image pairs

**Best for:** Understanding feature-based registration and debugging matching issues

**Key Features:**

1. **Overlap Visualization** - View the overlapping regions between adjacent tiles
2. **Interactive Parameters** - Modify parameters and see results immediately
3. **Keypoint Visualization** - Visualize detected keypoints and their matches

**Main Operations:**

```python
# Initialize with debug mode for detailed output
reg = MVSRegistration(operation='register', 
                      input_path='../data/S000/*.zarr', 
                      output_path='../../output/', 
                      ui='mpl', 
                      debug=True)
sims = reg.init_sims()
norm_sims, _, _ = reg.preprocess(sims)

# Get and visualize overlap
overlap1, overlap2, _ = get_overlap_images(norm_sims[0], norm_sims[1], 
                                            reg.source_transform_key)
draw_keypoints_matches(overlap1, [], overlap2, [])

# Configure and visualize feature matching
register_params = {
    'transform_type': 'rigid',
    'pairing': 'orthogonal',
    'name': 'sift',  # Try 'sift' or 'orb'
    'normalisation': True,
    'gaussian_sigma': 1,
    'max_keypoints': 5000,
    'inlier_threshold_factor': 0.05,
    'max_trials': 1000,
    'ransac_iterations': 3,
}
```

**Interactive Exploration:**
- Modify `name` parameter to use different feature detectors:
  - `'sift'` - Scale-Invariant Feature Transform (more robust, slower)
  - `'orb'` - Oriented FAST and Rotated BRIEF (faster, less robust)
- Adjust `gaussian_sigma` to smooth the image before feature detection
- Change `transform_type` to test different transformation models:
  - `'translation'` - Only translation
  - `'euclidean'` - Translation + rotation
  - `'affine'` - Full affine

**Output:** Visual overlays of keypoints and feature matches between tiles

---

## feature_matching_metrics.ipynb

**Purpose:** Interactive feature matching with alignment quality metrics

**Best for:** Optimizing registration parameters and evaluating match quality

**Key Features:**

1. **Feature Visualization** - Like `feature_matching.ipynb`
2. **Alignment Metrics** - Compute NCC, SSIM, ONMI on overlapping regions
3. **Transform Visualization** - Show the estimated transformation

**Main Operations:**

```python
# Initialize
reg = MVSRegistration(operation='register', 
                      input_path='../data/S000/*.zarr', 
                      output_path='../../output/', 
                      ui='mpl', 
                      debug=True)
sims = reg.init_sims()
norm_sims, _, _ = reg.preprocess(sims)

# Get overlap with metrics
overlap1, overlap2, sims_pixel_space = get_overlap_images(norm_sims[0], norm_sims[1], 
                                                           reg.source_transform_key)

# Compute metrics
metrics = calc_sims_metrics(norm_sims[0:2], overlap_sims=sims_pixel_space)

register_params = { /* ... */ }
# Run registration and get results
```

**Metrics Computed:**
- **NCC** (Normalized Cross Correlation) - 0 to 1, higher is better
- **SSIM** (Structural Similarity Index) - -1 to 1, higher is better  
- **ONMI** (Overlapping Normalized Mutual Information) - Higher is better

**Use Case:** Parameter tuning by iteratively adjusting parameters and checking improvement in metrics

---

## feature_matching_overlay.ipynb

**Purpose:** Visualize feature matching between specific tile pairs with overlay

**Best for:** Debugging specific tile pairs and understanding spatial relationships

**Key Features:**

1. **Custom Tile Selection** - Load specific tiles you want to analyze
2. **Transform Integration** - Load pre-computed transforms from JSON
3. **Overlay Visualization** - Visual comparison of tiles with estimated alignment

**Main Operations:**

```python
# Specify exact tiles to compare
input_path = ['../data/S000/S000_000_000.ome.zarr', 
              '../data/S001/S001_000_001.ome.zarr']
extra_metadata = '../aligned_stitched_mappings1.json'  # Pre-computed transforms

reg = MVSRegistration(input_path=input_path, 
                      extra_metadata=extra_metadata,
                      output_path='../../output/', 
                      ui='mpl', 
                      debug=True)
reg.file_labels = ['S000_000_000', 'S001_000_001']
sims = reg.init_sims()
norm_sims, _, _ = reg.preprocess(sims)

# Visualize overlap
overlap1, overlap2, sims_pixel_space = get_overlap_images(norm_sims[0], norm_sims[1], 
                                                           reg.source_transform_key)
```

**Use Case:** 
- Verify if pre-computed registrations are correct
- Inspect specific problematic tile pairs
- Validate transformation matrices from previous runs

---

## stitching2d_metrics.ipynb

**Purpose:** Complete 2D stitching with detailed metrics reporting

**Best for:** Quality assessment and registration verification

**Key Features:**

1. **Registration with Metrics** - Compute NCC, SSIM, ONMI during registration
2. **Quality Graph** - Visualize registration quality as a graph
3. **Detailed Reporting** - Various output formats and summaries

**Main Operations:**

```python
register_params = {
    'pairing': 'orthogonal',
    'transform_type': 'rigid',
    'method': 'sift',
    'gaussian_sigma': 2,
    'normalisation': True,
    'max_keypoints': 5000,
    'inlier_threshold_factor': 0.05,
    'max_trials': 1000,
    'ransac_iterations': 3,
    'metrics': ['ncc', 'ssim', 'onmi'],  # Request metric computation
    'n_parallel_pairwise_regs': 1,
}

results = reg.register(sims, reg_sims, reg_indices, register_params)

# Access metrics
pair_metrics = results['metrics']
# Analyze quality by examining the graph structure
```

**Output Analysis:**
- Edge weights represent registration quality
- Visualize the registration graph with matplotlib
- Export metrics to CSV for further analysis

---

## stitching2d_split_registration.ipynb

**Purpose:** Advanced registration with detailed control and multi-step workflow

**Best for:** Fine-tuning parameters and understanding registration steps

**Key Features:**

1. **Separate Pairwise/Global Steps** - Control each registration phase independently
2. **Intermediary Inspection** - Examine results between steps
3. **Advanced Filtering** - Apply quality thresholds and edge filtering
4. **Multi-method Comparison** - Easy comparison of different registration methods

**Main Operations:**

```python
# Step 1: Pairwise registration
register_params = { /* ... */ }
pairwise_results = reg.register_pairs(sims, register_sims=reg_sims, 
                                     params=register_params)

# Inspect pairwise results
pairs_graph = pairwise_results['pairs_graph']
msims_reg = pairwise_results['msims']
pair_metrics = pairwise_results['metrics']

# Step 2: Global registration (with optional quality filtering)
register_params['post_registration_quality_threshold'] = 0.7
global_results = reg.register_global(sims, msims_reg, 
                                    register_indices=reg_indices,
                                    params=register_params)

# Step 3: Inspect and save
transforms = global_results['transforms']
# ... further analysis or visualization
```

**Advanced Features:**
- Quality-based edge filtering: `post_registration_quality_threshold`
- Different groupwise resolution methods: `groupwise_resolution_method`
- Manual pair selection instead of automatic pairing
- Debug visualization of the registration graph

**Use Case:** Parameter optimization by isolating and testing individual registration components

---

## tile_staging_napari.ipynb

**Purpose:** Interactive visualization of tiles and fused images using napari viewer

**Best for:** Results inspection and interactive exploration

**Key Features:**

1. **napari Integration** - Open interactive viewer for spatial exploration
2. **Tile Staging** - Visualize individual tile positions
3. **Fused Image** - View the final stitched result
4. **Interactive Navigation** - Zoom, pan, and inspect details

**Main Operations:**

```python
# Initialize
reg = MVSRegistration(operation='register', 
                      input_path='../data/S000/*.zarr',
                      output_path='../../output/', 
                      ui='mpl', 
                      debug=True)
sims = reg.init_sims()

# Calculate tile shapes
shapes = [get_sim_shape_2d(sim, transform_key=reg.source_transform_key) 
          for sim in sims]

# Calculate fused image bounds
fused_shape, fused_offset, fused_scale = reg.fuse(sims)

# Open in napari viewer
viewer = napari.Viewer()
for label, sim in zip(reg.file_labels, sims):
    viewer.add_image(sim.data, name=label, scale=si_utils.get_spacing_from_sim(sim))
```

**napari Features:**
- Multiple image layers for side-by-side comparison
- Zoom and pan to inspect details
- Measure distances and angles
- Export visualizations as images

**Workflow:**
1. Inspect individual tiles for quality
2. View fused result to verify stitching
3. Identify problem areas for further debugging
4. Export screenshots for reports/documentation

---

## General Tips for All Notebooks

### Setting Up Your Data

1. **Modify input paths** to point to your data:
   ```python
   input_path = '/path/to/your/data/*.zarr'  # Supports wildcards
   ```

2. **Adjust output directory**:
   ```python
   output_path = '/path/to/output/'
   ```

### Common Parameter Adjustments

**For faster processing (lower quality):**
```python
register_params = {
    'max_keypoints': 1000,      # Reduce from 5000
    'gaussian_sigma': 2,        # Increase blur
    'scale': 8,                 # Downsample during processing
}
```

**For better accuracy (slower):**
```python
register_params = {
    'max_keypoints': 10000,       # Increase from 5000
    'ransac_iterations': 5,       # Increase from 3
    'inlier_threshold_factor': 0.01,  # Stricter threshold
}
```

**For different image types:**
```python
# For very noisy images
register_params = {
    'gaussian_sigma': 3,        # More blur
    'normalisation': 'global',  # Global normalization
}

# For high-contrast images
register_params = {
    'method': 'orb',           # Faster, good for high contrast
    'max_keypoints': 3000,     # Fewer keypoints needed
}
```

### Debugging Registration Issues

1. **Start with `feature_matching.ipynb`** to verify feature detection
2. **Check metrics with `feature_matching_metrics.ipynb`** to assess alignment quality
3. **Use `stitching2d_split_registration.ipynb`** to isolate problem steps
4. **Visualize with `tile_staging_napari.ipynb`** to inspect spatial results

### Performance Optimization

- Use `n_parallel_pairwise_regs` > 1 to parallelize pairwise registrations
- Reduce `scale` to process downsampled images (faster but less accurate)
- Limit `max_keypoints` for faster feature detection
- Use `orb` method instead of `sift` for speed

---

## Running the Notebooks

### Requirements
- Jupyter or JupyterLab
- muvis-align installed
- multiview-stitcher package
- napari (for tile_staging_napari.ipynb)

### Start Jupyter
```bash
jupyter notebook
```

### Navigate to notebooks folder and select a notebook

### Run cells sequentially (Shift+Enter) or all at once (Kernel > Run All)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No files matched" | Check input path pattern and file extensions |
| "ImportError" | Ensure muvis-align is installed and parent directory is in sys.path |
| "Out of memory" | Reduce `max_keypoints`, increase `gaussian_sigma`, or use `scale` parameter |
| "Poor registration" | Check `feature_matching.ipynb`, adjust parameters, verify data quality |
| "napari won't open" | Install napari: `pip install napari[all]` |


