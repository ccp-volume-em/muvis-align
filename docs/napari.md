# The napari plugin

The napari plugin drives the same registration pipeline interactively: load a dataset,
check how the sources are laid out, preview a pair before committing to a full run, and
watch each step's progress in napari's own progress bar.

Start napari and open the plugin from **Plugins → muvis-align**.

## The project file

The plugin keeps its settings in a project file (`muvis_align_project.yml` by default),
chosen on the **project** tab. This is *not* the `general`/`operations` file the
[command-line pipeline](pipeline.md) takes - it is one section per tab:

```yaml
input_output:
  input_path: data/*/*.tiff
  output_path: output
  source_scale_x: '0.004'
  source_position_x: fn[-2]*24
  overwrite: true
pre_processing:
  scale: 8
  normalisation: none
registration:
  operation: register
  method: phase_correlation
  transform_type: rigid
fusion:
  method: average
  spacing: mean
  ome_version: '0.5'
```

Every widget writes straight back to this file as you change it, so a project can be
reopened exactly as it was left. Paths are stored relative to the directory the project
file sits in.

## Tabs

The tabs run left to right, and each is enabled once the previous one has been run -
the plugin will not let you fuse before the sources have been read. Each tab has a
**Process** button that performs that step.

### input output

Input path (a file, a folder or a wildcard pattern), output path, and how to read each
source's metadata. The `source_position_*` and `source_scale_*` fields accept either
`source` (take it from the file) or an expression over the filename, so
`fn[-2]*24` reads the second-to-last numeric field in the filename and multiplies it by
24. **Process** reads the sources and draws their layout.

The **metadata** table lists what was read from each source, ordered by `(z, y, x)`
position rather than file-loading order, with position and size columns printed in that
same order. The **channels** table sets each channel's label and colour; clicking a
channel's colour cell opens a colour picker instead of requiring a raw `r,g,b` value.

### pre processing

Applied to the images before registration only - it does not affect the exported result.

| Parameter | Description |
|-----------|-------------|
| Target downscale factor | Downsample sources before registering |
| Flat-field quantiles | One or two quantiles for flat-field correction |
| Normalisation | `single` (per tile), `global` or `none` |
| Gaussian sigma | Gaussian blur before registration; blank or 0 for none |
| Filter foreground | Drop tiles with little image signal, such as empty tiles |

An option that is not recognised is reported rather than silently ignored.

### registration

Pick the **Operation** first, since it decides what the rest of the run does:

- **Register** - register the sources, then fuse them on the fusion tab.
- **Merge** - intended to fuse at the source metadata positions without registering.
  Currently this only changes the output name to `merged`: **Process** still runs the
  registration, and the fusion tab is unlocked only once global registration has
  completed, so there is no way to skip it from the plugin.
- **Convert** - see [Convert](#convert) below.

The remaining parameters (method, transform type, pairing, feature and RANSAC settings)
match the [pipeline's registration options](pipeline.md#registration-configuration).

**Preview** registers a single pair and shows the result, which is much faster than a
full run when tuning parameters. Choose the pair from the two dropdowns, or click an
overlap region in the layout view to select that pair directly. **Pair registration**
registers every pair; **Process** runs the global optimisation across all of them
(offering to run pair registration first if it has not been done).

The **metrics** table below reports each pair's registration quality, ordered by
position in the same way as the metadata table.

### fusion

| Parameter | Description |
|-----------|-------------|
| Method | `average`, `exclusive`, `additive` or `compose` |
| Spacing | Output pixel size across sources: `mean`, `min` or `max` |
| Tile size(s) | Leave empty to size automatically - see [Tile size](pipeline.md#tile-size) |
| OME version | OME-Zarr version to write: `0.4` or `0.5` |

**Preview** fuses a reduced version for inspection; the preview is bounded by the size
of the result, so a large dataset still previews quickly. **Process** exports the full
fused result, fusing its blocks in parallel.

## Convert

**Convert** writes each source out individually as OME-Zarr, at its own metadata
position, with no registration and no fusion. Each source keeps its native pyramid
levels exactly - nothing is resampled - and the sources are never combined into a
shared output canvas or a multichannel image. It is the quickest way to get a set of
TIFFs into OME-Zarr before working with them.

Convert runs from the registration tab and produces one file per source, named after
that source, under a `converted/` output folder. It never reaches the fusion tab.

Giving OME-Zarr sources a real pyramid this way is also worth doing before a large
registration run: without one, every preview and every coarse-level read falls back to
re-reading full-resolution chunks.

## Progress

Each operation reports one progress bar, following the work itself rather than the
number of dask tasks, so the bar does not restart partway through. Where a step's
progress cannot be observed directly - the global optimisation in particular, which
previously sat at 0% for the whole run - it is followed through its own log output and
counted against the number of edges it can remove.

Exporting reports two bars in sequence: one for the export, then one for drawing the
result.

## Performance notes

- A project opens without fusing anything: the layout view is drawn from the source
  geometry, read from metadata alone.
- Leave **Tile size(s)** empty unless the on-disk layout matters. Blocks are then sized
  against the memory actually available to the process, which on a cluster means the
  job's allocation rather than the size of the node.
- Sources without a usable pyramid make every preview expensive. Convert them first.
- The **Target downscale factor** on the pre-processing tab is usually the largest
  single win when tuning registration: it applies to registration only, so the exported
  result stays at full resolution.
