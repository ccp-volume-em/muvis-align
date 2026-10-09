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

## Using the plugin

[Step by step in napari](napari_guide.md) goes through the tabs in order, with every
parameter and what each button does.

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
