# Parameters hierarchy

A pipeline file is the run's `general` settings and a list of operations, each operation in the sections of the
project template (`src/muvis_align/ui/project_template.yaml`), as a plugin project file. Settings without a template
entry stay in the section they belong to (warned about in the RO-Crate, but kept). See `docs/pipeline.md`.

[general]
- break_on_error
- metadata_summary
- verbose, debug, ui
- logging: filename, format, verbose, debug, time, dask

[operations]
- input_output
  - input_path, output_path, overwrite, clear
  - source_position_x/y/z, source_scale_x/y/z, source_rotation, normalise_rotated_positions
  - source_metadata (flags, per-file values or a file), extra_metadata, channels_table, labels
  - preview_scale, n_set_workers, mappings, pair_mappings
- pre_processing
  - scale, flatfield_quantiles, normalisation, gaussian_sigma, filter_foreground
- registration
  - operation (register / register match LABEL / transition / fuse / merge / convert)
  - pairing, method, transform_type, normalisation, gaussian_sigma, max_keypoints, inlier_threshold_factor,
    max_trials, ransac_iterations, channel, split_binning, n_parallel_pairwise_regs, groupwise_resolution_method
  - metrics, overlap_threshold, post_registration_quality_threshold
- fusion
  - method, spacing, block_size, ome_version
  - format, compression, npyramid_add, pyramid_downsample, preview, save_images
  - transition: frames, fps, spacing, scale
