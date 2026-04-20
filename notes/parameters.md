# Parameters hierarchy

[general]

- break_on_error
- verbose
- debug
- logging
- chunk_size
- output_options
  - zarr_options
  - compression

[operations]
  - operation: register / match / stack

  - input
    - path
    - source_metadata
    - extra_metadata

  - output
    - path
    - format
    - overwrite

  - preprocessing
    - scale
    - flat_field
    - sigma
    - normalisation
    - foreground filtering

  - pair registration
    - pairing
    - overlap_threshold
    - sigma
    - normalisation
    - method
    - max_keypoints
    - inlier_threshold
    - max_trials
    - ransac_iterations 

  - global registration
    - method
    - transform_type
    - quality_filter

  - fusion preview
    - method

  - export fusion/save
    - output_spacing
    - path
    - thumbnail
