cwlVersion: v1.2
class: CommandLineTool

doc: |
  Deep Learning algorithm for cell segmentation in microscopy images (Cellpose)
  Based on https://github.com/bilayer-containers/bilayers/blob/main/algorithms/cellpose_inference/config.yaml

requirements:
  DockerRequirement:
    dockerPull: cellprofiler/runcellpose_no_pretrained:2.3.2

baseCommand:
  - python
  - -m
  - cellpose
  - --verbose

inputs:
  dir:
    type: Directory
    inputBinding:
      position: 1
      prefix: --dir
    doc: Path to the directory of input images
  custom_model:
    type: File?
    inputBinding:
      position: 2
      prefix: --add_model
    doc: Custom model to be used for segmentation, if not using pretrained model
  channel_axis:
    type: int?
    inputBinding:
      prefix: --channel_axis
    doc: axis of image which corresponds to image channels
    default: 0
  pretrained_model:
    type: string?
    inputBinding:
      prefix: --pretrained_model
    doc: type of model to use
    default: cyto
  diameter:
    type: float?
    inputBinding:
      prefix: --diameter
    doc: estimated diameter of cells in pixels
    default: 30
  stitch_threshold:
    type: float?
    inputBinding:
      prefix: --stitch_threshold
    doc: stitching threshold
    default: 0.0
  min_size:
    type: int?
    inputBinding:
      prefix: --min_size
    doc: minimum size of objects in pixels
    default: 15
  save_omezarr:
    type: boolean?
    inputBinding:
      prefix: --save_omezarr
    doc: save segmentation as Ome-zarr
    default: false
  save_dir:
    type: string?
    inputBinding:
      prefix: --savedir
    doc: directory to save output files
    default: /output_images

outputs:
  omezarr_images:
    type: Directory?
    outputBinding:
      glob: /output_images/*_cp_masks
    doc: Segmented image if --save_omezarr flag is true.

