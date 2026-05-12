# https://pypi.org/project/rocrate/
# https://github.com/ome/ome2024-ngff-challenge/tree/main/src/ome2024_ngff_challenge/zarr_crate
# https://github.com/clbarnes/rembi-mifa-py/blob/main/examples/rembi.py


import os.path
from rocrate.model import ComputationalWorkflow

from src.muvis_align.constants import NAPARI_PROJECT_TEMPLATE
from src.muvis_align.file.rembi_extension import ImageAcquistion
from src.muvis_align.file.zarr_extension import ZarrCrate


def create_ro_crate(source, dest_path, image_paths=[]):
    crate = ZarrCrate()

    for image_path in image_paths:
        crate.add_dataset(dest_path=image_path)

    properties = {"fbbi_id": {"@id": 'obo:FBbi_00000257'}}
    crate.add(ImageAcquistion(crate, properties=properties))

    workflow_schema_filename = os.path.join('src', 'muvis_align/', NAPARI_PROJECT_TEMPLATE)
    crate.add(ComputationalWorkflow(crate, workflow_schema_filename))

    crate.write(dest_path)
    return crate


def create_zarr_ro_crate(source, dest_path):
    crate = ZarrCrate()

    crate.add_dataset(dest_path='.')

    properties = {"fbbi_id": {"@id": 'obo:FBbi_00000257'}}
    crate.add(ImageAcquistion(crate, properties=properties))

    crate.write(dest_path)
    return crate
