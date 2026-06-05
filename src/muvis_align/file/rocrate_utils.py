# https://pypi.org/project/rocrate/
# https://github.com/ome/ome2024-ngff-challenge/tree/main/src/ome2024_ngff_challenge/zarr_crate
# https://github.com/clbarnes/rembi-mifa-py/blob/main/examples/rembi.py


import os.path
from rocrate.model import ContextEntity

from src.muvis_align.constants import NAPARI_PROJECT_TEMPLATE
from src.muvis_align.file.rembi_extension import ImageAcquistion
from src.muvis_align.file.zarr_extension import ZarrCrate
from src.muvis_align.util import get_filetitle


def create_ro_crate(source, dest_path, image_paths=[]):
    crate = ZarrCrate()

    for image_path in image_paths:
        crate.add_dataset(dest_path=image_path)

    properties = {"fbbi_id": {"@id": 'obo:FBbi_00000257'}}
    crate.add(ImageAcquistion(crate, properties=properties))

    workflow_schema_filename = os.path.join('src', 'muvis_align/', NAPARI_PROJECT_TEMPLATE)
    #crate.add(ComputationalWorkflow(crate, workflow_schema_filename))
    crate.add_workflow(workflow_schema_filename)
    #crate.add_formal_parameter('bla', 'PropertyValue', '#acq:001')

    crate.write(dest_path)
    return crate


def create_zarr_ro_crate(dest_path):
    crate = ZarrCrate()

    properties = {}
    properties['name'] = get_filetitle(dest_path)
    #properties["description"] = ...
    #properties["license"] = ...
    dataset_entity = crate.add_dataset(dest_path='.', properties=properties)

    acquisition_properties = {
        '@type': 'image_acquisition',
        'fbbi_id': {'@id': 'obo:FBbi_00000257'},
    }
    acquisition_entity = ContextEntity(crate, '#acquisition-001', acquisition_properties)
    crate.add(acquisition_entity)

    dataset_entity['resultOf'] = acquisition_entity

    crate.write(dest_path)
    return crate

