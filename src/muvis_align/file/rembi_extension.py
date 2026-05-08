# Based on: https://github.com/ome/ome2024-ngff-challenge/tree/main/src/ome2024_ngff_challenge/zarr_crate

from __future__ import annotations

from rocrate.model.contextentity import ContextEntity


class ImageAcquistion(ContextEntity):
    def __init__(self, crate, identifier=None, properties=None):
        image_acquisition_type_path = "image_acquisition"
        if properties:
            image_acquisition_properties = {}
            image_acquisition_properties.update(properties)
            if "@type" in properties:
                image_acquisition_type = image_acquisition_properties["@type"]
                if image_acquisition_type_path not in image_acquisition_type:
                    try:
                        image_acquisition_type.append(image_acquisition_type_path)
                    except Exception:
                        image_acquisition_type = [image_acquisition_type]
                        image_acquisition_type.append(image_acquisition_type_path)
                    image_acquisition_properties["@type"] = image_acquisition_type
            else:
                image_acquisition_properties.update(
                    {"@type": image_acquisition_type_path}
                )
        else:
            image_acquisition_properties = {"@type": image_acquisition_type_path}

        super().__init__(crate, identifier, image_acquisition_properties)

    def popitem(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
