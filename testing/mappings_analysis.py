import numpy as np

from muvis_align.util import import_json


def calc_offsets(mappings):
    translations = [np.array(mapping)[:-1, -1] for key, mapping in mappings.items()]
    x_translations = np.array([translation[1] for translation in translations]).reshape(9, -1)
    y_translations = np.array([translation[0] for translation in translations]).reshape(9, -1)
    delta_x_in_x_direction = np.mean(np.mean(np.diff(x_translations, axis=0), axis=0))
    delta_x_in_y_direction = np.mean(np.mean(np.diff(x_translations, axis=1), axis=1))
    delta_y_in_x_direction = np.mean(np.mean(np.diff(y_translations, axis=0), axis=0))
    delta_y_in_y_direction = np.mean(np.mean(np.diff(y_translations, axis=1), axis=1))
    print('tile delta x in x direction', delta_x_in_x_direction)
    print('tile delta x in y direction', delta_x_in_y_direction)
    print('tile delta y in x direction', delta_y_in_x_direction)
    print('tile delta y in y direction', delta_y_in_y_direction)

if __name__ == "__main__":
    filename = 'D:/slides/12193/stitched_hpc/test2/mappings.json'
    mappings = import_json(filename)

    calc_offsets(mappings)
