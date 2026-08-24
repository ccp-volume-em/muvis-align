import glob
import numpy as np

from muvis_align.image.ome_tiff_helper import load_tiff
from muvis_align.image.util import *
from muvis_align.metrics import *


def calc_metrics(images, offsets):
    images2 = []
    for image, offset in zip(images, reversed(offsets)):
        #image = np.pad(image, offset)
        image = image[offset[0][0]:image.shape[0] - offset[0][1],
                      offset[1][0]:image.shape[1] - offset[1][1]]
        #show_image(image)
        images2.append(image)
    metrics = {
        'ncc': calc_ncc(*images2),
        'ssim': calc_ssim(*images2),
    }
    return metrics


def calc_metrics_range(images, offsets, offset_range=5, nsamples=3):
    results = []
    for mag in range(1, offset_range + 1):
        for n in range(nsamples):
            var0 = np.array([np.random.randint(-mag, mag + 1), np.random.randint(-mag, mag + 1)])
            var1 = np.array([np.random.randint(-mag, mag + 1), np.random.randint(-mag, mag + 1)])
            var_offsets = np.abs(var0 + offsets[0]), np.abs(var1 + offsets[1])
            results.append(calc_metrics(images, var_offsets))
    return results


if __name__ == "__main__":
    path = 'D:/slides/EM04654_slice011/overlaps/*.ome.tiff'
    images = [load_tiff(filename) for filename in glob.glob(path)]

    offsets0 = [
        ((0, 0), (0, 0)),
        ((0, 0), (0, 0)),
    ]

    offsets = [
        ((38, 0), (0, 47)),
        ((0, 38), (47, 0)),
    ]
    #print(f'control {calc_metrics([images[0], images[0]], offsets0)}')
    print(f'optimal {calc_metrics(images, offsets)}')
    print(f'suboptimal {calc_metrics(images, offsets0)}')
    #metric_range = ' '.join([f'{x:.3f}' for x in calc_metrics_range(images, offsets)])
    print(f'range {calc_metrics_range(images, offsets)}')
