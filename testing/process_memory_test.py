import logging
import yaml

from src.muvis_align.MVSRegistration import MVSRegistration


def test_process_memory(params, filenames):
    with open(params, 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    logging.basicConfig(level=logging.INFO, encoding='utf-8')
    logging.getLogger('ome_zarr').setLevel(logging.WARNING)

    reg = MVSRegistration(params['general'])
    reg.run('1', filenames, params['operations'][0])


if __name__ == '__main__':
    params = 'resources/params_test.yml'
    filenames = ['D:/slides/13457227.zarr'] * 2
    test_process_memory(params, filenames)
