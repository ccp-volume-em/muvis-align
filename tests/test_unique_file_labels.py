from muvis_align.util import get_unique_file_labels, strip_common_path_prefix


def test_get_unique_file_labels_simple_same_dir():
    filenames = [
        '/data/proj/subset/sample_ov000_s00400.ome.tif',
        '/data/proj/subset/sample_r0005_t0002_s00400.ome.tif',
        '/data/proj/subset/sample_r0005_t0003_s00400.ome.tif',
    ]
    assert get_unique_file_labels(filenames) == ['ov000', 'r0005_t0002', 'r0005_t0003']


def test_get_unique_file_labels_trims_shared_prefix_before_falling_back():
    # same basenames repeated across several subdirectories with no digits of their own -
    # numeric-only differentiation can't tell them apart, so the fallback must use the path
    # relative to the shared root, not the raw absolute filename
    base = '/nemo/project/proj-mrc-mm/raw/em/EM04652/EM04652_02_slice017/EM04652-02_slice17_meatballs'
    subdirs = ['subset', 'tiles', 'stitched', 'stitched_hpc']
    name = 'EM04652-02_slice17_meatballs_ov000_s00400.ome.tif'
    filenames = [f'{base}/{sd}/{name}' for sd in subdirs]

    labels = get_unique_file_labels(filenames)

    assert len(set(labels)) == len(labels)
    assert all(not label.startswith(base) for label in labels)
    assert labels == [f'{sd}/{name}' for sd in subdirs]


def test_strip_common_path_prefix():
    filenames = ['/a/b/c/x.tif', '/a/b/c/y.tif', '/a/b/d/x.tif']
    assert strip_common_path_prefix(filenames) == ['c/x.tif', 'c/y.tif', 'd/x.tif']


def test_strip_common_path_prefix_no_shared_root():
    filenames = ['a/x.tif', 'b/x.tif']
    assert strip_common_path_prefix(filenames) == filenames
