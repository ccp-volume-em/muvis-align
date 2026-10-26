import os

import zarr

from muvis_align.zarr_compat import apply_windows_atomic_write_retry

zarr_extension = '.ome.zarr'
# every position and pixel size is converted to this on reading; napari's layers are given it to show
spatial_unit = 'micrometer'
tiff_extension = '.ome.tiff'

default_ome_zarr_version = '0.5'

default_chunk_size = 1024
try:
    # the process' cpuset reflects a SLURM job's allocation; os.cpu_count() reports the whole node
    _available_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    _available_cpus = os.cpu_count() or 8
def _cgroup_memory_limit(proc_cgroup='/proc/self/cgroup', cgroup_root='/sys/fs/cgroup'):
    """The tightest memory limit on this process's cgroup or any parent (v2, then v1), in bytes, or None."""
    paths = []
    try:
        with open(proc_cgroup) as file:
            lines = file.read().splitlines()
    except OSError:
        lines = []
    for line in lines:
        _, controllers, path = line.split(':', 2)
        # a job's limit sits on a parent of its task's cgroup (Slurm: job_N above step and task)
        parts = [part for part in path.split('/') if part]
        if controllers == '':
            paths += [os.path.join(cgroup_root, *parts[:depth], 'memory.max') for depth in range(len(parts) + 1)]
        elif 'memory' in controllers.split(','):
            paths += [os.path.join(cgroup_root, 'memory', *parts[:depth], 'memory.limit_in_bytes')
                      for depth in range(len(parts) + 1)]
    # the root's own files, where /proc/self/cgroup is missing or names a path not mounted here
    paths += [os.path.join(cgroup_root, 'memory.max'), os.path.join(cgroup_root, 'memory', 'memory.limit_in_bytes')]
    limits = []
    for path in paths:
        try:
            with open(path) as file:
                limits.append(int(file.read().strip()))
        except (OSError, ValueError):
            pass
    # 'max', or an implausibly huge sentinel, means unset
    limits = [limit for limit in limits if 0 < limit < 1 << 60]
    return min(limits) if limits else None


def _available_memory():
    """Memory allocated to this process in bytes (SLURM, then cgroup, then hardware), or None if unknown."""
    # SLURM values are in MB; SLURM_MEM_PER_CPU is per allocated core
    for variable, multiplier in (('SLURM_MEM_PER_NODE', 1), ('SLURM_MEM_PER_CPU', _available_cpus)):
        value = os.environ.get(variable)
        if value:
            try:
                return int(float(value)) * multiplier * 1024 ** 2
            except ValueError:
                pass
    limit = _cgroup_memory_limit()
    if limit is not None:
        return limit
    try:
        return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, ValueError, OSError):
        pass
    try:
        # Windows has no sysconf; psutil is not a declared dependency
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except Exception:
        pass
    return None


_available_memory_bytes = _available_memory()
# per-worker fusion chunk budget: a quarter of memory across the workers, capped so one chunk
# doesn't hold a whole view while the other cores idle
default_fusion_chunk_bytes = min(4 * 1024 ** 3, max(
    64 * 1024 ** 2,
    int((_available_memory_bytes or 16 * 1024 ** 3) * 0.25 / max(1, _available_cpus))))
# napari's own default share, applied to this job's memory rather than the machine's
napari_dask_cache_share = 0.25
# memory an export's blocks share, each reserving its own need before it runs: a busy block waits for room
# rather than every block being sized for the busiest one on every worker at once
default_fusion_memory_pool = max(64 * 1024 ** 2, int((_available_memory_bytes or 16 * 1024 ** 3) * 0.25))
# chunk-shaped float32 arrays multiview_stitcher holds per overlapping source: transformed sources,
# blending weights and their product (fusion._core's field_ims_t / field_ws_t)
fusion_stack_arrays = 3
# get_contrast_limits() computes the lazy coarsest level's whole fusion graph; above this many
# tasks a dtype-range guess is used instead
default_contrast_limits_max_tasks = 4096


def _source_init_worker_ceiling(default=64):
    value = os.environ.get('MUVIS_SOURCE_INIT_WORKERS')
    if value:
        try:
            return max(int(value), 1)
        except ValueError:
            pass
    return default


# init_sources() is I/O-bound, so threads exceed the core count; MUVIS_SOURCE_INIT_WORKERS raises
# the ceiling for slow shared filesystems
default_source_init_workers = min(_source_init_worker_ceiling(), _available_cpus * 8)
# msim building is GIL-bound xarray work plus a slow header read: more threads only add contention
default_msim_build_workers = min(default_source_init_workers, 32)
# zarr v3 routes all I/O through one process-wide pool, which would otherwise bottleneck our workers;
# set globally so it holds for the life of the process
zarr.config.set({'threading.max_workers': default_source_init_workers})
# Windows: zarr's rename of metadata into place fails if anything holds the destination open
apply_windows_atomic_write_retry()
# multiview_stitcher fuses an export's blocks sequentially unless given our own batch_func
default_fusion_workers = _available_cpus
# an export's blocks are sized so this many fit the pool together: fusion keeps only a handful of cores busy, so
# sizing for every worker would only shrink the blocks, and a coarse level must still split among the workers
fusion_blocks_at_once = 16
default_fusion_block_bytes = (default_fusion_memory_pool
                              // min(max(1, default_fusion_workers), fusion_blocks_at_once))
# pairs registered at once when n_parallel_pairwise_regs is blank, one per process: threads are GIL-bound
default_pair_workers = _available_cpus
# pairs a registration worker process handles before it is replaced, handing back what it kept
default_pair_worker_tasks = 1000
# smoothing of a fused z-plane or channel before it is registered, in tile sizes: a pattern every tile repeats
# (e.g. a grid) would otherwise match at a shift of that pattern
split_smoothing = 1 / 25
# binning of a fused z-plane or channel registered against the next, when the setting is blank;
# MVS's own binning would leave a 2D section at full size
default_split_binning = 8
# background margin around the fused z-planes' grid, a fraction of each side: FFT registration wraps
# around, and content touching the edges pulls the shift towards zero
split_grid_margin = 0.05
# an export block's maximum size (also a zarr export's on-disk chunk size): an export pays a fixed
# cost per block, while a preview gains nothing from chunks larger than the screen
default_export_chunk_size = 4096
# on-screen overview reduction per source, matching create_preview()'s 'preview_scale'
default_interactive_preview_scale = 16
# upper bound on the fused preview, only a few hundred pixels on screen; preview_scale is relative to
# each source, so the bound is on the result (see image.util.reduce_msims_to_fused_size)
default_preview_max_bytes = 4 * 1024 ** 3

prereg_mappings_name = 'prereg_mappings.csv'
default_pair_mappings_name = 'pair_mappings.json'
# the 'kind' of a split group pair (e.g. section to section) in the pair mappings file
split_group_pair_kind = 'split_group'
default_mappings_name = 'mappings.json'
default_mappings_tabular_name = 'mappings.csv'
original_positions_name = 'positions_original.pdf'
registered_positions_name = 'positions_registered.pdf'
metrics_name = 'metrics.json'

# file name labels of a section's number, most specific first ('S000', SBEMimage's 's00538')
section_number_labels = ('section', 'slice', 's', 'z')

default_transform_key = 'transform'
default_quality_key = 'quality'

NAPARI_PROJECT_TEMPLATE = 'ui/project_template.yaml'

# registration params read only by some methods (registration_method_kind()), greyed out in the plugin for the rest
registration_method_params = {
    'normalisation': ('feature', 'cpd'),
    'gaussian_sigma': ('feature', 'cpd'),
    'max_keypoints': ('feature', 'cpd'),
    'inlier_threshold_factor': ('feature',),
    'max_trials': ('feature',),
    'ransac_iterations': ('feature',),
}
