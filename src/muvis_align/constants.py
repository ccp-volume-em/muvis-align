import os

import zarr

from muvis_align.zarr_compat import apply_windows_atomic_write_retry

zarr_extension = '.ome.zarr'
tiff_extension = '.ome.tiff'

default_ome_zarr_version = '0.5'

default_chunk_size = 1024
try:
    # sched_getaffinity (Linux-only) reads the process' real cpuset, which on a SLURM node
    # reflects the job's actual allocation - unlike os.cpu_count(), which reports the whole node
    # regardless of what was allocated to this job.
    _available_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    _available_cpus = os.cpu_count() or 8
def _available_memory():
    """Total memory this process may actually use, in bytes - the counterpart to
    _available_cpus above, and read in the same spirit: what was *allocated* to this job, not
    what the machine happens to have. A 2TB HPC node handed a 64GB job allocation must budget
    against the 64GB, so the batch-system and cgroup limits are checked before the hardware.
    Returns None if nothing here can tell, leaving callers to fall back to a fixed default.
    """
    # SLURM's own allocation, in MB (SLURM_MEM_PER_NODE wins; SLURM_MEM_PER_CPU is per
    # allocated core, so scale it by the cpuset _available_cpus already reads)
    for variable, multiplier in (('SLURM_MEM_PER_NODE', 1), ('SLURM_MEM_PER_CPU', _available_cpus)):
        value = os.environ.get(variable)
        if value:
            try:
                return int(float(value)) * multiplier * 1024 ** 2
            except ValueError:
                pass
    # container/cgroup limit (v2 then v1) - 'max', or an implausibly huge sentinel, means unset
    for path in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            with open(path) as file:
                limit = int(file.read().strip())
            if 0 < limit < 1 << 60:
                return limit
        except (OSError, ValueError):
            pass
    try:
        return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, ValueError, OSError):
        pass
    try:
        # Windows has no sysconf - ask the kernel directly rather than depend on psutil, which
        # is not one of this package's declared dependencies
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
# Per-output-chunk memory budget for fusion (see image.util.get_chunk_sizes), per worker: one
# chunk is held per worker at once, so a quarter of the allocation spread across them leaves the
# rest for source data, napari and the fused result. The ceiling matters as much as the budget -
# past a few GB a chunk stops being a useful unit of parallel work, one task holding a whole view
# while the other cores idle - so extra headroom buys more chunks, not bigger ones.
default_fusion_chunk_bytes = min(4 * 1024 ** 3, max(
    64 * 1024 ** 2,
    int((_available_memory_bytes or 16 * 1024 ** 3) * 0.25 / max(1, _available_cpus))))
# ...except for an export, which gets the same per-worker share without that ceiling. The ceiling
# is a parallelism argument (one task holding a whole view leaves the other cores idle) and an
# export has no such parallelism to protect: threads over its blocks were measured not to scale,
# the per-block work being Python holding the GIL. It rarely raises a block above
# default_export_chunk_size; what it stops is a source-dense one having to shrink below it.
default_export_fusion_chunk_bytes = max(default_fusion_chunk_bytes, min(16 * 1024 ** 3, max(
    64 * 1024 ** 2,
    int((_available_memory_bytes or 16 * 1024 ** 3) * 0.25 / max(1, _available_cpus)))))
# same-shaped float32 arrays multiview_stitcher holds per output chunk: every overlapping source
# transformed into the chunk's grid, the blending-weight stack, and their product (fusion._core's
# field_ims_t / field_ws_t). What makes a chunk's real cost differ from its own byte size by
# orders of magnitude, for thousands of sources.
fusion_stack_arrays = 3
# get_contrast_limits() reads the coarsest pyramid level, which is lazy - so the compute runs its
# whole fusion graph. Above this many tasks it is no longer the cheap up-front step it is meant
# to be, and a naive dtype-range guess is used instead.
default_contrast_limits_max_tasks = 4096


def _source_init_worker_ceiling(default=64):
    value = os.environ.get('MUVIS_SOURCE_INIT_WORKERS')
    if value:
        try:
            return max(int(value), 1)
        except ValueError:
            pass
    return default


# init_sources() mostly waits on a file open/header read per source, so this is deliberately not
# capped at core count: threads blocked on I/O consume no CPU, and a 4733-source run was
# near-perfectly I/O-bound. It still scales with cores up to a ceiling, so a small machine (and
# likely a modest network link) doesn't take the same 64 threads a big one would.
# MUVIS_SOURCE_INIT_WORKERS raises that ceiling, which a shared HPC filesystem needs - the
# per-source open there is slower still, and more threads is the only way to overlap the wait.
default_source_init_workers = min(_source_init_worker_ceiling(), _available_cpus * 8)
# zarr v3 routes its I/O through one process-wide thread pool (zarr.core.sync._get_executor()),
# independent of the workers above. Unraised, reads stay bottlenecked on it however many of our
# own threads are waiting to submit one - which is why OME-Zarr sources parallelized worse than
# OME-TIFF. Set globally, not inside a with block, so it holds for the life of the process.
zarr.config.set({'threading.max_workers': default_source_init_workers})
# Windows only: zarr renames each metadata document into place, which fails if anything holds the
# destination open for that instant. Applied globally, before any store is written.
apply_windows_atomic_write_retry()
# per-source preview/fusion prep is CPU-bound, not I/O wait, so unlike the workers above it has
# no file-handle concern capping it
default_preview_workers = _available_cpus
# multiview_stitcher's loop over an export's blocks is sequential (fusion._core: batch_func None,
# n_batch 1), so without a batch_func of our own an export runs on one core. The per-chunk budget
# above is already per-worker, sized for this many holding a chunk at once.
default_fusion_workers = _available_cpus
# pairs per dask compute when n_parallel_pairwise_regs is blank - multiview_stitcher would plan
# them all in one graph, which for 115549 pairs took 110GB and reported nothing in 9 hours
default_pair_batch_size = max(_available_cpus * 4, 64)
# What one output block of an *export* may span, where default_chunk_size (1024) is what a preview
# wants: x/y chunks larger than the screen buy a preview nothing, while an export pays a fixed
# cost (~0.5s measured) per block however small. Caps the block, does not overrule the budget -
# and for a zarr export the block size is also the on-disk chunk size, so this bounds that too.
default_export_chunk_size = 4096
# what the on-screen overview reduces each source by, matching create_preview()'s own
# 'preview_scale' so a preview stays proportionate to an exported one
default_interactive_preview_scale = 16
# ...and how large that overview may be. Unlike a fused preview (a lazy graph napari computes
# only the coarsest level of), it is pasted eagerly into one array, so this is real memory.
default_overview_max_bytes = 512 * 1024 ** 2
# ...and an upper bound on the fused preview however it was reached. It is a few hundred pixels
# on screen whatever is behind it, so fusing more is wasted: one run fused 396.9GB over 55
# minutes to show what an 8x-reduced one showed in 9. preview_scale cannot prevent that (it is
# relative to each source's own pyramid, and the post-pre-processing preview skips it), so the
# guard is on the resulting size - see image.util.reduce_msims_to_fused_size.
default_preview_max_bytes = 4 * 1024 ** 3

prereg_mappings_name = 'prereg_mappings.csv'
default_pair_mappings_name = 'pair_mappings.json'
default_mappings_name = 'mappings.json'
default_mappings_tabular_name = 'mappings.csv'
original_positions_name = 'positions_original.pdf'
registered_positions_name = 'positions_registered.pdf'
metrics_name = 'metrics.json'

default_transform_key = 'transform'
default_quality_key = 'quality'

NAPARI_PROJECT_TEMPLATE = 'ui/project_template.yaml'
