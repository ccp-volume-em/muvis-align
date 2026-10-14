from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from configparser import ConfigParser
from contextlib import contextmanager, nullcontext
import csv
import cv2 as cv
import functools
from datetime import datetime
import gc
import glob
import itertools
import json
import logging
import math
import multiprocessing
import numpy as np
import pickle
import os.path
import re
import sys
import threading
import time
import warnings
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree
from xarray import DataArray

from muvis_align.image.tiff_metadata import UM_CONVERSIONS


def single_threaded_highs():
    """Run scipy's HiGHS (linprog, as multiview-stitcher's overlap tests use) on the calling thread alone."""
    # HiGHS starts and tears down workers per calling thread, which can deadlock on the Windows loader lock;
    # these problems are tiny
    try:
        from scipy.optimize import _linprog_highs
    except ImportError:
        return
    highs_wrapper = getattr(_linprog_highs, '_highs_wrapper', None)
    if highs_wrapper is None or getattr(highs_wrapper, '_muvis_single_threaded', False):
        return

    def single_threaded(*args):
        *arrays, options = args
        return highs_wrapper(*arrays, dict(options, threads=1))

    single_threaded._muvis_single_threaded = True
    _linprog_highs._highs_wrapper = single_threaded


single_threaded_highs()



def ensure_list(x) -> list:
    if x is None:
        return []
    elif isinstance(x, list):
        return x
    else:
        return [x]



def _paths_match_by_suffix(path1, path2, min_parts=2):
    # a project reopened on another mount sees a file under an unrelated absolute path: parent folder
    # and filename are unique in a dataset, so compare those
    parts1 = path1.replace('\\', '/').rstrip('/').split('/')
    parts2 = path2.replace('\\', '/').rstrip('/').split('/')
    if min(len(parts1), len(parts2)) < min_parts:
        return False
    return parts1[-min_parts:] == parts2[-min_parts:]


def find_file_list_index(filenames, match):
    if match in filenames:
        return filenames.index(match)
    for index, filename in enumerate(filenames):
        if match.lstrip('.') in filename:
            return index
    for index, filename in enumerate(filenames):
        if _paths_match_by_suffix(filename, match):
            return index
    return None


def contrasting_color(background):
    """A saturated colour, apart from greyscale by its hue, that stands out on a background of `background` luminance
    (0 black to 1 white): dark on a bright one, bright on a dark one, magenta between or when unknown (None).
    Hex: napari reads an RGB tuple as three colours."""
    if background is not None and background > 0.6:
        return '#990066'
    if background is not None and background < 0.4:
        return '#ffd700'
    return '#ff00ff'


def background_luminance(levels, colors):
    """The luminance channels shown additively at their display `levels` (get_display_level) in RGB `colors` give
    their background, clipped to 1; None if any level is unknown."""
    if not levels or any(level is None for level in levels):
        return None
    return float(min(1.0, sum(level * (0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2])
                              for level, color in zip(levels, colors))))


def file_list_index_finder(filenames):
    """find_file_list_index(filenames, match) for many matches: an exact name by dict, each other found once."""
    found = {filename: index for index, filename in reversed(list(enumerate(filenames)))}

    def find(match):
        if match not in found:
            found[match] = find_file_list_index(filenames, match)
        return found[match]
    return find


def find_file_dict_item(dct, match):
    if match in dct:
        return dct[match]
    for key, value in dct.items():
        if key.lstrip('.') in match:
            return value
    for key, value in dct.items():
        if _paths_match_by_suffix(key, match):
            return value
    return None


def numpy_to_native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, list):
        return [numpy_to_native(v) for v in value]
    elif isinstance(value, tuple):
        return (numpy_to_native(v) for v in value)
    elif isinstance(value, dict):
        return {k: numpy_to_native(v) for k, v in value.items()}
    elif hasattr(value, 'dtype'):
        return value.item()
    else:
        return value


def is_valid_value(value):
    return value is not None and value != ''


def set_dict_value(dct, keys, value):
    try:
        value = float(value)
    except:
        pass
    for index, key in enumerate(keys):
        if index == len(keys) - 1:
            dct[key] = value
        else:
            if key not in dct:
                dct[key] = {}
            dct = dct[key]


def filter_dict(dict0: dict) -> dict:
    new_dict = {}
    for key, value0 in dict0.items():
        if value0 is not None:
            values = []
            for value in ensure_list(value0):
                if isinstance(value, dict):
                    value = filter_dict(value)
                values.append(value)
            if len(values) == 1:
                values = values[0]
            new_dict[key] = values
    return new_dict



def print_dict(dct: dict, indent: int = 0) -> str:
    s = ''
    if isinstance(dct, dict):
        for key, value in dct.items():
            s += '\n'
            if not isinstance(value, list):
                s += '\t' * indent + str(key) + ': '
            if isinstance(value, dict):
                s += print_dict(value, indent=indent + 1)
            elif isinstance(value, list):
                for v in value:
                    s += print_dict(v)
            else:
                s += str(value)
    else:
        s += str(dct)
    return s


def print_dict_simple(dct: dict, dims: str='zyx', significant_digits: int=3) -> str:
    items = []
    # a list, not a string: keys need not be single letters (e.g. a pair's (index, index))
    dims = list(dims)
    dims += [key for key in dct if key not in dims]
    for dim in dims:
        if dim in dct:
            value = dct[dim]
            if isinstance(value, float):
                value = print_significants(value, significant_digits)
            items.append(f'{dim}: {value}')
    return ' '.join(items)


def print_significants(value: float, significant_digits: int) -> str:
    # fixed-point rather than 'g', which turns 64863 into 6.49e+04: every integer digit is kept
    if value == 0 or not np.isfinite(value):
        return f'{value:g}'
    decimals = max(significant_digits - int(np.floor(np.log10(abs(value)))) - 1, 0)
    text = f'{round(value, decimals):.{decimals}f}'
    return text.rstrip('0').rstrip('.') if '.' in text else text


def print_dict_xyz(dct: dict, dims='xyz', decimals=3, as_tuple=False) -> str:
    s = ''
    for dim in dims:
        if dim in dct:
            if s:
                s += ' '
            if as_tuple:
                s += f'{dct[dim]:.{decimals}f}'
            else:
                s += f'{dim}:{dct[dim]:.{decimals}f}'
    return s


def print_hbytes(nbytes: int) -> str:
    exps = ['', 'K', 'M', 'G', 'T', 'P', 'E']
    div = 1024
    exp = 0

    while nbytes > div:
        nbytes /= div
        exp += 1
    if exp < len(exps):
        e = exps[exp]
    else:
        e = f'e{exp * 3}'
    return f'{nbytes:.1f} {e}B'



def round_significants(a: float, significant_digits: int) -> float:
    if a != 0:
        round_decimals = significant_digits - int(np.floor(np.log10(abs(a)))) - 1
        return round(a, round_decimals)
    return a



def get_filetitle(filename: str) -> str:
    filebase = os.path.basename(filename)
    # removesuffix, not rstrip: rstrip('.ome') strips any trailing '.', 'o', 'm' or 'e'
    title = os.path.splitext(filebase)[0].removesuffix('.ome')
    return title


def dir_regex(pattern):
    files = []
    for pattern_item in ensure_list(pattern):
        files.extend(glob.glob(pattern_item, recursive=True))
    files_sorted = sorted(files, key=lambda file: find_all_numbers(get_filetitle(file)))
    return files_sorted


def find_sbemimage_meta_dir(filename, max_hops=5):
    path = os.path.dirname(filename)
    for _ in range(max_hops):
        metapath = os.path.join(path, 'meta')
        if os.path.exists(metapath):
            return metapath
        path = os.path.join(path, '..')
    return None


def pattern_base_dir(pattern):
    """The real directory an input pattern sits in, without wildcard components (os.path.dirname of
    'data/*/*.tiff' is 'data/*'): what a relative output path is relative to."""
    directory = os.path.dirname(pattern)
    while directory and any(char in os.path.basename(directory) for char in '*?['):
        parent = os.path.dirname(directory)
        if parent == directory:
            return ''
        directory = parent
    return directory


def find_all_numbers(text: str) -> list:
    return list(map(int, re.findall(r'\d+', text)))


def find_labelled_numbers(text: str) -> dict:
    """Each number led by letters, by its lower-case label: 'x_s00399_ov0' -> {'s': 399, 'ov': 0}."""
    return {label.lower(): int(number) for label, number in re.findall(r'([A-Za-z]+)(\d+)', text)}


def split_path_parts(text: str) -> list:
    return [part for part in
            text.replace('/', '_').replace('\\', '_').replace('.', '_').replace('-', '_').split('_')
            if part]


def split_numeric(text: str) -> list:
    num_parts = []
    parts = split_path_parts(text)
    for part in parts:
        num_span = re.search(r'\d+', part)
        if num_span:
            num_parts.append(part)
    return num_parts


def find_target_numeric(text: str, target: str) -> int|None:
    parts = split_path_parts(text)
    for part in parts:
        if part.startswith(target):
            part = part.lstrip(target)
            if part.isdecimal():
                return int(part)
    return None


def split_numeric_dict(text: str) -> dict:
    num_parts = {}
    parts = split_path_parts(text)
    parti = 0
    for part in parts:
        num_span = re.search(r'\d+', part)
        if num_span:
            index = num_span.start()
            label = part[:index]
            if label == '':
                label = parti
            num_parts[label] = num_span.group()
            parti += 1
    return num_parts


def get_unique_nums(all_parts: list) -> list:
    keys = []
    for parts in all_parts:
        for key in parts:
            if key not in keys:
                keys.append(key)

    changing_keys = []
    for key in keys:
        values = [parts.get(key) for parts in all_parts]
        if len(set(values)) > 1:
            changing_keys.append(key)

    # each file's own key order: a global one would let one naming scheme dictate another's
    changing = set(changing_keys)
    final_parts = [{key: value for key, value in parts.items() if key in changing} for parts in all_parts]
    return final_parts


def strip_common_path_prefix(filenames: list) -> list:
    # leading directories shared by every file are just noise for labelling
    split = [filename.replace('\\', '/').split('/') for filename in filenames]
    common_len = 0
    for parts in zip(*split):
        if len(set(parts)) > 1:
            break
        common_len += 1
    if common_len == 0:
        return filenames
    return ['/'.join(parts[common_len:]) for parts in split]


def get_unique_file_labels(filenames: list) -> list:
    file_labels = []

    ntot = len(filenames)
    common_paths = strip_common_path_prefix(filenames)
    parts_dic = get_unique_nums([split_numeric_dict(get_filetitle(filename)) for filename in filenames])
    parts_dic_full = get_unique_nums([split_numeric_dict(filename) for filename in common_paths])
    parts_num_full = get_unique_nums([{index: value for index, value in enumerate(split_numeric(filename))} for filename in common_paths])

    dic_ok = (len(set(['_'.join(parts.values()) for parts in parts_dic])) == ntot)
    full_dic_ok = (len(set(['_'.join(parts.values()) for parts in parts_dic_full])) == ntot)
    full_num_ok = (len(set(['_'.join(parts.values()) for parts in parts_num_full])) == ntot)

    if dic_ok:
        all_parts = parts_dic
    elif full_dic_ok:
        all_parts = parts_dic_full
    elif full_num_ok:
        all_parts = parts_num_full
    else:
        # no digit-based split told them apart - use the path relative to the shared root
        all_parts = [{0: common_path} for common_path in common_paths]

    for parts in all_parts:
        file_label = '_'.join([key + part if isinstance(key, str) else part for key, part in parts.items()])
        file_labels.append(file_label)

    if len(set(file_labels)) < len(file_labels):
        # fallback for duplicate labels
        file_labels = [get_filetitle(filename) for filename in filenames]

    return file_labels


def split_num_text(text: str) -> list:
    num_texts = []
    block = ''
    is_num0 = None
    if text is None:
        return []

    for c in text:
        is_num = (c.isnumeric() or c == '.')
        if is_num0 is not None and is_num != is_num0:
            num_texts.append(block)
            block = ''
        block += c
        is_num0 = is_num
    if block != '':
        num_texts.append(block)

    num_texts2 = []
    for block in num_texts:
        block = block.strip()
        try:
            block = float(block)
        except:
            pass
        if block not in [' ', ',', '|']:
            num_texts2.append(block)
    return num_texts2



def check_contains_value(value, contains_value):
    return isinstance(value, (dict, str)) and contains_value in value


def is_source_value(value):
    # 'invert' alone inverts the source's own value, as 'source invert' does
    return check_contains_value(value, 'source') or check_contains_value(value, 'invert')


def get_metadata_z_scale(metadata):
    """The configured z spacing from a source/extra metadata dict, as a number - or None when it
    is unset, blank, or 'source' (the file's own, already applied by ImageSource.fix_metadata).
    """
    if not isinstance(metadata, dict):
        return None
    scales = metadata.get('scale', {})
    # a list holds one scale per file: the run steps by the finest z, as init_data does by z positions
    values = []
    for scale in (scales if isinstance(scales, list) else [scales]):
        value = scale.get('z') if isinstance(scale, dict) else None
        if value is not None and not check_contains_value(value, 'source'):
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
    return min(values, key=abs) if values else None


def eval_context(data, key, default_value, context):
    value = data.get(key, default_value)
    if isinstance(value, str):
        try:
            value = value.format_map(context)
        except:
            pass
        try:
            value = eval(value, context)
        except Exception as error:
            # napari shows a warning once, as a notification, however many sources repeat it
            warnings.warn(f'Invalid source metadata {key}: {value!r} ({type(error).__name__}: {error}),'
                          f' using {default_value}')
    if not isinstance(value, (float, int)):
        value = default_value
    return value



# tiff_metadata's table, plus the cased spellings OME writes
um_conversions = UM_CONVERSIONS | {'Å': 1e-4, 'A': 1e-4}


def format_phase_timing(wall_time, item_times, item_cpu_times, max_workers, process_cpu_time=None,
                        processes=False):
    """One timing line for a threaded per-item phase: wall time against summed per-item CPU time.

    Summed per-item wall time inflates with GIL waits, but per-item CPU (time.thread_time) is a floor:
    wall ~= cpu means CPU/GIL-bound, more workers cannot help; wall >> cpu means waiting (I/O) that more
    workers can overlap. `process_cpu_time` (time.process_time over the phase) also shows CPU spent outside
    the timed items, such as nested pools or other threads.
    """
    total_time = sum(item_times)
    total_cpu = sum(item_cpu_times) if item_cpu_times else 0.0
    summary = (f'with {max_workers} workers, wall {wall_time:.1f}s,'
               f' per-item total {total_time:.1f}s, cpu {total_cpu:.1f}s'
               f' (mean {1000 * total_time / len(item_times):.0f}ms,'
               f' max {1000 * max(item_times):.0f}ms)')
    if process_cpu_time is not None and wall_time > 0:
        summary += (f', process cpu {process_cpu_time:.1f}s'
                    f' ({process_cpu_time / wall_time:.1f} cores)')
        if total_cpu > 0 and process_cpu_time > 2 * total_cpu:
            summary += (f' - only {total_cpu / process_cpu_time:.0%} of the process CPU is this'
                        f' phase: the rest is elsewhere in the process')
    if processes and total_cpu > 0 and wall_time > 0:
        # no shared GIL: the items' cpu against the wall is how many of the workers were busy
        summary += f' - items in worker processes, {total_cpu / wall_time:.1f} of {max_workers} busy'
    elif total_cpu > 0:
        ratio = wall_time / total_cpu
        regime = ('CPU-bound: at the single-thread floor, more workers will not help'
                  if ratio < 1.3 else
                  f'{1 - 1 / ratio:.0%} of wall time is not CPU: more workers can overlap it')
        summary += f' - {regime}'
    return summary



def timed_calls(func, times, cpu_times):
    """`func`, appending each call's wall and thread CPU time to `times` and `cpu_times` (for
    format_phase_timing()). Keeps func's signature for callers that inspect it.
    """
    @functools.wraps(func)
    def timed(*args, **kwargs):
        start, cpu_start = time.perf_counter(), time.thread_time()
        try:
            return func(*args, **kwargs)
        finally:
            times.append(time.perf_counter() - start)
            cpu_times.append(time.thread_time() - cpu_start)
    return timed


def progress_phase(progress_factory, total=None, desc=None, weight=1):
    """One reporting phase of the caller's operation, or nothing to report into."""
    return progress_factory(total=total, desc=desc, weight=weight) if progress_factory is not None \
        else nullcontext(None)


def pooled_map(func, count, workers, progress_factory=None, desc=None, weight=1, timing_label=None):
    """[func(index) for index in range(count)], on `workers` threads (one: none), each on the bar as it finishes; with
    timing_label, the phase's timing logged (format_phase_timing). Rolling: a cancel need not wait for every item."""
    results = [None] * count
    times, cpu_times = [], []
    timed = timed_calls(func, times, cpu_times)
    workers = max(1, min(workers, count))
    phase_start, phase_cpu_start = time.time(), time.process_time()
    with progress_phase(progress_factory, total=count, desc=desc, weight=weight) as pbar:
        items = ((index, timed(index)) for index in range(count)) if workers == 1 \
            else rolling_map(timed, range(count), workers)
        for index, result in items:
            results[index] = result
            if pbar is not None:
                pbar.update(1)
    if timing_label and times:
        logging.info(f'{timing_label}: {len(times)} items {format_phase_timing(
            time.time() - phase_start, times, cpu_times, workers, time.process_time() - phase_cpu_start)}')
    return results


class OperationCancelled(Exception):
    """The user cancelled the running operation."""


_cancel_requested = threading.Event()


def request_cancel():
    _cancel_requested.set()


def raise_if_cancelled():
    if _cancel_requested.is_set():
        raise OperationCancelled('Cancelled')


@contextmanager
def cancellable():
    """One operation's span: a cancel requested within it stops it; one left from before does not."""
    _cancel_requested.clear()
    try:
        yield
    finally:
        _cancel_requested.clear()


def single_threaded_worker():
    """Worker process initializer: one thread per native pool, as the processes already use every core
    between them - left at their defaults, each worker's pools spin on cores the others need."""
    from threadpoolctl import threadpool_limits
    cv.setNumThreads(1)
    threadpool_limits(1)


def picklable(value):
    try:
        pickle.dumps(value)
        return True
    except Exception as error:
        logging.info(f'Not picklable ({type(error).__name__}: {error})')
        return False


_worker_environment = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
_spawn_lock = threading.Lock()


class _WorkerProcessPool(ProcessPoolExecutor):
    def _spawn_process(self):
        # set before numpy/scipy load: OpenBLAS commits a buffer per thread at import, exhausting Windows commit
        # across workers; the initializer's limits come too late
        with _spawn_lock:
            saved = {name: os.environ.get(name) for name in _worker_environment}
            os.environ.update(_worker_environment)
            try:
                super()._spawn_process()
            finally:
                for name, value in saved.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

    def __exit__(self, exc_type, exc_value, traceback):
        # cancelled: not waited for - its workers finish their current item and exit on their own
        if exc_type is not None and issubclass(exc_type, OperationCancelled):
            self.shutdown(wait=False, cancel_futures=True)
            return False
        return super().__exit__(exc_type, exc_value, traceback)


def thread_map(func, items, workers):
    """[func(item) for item in items], in order, on `workers` threads (one: in the calling thread)."""
    items = list(items)
    workers = max(1, min(workers, len(items)))
    if workers == 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, items))


def worker_process_pool(workers, max_tasks_per_child=None):
    """Worker processes for GIL-bound per-item work. Spawned, never forked: forking a process running
    Qt and dask threads can copy a lock another thread holds."""
    return _WorkerProcessPool(workers, mp_context=multiprocessing.get_context('spawn'),
                              initializer=single_threaded_worker, max_tasks_per_child=max_tasks_per_child)


def result_unless_cancelled(future, poll_seconds=0.5):
    """future.result(), or OperationCancelled as soon as a cancel is requested while waiting for it."""
    while not wait([future], timeout=poll_seconds).done:
        raise_if_cancelled()
    return future.result()


def rolling_map(func, items, workers):
    """(item, func(item)) for each of `items` as it finishes, on `workers` threads (one: in the calling thread). Only
    twice that many are submitted at once, so memory stays bounded and a slow item holds up only its own thread."""
    if workers <= 1:
        for item in items:
            yield item, func(item)
            raise_if_cancelled()
        return
    items = iter(items)
    pending = {}
    with ThreadPoolExecutor(workers) as pool:
        def submit(count):
            for item in itertools.islice(items, count):
                pending[pool.submit(func, item)] = item

        submit(2 * workers)
        try:
            while pending:
                # woken twice a second even with nothing done, so a cancel need not wait for an item
                done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done:
                    yield pending.pop(future), future.result()
                raise_if_cancelled()
                submit(len(done))
        finally:
            # on an error or an early stop, leaving the pool would otherwise still run every queued item
            for future in pending:
                future.cancel()


def convert_to_um(value, unit):
    """`value` in `unit`, converted to um; an unrecognised unit is left unscaled, and logged.
    Takes both OME abbreviations and spelled-out NGFF unit names.
    """
    if unit is None:
        return value
    factor = um_conversions.get(unit)
    if factor is None:
        factor = um_conversions.get(str(unit).lower())
    if factor is None:
        if str(unit).strip():
            logging.warning(f'Unrecognised physical unit {unit!r}, treating values as um')
        return value
    return value * factor


_pixel_size_pattern = re.compile(r'([-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)\s*(\S+)')


def pixel_size_to_um(text):
    """A pixel size written with its unit ('10um', '0.5 mm', '250nm') in um, or None if it is not one."""
    match = _pixel_size_pattern.fullmatch(text.strip())
    if match is None:
        return None
    unit = match.group(2)
    if unit not in um_conversions and unit.lower() not in um_conversions:
        return None
    return convert_to_um(float(match.group(1)), unit)


def parse_scale(value, default=1):
    """A downscale setting as get_level_from_scale() takes it: a factor - a number, or a number as the
    text a text field gives ('2') - or a pixel size with its unit ('10um'), kept as text. Blank is
    `default`; other text is an error naming the units a pixel size can have."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    if not isinstance(value, str):
        return value
    try:
        number = float(value)
    except ValueError:
        if pixel_size_to_um(value) is None:
            raise ValueError(f'{value!r} is neither a downscale factor nor a pixel size with a unit'
                             f' ({", ".join(sorted(set(um_conversions)))})') from None
        return value.strip()
    return int(number) if number.is_integer() else number



def get_moments(data, offset=(0, 0)):
    moments = cv.moments((np.array(data) + offset).astype(np.float32))    # doesn't work for float64!
    return moments


def get_moments_center(moments, offset=(0, 0)):
    return np.array([moments['m10'], moments['m01']]) / moments['m00'] + np.array(offset)


def get_center(data, offset=(0, 0)):
    moments = get_moments(data, offset=offset)
    if moments['m00'] != 0:
        center = get_moments_center(moments)
    else:
        center = np.mean(data, 0).flatten()  # close approximation
    return center.astype(np.float32)



def create_transform(center, angle, translation=None, matrix_size=3):
    if not center:
        center = np.array([0] * (matrix_size - 1))
    if isinstance(center, dict):
        dims = 'xyz' if matrix_size >= 4 else 'xy'
        center = dict_to_xyz(center, dims=dims, add_zeros=True)
    if len(center) == 2:
        center = np.array(list(center) + [0])
    if angle is None:
        angle = 0
    r = Rotation.from_euler('z', angle, degrees=True)
    t = center - r.apply(center, inverse=True)
    if translation is not None:
        if len(translation) < len(t):
            translation = list(translation) + [0] * (len(t) - len(translation))
        t += translation
    transform = np.eye(matrix_size)
    transform[:3, :3] = np.transpose(r.as_matrix())
    transform[:3, -1] += t
    return transform


def apply_transform(points, transform):
    new_points = []
    for point in points:
        point_len = len(point)
        while len(point) < len(transform):
            point = list(point) + [1]
        new_point = np.dot(point, np.transpose(np.array(transform)))
        new_points.append(new_point[:point_len])
    return new_points


def apply_transform_dict(points, transform, transform_dims='xyz'):
    new_points = []
    for point in points:
        point = dict_to_xyz(point, dims=transform_dims)
        while len(point) < max(len(transform), 3):
            point = list(point) + [1]
        new_point = np.dot(point, np.transpose(np.array(transform)))
        new_point = xyz_to_dict(new_point, dims=transform_dims)
        new_point.pop('1', None)
        new_points.append(new_point)
    return new_points


def calculate_rigid_difference(m1, m2):
    return np.matmul(np.linalg.inv(m1), m2)


def validate_transform(transform, max_scale = 1.25, max_rotation=None):
    try:
        if transform is None or isinstance(transform, str):
            return False
        transform = np.array(transform)
        if np.any(np.isnan(transform)):
            return False
        if np.any(np.isinf(transform)):
            return False
        if np.linalg.det(transform) == 0:
            return False
        scale = get_scale_from_transform(transform)
        if scale < 1 / max_scale or scale > max_scale:
            return False
        if  max_rotation is not None and abs(normalise_rotation(get_rotation_from_transform(transform))) > max_rotation:
            return False
        return True
    except:
        return False


def get_scale_from_transform(transform):
    scale = np.mean(np.linalg.norm(transform, axis=0)[:-1])
    return float(scale)


def get_translation_from_transform(transform):
    ndim = len(transform) - 1
    #translation = transform[:ndim, ndim]
    translation = apply_transform([[0] * ndim], transform)[0]
    return translation



def get_rotation_from_transform(transform, dims='xyz'):
    # TODO: assume 2D rotation, expand to 3D
    # Rotation.from_matrix(transform).as_euler() only works for simple rotation matrices
    if isinstance(transform, DataArray):
        dims = transform['x_in'].data.tolist()
    x_index, y_index = dims.index('x'), dims.index('y')
    transform = np.array(transform)
    if y_index > x_index:
        rotation = np.arctan2(transform[0][1], transform[0][0])
    else:
        rotation = np.arctan2(transform[1][0], transform[1][1])
    return float(np.rad2deg(rotation))


def normalise_rotation(rotation):
    """
    Normalise rotation to be in the range [-180, 180].
    """
    while rotation < -180:
        rotation += 360
    while rotation > 180:
        rotation -= 360
    return rotation


def points_to_3d(points):
    return [list(point) + [0] for point in points]


def xyz_to_dict(xyz, dims='xyz'):
    dct = {dim: float(value) for dim, value in zip(dims, xyz)}
    return dct


def dict_to_xyz(dct, dims='xyz', add_zeros=False):
    if add_zeros:
        array = [dct.get(dim, 0) for dim in dims]
    else:
        array = [dct[dim] for dim in dims if dim in dct]
    return array


def create_chunk_dict(chunk_size, dims):
    if isinstance(chunk_size, dict):
        chunk_dict = chunk_size
    else:
        chunk_dict = {dim: chunk_size if dim in 'xyz' else 1 for dim in dims}
    return chunk_dict


def normalise_rotated_positions(positions0, rotations0, sizes, center, ndims):
    # in [xy(z)]; each source turns about its own centre, as its image is drawn unrotated
    positions = []
    rotations = []
    grid_angle = get_grid_angle(positions0, sizes)
    for position0, size, rotation in zip(positions0, sizes, rotations0):
        if rotation is None and grid_angle is not None:
            rotation = -grid_angle
        angle = -rotation if rotation is not None else None
        transform = create_transform(center=center, angle=angle, matrix_size=ndims + 1)
        half_size = {dim: size.get(dim, 0) / 2 for dim in 'xy'}
        source_center = {dim: position0.get(dim, 0) + half_size[dim] for dim in 'xy'}
        turned_center = apply_transform_dict([source_center], transform, transform_dims='xy')[0]
        positions.append(dict(position0) | {dim: turned_center[dim] - half_size[dim] for dim in 'xy'})
        rotations.append(rotation)
    return positions, rotations


def get_grid_angle(positions, sizes):
    """How far a tile grid is turned, in degrees within (-45, 45], or None without neighbours to tell.
    Only same-sized neighbours at distinct xy count.
    """
    pairs, _ = get_pairs(positions, sizes)
    directions = []
    for first, second in pairs:
        vector = [positions[first].get(dim, 0) - positions[second].get(dim, 0) for dim in 'xy']
        is_same_size = all(math.isclose(sizes[first].get(dim, 0), sizes[second].get(dim, 0), rel_tol=0.01)
                           for dim in 'xy')
        if is_same_size and math.hypot(*vector) > 1e-3 * max(sizes[first].values(), default=0):
            directions.append(math.atan2(vector[1], vector[0]))
    if not directions:
        return None
    # four times the angle: rows and columns, either way round, give the same grid direction
    return math.degrees(math.atan2(np.mean(np.sin(4 * np.array(directions))),
                                   np.mean(np.cos(4 * np.array(directions)))) / 4)


def get_nn_distance(points0):
    points = list(set(map(tuple, points0)))     # get unique points
    if len(points) >= 2:
        tree = KDTree(points, leaf_size=2)
        dist, ind = tree.query(points, k=2)
        nn_distance = np.median(dist[:, 1])
    else:
        nn_distance = 1
    return nn_distance


def get_mean_nn_distance(points1, points2):
    return np.mean([get_nn_distance(points1), get_nn_distance(points2)])


def filter_edge_points(points, bounds, filter_factor=0.1, threshold=0.5):
    center = np.array(bounds) / 2
    dist_center = np.abs(points / center - 1)
    position_weights = np.clip((1 - np.max(dist_center, axis=-1)) / filter_factor, 0, 1)
    order_weights = 1 - np.array(range(len(points))) / len(points) / 2
    weights = position_weights * order_weights
    return weights > threshold



def get_pairs(positions, sizes):
    """
    Get pairs of orthogonal neighbors from a list of tiles.
    Tiles don't have to be placed on a regular grid.
    """
    pairs = []
    angles = []
    z_positions = [position['z'] for position in positions if 'z' in position]
    ordered_z = sorted(set(z_positions))
    z_index = {z: index for index, z in enumerate(ordered_z)}
    is_mixed_3dstack = len(ordered_z) < len(z_positions)
    for i, j in _get_pairs_candidates(positions, sizes, is_mixed_3dstack, z_index):
        posi, posj = positions[i], positions[j]
        sizei, sizej = sizes[i], sizes[j]
        if is_mixed_3dstack:
            # ignore z value for distance
            distance = math.dist([posi[dim] for dim in 'xy'], [posj[dim] for dim in 'xy'])
            min_distance = max([size[dim] for size in [sizei, sizej] for dim in 'xy'])
            is_same_z = (posi['z'] == posj['z'])
            is_close_z = abs(z_index[posi['z']] - z_index[posj['z']]) <= 1
            if not is_close_z:
                # if not close, discard as pair
                min_distance = 0
            elif not is_same_z:
                # for tiles in different z stack, require greater overlap
                min_distance *= 0.8
        else:
            distance = math.dist(posi.values(), posj.values())
            min_distance = max(list(sizei.values()) + list(sizej.values()))

        if distance < min_distance:
            pairs.append((int(i), int(j)))
            vector = np.array(list(posi.values())) - np.array(list(posj.values()))
            angle = math.degrees(math.atan2(vector[1], vector[0]))
            if distance < min(list(sizei.values()) + list(sizej.values())):
                angle += 90
            while angle < -90:
                angle += 180
            while angle > 90:
                angle -= 180
            angles.append(angle)
    return pairs, angles


def _get_pairs_candidates(positions, sizes, is_mixed_3dstack, z_index):
    """(i, j), i < j, of every pair get_pairs() could accept, in the order its full loop visits
    them: each source's box of its own search distance, one section deep in z for a stack.
    """
    from muvis_align.image.util import _sweep_candidate_pairs

    if is_mixed_3dstack:
        dims = ['x', 'y']
        usable = (all({'x', 'y', 'z'} <= set(position) for position in positions)
                  and all({'x', 'y'} <= set(size) for size in sizes))
    else:
        # the full loop pairs positions' values up in order, so every one must list the same dims
        dims = list(positions[0]) if len(positions) else []
        usable = all(list(position) == dims for position in positions)
    if not usable:
        return np.transpose(np.triu_indices(len(positions), 1))
    centers = np.array([[position[dim] for dim in dims] for position in positions], dtype=float)
    # a pair is only accepted closer than the larger of its two sizes, so within either box
    reach = np.array([max(size[dim] for dim in dims) if is_mixed_3dstack else max(size.values())
                      for size in sizes], dtype=float)[:, None]
    mins, maxs = centers - reach, centers + reach
    if is_mixed_3dstack:
        # half a section each way: boxes of neighbouring sections touch, further ones do not
        sections = np.array([[z_index[position['z']]] for position in positions], dtype=float)
        mins, maxs = np.hstack([mins, sections - 0.5]), np.hstack([maxs, sections + 0.5])
    return _sweep_candidate_pairs(mins, maxs)


def retuple(chunks, shape):
    # from ome-zarr-py
    """
    Expand chunks to match shape.

    E.g. if chunks is (64, 64) and shape is (3, 4, 5, 1028, 1028)
    return (3, 4, 5, 64, 64)

    If chunks is an integer, it is applied to all dimensions, to match
    the behaviour of zarr-python.
    """

    if isinstance(chunks, int):
        return tuple([chunks] * len(shape))

    dims_to_add = len(shape) - len(chunks)
    return *shape[:dims_to_add], *chunks



def get_process_memory():
    """This process's (resident, peak resident) memory in bytes, or None where the platform
    will not say (psutil is not a dependency).
    """
    resident = peak = None
    try:
        with open('/proc/self/statm') as file:
            resident = int(file.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, AttributeError, IndexError):
        pass
    try:
        import resource
        # ru_maxrss is kilobytes on Linux but bytes on macOS
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = max_rss if sys.platform == 'darwin' else max_rss * 1024
    except (ImportError, AttributeError, OSError, ValueError):
        pass
    if resident is None and peak is None:
        resident, peak = _windows_process_memory()
    if resident is not None and peak is not None:
        # Linux RSS counters are per-CPU approximations read at different moments, so peak can lag
        peak = max(peak, resident)
    return resident, peak


def _windows_process_memory():
    # neither /proc nor resource exists here - ask the kernel, as _available_memory() does
    try:
        import ctypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        kernel32 = ctypes.windll.kernel32
        # the pseudo-handle is -1, which ctypes' default int restype truncates on 64-bit
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                                     ctypes.POINTER(_ProcessMemoryCounters),
                                                     ctypes.c_ulong]
        if kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                            ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    except Exception:
        pass
    return None, None


def print_memory_usage():
    """' rss 1.2GB (peak 3.4GB)' for a log line, or '' where the platform will not say."""
    resident, peak = get_process_memory()
    if resident is None and peak is None:
        return ''
    parts = []
    if resident is not None:
        parts.append(f'rss {print_hbytes(resident)}')
    if peak is not None:
        parts.append(f'peak {print_hbytes(peak)}')
    return ' ' + ' '.join(parts)


def release_memory(label=None, generation=2):
    """Hand memory the process has freed back to the OS, logging what that recovered.

    glibc's per-thread heaps keep freed buffers below anything still in use; malloc_trim() returns them,
    elsewhere this only collects garbage. `generation` 1 collects only young objects, enough after one
    batch, where a full collection scans every live object.
    """
    gc.collect(generation)
    if not sys.platform.startswith('linux'):
        return
    try:
        import ctypes
        before, _ = get_process_memory()
        ctypes.CDLL('libc.so.6').malloc_trim(0)
        after, _ = get_process_memory()
    except (OSError, AttributeError):
        return
    if label and before is not None and after is not None:
        logging.info(f'{label}: released {print_hbytes(max(before - after, 0))} to the OS,'
                     f' rss {print_hbytes(before)} -> {print_hbytes(after)}')


def to_posix_path(path):
    # one separator everywhere: Windows reports '\' but accepts '/'
    return path.replace('\\', '/') if isinstance(path, str) else path


def path_param_to_text(value):
    # a multi-path param is stored as a comma-separated string or a yaml list; the helpers take one line
    if isinstance(value, (list, tuple)):
        return ', '.join(str(part).strip() for part in value)
    return '' if value is None else str(value)


def eval_path(path):
    if ',' in path:
        parts = path.split(',')
        new_path = ','.join(['"' + part.strip() + '"' for part in parts])
        new_path = eval('[' + new_path + ']')
    else:
        new_path = path
    return new_path


def resolve_to_project_dir(path, base_dir):
    # path params are stored relative to the project dir for portability, but consumers resolve
    # against the cwd; returned with forward slashes
    if not path or not base_dir:
        return to_posix_path(path)
    parts = [part.strip() for part in path.split(',')]
    resolved = []
    for part in parts:
        if not part:
            resolved.append(part)
        else:
            joined = part if os.path.isabs(part) else os.path.join(base_dir, part)
            resolved.append(to_posix_path(os.path.normpath(joined)))
    return ', '.join(resolved)


def relativize_to_project_dir(path, base_dir):
    # counterpart to resolve_to_project_dir(): stored paths stay relative and portable; forward slashes
    if not path or not base_dir:
        return to_posix_path(path)
    parts = [part.strip() for part in path.split(',')]
    relativized = []
    for part in parts:
        if not part:
            relativized.append(part)
        elif not os.path.isabs(part):
            relativized.append(to_posix_path(part))
        else:
            try:
                relativized.append(to_posix_path(os.path.relpath(part, base_dir)))
            except ValueError:
                # on a different drive than the project dir (Windows): can't be made relative
                relativized.append(to_posix_path(part))
    return ', '.join(relativized)


def import_metadata(content, fields=None, input_path=None):
    # return dict[id] = {values}
    if isinstance(content, str):
        ext = os.path.splitext(content)[1].lower()
        if input_path and not isinstance(content, dict):
            if isinstance(input_path, list):
                input_path = input_path[0]
            content = to_posix_path(os.path.normpath(os.path.join(os.path.dirname(input_path), content)))
        if ext == '.csv':
            content = import_csv(content)
        elif ext in ['.json', '.ome.json']:
            content = import_json(content)
    if fields is not None:
        content = [[data[field] for field in fields] for data in content]
    return content


def import_json(filename):
    with open(filename, encoding='utf8') as file:
        data = json.load(file)
    return data


def export_json(filename, data):
    with open(filename, 'w', encoding='utf8') as file:
        json.dump(data, file, indent=4)


def import_csv(filename):
    with open(filename, encoding='utf8') as file:
        data = csv.reader(file)
    return data


def export_csv(filename, data, header=None):
    with open(filename, 'w', encoding='utf8', newline='') as file:
        csvwriter = csv.writer(file)
        if header is not None:
            csvwriter.writerow(header)
        for row in data:
            csvwriter.writerow(row)


def load_sbemimage_best_config(metapath, filename):
    target_datetime = datetime.fromtimestamp(os.path.getmtime(filename))

    for config_filename in sorted(glob.glob(os.path.join(metapath, 'logs/config_*.txt')), reverse=True):
        match = re.split(r'config_(\d+-\d+-\d+).txt', config_filename)
        if len(match) >= 2:
            file_date = datetime.strptime(match[1], '%Y-%m-%d%H%M%S%f')
            if file_date <= target_datetime:
                with open(config_filename, 'r') as file:
                    sbemimage_config = file.read()
                return sbemimage_config
    return None


def adjust_sbemimage_properties(translation, scale, size, filename, sbemimage_config):
    cfg = ConfigParser()
    cfg.read_string(sbemimage_config)
    if bool(cfg['sys'].get('use_microtome')):
        props = cfg['microtome']
    else:
        props = cfg['sem']
    scale_x = float(props.get('stage_scale_factor_x'))
    scale_y = float(props.get('stage_scale_factor_y'))
    rotation_x = float(props.get('stage_rotation_angle_x'))
    rotation_y = float(props.get('stage_rotation_angle_y'))
    rotation_diff = rotation_x - rotation_y
    rot_mat_a = math.cos(rotation_y) / math.cos(rotation_diff)
    rot_mat_b = -math.sin(rotation_y) / math.cos(rotation_diff)
    rot_mat_c = math.sin(rotation_x) / math.cos(rotation_diff)
    rot_mat_d = math.cos(rotation_x) / math.cos(rotation_diff)
    rot_mat_determinant = rot_mat_a * rot_mat_d - rot_mat_b * rot_mat_c

    pixel_size = None
    parts = split_numeric_dict(filename)
    if 't' in parts:
        grids = cfg['grids']
        if 'r' in parts:
            grid_index = json.loads(grids.get('roi_index')).index(int(parts['r']))
        else:
            grid_index = int(parts['g'])
        pixel_size = json.loads(grids.get('pixel_size'))[grid_index] * 1e-3
    else:
        ov = cfg['overviews']
        if 'ov' in parts:
            ov_index = int(parts['ov'])
            pixel_size = json.loads(ov.get('ov_pixel_size'))[ov_index] * 1e-3
        elif '_stubov_' in filename.lower():
            pixel_size = float(ov.get('stub_ov_pixel_size')) * 1e-3

    if pixel_size:
        scale = {'x': pixel_size, 'y': pixel_size}
    else:
        scale = None

    stage_x, stage_y = translation['x'], translation['y']
    stage_x /= scale_x
    stage_y /= scale_y
    dx = ((rot_mat_d * stage_x - rot_mat_b * stage_y) / rot_mat_determinant)
    dy = ((-rot_mat_c * stage_x + rot_mat_a * stage_y) / rot_mat_determinant)
    # convert center to top/left
    physical_size = {dim: size[dim] * scale[dim] for dim in size}
    translation['x'] = dx - physical_size['x'] / 2
    translation['y'] = dy - physical_size['y'] / 2

    return translation, scale



def operation_to_past_participle(operation):
    verb = operation.split()[0]
    if verb.endswith('e'):
        return verb + 'd'
    elif verb.endswith('y'):
        return verb[:-1] + 'ied'
    else:
        return verb + 'ed'
