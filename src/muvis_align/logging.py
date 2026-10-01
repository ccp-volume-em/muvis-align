import faulthandler
import logging
import os
from  multiview_stitcher import __version__ as mvs_version

from muvis_align._version import version

_fault_file = None


def init_logging(log_filename='log/muvis-align.log', log_format='%(asctime)s %(levelname)s: %(message)s',
                 verbose=False, debug=False):
    """verbose: muvis-align's own logging (echoed to console, not just the log file).
    debug: external libraries (multiview_stitcher, zarr) at DEBUG level - off by default,
    since unlisted loggers also inherit root's level and get noisy otherwise.
    """
    basepath = os.path.dirname(log_filename)
    if basepath and not os.path.exists(basepath):
        os.makedirs(basepath, exist_ok=True)

    handlers = [logging.FileHandler(log_filename, encoding='utf-8')]
    if verbose:
        handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                        format=log_format, handlers=handlers, encoding='utf-8', force=True)

    if debug:
        # expose multiview_stitcher.registration logger and make more verbose
        mvsr_logger = logging.getLogger('multiview_stitcher.registration')
        mvsr_logger.setLevel(logging.DEBUG)
        if len(mvsr_logger.handlers) == 0:
            mvsr_logger.addHandler(logging.StreamHandler())
        for module in ['multiview_stitcher', 'multiview_stitcher.fusion', 'ome_zarr', 'zarr']:
            logging.getLogger(module).setLevel(logging.DEBUG)
    else:
        for module in ['multiview_stitcher', 'multiview_stitcher.registration', 'multiview_stitcher.fusion',
                       'ome_zarr', 'zarr']:
            logging.getLogger(module).setLevel(logging.WARNING)

    enable_fault_log(log_filename)

    logging.info(f'muvis-align version {version}')
    logging.info(f'Multiview-stitcher version: {mvs_version}')


def enable_fault_log(log_filename):
    """Every thread's stack appended to the log on a fatal signal: a native crash otherwise leaves no trace at all."""
    global _fault_file
    # faulthandler holds only the descriptor: the file must stay open for as long as it is enabled
    fault_file = open(log_filename, 'a', encoding='utf-8')
    faulthandler.enable(file=fault_file, all_threads=True)
    if _fault_file is not None:
        _fault_file.close()
    _fault_file = fault_file
