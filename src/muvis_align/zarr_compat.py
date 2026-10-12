"""Windows-only retry for zarr's atomic metadata write (zarr.storage._local._atomic_write).

On Windows the temporary file's rename onto the real name fails (PermissionError) if anything briefly holds the
destination open. The holder is unidentified and the hold transient, so the rename is retried for a few seconds.
Does nothing on other platforms, or if zarr's private helper has changed.
"""
import contextlib
import logging
import os
import time
import uuid
from pathlib import Path

_MAX_WAIT_SECONDS = 4.0
_FIRST_DELAY_SECONDS = 0.01

_applied = False


def apply_windows_atomic_write_retry():
    """Install the retry. Idempotent, and a no-op where it does not apply."""
    global _applied
    if _applied or os.name != 'nt':
        return False

    try:
        import zarr.storage._local as zarr_local
    except ImportError:
        return False
    original = getattr(zarr_local, '_atomic_write', None)
    safe_move = getattr(zarr_local, '_safe_move', None)
    if original is None or safe_move is None:
        # zarr's internals have moved - leave them alone rather than guess
        return False

    @contextlib.contextmanager
    def _atomic_write_with_retry(path, mode, exclusive=False):
        path = Path(path)
        tmp_path = path.with_suffix(f'.{uuid.uuid4().hex}.partial')
        # cleanup on failure only, as the original: on success the temp file has been renamed away
        try:
            with tmp_path.open(mode) as file:
                yield file
            delay, waited, attempts = _FIRST_DELAY_SECONDS, 0.0, 0
            while True:
                try:
                    if exclusive:
                        safe_move(tmp_path, path)
                    else:
                        tmp_path.replace(path)
                    break
                except PermissionError:
                    if waited >= _MAX_WAIT_SECONDS:
                        # a hold this long is not the transient one this exists for
                        raise
                    time.sleep(delay)
                    waited += delay
                    attempts += 1
                    delay = min(delay * 2, 0.2)
            if attempts:
                logging.debug(f'{path.name}: atomic write retried {attempts}x'
                              f' ({waited:.2f}s) - destination was briefly held open')
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    zarr_local._atomic_write = _atomic_write_with_retry
    _applied = True
    return True
