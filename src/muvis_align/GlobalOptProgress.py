"""Progress for multiview_stitcher's global optimization, read out of its own debug log.

groupwise_resolution() is one long blocking call with nothing to hook (no dask tasks, no tqdm), but it logs
each inner-loop iteration. The bar counts outer-loop passes, each removing one edge, so the edge count bounds
them; a pass also advances with its iterations. Headless, the progress and residual are logged every
`heartbeat_seconds`, a falling residual telling slow from stuck.
"""
import logging
import time

from muvis_align.util import raise_if_cancelled


# upstream's logger and messages: if they change, this degrades to a single phase rather than failing
GLOBAL_OPT_LOGGER = 'multiview_stitcher.param_resolution.global_optimization'
ITERATION_MESSAGE = 'Glob opt iter %s, node %s, mean residual %s, max residual %s'
MAX_ITER_MESSAGE = 'Global optimization: setting max_iter to %s'
FINISHED_MESSAGE = 'Finished glob opt. Max and mean residuals: %s \t %s'

DEFAULT_MAX_ITER = 500


class GlobalOptProgress:
    """Reports multiview_stitcher's global optimization into `progress_factory`'s bar, or only the log without one."""

    heartbeat_seconds = 30

    def __init__(self, progress_factory=None, desc='Global registration', max_passes=None,
                 max_iter=None, weight=1, heartbeat_seconds=None):
        self.progress_factory = progress_factory
        self.desc = desc
        self.max_passes = max(int(max_passes), 1) if max_passes else None
        self.max_iter = max_iter or DEFAULT_MAX_ITER
        self.weight = weight
        if heartbeat_seconds is not None:
            self.heartbeat_seconds = heartbeat_seconds
        self._logger = None
        self._handler = None
        self._prior_level = None
        self._forward_level = logging.NOTSET
        self._prior_propagate = None
        self._phase = None
        self._pass = 0
        self._iteration = None
        self._reported = 0.0
        self._reported_at = None
        self._saw_iterations = False

    def __enter__(self):
        self._logger = logging.getLogger(GLOBAL_OPT_LOGGER)
        # debug level is needed for the iteration lines, so propagation is off and forward() passes on
        # only what the level in force would have shown
        self._prior_level = self._logger.level
        self._forward_level = self._logger.getEffectiveLevel()
        self._prior_propagate = self._logger.propagate
        self._handler = _RecordListener(self)
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._pass = 1
        self._saw_iterations = False
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._logger is not None:
            self._logger.removeHandler(self._handler)
            self._logger.setLevel(self._prior_level)
            self._logger.propagate = self._prior_propagate
        self._handler = None
        if not self._saw_iterations and exc_type is None:
            # nothing recognised: still cross one phase
            self._open_phase()
        self._end_phase(exc_type)
        return False

    def forward(self, record):
        if record.levelno < self._forward_level:
            return
        parent = self._logger.parent if self._logger is not None else None
        if parent is not None:
            parent.handle(record)

    def note_max_iter(self, max_iter):
        try:
            self.max_iter = int(max_iter)
        except (TypeError, ValueError):
            pass

    def note_iteration(self, iteration, max_residual=None):
        try:
            iteration = int(iteration)
        except (TypeError, ValueError):
            return
        if self._iteration is not None and iteration <= self._iteration:
            # the count restarting means the outer loop dropped an edge and began again
            self._pass += 1
        self._iteration = iteration
        self._saw_iterations = True
        self._open_phase()
        self._advance(self._pass - 1 + min((iteration + 1) / self.max_iter, 1.0))
        self._log_heartbeat(max_residual)

    def note_finished(self):
        if self._saw_iterations:
            self._advance(self._pass)

    def _open_phase(self):
        # opened at the first iteration: max_iter is only logged after __enter__
        if self._phase is not None or self.progress_factory is None:
            return
        self._phase = self.progress_factory(total=self.max_passes, desc=self.desc,
                                            weight=self.weight)
        self._phase.__enter__()

    def _end_phase(self, exc_type):
        if self._phase is not None:
            self._phase.__exit__(exc_type, None, None)
            self._phase = None

    def _advance(self, progress):
        """Move the phase to `progress` passes done, which only ever grows."""
        progress = max(progress, self._reported)
        if self.max_passes is not None:
            progress = min(progress, self.max_passes)
        step = progress - self._reported
        self._reported = progress
        if self._phase is not None and step > 0:
            self._phase.update(step)

    def _log_heartbeat(self, max_residual):
        if not self.heartbeat_seconds:
            return
        now = time.monotonic()
        if self._reported_at is not None and now - self._reported_at < self.heartbeat_seconds:
            return
        self._reported_at = now
        residual = ''
        if max_residual is not None:
            try:
                residual = f', max residual {float(max_residual):.4g}'
            except (TypeError, ValueError):
                pass
        of_passes = f'/{self.max_passes}' if self.max_passes else ''
        logging.info(f'{self.desc}: optimisation pass {self._pass}{of_passes},'
                     f' iteration {self._iteration + 1}/{self.max_iter}{residual}')


class _RecordListener(logging.Handler):
    """Takes the optimiser's progress lines and hands everything else back (forward())."""

    def __init__(self, progress):
        super().__init__(level=logging.DEBUG)
        self.progress = progress

    def emit(self, record):
        # raised from inside the optimiser's own log call: the one way to stop that blocking call
        raise_if_cancelled()
        try:
            if record.msg == ITERATION_MESSAGE and record.args:
                max_residual = record.args[3] if len(record.args) > 3 else None
                self.progress.note_iteration(record.args[0], max_residual)
                return
            if record.msg == MAX_ITER_MESSAGE and record.args:
                self.progress.note_max_iter(record.args[0])
            elif record.msg == FINISHED_MESSAGE:
                self.progress.note_finished()
            self.progress.forward(record)
        except Exception:  # pragma: no cover - progress must never break the run it watches
            self.handleError(record)
