"""Progress for multiview_stitcher's global optimization, read out of its own log.

groupwise_resolution() is one blocking call that can run for a day on a large project (4733
sources, 15576 edges: 22.9 hours in one run) and reports nothing at all while it does. There is
nothing to hook: its optimiser is plain numpy and networkx, so the dask callback around it never
sees a task, and multiview_stitcher.param_resolution contains no tqdm to patch. Wrapped in a
progress phase of its own it still only moved the bar when it returned - which is what left
'Global registration: 0% (1372.8 minutes so far)' in the heartbeat log, hour after hour.

What it does emit is a debug line per pass of its inner optimisation loop, carrying the iteration
number and the residuals. That is a real measure of where it has got to, so this listens for it:
the loop's iteration count against the max_iter ceiling upstream logs on its way in.

What that measures is a pass of the inner loop, and the optimiser runs many: the outer loop drops
the worst edge and runs the inner loop again, 505 times in a 328-source run. So the bar counts
passes, not iterations - and it can count them against a real total, because a pass removes
exactly one edge and the loop stops when there is none worth removing. However long it runs, it
cannot run more passes than the graph has edges.

Giving each pass a phase of its own instead, with no total to count against, is what a first cut
did: every pass took most of what was left of the bar, so 505 of them had it reading 100% thirty
seconds in and for the eighty-six minutes that followed. A bar stuck at 100% is worse than one
stuck at 0% - it says the work is done.

A headless run has no bar, so the same records also go to the log every `heartbeat_seconds`, with
the residual. Over a run this long, a falling residual is the difference between slow and stuck -
the elapsed time alone cannot tell them apart.
"""
import logging
import time


# the optimiser's own logger, and the two messages of its that say where it has got to. They are
# upstream's, so they can change: everything here degrades to a single phase that moves when the
# call returns (which is all there was before) rather than failing, if they ever do.
GLOBAL_OPT_LOGGER = 'multiview_stitcher.param_resolution.global_optimization'
ITERATION_MESSAGE = 'Glob opt iter %s, node %s, mean residual %s, max residual %s'
MAX_ITER_MESSAGE = 'Global optimization: setting max_iter to %s'
FINISHED_MESSAGE = 'Finished glob opt. Max and mean residuals: %s \t %s'

# what upstream falls back to when the caller names no ceiling, used here only if the message
# above never arrives to say so
DEFAULT_MAX_ITER = 500


class GlobalOptProgress:
    """Reports multiview_stitcher's global optimization into `progress_factory`'s bar.

    Used in place of a plain progress phase around groupwise_resolution(); with no factory it
    still logs, which is the whole of what a headless run can show.
    """

    heartbeat_seconds = 30

    def __init__(self, progress_factory=None, desc='Global registration', max_passes=None,
                 max_iter=None, weight=1, heartbeat_seconds=None):
        self.progress_factory = progress_factory
        self.desc = desc
        # what this phase is worth beside the operation's others: it is the great majority of the
        # run (66 of one run's 86 minutes), and equal phases would crawl to a fifth of the bar
        # over an hour and then cross the rest in twenty minutes
        self.weight = weight
        # a pass removes one edge, so the graph's edge count is the most passes there can be. It
        # is a ceiling, not an estimate - the optimiser stops as soon as no edge is worth removing
        # (505 passes of a possible 831 in one run), and the phase's own end covers the rest.
        self.max_passes = max(int(max_passes), 1) if max_passes else None
        self.max_iter = max_iter or DEFAULT_MAX_ITER
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
        # the iteration line is debug, and an app logging at info would never deliver it. Taking
        # the logger down to debug to hear it would also flood the app's own log with every other
        # debug line the optimiser writes, so this takes the records instead of sharing them:
        # propagation off, and anything not consumed here passed on to the handlers it would have
        # reached anyway, if the level it was running at would have shown it. The level to put
        # back is this logger's own - normally unset, so that it follows the app's - while the
        # one to judge a forwarded record by is the level that was actually in force.
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
            # nothing recognised the whole way through: still cross one phase, so the operation is
            # left no worse off than the single phase this replaced
            self._open_phase()
        self._end_phase(exc_type)
        return False

    def forward(self, record):
        """Pass on a record this is not interested in, to the handlers it would have reached had
        the level not been lowered to hear the iteration line."""
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
        """One iteration of the inner optimisation loop finished."""
        try:
            iteration = int(iteration)
        except (TypeError, ValueError):
            return
        if self._iteration is not None and iteration <= self._iteration:
            # the count restarting means the inner loop converged, the outer loop dropped an edge
            # and began it again: one more of the passes the bar is counting
            self._pass += 1
        self._iteration = iteration
        self._saw_iterations = True
        self._open_phase()
        # a pass can be 500 iterations and minutes long, so the bar crosses its own share of one
        # as it goes rather than standing still between pass boundaries
        self._advance(self._pass - 1 + min((iteration + 1) / self.max_iter, 1.0))
        self._log_heartbeat(max_residual)

    def note_finished(self):
        """The optimiser said it was done - the last pass is a whole one, whatever it reached."""
        if self._saw_iterations:
            self._advance(self._pass)

    def _open_phase(self):
        """Opened at the first iteration, not at __enter__: what the passes count against is the
        max_iter ceiling upstream logs on its way in, which it writes after this is already
        listening. A phase opened any earlier would size itself against the default."""
        if self._phase is not None or self.progress_factory is None:
            return
        self._phase = self.progress_factory(total=self.max_passes, desc=self.desc,
                                            weight=self.weight)
        self._phase.__enter__()

    def _advance(self, progress):
        """Move the phase to `progress` passes done, which only ever grows."""
        progress = max(progress, self._reported)
        if self.max_passes is not None:
            progress = min(progress, self.max_passes)
        step = progress - self._reported
        self._reported = progress
        if self._phase is not None and step > 0:
            self._phase.update(step)

    def _end_phase(self, exc_type):
        if self._phase is not None:
            self._phase.__exit__(exc_type, None, None)
            self._phase = None

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
        try:
            if record.msg == ITERATION_MESSAGE and record.args:
                iteration = record.args[0]
                max_residual = record.args[3] if len(record.args) > 3 else None
                self.progress.note_iteration(iteration, max_residual)
                return
            if record.msg == MAX_ITER_MESSAGE and record.args:
                self.progress.note_max_iter(record.args[0])
            elif record.msg == FINISHED_MESSAGE:
                self.progress.note_finished()
            self.progress.forward(record)
        except Exception:  # pragma: no cover - a progress bar must never break the run it watches
            self.handleError(record)
