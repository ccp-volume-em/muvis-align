import logging
import threading
import time

from muvis_align.util import print_memory_usage, raise_if_cancelled


def _paint_now():
    """Give Qt one pass to paint what was just shown: a bar put up right before a blocking call on the Qt
    thread would otherwise only appear once it returned.
    """
    from muvis_align.ui._utils import flush_paint_events

    flush_paint_events()


class NapariPhaseProgress:
    """One napari progress bar, filling once, for a whole user-facing operation.

    Used as a progress_factory: each phase asking for a bar gets a slice of this one, so the bar runs
    empty to full once, never restarting or going backwards.

    `phases` is the number of phase-sized units the operation expects and sizes the slices; extra phases
    take from what is left, missing ones are filled in at the end. A phase worth several siblings claims
    weight=n units. The bar is painted when the operation starts, before its first phase reports.

    `emit` makes a headless twin for work off the Qt thread, reporting positions instead of moving a bar,
    as Qt widgets may only be touched from their own thread; build it with worker_twin() so it starts
    where the bar is.

    A long operation logs its progress every `heartbeat_seconds`, to tell a slow phase from a hung one.
    The bar keeps the operation's description; only a long step (see set_step()) is named after it.
    """

    # just the time estimate: napari's eta label is everything after the last '|'
    bar_format = '{desc}|{elapsed}<{remaining}'

    # ticks of the whole operation; a phase maps its own steps onto its slice of these
    ticks = 1000

    # long enough that a short operation never logs, short enough to see a long one is alive
    heartbeat_seconds = 30

    # the bar, and a worker's position, shown at most this often: each move is a redraw, or a signal to the Qt thread
    min_update_seconds = 1.0

    # share of the remainder for the last expected phase: only the operation's end fills the bar,
    # and an unexpected extra phase still has room
    last_phase_share = 0.9

    def __init__(self, progress_class=None, desc=None, phases=1, min_duration=0.0, emit=None,
                 emit_step=None, **progress_kwargs):
        from napari.utils import progress

        self.progress_class = progress_class or progress
        self.emit = emit
        self.emit_step = emit_step
        self.desc = desc
        self.step = None
        progress_kwargs.setdefault('bar_format', self.bar_format)
        self.phases = max(int(phases), 1)
        self.phases_left = self.phases
        self.min_duration = max(float(min_duration), 0.0)
        self.progress_kwargs = progress_kwargs
        self._pbar = None
        self._started_at = None
        self._position = 0.0
        self._target = 0.0
        self._moving = False
        self._shown_at = None
        self._heartbeat = None
        self._done = threading.Event()

    def __enter__(self):
        self._started_at = time.monotonic()
        if self.emit is None:
            kwargs = dict(self.progress_kwargs)
            kwargs['total'] = self.ticks
            if self.desc is not None:
                kwargs['desc'] = self.desc
            self._pbar = self.progress_class(**kwargs)
            _paint_now()
        self._start_heartbeat()
        return self

    def __call__(self, total=None, desc=None, weight=1, **_):
        return _ProgressPhase(self, total, desc, weight)

    def worker_twin(self, emit, emit_step=None):
        """A bar-less stand-in for work on another thread, starting where this bar is. Hand its final state
        back with continue_from().
        """
        twin = NapariPhaseProgress(emit=emit, emit_step=emit_step, phases=self.phases)
        twin._position = self._position
        twin._target = self._target
        twin.min_update_seconds = self.min_update_seconds
        twin.last_phase_share = self.last_phase_share
        twin.phases_left = self.phases_left
        return twin

    def continue_from(self, twin):
        """Take over a twin's position and phases used, so later phases do not re-divide used slices."""
        self.phases_left = twin.phases_left
        self._move_to(twin._target, force=True)

    def set_step(self, step):
        """Name the long step the operation is on (a fusion level) in brackets after its description, on the bar and
        in the log."""
        if self.emit is not None:
            if self.emit_step is not None:
                self.emit_step(step)
            return
        self.step = step
        if self._pbar is not None:
            self._pbar.set_description(self.label)

    @property
    def label(self):
        if self.desc and self.step:
            return f'{self.desc} ({self.step})'
        return self.desc or self.step or 'Working'

    def ensure_phases(self, phases):
        """Make room for a nested operation reporting `phases` of its own into this bar, so it does not keep
        re-dividing the same remainder.
        """
        self.phases_left = max(self.phases_left, float(phases))

    # a filled bar is held before closing: one painted frame is not reliably seen over a remote display
    completion_dwell_seconds = 0.4

    def _report_completed(self):
        """Show, and log, that the operation reached 100%, also when reserved phases never reported."""
        if self.emit is not None:
            return
        elapsed = time.monotonic() - (self._started_at or time.monotonic())
        if self.heartbeat_seconds and elapsed >= self.heartbeat_seconds:
            # only after a heartbeat: a short operation logging just 100% says nothing
            so_far = (f'{elapsed / 60:.1f} minutes' if elapsed >= 60 else f'{elapsed:.0f} seconds')
            logging.info(f'{self.label}: 100% ({so_far}){print_memory_usage()}')
        if elapsed >= self.completion_dwell_seconds:
            time.sleep(self.completion_dwell_seconds)

    def _start_heartbeat(self):
        if self.emit is not None or not self.heartbeat_seconds:
            # a twin reports through the one that owns the bar, which does the logging
            return
        self._done.clear()

        def heartbeat():
            while not self._done.wait(self.heartbeat_seconds):
                elapsed = time.monotonic() - (self._started_at or time.monotonic())
                so_far = (f'{elapsed / 60:.1f} minutes' if elapsed >= 60
                          else f'{elapsed:.0f} seconds')
                logging.info(f'{self.label}:'
                             f' {self._position / self.ticks * 100:.0f}%'
                             f' ({so_far} so far){print_memory_usage()}')

        self._heartbeat = threading.Thread(target=heartbeat, daemon=True,
                                           name='muvis-align progress')
        self._heartbeat.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self._done.set()
        self._heartbeat = None
        if self._pbar is not None:
            if exc_type is None:
                self._move_to(self.ticks, force=True)
                # filling and closing in one pass would read as giving up rather than finishing
                _paint_now()
                self._report_completed()
            self._pbar.close()
            self._pbar = None
        if self.emit is None:
            # reset so a factory outliving its operation reports on a new bar, not a closed one
            # (a twin is never reused)
            self.phases_left = self.phases
            self.step = None
            self._position = 0.0
            self._target = 0.0
            self._shown_at = None
        if self._started_at is not None and self.min_duration > 0:
            # an instant operation would otherwise flash the activity dock open and shut
            wait_s = self.min_duration - (time.monotonic() - self._started_at)
            if wait_s > 0:
                time.sleep(wait_s)
        self._started_at = None
        return False

    @property
    def tqdm_class(self):
        """A tqdm stand-in whose bars are phases of this one, so a third-party library moves this bar
        instead of opening its own.
        """
        owner = self

        class _PhaseTqdm(_ProgressPhase):
            def __init__(self, iterable=None, desc=None, total=None, **kwargs):
                if total is None:
                    try:
                        total = len(iterable)
                    except TypeError:
                        total = None
                super().__init__(owner, total=total, desc=desc)
                self.iterable = iterable
                self._open = False

            def __iter__(self):
                self._start()
                for item in self.iterable:
                    yield item
                    self.update(1)
                self.close()

            def _start(self):
                if not self._open:
                    self.__enter__()
                    self._open = True
                    if self.desc:
                        owner.set_step(self.desc)

            def update(self, n=1):
                self._start()
                super().update(n)

            def set_description(self, desc=None, refresh=True):
                # a library's bar over several steps (fusion's levels) names each as it starts it
                if desc:
                    self.desc = desc
                    if self._open:
                        owner.set_step(desc)

            def close(self):
                if self._open:
                    self._open = False
                    self.__exit__(None, None, None)

            def __getattr__(self, name):
                # a library may touch any of tqdm's wide surface: a no-op beats an AttributeError mid-loop
                return lambda *args, **kwargs: None

        return _PhaseTqdm

    # share of its remaining slice an update of a phase without a step count crosses: moving, never arriving
    undeclared_step_share = 0.25

    def _check_cancelled(self):
        # only a worker twin stops: the Qt side's own phases (building the view) are never left half done
        if self.emit is not None:
            raise_if_cancelled()

    def _begin_phase(self, total, desc=None, weight=1):
        self._check_cancelled()
        remaining = self.ticks - self._position
        weight = max(float(weight), 1.0)
        if self.phases_left > weight:
            span = remaining * weight / self.phases_left
        else:
            # the last expected phase, or one never expected at all, takes most of the remainder
            span = remaining * self.last_phase_share
        self.phases_left = max(self.phases_left - weight, 0)
        return _PhaseSlice(start=self._position, span=span, total=total)

    def _advance_phase(self, phase_slice, n=1):
        self._check_cancelled()
        phase_slice.done += n
        if phase_slice.total:
            fraction = min(phase_slice.done / phase_slice.total, 1.0)
        else:
            # still moves on every update: parked at a fixed fraction it would look hung
            fraction = 1.0 - (1.0 - self.undeclared_step_share) ** phase_slice.done
        self._move_to(phase_slice.start + phase_slice.span * fraction)

    def _end_phase(self, phase_slice):
        # shown at once: a phase's last update may have been held back, and the next phase can be long
        self._move_to(phase_slice.start + phase_slice.span, force=True)

    def set_position(self, position):
        """Move the bar to a position reported by a twin on a worker thread, which already reports at most once a
        second."""
        self._move_to(position, force=True)

    def _move_to(self, position, force=False):
        # never re-entered: a repaint pumps the Qt loop, which can deliver the next position back here and
        # overflow the stack; a move in progress just raises its target
        self._target = min(max(position, self._position, self._target), self.ticks)
        now = time.monotonic()
        if self._moving or (not force and self._shown_at is not None
                            and now - self._shown_at < self.min_update_seconds):
            return
        self._shown_at = now
        self._moving = True
        try:
            while True:
                # the bar counts in whole ticks, so only a move that crosses one shows up
                step = int(self._target) - int(self._position)
                self._position = self._target
                if step <= 0:
                    break
                if self.emit is not None:
                    self.emit(int(self._position))
                elif self._pbar is not None:
                    self._pbar.update(step)
                if self._target <= self._position:
                    break
        finally:
            self._moving = False


class _PhaseSlice:
    """What one phase was given: where its slice starts, how much of the bar it spans, and how
    many of the phase's own steps that is worth (None if it never said)."""

    def __init__(self, start, span, total):
        self.start = start
        self.span = span
        self.total = total
        self.done = 0


class _ProgressPhase:
    """One phase's view of the bar, with the same interface a napari progress bar offers, so a
    phase needs no knowledge of the sharing."""

    def __init__(self, owner, total=None, desc=None, weight=1):
        self.owner = owner
        self.total = total
        self.desc = desc
        self.weight = weight
        self._slice = None

    def __enter__(self):
        self._slice = self.owner._begin_phase(self.total, self.desc, self.weight)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None and self._slice is not None:
            self.owner._end_phase(self._slice)
        return False

    def update(self, n=1):
        if self._slice is not None:
            self.owner._advance_phase(self._slice, n)

    def set_description(self, desc):
        # ignored: the bar shows the operation's description throughout
        pass


class SilentProgress:
    """A progress factory that moves no bar but still stops on a cancel: for a short step inside an operation whose
    bar is sized for its long one - its phases would otherwise take that one's share."""

    def __call__(self, total=None, desc=None, weight=1, **_):
        return _SilentPhase(total=total)

    def ensure_phases(self, phases):
        pass

    def worker_twin(self, emit, emit_step=None):
        return self

    def continue_from(self, twin):
        pass

    def set_position(self, *_):
        pass

    def set_step(self, *_):
        pass

    @property
    def tqdm_class(self):
        return _SilentPhase

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _SilentPhase:
    def __init__(self, iterable=None, total=None, **_):
        self.iterable, self.total, self.n = iterable, total, 0

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)

    def update(self, n=1):
        raise_if_cancelled()
        self.n += n

    def __getattr__(self, name):
        # set_description, refresh, close and the like: nothing to show
        return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False
