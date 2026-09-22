import logging
import threading
import time


def _paint_now():
    """Give Qt one pass to paint what was just shown.

    Everything here runs on the Qt thread, so a bar put up immediately before a long blocking
    call would not actually appear until that call returned - which is the one time it is worth
    having.
    """
    from muvis_align.ui._utils import flush_paint_events

    flush_paint_events()


class NapariPhaseProgress:
    """One napari progress bar, filling once, for a whole user-facing operation.

    Used as the progress_factory the phases already expect (MVSRegistration._build_msims(),
    preprocess(), init_progress(), Interface._build_view_msims()): a phase asks for a bar of its
    own and gets a slice of this one. It runs empty to full exactly once, each phase moving it
    across its own slice, so it never restarts and never goes backwards - phases sharing a bar by
    adding to its total each looked like a reset instead, 2/2 becoming 2/330.

    `phases` is what the operation expects to report, in phase-sized units, and sizes the slices.
    An operation running more phases takes them out of what is left; one running fewer has the
    remainder filled in at the end. A phase worth several of its siblings claims that many units
    at once (weight=n) - equal slices for unequal steps is what sat a refresh at 18% for ten
    minutes and then crossed most of the bar in a second.

    The bar goes up when the operation starts, not when its first phase reports, and the Qt loop
    is pumped once so it is painted before the work begins: an operation whose first stretch
    reports nothing would otherwise show nothing until it was nearly done.

    `emit` makes a headless twin for work off the Qt thread (Interface._run_off_thread()), which
    drives no bar and instead reports the position it would have moved one to - Qt widgets may
    only be touched from the thread that owns them. Build one with worker_twin(), which starts it
    where the bar has got to: a twin built at zero re-plans from empty, so an operation's second
    off-thread call reports positions the bar is already past and moves it nothing.

    A long operation also says where it has got to in the log every `heartbeat_seconds`, which is
    all a headless run has, and the only way to tell a slow phase from a hung one.

    The bar keeps the operation's description throughout - a phase naming itself would turn one
    bar into a flicker of labels - and its tick count is internal, left out of what napari shows.
    """

    # no counts, no rate, just the time estimate (napari's eta label is everything after the
    # last '|' of tqdm's formatted line)
    bar_format = '{desc}|{elapsed}<{remaining}'

    # the bar counts in ticks of the whole operation, not in any phase's own units - a phase
    # maps its own steps onto its slice of these
    ticks = 1000

    # long enough that a short operation never logs, short enough to see a long one is alive
    heartbeat_seconds = 30

    # what the last expected phase may take of the remainder, so no phase ever fills the bar:
    # only the operation's end does that, and an unexpected extra phase always has somewhere to go
    last_phase_share = 0.9

    def __init__(self, progress_class=None, desc=None, phases=1, min_duration=0.0, emit=None,
                 **progress_kwargs):
        from napari.utils import progress

        self.progress_class = progress_class or progress
        self.emit = emit
        self.desc = desc
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

    def worker_twin(self, emit):
        """A bar-less stand-in for work about to run on another thread, continuing this bar
        rather than re-planning it from empty. Hand its final state back with continue_from().
        """
        twin = NapariPhaseProgress(emit=emit, phases=self.phases)
        twin._position = self._position
        twin._target = self._target
        twin.phases_left = self.phases_left
        return twin

    def continue_from(self, twin):
        """Take over where a twin left off: its position, and how much of the operation it
        accounted for, so a later phase does not re-divide slices this one has used.
        """
        self.phases_left = twin.phases_left
        self._move_to(twin._position)

    def ensure_phases(self, phases):
        """Make room for a nested operation reporting `phases` of its own into this bar.

        Slices are sized against what is left, so without this a nested operation divides the
        same remainder again and again and the bar crawls toward full without arriving.
        """
        self.phases_left = max(self.phases_left, float(phases))

    # how long a filled bar is held before closing: one painted frame is not reliably seen over
    # a remote display, which is what made every operation look like it stopped where it was
    completion_dwell_seconds = 0.4

    def _report_completed(self):
        """Show, and log, that the operation reached 100% - the counterpart to the heartbeat.

        Reserved phases that never report leave the bar short, and filling it silently at the
        last instant is indistinguishable from giving up at whatever it last showed.
        """
        if self.emit is not None:
            return
        elapsed = time.monotonic() - (self._started_at or time.monotonic())
        if self.heartbeat_seconds and elapsed >= self.heartbeat_seconds:
            # only for operations long enough to have logged their way up - a short one would
            # only ever log 100%, which says nothing
            so_far = (f'{elapsed / 60:.1f} minutes' if elapsed >= 60 else f'{elapsed:.0f} seconds')
            logging.info(f'{self.desc or "Working"}: 100% ({so_far})')
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
                logging.info(f'{self.desc or "Working"}:'
                             f' {self._position / self.ticks * 100:.0f}%'
                             f' ({so_far} so far)')

        self._heartbeat = threading.Thread(target=heartbeat, daemon=True,
                                           name='muvis-align progress')
        self._heartbeat.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self._done.set()
        self._heartbeat = None
        if self._pbar is not None:
            if exc_type is None:
                self._move_to(self.ticks)
                # filling and closing in one pass shows the operation part-done and then gone,
                # which reads as having given up rather than finished
                _paint_now()
                self._report_completed()
            self._pbar.close()
            self._pbar = None
        if self.emit is None:
            # back to how this started, so a factory outliving its operation reports on a new bar
            # rather than silently on a closed one. A twin is never reused, so it is exempt.
            self.phases_left = self.phases
            self._position = 0.0
            self._target = 0.0
        if self._started_at is not None and self.min_duration > 0:
            # an instant operation would otherwise flash the activity dock open and shut
            wait_s = self.min_duration - (time.monotonic() - self._started_at)
            if wait_s > 0:
                time.sleep(wait_s)
        self._started_at = None
        return False

    @property
    def tqdm_class(self):
        """A tqdm stand-in whose bars are phases of this one, so a third-party library (the
        fusion loop in multiview_stitcher, patched in by NapariMVSProgress) moves this bar along
        instead of opening one of its own beside it.
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

            def update(self, n=1):
                self._start()
                super().update(n)

            def close(self):
                if self._open:
                    self._open = False
                    self.__exit__(None, None, None)

            def __getattr__(self, name):
                # tqdm has a wide surface a library may touch anywhere in its loop, so anything
                # beyond the bar interface above is a no-op rather than an AttributeError raised
                # in the middle of someone else's fusion
                return lambda *args, **kwargs: None

        return _PhaseTqdm

    # what one update of a phase that declared no step count crosses of what is left of its
    # slice, so it keeps moving without ever claiming to have arrived
    undeclared_step_share = 0.25

    def _begin_phase(self, total, desc=None, weight=1):
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
        phase_slice.done += n
        if phase_slice.total:
            fraction = min(phase_slice.done / phase_slice.total, 1.0)
        else:
            # a phase that never said how many steps it has still moves on every update: parked
            # at a fixed fraction it is indistinguishable from a hung one for as long as it runs
            fraction = 1.0 - (1.0 - self.undeclared_step_share) ** phase_slice.done
        self._move_to(phase_slice.start + phase_slice.span * fraction)

    def _end_phase(self, phase_slice):
        self._move_to(phase_slice.start + phase_slice.span)

    def set_position(self, position):
        """Move the bar to a position reported by a twin on a worker thread."""
        self._move_to(position)

    def _move_to(self, position):
        # Never re-entered. Updating a napari bar repaints it, which pumps the Qt event loop,
        # which can deliver a worker's next position straight back into here - and a load
        # reporting hundreds of them would run the stack out. A move already in progress just
        # raises the target it is heading for.
        self._target = min(max(position, self._position, self._target), self.ticks)
        if self._moving:
            return
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
        # accepted (phases and third-party bars call it) but ignored: the bar shows the
        # operation's own description throughout
        pass
