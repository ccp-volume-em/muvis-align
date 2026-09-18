"""One bar per operation - NapariPhaseProgress.

The cases here are the ones a 4733-source run got wrong: a bar that froze partway through an
operation and one that sat on a single number for ten minutes.
"""
import pytest

from muvis_align.ui.NapariPhaseProgress import NapariPhaseProgress


class FakeBar:
    """Stands in for napari's progress bar - these tests need no Qt."""

    def __init__(self, **kwargs):
        self.total = kwargs.get('total')
        self.n = 0
        self.closed = False

    def update(self, step=1):
        self.n += step

    def close(self):
        self.closed = True


def make_factory(phases=1, **kwargs):
    factory = NapariPhaseProgress(progress_class=FakeBar, desc='Operation', phases=phases, **kwargs)
    # the heartbeat only logs; nothing here waits long enough for it to fire
    factory.heartbeat_seconds = 0
    return factory


def percent(factory):
    return factory._position / factory.ticks * 100


def test_successive_off_thread_calls_each_move_the_bar():
    """Every off-thread call of an operation gets its own twin (Interface._run_off_thread). Built
    fresh at zero, the second and later ones re-planned the whole bar from empty and so only ever
    reported positions the bar was already past - which it ignores, leaving it frozen."""
    owner = make_factory(phases=4)
    reached = []
    with owner:
        for _ in range(3):
            twin = owner.worker_twin(owner.set_position)
            twin.heartbeat_seconds = 0
            with twin:
                with twin(total=10) as phase:
                    for _ in range(10):
                        phase.update(1)
            owner.continue_from(twin)
            reached.append(percent(owner))

    assert reached == sorted(reached)
    assert len(set(reached)) == 3, f'bar stalled across off-thread calls: {reached}'
    assert reached[0] == pytest.approx(25, abs=1)
    assert reached[-1] > 70


def test_a_twin_reports_into_its_owners_remaining_space():
    owner = make_factory(phases=2)
    with owner:
        with owner(total=1) as phase:
            phase.update(1)
        half = percent(owner)
        twin = owner.worker_twin(owner.set_position)
        twin.heartbeat_seconds = 0
        with twin:
            with twin(total=4) as phase:
                phase.update(1)
                # the twin continues from where the bar is, rather than starting again at 0
                assert percent(owner) > half


def test_a_heavy_phase_can_claim_more_of_the_bar_than_its_siblings():
    """Equal slices for steps that are nothing like equal is what left a refresh at 18% for ten
    minutes: building the view data is most of the work but was one step of five."""
    # five units, so the weighted phase below is not also the last one the operation expects -
    # that takes most of whatever is left (last_phase_share) regardless of its weight
    factory = make_factory(phases=5)
    with factory:
        with factory(total=1) as phase:
            phase.update(1)
        light = percent(factory)
        with factory(total=1, weight=3) as phase:
            phase.update(1)
        heavy = percent(factory) - light

    assert light == pytest.approx(20, abs=1)
    assert heavy == pytest.approx(3 * light, rel=0.02)


def test_a_phase_without_a_step_count_keeps_moving():
    """It used to park at half its slice and stay there, which is indistinguishable from a hung
    operation for as long as the phase runs."""
    factory = make_factory(phases=1)
    seen = []
    with factory:
        with factory() as phase:
            for _ in range(5):
                phase.update(1)
                seen.append(percent(factory))

    assert seen == sorted(seen)
    assert len(set(seen)) == len(seen), f'position repeated: {seen}'
    assert seen[-1] < 100


def test_a_nested_operation_makes_room_for_its_own_phases():
    """Without ensure_phases() a nested operation divides whatever the outer one had left, each
    of its phases taking most of the remainder, so the bar creeps toward full without arriving."""
    factory = make_factory(phases=1)
    with factory:
        factory.ensure_phases(4)
        positions = []
        for _ in range(4):
            with factory(total=1) as phase:
                phase.update(1)
            positions.append(percent(factory))

    gaps = [b - a for a, b in zip([0] + positions, positions)]
    assert max(gaps) < 2 * min(gaps), f'slices wildly uneven: {gaps}'


def test_the_bar_is_filled_before_it_closes():
    """An operation that only ever showed part-done and then vanished reads as having given up."""
    bars = []

    class RecordingBar(FakeBar):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            bars.append(self)

    factory = NapariPhaseProgress(progress_class=RecordingBar, desc='Operation', phases=2)
    factory.heartbeat_seconds = 0
    with factory:
        with factory(total=1) as phase:
            phase.update(1)

    assert bars[0].n == bars[0].total
    assert bars[0].closed


def test_a_finished_factory_starts_over_but_a_twin_keeps_its_state():
    factory = make_factory(phases=2)
    with factory:
        with factory(total=1) as phase:
            phase.update(1)
    assert factory._position == 0
    assert factory.phases_left == factory.phases

    owner = make_factory(phases=2)
    with owner:
        twin = owner.worker_twin(owner.set_position)
        twin.heartbeat_seconds = 0
        with twin:
            with twin(total=1) as phase:
                phase.update(1)
        # left readable on purpose: continue_from() reads it after the twin has exited
        assert twin._position > 0
        assert twin.phases_left < twin.phases
