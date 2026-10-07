"""One bar per operation - NapariPhaseProgress.

The cases here are a bar that froze partway through an operation and one that sat on a single
number while most of the work ran.
"""
import pytest

from tests.data_builders import FakeBar, make_phase_factory, percent


def test_successive_off_thread_calls_each_move_the_bar():
    """Every off-thread call of an operation gets its own twin (Interface._run_off_thread). Built
    fresh at zero, the second and later ones re-planned the whole bar from empty and so only ever
    reported positions the bar was already past - which it ignores, leaving it frozen."""
    owner = make_phase_factory(phases=4)
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
    owner = make_phase_factory(phases=2)
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
    """Equal slices for steps that are nothing like equal stall the bar: building the view data is
    most of a refresh's work but was one step of five."""
    # five units, so the weighted phase below is not also the last one the operation expects -
    # that takes most of whatever is left (last_phase_share) regardless of its weight
    factory = make_phase_factory(phases=5)
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
    factory = make_phase_factory(phases=1)
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
    factory = make_phase_factory(phases=1)
    with factory:
        factory.ensure_phases(4)
        positions = []
        for _ in range(4):
            with factory(total=1) as phase:
                phase.update(1)
            positions.append(percent(factory))

    gaps = [after - before for before, after in zip([0] + positions, positions)]
    assert max(gaps) < 2 * min(gaps), f'slices wildly uneven: {gaps}'


@pytest.mark.parametrize(('phases', 'steps_done'), [(2, 4), (1, 0)], ids=['phase-left-unused', 'phase-undercounts'])
def test_the_bar_is_filled_before_it_closes(phases, steps_done):
    """An operation that only ever showed part-done and then vanished reads as having given up."""
    FakeBar.instances.clear()
    factory = make_phase_factory(phases=phases)
    with factory:
        with factory(total=4) as phase:
            phase.update(steps_done)

    bar = FakeBar.instances[0]
    assert (bar.n, bar.total) == (factory.ticks, factory.ticks)
    assert bar.closed


def test_a_finished_factory_starts_over_on_a_new_bar_but_a_twin_keeps_its_state():
    """A factory handed on to work that runs after its operation must report on a new bar, not the closed one."""
    FakeBar.instances.clear()
    factory = make_phase_factory(phases=2)
    with factory:
        with factory(total=1) as phase:
            phase.update(1)
    assert factory._position == 0
    assert factory.phases_left == factory.phases
    with factory:
        with factory(total=2) as phase:
            phase.update(2)
    first, second = FakeBar.instances
    assert first.closed
    assert second.n > 0

    owner = make_phase_factory(phases=2)
    with owner:
        twin = owner.worker_twin(owner.set_position)
        twin.heartbeat_seconds = 0
        with twin:
            with twin(total=1) as phase:
                phase.update(1)
        # left readable on purpose: continue_from() reads it after the twin has exited
        assert twin._position > 0
        assert twin.phases_left < twin.phases


def test_a_cancel_stops_work_reporting_off_thread_but_never_the_bar_itself():
    """Worker twins raise at their next step once cancelled; the Qt side's own phases (building the view)
    are never left half done."""
    from muvis_align.util import OperationCancelled, cancellable, request_cancel

    owner = make_phase_factory(phases=2)
    with cancellable(), owner:
        twin = owner.worker_twin(owner.set_position)
        twin.heartbeat_seconds = 0
        with pytest.raises(OperationCancelled):
            with twin:
                with twin(total=10) as phase:
                    phase.update(1)
                    request_cancel()
                    phase.update(1)
        with owner(total=2) as phase:
            phase.update(2)


def test_phases_fill_one_bar_once():
    """Every phase of one operation moves a single bar across its own slice, so the bar never
    restarts - phases adding to its total made 2/2 become 2/330, which reads as a new bar."""
    FakeBar.instances.clear()
    factory = make_phase_factory(phases=2, desc='Loading project')
    with factory:
        with factory(total=3, desc='Building sources') as phase:
            for _ in range(3):
                phase.update(1)
            after_first_phase = FakeBar.instances[0].n
        with factory(total=2, desc='Loading pair registration') as phase:
            phase.update(1)
            phase.set_description('Building pair graph')
            phase.update(1)

    assert len(FakeBar.instances) == 1
    bar = FakeBar.instances[0]
    assert bar.total == factory.ticks
    assert after_first_phase == factory.ticks // 2
    # one description for the whole operation - phases naming themselves would make it flicker
    assert bar.descriptions == ['Loading project']


def test_an_unexpected_extra_phase_moves_the_bar_on_without_filling_it():
    """The bar only advances, and no phase fills it: an extra phase (another dask compute, a second
    registration) still needs somewhere to go. Only the end of the operation fills it."""
    FakeBar.instances.clear()
    values, after_phases = [], []
    factory = make_phase_factory(phases=2)
    with factory:
        for total in [4, 300, 2]:
            with factory(total=total) as phase:
                for _ in range(total):
                    phase.update(1)
                    values.append(FakeBar.instances[0].n)
            after_phases.append(FakeBar.instances[0].n)

    assert values == sorted(values)
    assert after_phases[1] < after_phases[2] < factory.ticks
    assert FakeBar.instances[0].n == factory.ticks


def test_the_bar_shows_before_any_phase_reports():
    """Global registration spends most of itself inside one blocking call that reports nothing,
    and showed no bar at all until it was nearly done."""
    FakeBar.instances.clear()
    with make_phase_factory(desc='Global registration'):
        assert len(FakeBar.instances) == 1
        assert FakeBar.instances[0].descriptions == ['Global registration']

    assert FakeBar.instances[0].closed


def test_a_library_tqdm_loop_reports_into_the_same_bar():
    """multiview_stitcher's fusion loop (patched in by NapariMVSProgress) is one more phase, not a bar beside it."""
    FakeBar.instances.clear()
    factory = make_phase_factory(phases=2, desc='Fusion')
    with factory:
        with factory(total=1, desc='Preparing fusion') as phase:
            phase.update(1)
        for _ in factory.tqdm_class(range(3), desc='Fusing blocks'):
            pass

    assert len(FakeBar.instances) == 1
    bar = FakeBar.instances[0]
    assert (bar.n, bar.total) == (factory.ticks, factory.ticks)
    assert bar.descriptions == ['Fusion']


def test_the_tqdm_stand_in_tolerates_the_rest_of_tqdm():
    """A library may touch any of tqdm's wide surface mid-loop - the stand-in must no-op, not raise."""
    FakeBar.instances.clear()
    factory = make_phase_factory()
    with factory:
        tqdm_bar = factory.tqdm_class(total=2, desc='Fusing blocks')
        tqdm_bar.set_postfix(loss=1)
        tqdm_bar.refresh()
        tqdm_bar.update(2)
        tqdm_bar.close()

    assert FakeBar.instances[0].n == factory.ticks


def test_the_bar_does_not_recurse_when_it_reports_back():
    """Repainting a napari bar pumps the Qt event loop, which can deliver the worker's next position
    back into the bar mid-update - left to recurse that runs the stack out."""
    updates = []

    class ReportsBackWhileUpdating(FakeBar):
        def update(self, step=1):
            super().update(step)
            updates.append(self.n)
            if self.n < factory.ticks:
                # as a queued position from the worker would arrive, inside processEvents()
                factory.set_position(self.n + 50)

    FakeBar.instances.clear()
    factory = make_phase_factory(progress_class=ReportsBackWhileUpdating)
    with factory:
        factory.set_position(50)

    assert FakeBar.instances[0].n == factory.ticks
    # each position was applied by the loop, not by re-entering it
    assert len(updates) <= factory.ticks // 50 + 1
