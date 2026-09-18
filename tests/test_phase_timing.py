"""The threaded-phase timing line has to say which regime a run is in.

Reporting wall against the summed per-item time - the obvious summary, and what these logs used
to print - cannot distinguish "the threads overlapped real waiting" from "the threads queued
behind each other". Under a pool an item's wall time also counts its wait for the GIL, so that
sum inflates with the worker count either way: on 328 sources it went 15s -> 1028s from 1 to 64
workers while wall time went 14.9s -> 16.6s, so the 60x "overlap" it implied was worth nothing.
"""
import pytest

from muvis_align.util import format_phase_timing

CPU_BOUND = 'more workers will not help'
IO_BOUND = 'more workers can overlap it'


@pytest.mark.parametrize('label, wall, item_times, cpu_times, workers, expected', [
    # the real measurement that motivated this: 328 sources at 64 workers, wall 16.6s, summed
    # per-source 1027.9s, but only 16.5s of CPU - and 14.9s sequentially, so the threads bought
    # nothing
    ('the motivating run', 16.64, [1027.9 / 328] * 328, [16.5 / 328] * 328, 64, CPU_BOUND),
    # each item waits 3.0s but needs 0.05s of CPU, and 8 workers cannot overlap it all
    ('under-parallelised I/O', 328 * 3.0 / 8, [3.0] * 328, [0.05] * 328, 8, IO_BOUND),
    # the same work with enough workers reaches the CPU floor, which no arrangement of threads
    # beats - so it must stop advising more
    ('I/O at the floor', 16.4, [3.0] * 328, [0.05] * 328, 64, CPU_BOUND),
])
def test_the_regime_is_named_from_wall_against_cpu(label, wall, item_times, cpu_times, workers,
                                                   expected):
    line = format_phase_timing(wall, item_times, cpu_times, workers)

    assert expected in line
    assert (CPU_BOUND if expected is IO_BOUND else IO_BOUND) not in line


def test_no_cpu_times_reports_without_a_verdict():
    """A caller that cannot measure CPU time must not get a made-up regime."""
    line = format_phase_timing(5.0, [1.0, 3.0], [], 2)

    assert 'wall 5.0s' in line
    assert CPU_BOUND not in line and IO_BOUND not in line


def test_the_numbers_themselves_are_reported():
    line = format_phase_timing(10.0, [1.0, 3.0, 1.0, 3.0], [0.5] * 4, 4)

    assert 'wall 10.0s' in line
    assert 'per-item total 8.0s' in line
    assert 'cpu 2.0s' in line
    assert 'with 4 workers' in line
    assert 'mean 2000ms' in line and 'max 3000ms' in line    # from the wall times
    assert 'process cpu' not in line                          # optional, and not given here


def test_process_cpu_separates_this_phase_from_the_rest_of_the_process():
    """The 4733-source run measured 468.5s of CPU inside a phase whose process burned 2268s over
    the same stretch - ~3.8 cores, almost none of it this phase. time.thread_time counts only
    the thread running the item, so without the process figure the phase looks merely slow."""
    busy = format_phase_timing(598.5, [38254.3 / 4733] * 4733, [468.5 / 4733] * 4733, 64,
                               process_cpu_time=2268.0)
    assert 'process cpu 2268.0s' in busy
    assert '3.8 cores' in busy
    assert 'the rest is elsewhere in the process' in busy

    # ...and a process whose CPU is all this phase is not flagged as elsewhere
    contained = format_phase_timing(20.0, [1.0] * 16, [1.0] * 16, 16, process_cpu_time=17.0)
    assert 'process cpu 17.0s' in contained
    assert 'the rest is elsewhere in the process' not in contained
