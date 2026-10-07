"""Progress for the global optimisation - GlobalOptProgress.

The call it waits on says nothing a progress phase can hear, and counting a phase per pass filled
the bar long before the end. What the optimiser does say goes to its own logger, so these cover both
halves: that the records become a bar that keeps moving without filling, and
(test_upstream_still_logs_its_iterations) that multiview_stitcher still emits them at all - the
one thing here no local care can keep true.
"""
import logging

import pytest
from multiview_stitcher.param_resolution import groupwise_resolution

from muvis_align.GlobalOptProgress import (
    GlobalOptProgress, GLOBAL_OPT_LOGGER, ITERATION_MESSAGE, MAX_ITER_MESSAGE, FINISHED_MESSAGE,
    DEFAULT_MAX_ITER)
from tests.data_builders import grid_graph, make_phase_factory, percent


def make_factory():
    return make_phase_factory(phases=2, desc='Global registration')


def emit_iteration(iteration, max_residual=1.0):
    logging.getLogger(GLOBAL_OPT_LOGGER).debug(ITERATION_MESSAGE, iteration, 0, 1.0, max_residual)


def emit_pass(iterations=10):
    """One pass of the inner loop, converging after `iterations` of the max_iter allowed."""
    for iteration in range(iterations):
        emit_iteration(iteration)


def test_the_bar_moves_while_the_call_is_still_running():
    """A pass is up to 500 iterations, so the bar has to move inside one too, not only when
    groupwise_resolution() returns or at pass boundaries."""
    factory = make_factory()
    reached = []
    with factory:
        with GlobalOptProgress(factory, max_passes=10, heartbeat_seconds=0):
            logging.getLogger(GLOBAL_OPT_LOGGER).info(MAX_ITER_MESSAGE, 100)
            for iteration in range(100):
                emit_iteration(iteration)
                reached.append(percent(factory))

    assert reached[0] > 0, 'the bar did not move until the call returned'
    assert reached == sorted(reached)
    assert len(set(reached)) > 10, f'the bar barely moved across one pass: {set(reached)}'
    assert reached[-1] == pytest.approx(5, abs=1)    # one pass of ten, over half the bar


def test_many_passes_move_the_bar_without_filling_it():
    """Given a phase each, with no total to count them against, every pass took most of what was left.
    They are counted against the edges the optimiser can remove, and it cannot run more passes than that."""
    factory = make_factory()
    ends = []
    with factory:
        with GlobalOptProgress(factory, max_passes=831, heartbeat_seconds=0) as progress:
            for _ in range(505):
                emit_pass()
                ends.append(percent(factory))
            assert progress._pass == 505

    assert ends == sorted(ends), 'the bar went backwards across passes'
    assert len(set(ends)) > 100, 'later passes moved the bar nothing'
    # 505 passes of a possible 831, across a phase holding half the bar
    assert ends[-1] == pytest.approx(50 * 505 / 831, abs=1)
    assert ends[-1] < 50, 'the passes filled a phase they had not finished'


@pytest.mark.parametrize('emit, expected_pass, most_while_running', [
    # converged early: the phase is still done with, so the bar crosses the rest of it
    (lambda: emit_pass(iterations=7), 1, 50),
    # more passes than there are edges to remove, and iterations past max_iter
    (lambda: [emit_pass(iterations=600) for _ in range(5)], 5, 50),
    # upstream saying it has finished completes the last pass
    (lambda: (emit_pass(iterations=3),
              logging.getLogger(GLOBAL_OPT_LOGGER).info(FINISHED_MESSAGE, 0.05, 0.01)), 1, 50),
    # upstream silent, or writing other lines: one phase that moves when the call returns, not an error
    (lambda: (logging.getLogger(GLOBAL_OPT_LOGGER).debug('Glob opt iter %s', 3),
              logging.getLogger(GLOBAL_OPT_LOGGER).debug('something else entirely')), 1, 0),
], ids=['converged early', 'more passes than edges', 'upstream finished', 'upstream silent'])
def test_the_phase_finishes_however_far_the_passes_got(emit, expected_pass, most_while_running):
    factory = make_factory()
    with factory:
        with GlobalOptProgress(factory, max_passes=4, heartbeat_seconds=0) as progress:
            emit()
            assert percent(factory) <= most_while_running, 'a pass ran past its own phase'
            assert progress._pass == expected_pass
        assert percent(factory) == pytest.approx(50, abs=2)


def test_the_log_carries_the_progress_and_not_the_flood(caplog):
    """A headless run has no bar, and this call is the one place where the log is all there is.
    But hearing a debug line means taking the logger down to debug, which would otherwise put
    every debug line the optimiser writes - one per iteration - into a log kept at info.

    heartbeat_seconds is left at its default (0 turns the logging off, which is what the tests
    above want): the first record always reports, whatever the interval.
    """
    with caplog.at_level(logging.INFO):
        with GlobalOptProgress(None, max_passes=831):
            emit_iteration(4, max_residual=0.125)
            for iteration in range(50):
                emit_iteration(iteration)
            logging.getLogger(GLOBAL_OPT_LOGGER).debug('Glob opt iter %s', 3)
            logging.getLogger(GLOBAL_OPT_LOGGER).info(MAX_ITER_MESSAGE, 500)
            logging.getLogger(GLOBAL_OPT_LOGGER).warning('something worth seeing')

    assert any('optimisation pass 1/831, iteration 5/500' in message
               and 'max residual 0.125' in message for message in caplog.messages)
    assert not any('mean residual' in message for message in caplog.messages)
    assert not any(message == 'Glob opt iter 3' for message in caplog.messages)
    # what the app was logging at still gets through, from the same logger, while we listen
    assert any('setting max_iter' in message for message in caplog.messages)
    assert 'something worth seeing' in caplog.messages


def test_the_logger_is_left_as_it_was_found():
    logger = logging.getLogger(GLOBAL_OPT_LOGGER)
    level, propagate, handlers = logger.level, logger.propagate, list(logger.handlers)

    progress = GlobalOptProgress(None, heartbeat_seconds=0)
    assert progress.max_iter == DEFAULT_MAX_ITER
    with progress:
        logging.getLogger(GLOBAL_OPT_LOGGER).info(MAX_ITER_MESSAGE, 120)
    assert progress.max_iter == 120, 'max_iter is not taken from the optimiser'

    with pytest.raises(ValueError):
        with GlobalOptProgress(None, heartbeat_seconds=0):
            raise ValueError('the call it was watching failed')

    assert (logger.level, logger.propagate, logger.handlers) == (level, propagate, handlers)


def test_upstream_still_logs_its_iterations():
    """Everything above rests on multiview_stitcher writing this exact record. It is upstream's,
    so this runs the real optimiser and checks the records arrive - the rest degrades quietly to
    a bar that moves once, and nothing else would notice if they stopped."""
    factory = make_factory()
    with factory:
        with GlobalOptProgress(factory, heartbeat_seconds=0):
            groupwise_resolution(grid_graph(outlier=(5, 6)), method='global_optimization', transform='translation')
        reached = percent(factory)

    assert reached > 0, ('the optimiser reported no iteration this recognised:'
                         f' {ITERATION_MESSAGE!r} is no longer what it logs')
    assert reached < 100, 'the optimisation filled the whole bar on its own'


def test_a_cancel_stops_global_optimization_at_its_next_logged_iteration():
    """global_optimization is one blocking call; its own log call is where a cancel can reach it."""
    from muvis_align.util import OperationCancelled, cancellable, request_cancel

    with cancellable(), GlobalOptProgress(None, desc='Global registration'):
        request_cancel()
        with pytest.raises(OperationCancelled):
            groupwise_resolution(grid_graph(outlier=(5, 6)), method='global_optimization', transform='translation')
