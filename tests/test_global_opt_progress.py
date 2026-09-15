"""Progress for the global optimisation - GlobalOptProgress.

The case these cover is a bar that read 'Global registration: 0% (1372.8 minutes so far)' for a
whole day, because the one call it was waiting on says nothing a progress phase can hear. What it
does say goes to its own logger, so these check both halves of that: that the records are turned
into a bar that keeps moving, and (test_upstream_still_logs_its_iterations) that multiview_stitcher
still emits the records at all - the one thing here that no amount of local care can keep true.
"""
import logging
import os.path

import pytest
import yaml
from multiview_stitcher import msi_utils

from muvis_align.GlobalOptProgress import (
    GlobalOptProgress, GLOBAL_OPT_LOGGER, ITERATION_MESSAGE, MAX_ITER_MESSAGE, DEFAULT_MAX_ITER)
from muvis_align.MVSRegistration import MVSRegistration
from muvis_align.ui.NapariPhaseProgress import NapariPhaseProgress

from tests.test_phase_progress import FakeBar, percent


def make_factory(phases=1):
    factory = NapariPhaseProgress(progress_class=FakeBar, desc='Global registration', phases=phases)
    # the bar's own heartbeat only logs; nothing here runs long enough for it to fire
    factory.heartbeat_seconds = 0
    return factory


def emit_iteration(iteration, node=0, mean_residual=1.0, max_residual=1.0):
    """The optimiser's own per-iteration line, as it writes it."""
    logging.getLogger(GLOBAL_OPT_LOGGER).debug(
        ITERATION_MESSAGE, iteration, node, mean_residual, max_residual)


def test_iterations_move_the_bar_while_the_call_is_still_running():
    """The whole point: the bar used to move only when groupwise_resolution() returned, so a run
    of many hours showed 0% throughout and then jumped straight to done."""
    factory = make_factory(phases=2)
    reached = []
    with factory:
        with GlobalOptProgress(factory, heartbeat_seconds=0):
            logging.getLogger(GLOBAL_OPT_LOGGER).info(MAX_ITER_MESSAGE, 100)
            for iteration in range(100):
                emit_iteration(iteration)
                reached.append(percent(factory))

    assert reached[0] > 0, 'the bar did not move until the call returned'
    assert reached == sorted(reached)
    assert len(set(reached)) > 10, f'the bar barely moved across 100 iterations: {set(reached)}'
    # the first phase of two, so about half the bar, and the operation's own end fills the rest
    assert reached[-1] == pytest.approx(50, abs=2)


def test_each_optimisation_pass_takes_a_new_slice():
    """The outer loop drops an edge and runs the inner loop again, as often as it takes. A
    restarting iteration count is a new phase out of what is left - not a jump backwards, and not
    a bar that sat at the end of its one slice for every pass after the first."""
    factory = make_factory(phases=2)
    ends = []
    with factory:
        with GlobalOptProgress(factory, heartbeat_seconds=0) as progress:
            for _ in range(4):
                for iteration in range(10):
                    emit_iteration(iteration)
                ends.append(percent(factory))
            assert progress._pass == 4

    assert ends == sorted(ends), f'the bar went backwards across passes: {ends}'
    assert len(set(ends)) == 4, f'later passes moved the bar nothing: {ends}'
    assert ends[-1] < 100, 'a pass filled the bar - only the end of the operation may do that'


def test_a_converged_pass_still_finishes_its_slice():
    """The inner loop stops as soon as the residual settles, which is nearly always long before
    max_iter. Its slice is still done with, so the bar crosses it."""
    factory = make_factory(phases=2)
    with factory:
        with GlobalOptProgress(factory, heartbeat_seconds=0):
            logging.getLogger(GLOBAL_OPT_LOGGER).info(MAX_ITER_MESSAGE, 500)
            for iteration in range(7):    # converged at 7 of 500
                emit_iteration(iteration)
        assert percent(factory) == pytest.approx(50, abs=2)


def test_silence_degrades_to_one_phase():
    """If upstream ever stops writing the line, or writes a different one, this must fall back to
    what there was before - a phase that moves when the call returns - and not to an error."""
    factory = make_factory(phases=2)
    with factory:
        with GlobalOptProgress(factory, heartbeat_seconds=0):
            logging.getLogger(GLOBAL_OPT_LOGGER).debug('Glob opt iter %s', 3)    # the outer line
            logging.getLogger(GLOBAL_OPT_LOGGER).debug('something else entirely')
            assert percent(factory) == 0
        assert percent(factory) == pytest.approx(50, abs=2)


def test_max_iter_is_taken_from_the_optimiser():
    progress = GlobalOptProgress(None, heartbeat_seconds=0)
    assert progress.max_iter == DEFAULT_MAX_ITER
    with progress:
        logging.getLogger(GLOBAL_OPT_LOGGER).info(MAX_ITER_MESSAGE, 120)
    assert progress.max_iter == 120


def test_without_a_factory_it_still_logs(caplog):
    """A headless run has no bar, and this call is the one place where the log is all there is.

    heartbeat_seconds is left at its default here (0 turns the logging off, which is what the
    tests above want): the first record always reports, whatever the interval."""
    with caplog.at_level(logging.INFO):
        with GlobalOptProgress(None, desc='Global registration'):
            emit_iteration(4, max_residual=0.125)

    assert any('optimisation pass 1, iteration 5/500' in message and 'max residual 0.125' in message
               for message in caplog.messages)


def test_the_iteration_flood_stays_out_of_the_app_log(caplog):
    """Hearing a debug line means taking the logger down to debug, which would otherwise put every
    debug line the optimiser writes - one per iteration, hundreds of thousands of them - into the
    log the app is keeping at info."""
    with caplog.at_level(logging.INFO):
        with GlobalOptProgress(None, heartbeat_seconds=0):
            for iteration in range(50):
                emit_iteration(iteration)
            logging.getLogger(GLOBAL_OPT_LOGGER).debug('Glob opt iter %s', 3)
            logging.getLogger(GLOBAL_OPT_LOGGER).info(MAX_ITER_MESSAGE, 500)
            logging.getLogger(GLOBAL_OPT_LOGGER).warning('something worth seeing')

    assert not any('mean residual' in message for message in caplog.messages)
    assert not any(message == 'Glob opt iter 3' for message in caplog.messages)
    # what the app was logging at still gets through, from the same logger, while we listen
    assert any('setting max_iter' in message for message in caplog.messages)
    assert 'something worth seeing' in caplog.messages


def test_the_logger_is_left_as_it_was_found():
    logger = logging.getLogger(GLOBAL_OPT_LOGGER)
    level, propagate, handlers = logger.level, logger.propagate, list(logger.handlers)

    with GlobalOptProgress(None, heartbeat_seconds=0):
        pass
    with pytest.raises(ValueError):
        with GlobalOptProgress(None, heartbeat_seconds=0):
            raise ValueError('the call it was watching failed')

    assert logger.level == level
    assert logger.propagate == propagate
    assert logger.handlers == handlers


def test_upstream_still_logs_its_iterations():
    """Everything above rests on multiview_stitcher writing this exact record. It is upstream's,
    not ours, so this runs the real optimiser and checks the records arrive - the rest degrades
    quietly to a bar that moves once, so nothing else would notice if they stopped."""
    with open(os.path.join('resources', 'params_test_2d.yml'), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    operation_params = params['operations'][0]
    reg_params = operation_params['registration']
    reg = MVSRegistration()
    reg.init_params(params['general'], operation_params)
    reg.init_data()
    reg.preprocess(reg.msims)
    reg.register_pairs(reg.register_msims, params=reg_params)

    factory = make_factory(phases=2)
    sims = [msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in reg.msims]
    from muvis_align.image.util import wrap_sims_as_msims
    with factory:
        reg.register_global(wrap_sims_as_msims(sims), params=reg_params, progress_factory=factory)
        reached = percent(factory)

    assert reached > 0, ('the optimiser reported no iteration this recognised:'
                         f' {ITERATION_MESSAGE!r} is no longer what it logs')
