"""Pre-processing's bar is sized by the phases that will actually report.

It reserves one for the per-source msim build and one for pre-processing itself - but the build
only runs when the msims are not already there. With the build cached, nothing reported into
the first half, so the bar (and its time estimate) finished at the end of the second phase's
slice: half a bar, an estimate twice the real time, and then a jump to 100% at the close.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from muvis_align.MVSRegistration import MVSRegistration
import muvis_align.ui.Interface as interface_module


def _registration_with(msims=None, scaled=None):
    registration = MVSRegistration()
    registration._msims = msims
    registration._scaled_msims = scaled if scaled is not None else {}
    registration.sources = []
    return registration


def test_build_pending_when_no_msims_built():
    assert _registration_with(msims=None).msims_build_pending() is True


def test_build_not_pending_when_msims_already_built():
    assert _registration_with(msims=['msim']).msims_build_pending() is False


def test_build_not_pending_when_that_scale_is_cached():
    registration = _registration_with(msims=None, scaled={'2': ['msim']})

    assert registration.msims_build_pending(2) is False


def test_build_pending_when_a_coarser_level_exists_for_that_scale():
    registration = _registration_with(msims=['msim'])
    registration.sources = [object()]

    with patch('muvis_align.MVSRegistration.get_level_from_scale', return_value=(1, None)):
        assert registration.msims_build_pending(2) is True


def _run_pre_processing(build_pending, eager=False):
    """Returns (phases reserved, weight the msim build was given)."""
    interface = interface_module.Interface.__new__(interface_module.Interface)
    interface.params = {'pre_processing': {'scale': 2}}
    interface.reg = MagicMock()
    interface.reg.msims_build_pending.return_value = build_pending
    interface.reg.has_eager_pre_processing.return_value = eager
    interface.reg.preprocess.return_value = (None, None, True)
    interface._timing_verbose = lambda: False
    interface._run_off_thread = lambda work, factory: work(factory)

    declared = {}

    class _Factory:
        def __call__(self, *args, **kwargs):
            raise AssertionError('no phase should be opened by the mocked work')

    @contextmanager
    def operation_progress(desc, progress_factory=None, phases=1):
        declared['phases'] = phases
        yield _Factory()

    interface._operation_progress = operation_progress
    interface.run_pre_processing()
    return declared['phases'], interface.reg.ensure_msims.call_args.kwargs['weight']


def test_only_the_phases_that_will_run_are_reserved():
    assert _run_pre_processing(build_pending=False)[0] == 1


def test_the_build_takes_the_bar_in_proportion_to_what_it_costs():
    # opening every source is nearly all of a run whose only step is scaling - splitting the
    # bar evenly with it left the work finishing at the halfway mark
    phases, weight = _run_pre_processing(build_pending=True)
    assert (phases, weight) == (9, 8)

    # a step that computes over the data makes the rest of the run real work again
    phases, weight = _run_pre_processing(build_pending=True, eager=True)
    assert (phases, weight) == (3, 2)


def test_eager_pre_processing_is_recognised_from_the_project_params():
    eager = MVSRegistration.has_eager_pre_processing
    assert eager({'scale': 8, 'gaussian_sigma': 2.0}) is False
    # as a project file stores "no normalisation" - plain truthiness reads it as on
    assert eager({'normalisation': 'none'}) is False
    assert eager({'normalisation': 'global'}) is True
    assert eager({'flatfield_quantiles': '0.05, 0.95'}) is True
    assert eager({'filter_foreground': True}) is True
