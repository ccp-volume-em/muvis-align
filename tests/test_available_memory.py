"""_available_memory() must read this job's own allocation, not the machine's hardware: on a shared
HPC node the allocation is what a memory budget has to fit inside."""
import pytest

from muvis_align import constants
from muvis_align.constants import _available_memory


@pytest.fixture
def no_slurm(monkeypatch):
    for variable in ('SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU'):
        monkeypatch.delenv(variable, raising=False)


# values in MB, as SLURM reports them; per node wins over per cpu, which scales by the allocated cores
@pytest.mark.parametrize('environment, expected', [
    ({'SLURM_MEM_PER_NODE': '65536'}, 64 * 1024 ** 3),
    ({'SLURM_MEM_PER_NODE': '65536', 'SLURM_MEM_PER_CPU': '1024'}, 64 * 1024 ** 3),
    ({'SLURM_MEM_PER_CPU': '4096'}, 4 * 1024 ** 3 * constants._available_cpus),
])
def test_reads_the_slurm_allocation(monkeypatch, no_slurm, environment, expected):
    for variable, value in environment.items():
        monkeypatch.setenv(variable, value)
    assert _available_memory() == expected


def test_ignores_unparseable_slurm_value(monkeypatch, no_slurm):
    monkeypatch.setenv('SLURM_MEM_PER_NODE', 'unlimited')
    # falls through to the real machine rather than raising
    assert _available_memory() is None or _available_memory() > 0


def test_falls_back_to_the_machine_without_slurm(no_slurm):
    memory = _available_memory()
    # a plausible floor guards a silently wrong unit (bytes vs kB vs pages)
    assert memory is not None
    assert memory > 256 * 1024 ** 2
