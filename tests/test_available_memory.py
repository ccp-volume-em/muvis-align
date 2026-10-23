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


def write_files(root, files):
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)


# a Slurm job's limit sits on its job cgroup, above the step and task the process is in; the root is unlimited
@pytest.mark.parametrize('proc_cgroup, files, expected', [
    ('0::/system.slice/slurmstepd.scope/job_1/step_batch/user/task_0\n',
     {'memory.max': 'max', 'system.slice/slurmstepd.scope/job_1/memory.max': str(200 * 1024 ** 3),
      'system.slice/slurmstepd.scope/job_1/step_batch/user/task_0/memory.max': 'max'}, 200 * 1024 ** 3),
    ('12:memory:/slurm/uid_1/job_1/step_batch\n11:cpu,cpuacct:/slurm/uid_1/job_1/step_batch\n',
     {'memory/memory.limit_in_bytes': str(1 << 63), 'memory/slurm/uid_1/job_1/memory.limit_in_bytes': str(64 * 1024 ** 3),
      'memory/slurm/uid_1/job_1/step_batch/memory.limit_in_bytes': str(1 << 63)}, 64 * 1024 ** 3),
    # a container without its host's path mounted: only the root's file is there
    ('0::/docker/abc\n', {'memory.max': str(8 * 1024 ** 3)}, 8 * 1024 ** 3),
    ('0::/user.slice\n', {'memory.max': 'max'}, None),
])
def test_reads_the_tightest_limit_on_the_process_cgroup_path(tmp_path, proc_cgroup, files, expected):
    write_files(tmp_path / 'sys', files)
    (tmp_path / 'cgroup').write_text(proc_cgroup)
    assert constants._cgroup_memory_limit(str(tmp_path / 'cgroup'), str(tmp_path / 'sys')) == expected
