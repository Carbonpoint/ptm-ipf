"""Tests for the child process the analysis runs in.

Its whole purpose is that a long match can be killed, which no cooperative
flag can achieve: OVITO's compute() returns when it is finished and not
before.  The tests therefore care about two things above all, that a job
comes back correctly, and that a job can be stopped while it runs.
"""

import subprocess
import sys
import threading
import time

import pytest

from ptmipf.webui.worker import AnalysisWorker, Cancelled, WorkerUnavailable

pytest.importorskip("ase.build")
if subprocess.run(
    [sys.executable, "-c", "import ovito"], capture_output=True
).returncode != 0:
    pytest.skip("OVITO is unavailable", allow_module_level=True)


@pytest.fixture
def worker():
    made = AnalysisWorker()
    yield made
    made.close()


def _job(path, **analysis):
    settings = {
        "path": str(path),
        "structures": ["hcp", "fcc"],
        "rmsd_cutoff": 0.1,
        "frame_index": 0,
        "columns": None,
        "slab": None,
    }
    settings.update(analysis)
    colour = {
        "direction": "z",
        "axes": {},
        "color_only": [],
        "other_color": (0.4, 0.4, 0.4),
        "rotations": (),
    }
    return settings, colour


def test_a_job_comes_back_matched_and_coloured(worker, write_crystal, tmp_path):
    path = write_crystal("hcp", repeat=4)
    stages = []
    analysis, colour = _job(path)
    result = worker.run(
        analysis, colour, tmp_path / "out.pkl", on_stage=lambda s, n=None: stages.append(s)
    )
    assert result.n_atoms > 0
    assert result.counts["hcp"] == result.n_atoms
    assert result.colors.shape == (result.n_atoms, 3)
    assert result.direction_label
    # The stages are reported as they happen, which is what drives the bar.
    assert stages[0].startswith("reading")
    assert any("template matching" in s for s in stages)
    # The result file is the child's to write and the parent's to clean up.
    assert not (tmp_path / "out.pkl").exists()


def test_the_process_is_kept_warm_between_jobs(worker, write_crystal, tmp_path):
    path = write_crystal("hcp", repeat=4)
    analysis, colour = _job(path)
    worker.run(analysis, colour, tmp_path / "a.pkl")
    first = worker.process.pid
    worker.run(analysis, colour, tmp_path / "b.pkl")
    assert worker.process.pid == first


def test_stopping_a_job_raises_cancelled_and_leaves_the_worker_usable(
    worker, write_crystal, tmp_path
):
    """A stopped job must not poison the next one."""
    # Big enough that the match lasts long enough to be caught and killed.
    path = write_crystal("hcp", repeat=26)
    analysis, colour = _job(path)
    outcome = {}

    def run():
        try:
            worker.run(analysis, colour, tmp_path / "slow.pkl")
        except BaseException as exc:  # noqa: BLE001 - the type is the assertion
            outcome["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    # Stop it as soon as it is under way; before that there is nothing to kill.
    deadline = time.time() + 30
    while not worker.busy and time.time() < deadline:
        time.sleep(0.01)
    assert worker.stop() is True
    thread.join(timeout=60)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), Cancelled)
    # A fresh child takes the next job.
    result = worker.run(*_job(write_crystal("hcp", repeat=3, name="small.xyz")),
                        tmp_path / "after.pkl")
    assert result.n_atoms > 0


def test_stopping_when_nothing_runs_says_so(worker):
    assert worker.stop() is False


def test_a_failing_analysis_is_reported_not_a_crash(worker, tmp_path):
    analysis, colour = _job(tmp_path / "there-is-no-such-file.xyz")
    with pytest.raises(ValueError):
        worker.run(analysis, colour, tmp_path / "out.pkl")
    # The child survives its own error and takes the next job.
    assert worker.process.poll() is None


def test_an_unusable_interpreter_is_reported_once(tmp_path):
    """When the child cannot run, the caller is told to fall back, once."""
    broken = AnalysisWorker(python=str(tmp_path / "not-a-python"))
    with pytest.raises(WorkerUnavailable):
        broken.start()
    assert broken.available is False
    with pytest.raises(WorkerUnavailable):
        broken.start()
    assert "could not be started" in broken.reason or "not usable" in broken.reason
