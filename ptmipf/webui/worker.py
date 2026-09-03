"""A separate process that runs the analysis, so that it can be stopped.

OVITO offers no way to interrupt a running pipeline: once ``compute()`` is
called, the call returns when it is finished and not before.  A cooperative
flag can therefore only be checked between stages, which is no use at all to
somebody who has just started a twenty minute match on the wrong file and
wants their interface back.

So the matching runs in a child process, which can simply be killed.  The
child is started once and kept warm (importing OVITO costs a second or two
that nobody should pay per run), takes one job at a time over its standard
input and answers on its standard output:

    parent -> child   one line, base64 of a pickled job dict
    child  -> parent  lines of ``@@`` followed by JSON

The result is far too big for a pipe, so the child pickles it to a file the
parent names and reports the path.  Anything the child prints that is not a
protocol line (an OVITO banner, a warning) is ignored, which is why the
prefix is there.

If the child cannot be started, or dies for a reason of its own, the caller
falls back to running the analysis in this process; see
:meth:`AnalysisWorker.run`.  The interface keeps working, it just cannot be
interrupted mid-stage.
"""

from __future__ import annotations

import base64
import json
import pickle
import subprocess
import sys
import threading
from pathlib import Path

__all__ = ["AnalysisWorker", "Cancelled", "WorkerUnavailable", "main"]

#: Marks a protocol line, so that anything else the child writes is ignored.
PREFIX = "@@"


class Cancelled(Exception):
    """Raised in the parent when a job was stopped rather than finished."""


class WorkerUnavailable(Exception):
    """Raised when the child process cannot be used and the caller must fall back."""


def _emit(payload: dict) -> None:
    sys.stdout.write(PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()


def main(argv=None) -> int:
    """Run the child: warm OVITO up, then answer jobs until stdin closes."""
    try:
        import ovito  # noqa: F401  (imported for its start-up cost, not its name)

        from .state import run_ptm
    except Exception as exc:  # pragma: no cover - depends on a broken install
        _emit({"fatal": f"{exc}"})
        return 1
    _emit({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = pickle.loads(base64.b64decode(line))
        except Exception as exc:  # pragma: no cover - only a corrupted pipe
            _emit({"fatal": f"unreadable job: {exc}"})
            return 1
        try:
            result = run_ptm(
                job["analysis"],
                job["colour"],
                progress=lambda stage, n_atoms=None: _emit(
                    {"stage": stage, "n_atoms": n_atoms}
                ),
                path=job.get("path"),
                frame_index=job.get("frame_index"),
            )
            with open(job["out"], "wb") as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            _emit({"done": True})
        except Exception as exc:
            _emit({"error": str(exc) or exc.__class__.__name__})
    return 0


class AnalysisWorker:
    """The parent's handle on the child process.

    One job runs at a time, submitted from the thread that owns the analysis;
    :meth:`stop` may be called from any other thread, and kills the child.
    """

    def __init__(self, python: str | None = None):
        self.python = python or sys.executable
        self.process: subprocess.Popen | None = None
        self.available = True  # cleared when the child proves unusable
        self.reason = ""  # why it is unavailable, for the diagnostics page
        self.busy = False
        # Reentrant: start() disables the worker from inside the lock when the
        # child will not start, and _disable takes the lock again.
        self._lock = threading.RLock()
        self._stopped = False

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        """Start the child and wait for it to say it is ready.

        Raises :class:`WorkerUnavailable` if it cannot be used, and remembers
        that, so the caller falls back once rather than on every run.
        """
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                return
            if not self.available:
                raise WorkerUnavailable(self.reason)
            try:
                self.process = subprocess.Popen(
                    [self.python, "-u", "-m", "ptmipf.webui.worker"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            except Exception as exc:
                self._disable(f"the analysis process could not be started: {exc}")
        # Outside the lock: the child imports OVITO before it answers, which
        # takes a moment, and stop() must not be blocked meanwhile.
        message = self._read()
        if message is None or "ready" not in message:
            detail = (message or {}).get("fatal", "it stopped before it was ready")
            self._disable(f"the analysis process is not usable here: {detail}")

    def _disable(self, reason: str) -> None:
        with self._lock:
            self.available = False
            self.reason = reason
            self.process = None
        raise WorkerUnavailable(reason)

    def stop(self) -> bool:
        """Kill the child, if it is working. True when something was stopped.

        The next job starts a fresh one; killing is the only way to interrupt
        OVITO, so there is nothing to salvage from this one.
        """
        with self._lock:
            process, busy = self.process, self.busy
            self._stopped = True
            self.process = None
        if process is None or process.poll() is not None:
            return False
        process.kill()
        try:
            process.wait(timeout=10)
        except Exception:  # pragma: no cover - a process that will not die
            pass
        return busy

    def close(self) -> None:
        with self._lock:
            process, self.process = self.process, None
        if process is not None and process.poll() is None:
            try:
                process.stdin.close()
                process.wait(timeout=5)
            except Exception:
                process.kill()

    # -- running a job --------------------------------------------------
    def _read(self) -> dict | None:
        """The next protocol line from the child, or None if it has gone."""
        process = self.process
        if process is None:
            return None
        while True:
            try:
                line = process.stdout.readline()
            except Exception:
                return None
            if not line:
                return None
            if line.startswith(PREFIX):
                try:
                    return json.loads(line[len(PREFIX) :])
                except ValueError:  # pragma: no cover - a truncated line
                    return None

    def run(self, analysis: dict, colour: dict, out: Path, on_stage=None):
        """Match one configuration in the child and return the result.

        Raises :class:`Cancelled` if :meth:`stop` killed it, ``ValueError``
        with the child's message if the analysis itself failed, and
        :class:`WorkerUnavailable` if the child could not be used at all, in
        which case the caller runs the analysis itself.
        """
        self.start()
        with self._lock:
            self._stopped = False
            self.busy = True
            process = self.process
        try:
            job = {"analysis": analysis, "colour": colour, "out": str(out)}
            payload = base64.b64encode(pickle.dumps(job, protocol=4)).decode()
            try:
                process.stdin.write(payload + "\n")
                process.stdin.flush()
            except Exception as exc:
                raise WorkerUnavailable(
                    f"the analysis process would not take the job: {exc}"
                ) from exc
            while True:
                message = self._read()
                if message is None:
                    # No answer: either we killed it, or it died on its own.
                    if self._stopped:
                        raise Cancelled("the analysis was stopped")
                    raise WorkerUnavailable("the analysis process stopped unexpectedly")
                if "stage" in message:
                    if on_stage:
                        on_stage(message["stage"], message.get("n_atoms"))
                    continue
                if message.get("done"):
                    with open(out, "rb") as handle:
                        return pickle.load(handle)
                if "error" in message:
                    raise ValueError(message["error"])
                if "fatal" in message:
                    raise WorkerUnavailable(message["fatal"])
        finally:
            with self._lock:
                self.busy = False
            try:
                Path(out).unlink()
            except OSError:
                pass


if __name__ == "__main__":  # pragma: no cover - entry point of the child
    raise SystemExit(main())
