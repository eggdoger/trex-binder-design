"""Per-worker wall-clock enforcement independent of controller decision waits."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from ..backends.process_management import terminate_owned_process_group


class WorkerRuntimeBudget:
    """Own exactly one subprocess group; only the main thread parses artifacts."""

    def __init__(self, proc: Any, *, seconds: float, output_dir: Path, candidate_id: str):
        self.proc = proc
        self.seconds = seconds
        self.pgid = proc.pid
        if self.pgid <= 1 or self.pgid == os.getpgrp() or os.getpgid(proc.pid) != self.pgid:
            raise RuntimeError("runtime budget requires a worker launched in its own session")
        self.leader_start_ticks = int(Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        self.output_dir = output_dir
        self.candidate_id = candidate_id
        self.started_at = time.time()
        self.started_monotonic = time.monotonic()
        self.stopping = False
        self.exhausted = False
        self.error: str | None = None
        self.ended_at: float | None = None
        self.elapsed_seconds: float | None = None
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._watch, name=f"worker-budget-{self.pgid}", daemon=True)
        self._record("armed")
        self.thread.start()

    def _record(self, state: str) -> None:
        path = self.output_dir / "worker_runtime_budget.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps({
            "schema_version": "binder.worker-runtime-budget.v1",
            "candidate_id": self.candidate_id, "worker_pid": self.proc.pid,
            "worker_pgid": self.pgid, "max_runtime_seconds": self.seconds,
            "worker_pid_start_ticks": self.leader_start_ticks,
            "started_at": self.started_at, "deadline_at": self.started_at + self.seconds,
            "ended_at": self.ended_at, "elapsed_seconds": self.elapsed_seconds,
            "state": state, "completion_reason": "runtime_budget_exhausted" if self.exhausted else None,
            "actual_return_code": self.proc.returncode, "termination_error": self.error,
        }, indent=2) + "\n")
        temp.replace(path)

    def _watch(self) -> None:
        try:
            remaining = max(0.0, self.seconds - (time.monotonic() - self.started_monotonic))
            if self.cancelled.wait(remaining):
                return
            leader_running = self.proc.poll() is None
            try:
                os.killpg(self.pgid, 0)
                group_present = True
            except ProcessLookupError:
                group_present = False
            if not leader_running and not group_present:
                return
            self.stopping = True
            try:
                self._record("stopping")
            except OSError:
                pass  # A logging failure must not disable the runtime bound.
            terminate_owned_process_group(self.proc, pgid=self.pgid, leader_start_ticks=self.leader_start_ticks)
            self.exhausted = True
            self.ended_at = time.time()
            self.elapsed_seconds = time.monotonic() - self.started_monotonic
            self._record("bounded_batch_completed")
        except Exception as exc:  # surfaced by supervision; never reuse this GPU silently
            self.error = f"{type(exc).__name__}: {exc}"
            try:
                self._record("termination_failed")
            except OSError:
                pass
        finally:
            self.finished.set()

    def finish(self) -> None:
        """Cancel an unused timer, or wait for its already-started group cleanup."""
        self.cancelled.set()
        self.thread.join(timeout=21)
        if self.thread.is_alive():
            raise RuntimeError("runtime-budget process-group cleanup did not finish")
        if self.error:
            raise RuntimeError(self.error)
        if self.ended_at is None:
            was_running = self.proc.poll() is None
            terminate_owned_process_group(self.proc, pgid=self.pgid, leader_start_ticks=self.leader_start_ticks)
            self.ended_at = time.time()
            self.elapsed_seconds = time.monotonic() - self.started_monotonic
            self._record("campaign_stopped" if was_running else "backend_exited")


def worker_runtime_budget(proc: Any) -> WorkerRuntimeBudget | None:
    value = getattr(proc, "_trex_runtime_budget", None)
    return value if isinstance(value, WorkerRuntimeBudget) else None


def arm_worker_runtime_budget(proc: Any, *, seconds: int, output_dir: Path, candidate_id: str) -> None:
    """Attach the watchdog immediately after launch, before archive/queue work."""
    try:
        proc._trex_runtime_budget = WorkerRuntimeBudget(
            proc, seconds=seconds, output_dir=output_dir, candidate_id=candidate_id)
    except Exception:
        terminate_owned_process_group(proc, pgid=proc.pid)
        raise


__all__ = ["WorkerRuntimeBudget", "arm_worker_runtime_budget", "worker_runtime_budget"]
