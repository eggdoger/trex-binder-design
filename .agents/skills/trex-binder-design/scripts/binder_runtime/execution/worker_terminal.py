"""Best-effort process exit evidence, separate from parsed scientific results.

These records describe the Popen leader only. They do not certify descendant
cleanup, GPU idleness, result validity, or the success of a design method.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import tempfile
from typing import Any


@dataclass(frozen=True)
class WorkerTerminalBinding:
    candidate_id: str
    method_family: str
    worker_pid: int
    worker_pgid: int
    worker_pid_start_ticks: int
    worker_host: str
    slurm_job_id: str | None
    worker_slot: str
    gpu_id: str
    tick_id: str
    attempt: int
    output_dir: str
    launched_at: float


def _audit_error(slot: Any, exc: Exception) -> None:
    try:
        print(
            f"  [worker_terminal_journal_error][gpu={getattr(slot, 'gpu_id', '?')}] "
            f"{type(exc).__name__}: {exc} — process exit evidence is unavailable",
            flush=True,
        )
    except Exception:
        # A broken/full log sink must not interrupt worker cleanup or salvage.
        # The missing terminal journal continues to mean unknown exit evidence.
        pass


def bind_worker_terminal_journal(slot: Any, *, attempt: int) -> None:
    """Capture birth identity while the dispatched Popen child is still owned.

    A zombie child still has a valid birth identity until it is reaped. Unlike a
    liveness check, recording that identity must not reject state Z. No PID or
    start tick is guessed if /proc is unavailable or the child was already reaped.
    """
    slot.terminal_binding = None
    slot.terminal_process = None
    slot.terminal_journal_path = None
    try:
        if slot.proc.returncode is not None:
            raise ValueError("worker was already reaped before birth identity capture")
        pid = int(slot.proc.pid)
        pgid = os.getpgid(pid)
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        ticks = int(fields[19])
        if slot.proc.returncode is not None:
            raise ValueError("worker was reaped during birth identity capture")
        if pid <= 1 or pgid != pid or ticks <= 0:
            raise ValueError("worker does not have a captured isolated child identity")
        slot.terminal_process = slot.proc
        slot.terminal_binding = WorkerTerminalBinding(
            candidate_id=str(slot.cand.candidate_id),
            method_family=str(slot.cand.method_family),
            worker_pid=pid, worker_pgid=pgid, worker_pid_start_ticks=ticks,
            worker_host=socket.gethostname(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID") or None,
            worker_slot=str(slot.slot_id), gpu_id=str(slot.gpu_id),
            tick_id=str(slot.tick_id), attempt=int(attempt),
            output_dir=str(Path(slot.out_dir).resolve()),
            launched_at=float(slot.launched_at),
        )
    except Exception as exc:  # Audit failure must not orphan or reschedule a child.
        _audit_error(slot, exc)


def record_worker_terminal(slot: Any, *, observation: str) -> Path | None:
    """Persist a real reaped return code; never consume parser sentinel values.

    observed_end_at is when the controller saw the exit, not the kernel exit
    instant. If cleanup did not produce a return code, no terminal success is
    invented. Writes are atomic; each birth identity gets its own file so retries
    in a reused output directory cannot overwrite a previous process's evidence.
    """
    if getattr(slot, "terminal_journal_path", None) is not None:
        return slot.terminal_journal_path
    temporary: Path | None = None
    try:
        binding = getattr(slot, "terminal_binding", None)
        if not isinstance(binding, WorkerTerminalBinding):
            raise ValueError("missing captured worker birth identity")
        if (slot.proc is not getattr(slot, "terminal_process", None)
                or slot.proc.pid != binding.worker_pid
                or str(slot.cand.candidate_id) != binding.candidate_id
                or str(Path(slot.out_dir).resolve()) != binding.output_dir):
            raise ValueError("worker slot no longer matches captured process binding")
        rc = slot.proc.poll()
        if type(rc) is not int:
            raise ValueError("worker has no observed terminal return code")
        directory = Path(binding.output_dir)
        final = directory / (
            f"worker_terminal.{binding.worker_pid}.{binding.worker_pid_start_ticks}.json"
        )
        payload = {
            "schema_version": "binder.worker-terminal.v1",
            **asdict(binding),
            "actual_return_code": rc,
            "observed_end_at": datetime.now(timezone.utc).isoformat(),
            "observation": observation,
            "incremental": False,
            "return_code_source": "subprocess.Popen.poll",
            "process_scope": "worker_leader_only",
            "process_group_quiescence_verified": False,
        }
        # A repeat observation never replaces the first terminal timestamp.
        if final.exists():
            existing = json.loads(final.read_text())
            expected = {**asdict(binding), "actual_return_code": rc}
            if any(existing.get(key) != value for key, value in expected.items()):
                raise ValueError("conflicting terminal journal already exists")
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory,
                prefix=final.name + ".", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, final)
            temporary = None
        slot.terminal_journal_path = final
        return final
    except Exception as exc:  # No change to result retention or scheduling policy.
        _audit_error(slot, exc)
        return None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
