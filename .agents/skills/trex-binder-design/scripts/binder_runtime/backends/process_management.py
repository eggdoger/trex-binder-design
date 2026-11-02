"""Shared subprocess lifecycle and environment management for backends."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkerLaunch:
    """A fully resolved, inspectable command for one backend worker."""

    argv: tuple[str, ...]
    output_dir: Path
    environment: dict[str, str]
    cwd: Path | None = None


def popen_kwargs() -> dict[str, Any]:
    """Launch every worker in an isolated process group."""

    return {"start_new_session": True}


def terminate_process_group(proc: Any) -> None:
    """Terminate a worker tree, escalating to SIGKILL after a grace period."""

    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError, AttributeError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError, AttributeError):
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def terminate_owned_process_group(
    proc: Any,
    *,
    pgid: int,
    leader_start_ticks: int | None = None,
    term_grace_seconds: float = 15.0,
    kill_wait_seconds: float = 5.0,
) -> None:
    """Stop an isolated group whose leader is this exact launched child.

    Keep the captured PGID after the leader exits: grandchildren may ignore
    TERM. Never derive a replacement PGID or signal the controller's group.
    Failure to establish ownership or to stop live members is an error.
    """
    if pgid <= 1 or pgid != proc.pid or pgid == os.getpgrp():
        raise RuntimeError("refusing to signal a process group not owned by this worker")
    try:
        if os.getpgid(proc.pid) != pgid:
            raise RuntimeError("worker no longer leads its isolated process group")
        if leader_start_ticks is not None:
            fields = Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[19]) != leader_start_ticks:
                raise RuntimeError("worker PID has been reused by another process")
    except ProcessLookupError:
        # A child can exit between observation and TERM. The PGID captured at
        # launch remains reserved while its descendants exist. Only callers
        # holding that original leader's start identity may clean this group.
        if leader_start_ticks is None:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            raise RuntimeError("cannot establish ownership after worker leader exit")
    except FileNotFoundError:
        # The leader exited between getpgid and reading stat; the captured
        # start identity still belongs to the original isolated group.
        if leader_start_ticks is None:
            raise

    def group_present() -> bool:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        return True

    def only_zombies_remain() -> bool:
        # Linux can retain orphan zombies until its init reaps them; these do
        # not execute or hold GPU memory. Permission/read failures fail closed.
        procfs = Path("/proc")
        if not procfs.is_dir():
            return False
        for entry in procfs.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                return False
            if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                return False
        return True

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait(timeout=kill_wait_seconds)
        return
    until = time.monotonic() + term_grace_seconds
    while group_present() and time.monotonic() < until:
        proc.poll()
        if proc.returncode is not None and only_zombies_remain():
            return
        time.sleep(min(0.05, max(0.0, until - time.monotonic())))
    if group_present():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    until = time.monotonic() + kill_wait_seconds
    while group_present() and time.monotonic() < until:
        proc.poll()
        if proc.returncode is not None and only_zombies_remain():
            return
        time.sleep(min(0.05, max(0.0, until - time.monotonic())))
    proc.poll()
    if group_present() and not only_zombies_remain():
        raise RuntimeError(f"worker process group {pgid} remains live after SIGKILL")
    if proc.returncode is None:
        proc.wait(timeout=max(0.001, until - time.monotonic()))


def isolated_subprocess_environment(env_root: Path) -> dict[str, str]:
    """Build an environment that protects backend-bundled CUDA libraries.

    Cluster modules can inject a different CUDA runtime through
    ``LD_LIBRARY_PATH``. BindCraft, BoltzGen, and ProteinMPNN ship their own
    runtime libraries, so backend processes receive only their environment's
    library directory while retaining all unrelated campaign variables.
    """

    env = os.environ.copy()
    env["PATH"] = f"{env_root}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{env_root}/lib"
    if not env.get("MPLCONFIGDIR"):
        temporary_root = Path(env.get("TMPDIR", "/tmp"))
        env["MPLCONFIGDIR"] = str(
            temporary_root / f"v7_mplconfig_{os.getuid()}"
        )
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


__all__ = [
    "WorkerLaunch",
    "isolated_subprocess_environment",
    "popen_kwargs",
    "terminate_process_group",
    "terminate_owned_process_group",
]
