"""Run the skill's embedded deterministic workflow with decisions authored in Codex.

This adapter never creates a network model client. Its bounded synchronous
mailbox services existing workers on the controller thread while a decision
is pending. A scheduler allocation still enforces hard resource deadlines.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import fcntl
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import sys
from typing import Any, Sequence
import uuid

from .mailbox import (DEFAULT_DECISION_TIMEOUT_SECONDS, MAX_DECISION_TIMEOUT_SECONDS, DecisionAborted, Mailbox,
                      validate_decision_timeout)

BUNDLE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = BUNDLE_ROOT / "assets" / "runtime-manifest.json"
_MODEL_ENV = ("TREX_MODEL_DIGEST", "TREX_QWEN_MODEL_PATH", "TREX_MODEL_MANIFEST",
              "TREX_CROSS_CAMPAIGN_MEMORY", "TREX_REPO_ROOT")


def _bundle_info() -> dict[str, Any]:
    """Verify the embedded execution bundle, independently of any checkout."""
    import binder_runtime
    expected_package = BUNDLE_ROOT / "scripts" / "binder_runtime"
    if Path(binder_runtime.__file__).resolve().parent != expected_package.resolve():
        raise ValueError("another binder_runtime package was imported; use this skill's entrypoint")
    if (not MANIFEST.is_file() or MANIFEST.is_symlink()
            or MANIFEST.parent.is_symlink() or MANIFEST.stat().st_size > 4 * 1024 * 1024):
        raise ValueError("missing or invalid bundled runtime manifest")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "binder.runtime-bundle.v1":
        raise ValueError("unsupported bundled runtime manifest")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("runtime manifest has no file hashes")
    required = {"scripts/binder_runtime/controller.py", "scripts/binder_runtime/live_tick.py",
                "scripts/binder_runtime/decisions.py", "scripts/trex_codex/runner.py",
                "scripts/trex_codex/mailbox.py"}
    if not required.issubset(files):
        raise ValueError("runtime manifest omits required execution files")
    for name, digest in files.items():
        if not isinstance(name, str):
            raise ValueError("invalid runtime manifest file entry")
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("invalid runtime manifest file entry")
        path = BUNDLE_ROOT / relative
        if (any((BUNDLE_ROOT / Path(*relative.parts[:index])).is_symlink()
                for index in range(1, len(relative.parts) + 1))
                or not path.resolve().is_relative_to(BUNDLE_ROOT.resolve())
                or not path.is_file() or _hash(path) != digest):
            raise ValueError(f"bundled runtime integrity mismatch: {name}")
    script_paths = list((BUNDLE_ROOT / "scripts").rglob("*"))
    if any(path.is_symlink() for path in script_paths):
        raise ValueError("embedded runtime must not contain symlink files or directories")
    actual = {str(path.relative_to(BUNDLE_ROOT)) for path in script_paths
              if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    if not actual.issubset(files):
        raise ValueError("runtime manifest omits embedded modules or resources")
    return {"root": str(BUNDLE_ROOT), "manifest_sha256": _hash(MANIFEST),
            "upstream_commit": payload.get("upstream_commit"), "file_count": len(files),
            "schema_version": payload["schema_version"]}


def _workers(value: Sequence[str] | str) -> list[str]:
    workers = value.split(",") if isinstance(value, str) else list(value)
    workers = [str(item).strip() for item in workers]
    if (not workers or any(not item or any(c in item for c in ",\r\n") for item in workers)
            or len(set(workers)) != len(workers)):
        raise ValueError("worker_gpus must contain unique non-empty allocation GPU identifiers")
    return workers


def _model_label(model: str | None) -> str:
    if model is None:
        return "codex-external:unknown"
    model = model.strip()
    if not model or any(char in model for char in "\r\n"):
        raise ValueError("model label must be a non-empty single line")
    return model if model.startswith("codex-external:") else "codex-external:" + model


def init_campaign(out: str | Path, *, target: str,
                  archive_root: str, worker_gpus: Sequence[str] | str,
                  hours: float = 1.0, name: str | None = None,
                  asset_root: str | None = None, target_constraint: str | None = None,
                  target_pdb: str | None = None, families: Sequence[str] | str | None = None,
                  seed: int = 0, model: str | None = None) -> Path:
    """Create a normal campaign contract with every listed GPU used as a worker."""
    from binder_runtime.campaign.template import campaign_template
    import yaml
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("hours must be finite and positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    workers = _workers(worker_gpus)
    def absolute(value):
        return str(Path(value).expanduser().resolve()) if value is not None else None
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty name")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("name must be non-empty")
    payload = campaign_template(name=name or target, target=target,
        archive_root=absolute(archive_root), asset_root=absolute(asset_root),
        target_constraint=absolute(target_constraint), target_pdb=absolute(target_pdb),
        max_wall_hours=hours, total_gpus=len(workers))
    payload["run"].update(worker_gpus=workers, seed=seed)
    if families is not None:
        selected = families.split(",") if isinstance(families, str) else list(families)
        if not selected or any(not isinstance(item, str) or not item.strip() for item in selected):
            raise ValueError("families must contain at least one backend family")
        from binder_runtime.campaign.config import _effective_families
        selected = [item.strip() for item in selected]
        if len(selected) != len(set(selected)):
            raise ValueError("families must not contain duplicates")
        _effective_families(tuple(selected))
        payload["run"]["enabled_families"] = selected
    payload["llm"] = {"base_url": "codex://mailbox", "model": _model_label(model)}
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    path = Path(out).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
    return path


@contextmanager
def _environment(values: dict[str, str]):
    keys = set(values) | set(_MODEL_ENV)
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in _MODEL_ENV:
            os.environ.pop(key, None)
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _mailbox_lease(root: str | Path):
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path / ".codex-run.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"mailbox already has an active controller: {path}") from exc
        yield path
    finally:
        os.close(descriptor)


def _campaign(config: str | Path, model=None):
    from binder_runtime.campaign.config import load_campaign
    with _environment({}):
        campaign = load_campaign(config)
    current_model = campaign.config.llm.model
    effective_model = model if model is not None else (
        current_model if current_model.startswith("codex-external:") else None)
    effective = replace(campaign.config,
        llm=replace(campaign.config.llm, base_url="codex://mailbox", model=_model_label(effective_model)),
        backends=replace(campaign.config.backends, **{key: None for key in
            ("qwen_model_path", "qwen_model_manifest") if hasattr(campaign.config.backends, key)}))
    return replace(campaign, config=effective), BUNDLE_ROOT


def _archive_resume_error(archive: Path, campaign=None) -> str | None:
    if not archive.exists():
        return None
    if not archive.is_dir():
        return "archive path exists and is not a directory"
    entries = [path for path in archive.iterdir()
               if path.name not in {".campaign_run.lock", ".controller.lock"}]
    if not entries:
        return None
    expected_target = ({"target_pdb_sha256": _hash(campaign.target.pdb_path),
                        "target_constraint_sha256": _hash(campaign.target.config_path)}
                       if campaign is not None else None)
    expected_bundle = _hash(MANIFEST) if campaign is not None and MANIFEST.is_file() else None
    valid_provenance = False
    for path in archive.glob("codex_provenance_*.json"):
        try:
            if path.is_symlink() or path.stat().st_size > 1024 * 1024:
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if (isinstance(record, dict)
                    and record.get("schema_version") == "binder.codex-provenance.v1"
                    and record.get("provider") == "external Codex conversation"
                    and isinstance(record.get("session_id"), str) and record["session_id"]
                    and isinstance(record.get("model_label"), str)
                    and record["model_label"].startswith("codex-external:")
                    and "model_content_sha256" in record and record["model_content_sha256"] is None
                    and isinstance(record.get("runtime_bundle"), dict)
                    and isinstance(record["runtime_bundle"].get("manifest_sha256"), str)
                    and len(record["runtime_bundle"]["manifest_sha256"]) == 64
                    and isinstance(record.get("inputs"), dict)
                    and all(isinstance(record["inputs"].get(key), str)
                            and len(record["inputs"][key]) == 64
                            for key in ("campaign_sha256", "target_pdb_sha256", "target_constraint_sha256"))):
                if campaign is not None and record["inputs"]["campaign_sha256"] != campaign.source_sha256:
                    return ("archive provenance identifies a different campaign configuration; "
                            "use a fresh archive or an explicitly reviewed migration before resuming")
                if expected_target is not None and any(record["inputs"][key] != digest
                                                       for key, digest in expected_target.items()):
                    return ("archive provenance identifies a different target structure or constraint; "
                            "use a fresh archive to avoid resuming pending work for another target")
                if expected_bundle is not None and record["runtime_bundle"]["manifest_sha256"] != expected_bundle:
                    return ("archive uses a different embedded runtime bundle; use a fresh archive "
                            "or an explicitly reviewed migration before resuming pending work")
                valid_provenance = True
        except (OSError, ValueError, RecursionError):
            continue
    if valid_provenance:
        return None
    return ("nonempty archive has no valid external-Codex standalone provenance; use a fresh archive "
            "or an explicitly reviewed migration. Automatic migration of pending work from another runtime is unsupported")


def _check(campaign, *, strict: bool) -> dict[str, Any]:
    from binder_runtime.campaign.preflight import preflight_campaign, PreflightReport
    from binder_runtime.validation import Check
    with _environment(campaign.environment()):
        original = preflight_campaign(campaign, require_backends=strict, require_model=False,
                                      verify_backend_revisions=strict)
    checks = [item for item in original.checks if item.name != "LLM endpoint"]
    checks.append(Check("external Codex decisions", "ok", "file mailbox; no network model client"))
    checks.append(Check("fresh Codex decisions each tick", "fail" if campaign.config.policy.evidence_skip else "ok",
                        "policy.evidence_skip must be false" if campaign.config.policy.evidence_skip else "evidence_skip=false"))
    resume_error = _archive_resume_error(campaign.config.run.archive_root, campaign)
    checks.append(Check("external-Codex archive compatibility", "fail" if resume_error else "ok",
                        resume_error or "fresh archive or prior valid embedded-runtime provenance"))
    try:
        bundle = _bundle_info()
        checks.append(Check("embedded runtime integrity", "ok", bundle["manifest_sha256"]))
    except (OSError, ValueError, RecursionError) as exc:
        checks.append(Check("embedded runtime integrity", "fail", str(exc)))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        allowed = {value.strip() for value in visible.split(",") if value.strip()}
        assigned = set(campaign.config.run.worker_gpus)
        valid = assigned.issubset(allowed) and allowed not in ({"-1"}, {"NoDevFiles"})
        checks.append(Check("allocation GPU mapping", "ok" if valid else "fail",
                            f"workers={sorted(assigned)} CUDA_VISIBLE_DEVICES={visible}"))
    report = PreflightReport(tuple(checks))
    return {"schema_version": "binder.codex-check.v1", "ok": report.ok,
            "resolved_campaign": campaign.as_dict(), "preflight": report.as_dict(),
            "decision_provider": "codex-external", "local_model_required": False,
            "synchronous_wait_limit_seconds": MAX_DECISION_TIMEOUT_SECONDS,
            "default_decision_timeout_seconds": DEFAULT_DECISION_TIMEOUT_SECONDS,
            "worker_supervision_during_wait": True,
            "hard_deadline": "enforce with scheduler allocation; controller drains existing workers"}


def check_campaign(config: str | Path, *, strict=False, model=None) -> dict[str, Any]:
    campaign, _ = _campaign(config, model=model)
    return _check(campaign, strict=strict)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    from .mailbox import _atomic_json
    _atomic_json(path, payload, limit=4 * 1024 * 1024)


def _write_outcome(path: Path, payload: dict[str, Any]) -> None:
    """Keep the original failure visible if durable outcome storage also fails."""
    try:
        _write_json(path, payload)
    except Exception as exc:
        try:
            print(f"trex-skill: outcome audit unavailable: {type(exc).__name__}: {exc}; "
                  f"controller exit={payload.get('exit_code')}, error={payload.get('controller_error')!r}",
                  file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass  # A full/broken scheduler log must not mask the original error.
        if payload.get("exit_code") in (None, 0):
            raise RuntimeError("controller finished but its outcome audit could not be saved") from exc


def _runtime_environment() -> dict[str, Any]:
    """Record this controller environment without importing scientific packages."""
    packages = {}
    errors = {}
    for name in ("biopython", "PyYAML", "numpy"):
        try:
            packages[name] = metadata.version(name)
        except (metadata.PackageNotFoundError, OSError, ValueError) as exc:
            packages[name] = None
            errors[name] = f"{type(exc).__name__}: {exc}"
    return {
        "python": {"version": sys.version, "executable": sys.executable,
                   "implementation": platform.python_implementation()},
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": packages, "package_metadata_errors": errors,
    }


def _publish_active_session(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".codex-active-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _request_campaign_control(archive_root: str | Path, action: str) -> dict[str, Any]:
    """Publish an idempotent control signal bound to the held session lease."""
    if action not in {"stop", "drain"}:
        raise ValueError("unsupported campaign control action")
    archive = Path(archive_root).expanduser().resolve()
    active_path = archive / "codex_active_session.json"
    if not archive.is_dir() or not active_path.is_file() or active_path.is_symlink():
        raise ValueError("archive has no active Codex controller session")
    descriptor = os.open(archive / ".campaign_run.lock", os.O_RDWR | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError("archive has no active Codex controller session")
        from .mailbox import _atomic_json, _read_json
        active = _read_json(active_path, 16384)
        session_id = active.get("session_id")
        if (active.get("schema_version") != "binder.codex-active.v1"
                or not isinstance(session_id, str) or len(session_id) != 32
                or any(char not in "0123456789abcdef" for char in session_id)):
            raise ValueError("invalid active controller session metadata")
        lease_metadata = os.read(descriptor, 4096).decode("utf-8")
        if f"session_id={session_id}\n" not in lease_metadata:
            raise ValueError("archive has no matching active Codex controller session")
        path = archive / f"codex_{action}_{session_id}.json"
        schema = f"binder.codex-{action}.v1"
        request = {"schema_version": schema, "session_id": session_id,
                   "requested_at": datetime.now(timezone.utc).isoformat(),
                   "status": f"{action}_requested", "archive": str(archive),
                   "effect": ("stop new dispatch and drain running workers" if action == "stop" else
                              "stop generation and finish existing canonical evaluations")}
        try:
            _atomic_json(path, request, limit=16384)
        except ValueError:
            if not path.is_file():
                raise
            request = _read_json(path, 16384)
            if request.get("session_id") != session_id or request.get("schema_version") != schema:
                raise ValueError(f"existing {action} request is invalid")
        return request
    finally:
        os.close(descriptor)


def stop_campaign(archive_root: str | Path) -> dict[str, Any]:
    """Stop all new dispatch, then reap running workers."""
    return _request_campaign_control(archive_root, "stop")


def drain_campaign(archive_root: str | Path) -> dict[str, Any]:
    """Stop generation and continue canonical evaluations in the active session."""
    return _request_campaign_control(archive_root, "drain")


def run_campaign(config: str | Path, *, mailbox_root=None,
                 timeout_seconds=DEFAULT_DECISION_TIMEOUT_SECONDS,
                 model=None, maximum_iterations=None, evaluation_only=False) -> int:
    """Start embedded execution; return 75 for unavailable or invalid decisions."""
    timeout_seconds = validate_decision_timeout(timeout_seconds)
    campaign, _ = _campaign(config, model=model)
    checked = _check(campaign, strict=True)
    if not checked["preflight"]["ok"]:
        raise ValueError("campaign preflight failed: " + json.dumps(checked["preflight"]))
    import binder_runtime.controller as controller
    from binder_runtime.campaign.artifacts import write_campaign_artifacts
    from binder_runtime.campaign.run_lock import campaign_run_lock
    from binder_runtime.campaign.runtime import ControllerLoopConfig
    loop = ControllerLoopConfig(maximum_iterations=maximum_iterations)
    archive = campaign.config.run.archive_root
    session_id = uuid.uuid4().hex
    with campaign_run_lock(archive), _environment(campaign.environment()), ExitStack() as resources:
        # Bind the held lease before publishing active metadata. A stale active
        # file left by SIGKILL cannot identify the next launch as its old session.
        with (archive / ".campaign_run.lock").open("a", encoding="utf-8") as lease:
            lease.write(f"session_id={session_id}\n")
        (archive / "codex_active_session.json").unlink(missing_ok=True)
        # Recheck under the archive lease before creating mailbox or artifacts.
        bundle = _bundle_info()
        resume_error = _archive_resume_error(archive, campaign)
        if resume_error:
            raise ValueError(resume_error)
        leased_mailbox = resources.enter_context(_mailbox_lease(mailbox_root or archive / "codex_mailbox"))
        mailbox = Mailbox(leased_mailbox,
                          timeout_seconds=timeout_seconds, session_id=session_id,
                          stop_path=archive / f"codex_stop_{session_id}.json",
                          drain_path=archive / f"codex_drain_{session_id}.json",
                          mode_path=archive / "codex_execution_mode.json",
                          initial_evaluation_only=evaluation_only)
        evaluation_only_at_start = mailbox.evaluation_only_requested()
        artifacts = write_campaign_artifacts(campaign)
        provenance = {
            "schema_version": "binder.codex-provenance.v1", "session_id": session_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "provider": "external Codex conversation", "model_label": campaign.config.llm.model,
            "model_content_sha256": None, "token_usage": None,
            "publication_model_parity": False,
            "runtime_bundle": bundle,
            "runtime_environment": _runtime_environment(),
            "backend_verification": {
                "preflight": checked["preflight"],
                "resolved_paths": campaign.runtime_paths().as_dict(),
                "gpu_execution_tested": False,
            },
            "transport": {"runner_sha256": _hash(Path(__file__)),
                          "mailbox_sha256": _hash(Path(__file__).with_name("mailbox.py"))},
            "inputs": {"campaign_sha256": campaign.source_sha256,
                       "input_path": str(artifacts.input_path),
                       "target_pdb_sha256": _hash(campaign.target.pdb_path),
                       "target_constraint_sha256": _hash(campaign.target.config_path),
                       "resolved_sha256": _hash(artifacts.resolved_path),
                       "resolved_path": str(artifacts.resolved_path)},
            "worker_gpus": list(campaign.config.run.worker_gpus),
            "requested_execution_mode": "evaluation_only" if evaluation_only_at_start else "adaptive",
            "max_wall_hours": campaign.config.run.max_wall_hours,
            "mailbox_root": str(mailbox.root), "decision_timeout_seconds": timeout_seconds,
            "decision_wait_supervision": "same-thread worker supervision; no new dispatch; immutable prompt snapshot",
            "limitations": ["external Codex must service decisions before their deadline",
                            "wall limit stops new rounds; existing workers drain",
                            "scheduler must enforce hard allocation deadline"],
        }
        _write_json(archive / f"codex_provenance_{session_id}.json", provenance)
        print(json.dumps({"session_id": session_id, "mailbox": str(mailbox.root),
                          "model": campaign.config.llm.model}), flush=True)
        exit_code = None
        controller_error = None
        active_path = archive / "codex_active_session.json"
        _publish_active_session(active_path, {"schema_version": "binder.codex-active.v1",
                                            "session_id": session_id, "pid": os.getpid()})
        try:
            exit_code = int(controller.main(campaign.controller_argv(),
                runtime_paths=campaign.runtime_paths(), loop_config=loop,
                decision_provider=mailbox) or 0)
            return exit_code
        except BaseException as exc:
            controller_error = f"{type(exc).__name__}: {exc}"
            exit_code = (130 if isinstance(exc, KeyboardInterrupt) else
                         exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 2)
            raise
        finally:
            active_path.unlink(missing_ok=True)
            records = []
            for request_file in sorted(mailbox.root.glob("*.request.json")):
                record = {"file": request_file.name}
                try:
                    if request_file.is_symlink() or request_file.stat().st_size > 4 * 1024 * 1024:
                        raise ValueError("request file is a symlink or exceeds audit size limit")
                    record["request_sha256"] = _hash(request_file)
                    request = json.loads(request_file.read_text(encoding="utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("request is not an object")
                    if request.get("session_id") != session_id:
                        continue
                    record["request_id"] = request.get("request_id")
                    response_file = request_file.with_name(request_file.name.replace(".request.json", ".response.json"))
                    if response_file.is_symlink() or (response_file.exists() and response_file.stat().st_size > 1024 * 1024):
                        raise ValueError("response file is a symlink or exceeds audit size limit")
                    record["response_sha256"] = _hash(response_file) if response_file.is_file() else None
                except (OSError, ValueError, RecursionError) as exc:
                    # Transport corruption must not mask the controller's
                    # fail-closed exit or prevent an outcome audit being saved.
                    record.update(audit_error=f"{type(exc).__name__}: {exc}", session_verified=False)
                records.append(record)
            _write_outcome(archive / f"codex_outcome_{session_id}.json", {
                "schema_version": "binder.codex-outcome.v1", "session_id": session_id,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": exit_code, "controller_error": controller_error,
                "stop_requested": mailbox.stop_path.is_file(),
                "drain_requested": mailbox.drain_path.is_file(),
                "execution_mode_file": str(mailbox.mode_path),
                "evaluation_only_requested_at_start": evaluation_only_at_start,
                "evaluation_only_state_file": str(archive / "evaluation_only_state.json"),
                "decisions": records})


__all__ = ["init_campaign", "check_campaign", "run_campaign", "stop_campaign", "drain_campaign"]
