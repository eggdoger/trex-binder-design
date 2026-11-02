"""Bounded, local-file decision transport; no network or model client is used.

The controller writes immutable requests. Codex reads the prompt as task data,
authors a JSON decision and submits it through ``submit_response``. Missing,
cancelled, corrupt, or late answers terminate the decision via DecisionAborted,
which deliberately bypasses upstream ``except Exception`` fallback handlers.

Run these helpers on the same Linux host as the controller (including over SSH).
Atomic links and advisory flock serialize cooperating clients. This is an audit
protocol, not a security boundary against an owner who rewrites the mailbox.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
import uuid


PROTOCOL_VERSION = 1
DEFAULT_DECISION_TIMEOUT_SECONDS = 900.0
MAX_DECISION_TIMEOUT_SECONDS = 3600.0
WAIT_SERVICE_INTERVAL_SECONDS = 1.0
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
_ID = re.compile(r"[0-9a-f]{32}\Z")
_STATES = {"waiting", "accepted", "consumed", "cancelled", "expired"}
_BINDINGS = ("version", "request_id", "session_id", "role", "prompt_hash")


from binder_runtime.decisions import DecisionAborted, DecisionResponse, EvaluationOnlyRequested


class _LockWaitExpired(TimeoutError):
    """A short lock attempt timed out; distinct from failures inside the lock."""


def validate_decision_timeout(value: float) -> float:
    """Share one bounded, finite timeout contract across Python and CLI entrypoints."""
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or not 0 < value <= MAX_DECISION_TIMEOUT_SECONDS):
        raise ValueError("timeout_seconds must be finite and in (0, 3600]")
    return float(value)


def _json_bytes(value: Any, limit: int) -> bytes:
    try:
        data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("payload must be finite JSON data") from exc
    if len(data) > limit:
        raise ValueError(f"payload exceeds {limit} bytes")
    return data


def _read_json(path: Path, limit: int) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("mailbox files must not be symlinks")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"mailbox file exceeds {limit} bytes")
    try:
        value = json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(
            ValueError("non-finite JSON value")))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid mailbox JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("mailbox envelope must be a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any], *, replace: bool = False,
                 limit: int = MAX_REQUEST_BYTES) -> None:
    """Publish a complete file; immutable files are never overwritten."""
    data = _json_bytes(value, limit)
    fd, name = tempfile.mkstemp(prefix=".mailbox-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise ValueError(f"immutable mailbox file already exists: {path.name}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _root(root: str | Path, *, create: bool = False) -> Path:
    path = Path(root).expanduser().absolute()
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir():
        raise ValueError("mailbox directory does not exist")
    return path


def _paths(root: Path, request_id: str) -> tuple[Path, Path, Path, Path]:
    if not isinstance(request_id, str) or not _ID.fullmatch(request_id):
        raise ValueError("request ID must be 32 lowercase hexadecimal characters")
    return tuple(root / f"{request_id}.{suffix}" for suffix in
                 ("request.json", "response.json", "status.json", "lock"))


@contextmanager
def _locked(path: Path, deadline_monotonic: float | None = None):
    deadline = time.monotonic() + 5 if deadline_monotonic is None else deadline_monotonic
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    acquired = False
    try:
        while not acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _LockWaitExpired("mailbox lock deadline elapsed")
                time.sleep(min(0.01, remaining))
        yield
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _prompt_content(request: dict[str, Any]) -> dict[str, Any]:
    return {key: request[key] for key in
            ("role", "system", "messages", "max_tokens", "temperature")}


def _hash(value: Any, limit: int = MAX_REQUEST_BYTES) -> str:
    return hashlib.sha256(_json_bytes(value, limit)).hexdigest()


def _load_request(path: Path, request_id: str) -> dict[str, Any]:
    request = _read_json(path, MAX_REQUEST_BYTES)
    if type(request.get("version")) is not int or request["version"] != PROTOCOL_VERSION:
        raise ValueError("unsupported mailbox protocol version")
    if request.get("request_id") != request_id:
        raise ValueError("request ID mismatch")
    if not isinstance(request.get("session_id"), str) or not request["session_id"]:
        raise ValueError("request session missing")
    if request.get("role") not in {"planner", "supervisor"}:
        raise ValueError("unsupported decision role")
    try:
        if request.get("prompt_hash") != _hash(_prompt_content(request)):
            raise ValueError("request prompt hash mismatch")
        created = request["created_unix"]
        deadline = request["deadline_unix"]
        if (isinstance(created, bool) or isinstance(deadline, bool) or
                not math.isfinite(created) or not math.isfinite(deadline) or
                deadline <= created):
            raise ValueError("invalid request deadline")
    except (KeyError, TypeError) as exc:
        raise ValueError("incomplete mailbox request") from exc
    return request


def _status(path: Path, request: dict[str, Any]) -> dict[str, Any]:
    state = _read_json(path, 16384)
    if any(state.get(key) != request.get(key) for key in _BINDINGS):
        raise ValueError("status binding mismatch")
    if state.get("status") not in _STATES:
        raise ValueError("invalid request status")
    return state


def _write_status(path: Path, request: dict[str, Any], status: str,
                  reason: str | None = None) -> dict[str, Any]:
    state = {key: request[key] for key in _BINDINGS}
    state.update(status=status, status_updated_at=_utc(time.time()), reason=reason)
    _atomic_json(path, state, replace=True, limit=16384)
    return state


def _expire_if_due(path: Path, request: dict[str, Any],
                   state: dict[str, Any]) -> dict[str, Any]:
    if state["status"] in {"waiting", "accepted"} and time.time() >= request["deadline_unix"]:
        return _write_status(path, request, "expired", "decision deadline elapsed")
    return state


def read_request(root: str | Path, request_id: str) -> dict[str, Any]:
    """Read one verified request, including its current lifecycle status."""
    request_path, _, state_path, lock_path = _paths(_root(root), request_id)
    with _locked(lock_path):
        request = _load_request(request_path, request_id)
        state = _expire_if_due(state_path, request, _status(state_path, request))
        return {**request, **state}


def list_requests(root: str | Path) -> list[dict[str, Any]]:
    """List compact verified metadata, without repeating potentially large prompts."""
    path = _root(root)
    records = []
    for request_path in sorted(path.glob("*.request.json")):
        request_id = request_path.name.removesuffix(".request.json")
        try:
            request = read_request(path, request_id)
            records.append({key: request[key] for key in
                            (*_BINDINGS, "created_at", "deadline_at", "status",
                             "status_updated_at", "reason")})
        except (OSError, ValueError, KeyError, RecursionError) as exc:
            # A damaged historical record must not hide a current valid request.
            # Keep it visibly invalid; show/respond still reject it strictly.
            records.append({"request_id": request_id, "status": "invalid",
                            "created_at": "", "reason": f"{type(exc).__name__}: {exc}"})
    return sorted(records, key=lambda row: (row["created_at"], row["request_id"]))


def submit_response(root: str | Path, request_id: str,
                    payload: dict[str, Any]) -> dict[str, Any]:
    """Submit a decision object once; upstream applies its scientific schema checks."""
    if not isinstance(payload, dict):
        raise ValueError("decision payload must be a JSON object")
    _json_bytes(payload, MAX_RESPONSE_BYTES - 4096)
    request_path, response_path, state_path, lock_path = _paths(_root(root), request_id)
    with _locked(lock_path):
        request = _load_request(request_path, request_id)
        state = _expire_if_due(state_path, request, _status(state_path, request))
        if state["status"] != "waiting":
            raise ValueError(f"request is {state['status']}; response rejected")
        response = {key: request[key] for key in _BINDINGS}
        response.update(payload=payload, response_hash=_hash(payload, MAX_RESPONSE_BYTES),
                        submitted_at=_utc(time.time()), submitted_unix=time.time())
        _atomic_json(response_path, response, limit=MAX_RESPONSE_BYTES)
        return _write_status(state_path, request, "accepted")


def cancel_request(root: str | Path, request_id: str) -> dict[str, Any]:
    """Cancel a pending/accepted decision; the controller then drains its workers."""
    request_path, _, state_path, lock_path = _paths(_root(root), request_id)
    with _locked(lock_path):
        request = _load_request(request_path, request_id)
        state = _expire_if_due(state_path, request, _status(state_path, request))
        if state["status"] == "cancelled":
            return state
        if state["status"] not in {"waiting", "accepted"}:
            raise ValueError(f"request is {state['status']}; cancellation rejected")
        return _write_status(state_path, request, "cancelled", "cancelled by Codex/user")


class Mailbox:
    """One controller session with bounded synchronous decision waits."""

    def __init__(self, root: str | Path, timeout_seconds: float = DEFAULT_DECISION_TIMEOUT_SECONDS,
                 session_id: str | None = None, stop_path: str | Path | None = None,
                 drain_path: str | Path | None = None,
                 mode_path: str | Path | None = None, initial_evaluation_only: bool = False):
        timeout_seconds = validate_decision_timeout(timeout_seconds)
        self.root = _root(root, create=True)
        self.timeout_seconds = float(timeout_seconds)
        self.session_id = uuid.uuid4().hex if session_id is None else session_id
        if not isinstance(self.session_id, str) or not self.session_id or len(self.session_id) > 256:
            raise ValueError("session_id must be a nonempty string of at most 256 characters")
        self.deadline_monotonic: float | None = None
        self.stop_path = Path(stop_path) if stop_path is not None else None
        self.drain_path = Path(drain_path) if drain_path is not None else None
        self.mode_path = Path(mode_path) if mode_path is not None else None
        self.initial_evaluation_only = bool(initial_evaluation_only)
        self.controller_stop_requested = None
        # The controller owns this hook. It runs on this same thread, outside
        # mailbox locks, and may reap existing workers but never dispatch work.
        self.controller_wait_callback = None

    def _close_pending_request(self, request_id: str, reason: str) -> None:
        """Fail closed after callback/transport errors without hiding the cause."""
        try:
            request_path, _, state_path, lock_path = _paths(self.root, request_id)
            with _locked(lock_path, time.monotonic() + 0.1):
                request = _load_request(request_path, request_id)
                state = _expire_if_due(state_path, request, _status(state_path, request))
                if state["status"] in {"waiting", "accepted"}:
                    _write_status(state_path, request, "cancelled", reason)
        except Exception:
            pass  # Damaged/missing/locked state must still abort the controller.

    def stop_requested(self) -> bool:
        """Read a stop signal bound to this controller session only."""
        if callable(self.controller_stop_requested) and self.controller_stop_requested():
            return True
        if self.stop_path is None or not self.stop_path.exists():
            return False
        try:
            request = _read_json(self.stop_path, 16384)
            if (request.get("schema_version") != "binder.codex-stop.v1"
                    or request.get("session_id") != self.session_id):
                raise ValueError("stop request does not match the active session")
            return True
        except (OSError, ValueError) as exc:
            raise DecisionAborted(f"invalid campaign stop request: {exc}") from exc

    def execution_mode(self) -> dict[str, Any] | None:
        """Load the archive's persistent operational mode, failing closed."""
        if self.mode_path is None or not (self.mode_path.exists() or self.mode_path.is_symlink()):
            return None
        try:
            mode = _read_json(self.mode_path, 16384)
            if (mode.get("schema_version") != "binder.codex-mode.v1"
                    or mode.get("mode") != "evaluation_only"
                    or not isinstance(mode.get("session_id"), str)
                    or not isinstance(mode.get("activated_at"), str)):
                raise ValueError("invalid persistent execution mode")
            return mode
        except (OSError, ValueError) as exc:
            raise DecisionAborted(f"invalid campaign execution mode: {exc}") from exc

    def _drain_request(self) -> dict[str, Any] | None:
        if self.drain_path is None or not (self.drain_path.exists() or self.drain_path.is_symlink()):
            return None
        try:
            request = _read_json(self.drain_path, 16384)
            if (request.get("schema_version") != "binder.codex-drain.v1"
                    or request.get("session_id") != self.session_id):
                raise ValueError("drain request does not match the active session")
            return request
        except (OSError, ValueError) as exc:
            raise DecisionAborted(f"invalid campaign drain request: {exc}") from exc

    def evaluation_only_requested(self) -> bool:
        # Read both files even when a CLI flag was supplied: damage must not be
        # silently hidden, and a previous activation must retain its scope.
        mode, request = self.execution_mode(), self._drain_request()
        return bool(mode or request or self.initial_evaluation_only)

    def activate_evaluation_only(self) -> dict[str, Any]:
        mode = self.execution_mode()
        if mode is not None:
            return mode
        request = self._drain_request()
        if not self.initial_evaluation_only and request is None:
            raise DecisionAborted("evaluation-only mode was not requested")
        mode = {"schema_version": "binder.codex-mode.v1", "mode": "evaluation_only",
                "session_id": self.session_id, "activated_at": _utc(time.time()),
                "source": "start --evaluation-only" if request is None else "drain",
                "drain_request": request}
        if self.mode_path is None:
            raise DecisionAborted("persistent execution mode path is missing")
        _atomic_json(self.mode_path, mode, limit=16384)
        return mode

    def _check_control(self, request_id: str) -> None:
        if self.stop_requested():
            raise DecisionAborted("campaign stop requested", request_id)
        if self.evaluation_only_requested():
            raise EvaluationOnlyRequested(request_id)

    def chat(self, role: str, messages: list[dict[str, Any]], system: str = "",
             max_tokens: int = 4096, temperature: float | None = None,
             tools: list[dict[str, Any]] | None = None):
        """Expose a request and wait at most the timeout and campaign deadline."""
        request_id = uuid.uuid4().hex
        try:
            return self._chat(request_id, role, messages, system, max_tokens, temperature, tools)
        except DecisionAborted as exc:
            self._close_pending_request(request_id, str(exc.reason))
            raise
        except (Exception, KeyboardInterrupt) as exc:
            # Upstream catches Exception and dispatches fallback work. Transport
            # corruption and interruption must never follow that path.
            self._close_pending_request(request_id, "mailbox transport error")
            raise DecisionAborted(f"mailbox_error:{type(exc).__name__}:{exc}", request_id) from exc

    def _chat(self, request_id, role, messages, system, max_tokens, temperature, tools):
        if role not in {"planner", "supervisor"}:
            raise ValueError("role must be planner or supervisor")
        if tools:
            raise ValueError("mailbox supports structured decisions, not tool calls")
        if not isinstance(messages, list) or not isinstance(system, str):
            raise ValueError("messages must be a list and system must be text")
        self._check_control(request_id)
        started = time.monotonic()
        deadline = started + self.timeout_seconds
        if self.deadline_monotonic is not None:
            if not math.isfinite(self.deadline_monotonic):
                raise ValueError("campaign deadline must be finite")
            deadline = min(deadline, self.deadline_monotonic)
        remaining = deadline - started
        if remaining <= 0:
            raise DecisionAborted("campaign deadline elapsed", request_id)
        now = time.time()
        request = dict(version=PROTOCOL_VERSION, request_id=request_id,
                       session_id=self.session_id, role=role, system=system,
                       messages=messages, max_tokens=max_tokens, temperature=temperature,
                       created_at=_utc(now), created_unix=now,
                       deadline_at=_utc(now + remaining), deadline_unix=now + remaining)
        request["prompt_hash"] = _hash(_prompt_content(request))
        request_path, response_path, state_path, lock_path = _paths(self.root, request_id)
        with _locked(lock_path, deadline):
            _atomic_json(request_path, request)
            _write_status(state_path, request, "waiting")

        next_service_at = started
        while True:
            self._check_control(request_id)
            try:
                # A stalled client holding this lock must not suspend worker
                # supervision or stop handling for the whole decision window.
                with _locked(lock_path, min(deadline, time.monotonic() + 0.05)):
                    # Recheck immutable request bytes to reject altered prompt/hash
                    # pairs even when both were rewritten by an external process.
                    actual = _load_request(request_path, request_id)
                    if actual != request:
                        _write_status(state_path, request, "cancelled", "request changed after publication")
                        raise DecisionAborted("request changed after publication", request_id)
                    state = _status(state_path, request)
                    self._check_control(request_id)
                    if time.monotonic() >= deadline and state["status"] in {"waiting", "accepted"}:
                        state = _write_status(state_path, request, "expired", "decision deadline elapsed")
                    state = _expire_if_due(state_path, request, state)
                    if state["status"] in {"cancelled", "expired", "consumed"}:
                        raise DecisionAborted(state["status"], request_id)
                    if state["status"] == "accepted":
                        response = _read_json(response_path, MAX_RESPONSE_BYTES)
                        if (type(response.get("version")) is not int or
                                any(response.get(key) != request[key] for key in _BINDINGS)):
                            _write_status(state_path, request, "cancelled", "response binding mismatch")
                            raise DecisionAborted("response binding mismatch", request_id)
                        payload = response.get("payload")
                        submitted = response.get("submitted_unix")
                        if (not isinstance(payload, dict) or response.get("response_hash") != _hash(payload)
                                or not isinstance(submitted, (int, float)) or isinstance(submitted, bool)
                                or not math.isfinite(submitted) or submitted < request["created_unix"]
                                or submitted >= request["deadline_unix"]):
                            _write_status(state_path, request, "cancelled", "invalid response integrity/deadline")
                            raise DecisionAborted("invalid response integrity/deadline", request_id)
                        # Import only the inert response dataclass, never an SDK/client.
                        answer = DecisionResponse(text=_json_bytes(payload, MAX_RESPONSE_BYTES).decode("utf-8"),
                                             model="codex-external", stop_reason="end_turn", usage={})
                        self._check_control(request_id)
                        _write_status(state_path, request, "consumed")
                        return answer
            except _LockWaitExpired:
                if time.monotonic() >= deadline:
                    raise DecisionAborted("expired", request_id)
            # A request remains an immutable evidence snapshot while existing
            # workers are supervised. All archive writes stay on the controller
            # thread, and slow parsing never holds a response/stop file lock.
            callback = self.controller_wait_callback
            if callback is not None and time.monotonic() >= next_service_at:
                self._check_control(request_id)
                callback()
                self._check_control(request_id)
                next_service_at = time.monotonic() + WAIT_SERVICE_INTERVAL_SECONDS
                # Recheck state/deadline before consuming a reply accepted while
                # the callback ran; the callback cannot extend its deadline.
                continue
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


__all__ = ["Mailbox", "DecisionAborted", "list_requests", "read_request",
           "submit_response", "cancel_request", "PROTOCOL_VERSION",
           "DEFAULT_DECISION_TIMEOUT_SECONDS", "MAX_DECISION_TIMEOUT_SECONDS",
           "validate_decision_timeout"]
