"""Explicit external decisions for the embedded workflow, without model clients."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
import math
import time
from typing import Any, Protocol


class DecisionAborted(BaseException):
    """Stop new dispatch and drain workers instead of selecting fallback work."""

    def __init__(self, reason: str, request_id: str | None = None):
        self.reason, self.request_id = reason, request_id
        super().__init__(f"Codex decision aborted: {reason}" +
                         (f" (request {request_id})" if request_id else ""))


class EvaluationOnlyRequested(DecisionAborted):
    """Leave scientific planning and finish existing canonical evaluations."""

    def __init__(self, request_id: str | None = None):
        super().__init__("evaluation-only mode requested", request_id)


@dataclass
class DecisionResponse:
    """Response text and honest optional attribution; no SDK or token estimate."""
    text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = "codex-external:unknown"
    stop_reason: str = "end_turn"
    raw: Any = None


class DecisionProvider(Protocol):
    def chat(self, role: str, messages: list[dict], *, system: str = "",
             max_tokens: int = 4096, temperature: float | None = None) -> DecisionResponse: ...


def request_decision(provider, role: str, messages: list[dict], **kwargs) -> DecisionResponse:
    """Invoke only the supplied provider; an absent provider never creates a client."""
    try:
        call = getattr(provider, "chat", None) if provider is not None else None
        call = call if callable(call) else provider
        if not callable(call):
            raise DecisionAborted(f"{role} decision provider is missing")
        response = call(role, messages, **kwargs)
        if not isinstance(getattr(response, "text", None), str) or not response.text.strip():
            raise DecisionAborted(f"{role} response is empty or malformed")
        return response
    except DecisionAborted:
        raise
    except (Exception, KeyboardInterrupt) as exc:
        raise DecisionAborted(f"{role} provider error: {type(exc).__name__}: {exc}") from exc


def require_valid_decision(role: str):
    """Run schema repair/validation, then reject a final unusable decision."""
    def decorate(call):
        @wraps(call)
        def guarded(*args, **kwargs):
            try:
                result = call(*args, **kwargs)
                decisions = result.cards if role == "planner" else result.candidate_decisions
                if not result.valid or result.abstain or not decisions:
                    reason = result.fail_reason or ("abstain" if result.abstain else "empty decisions")
                    raise DecisionAborted(f"{role} decision rejected: {reason}")
                return result
            except DecisionAborted:
                raise
            except (Exception, KeyboardInterrupt) as exc:
                raise DecisionAborted(f"{role} validation error: {type(exc).__name__}: {exc}") from exc
        return guarded
    return decorate


@contextmanager
def decision_deadline(provider, remaining_wall_h: float):
    """Bound one tick's mailbox waits without extending an earlier deadline.

    The mailbox enforces the deadline. This is not an allocation hard cutoff;
    the controller still drains workers after the decision wait ends.
    """
    if provider is None:
        raise DecisionAborted("decision provider is missing")
    if (not isinstance(remaining_wall_h, (int, float)) or
            not math.isfinite(remaining_wall_h) or remaining_wall_h <= 0):
        raise DecisionAborted("campaign decision time exhausted or unspecified")
    supports_deadline = hasattr(provider, "deadline_monotonic")
    previous = getattr(provider, "deadline_monotonic", None)
    if supports_deadline:
        deadline = time.monotonic() + remaining_wall_h * 3600.0
        provider.deadline_monotonic = min(previous, deadline) if previous is not None else deadline
    try:
        yield
    finally:
        if supports_deadline:
            provider.deadline_monotonic = previous


def bounded_decision_tick(call):
    @wraps(call)
    def bounded(*args, **kwargs):
        if kwargs.get("evidence_only", False):
            return call(*args, **kwargs)
        with decision_deadline(kwargs.get("decision_provider"), kwargs.get("remaining_wall_h")):
            return call(*args, **kwargs)
    return bounded
