"""Finite, deterministic canonical scoring after generation is closed.

This module never creates a scientific decision. The immutable scope admits
native results already present at activation and results from generators that
were actually running at that point. Selected work that has not started, or
interrupted generation with no result that is not resumed, is suspended.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import uuid

from ..capability_registry import default_registry
from ..dispatch_outcomes import classify_dispatch_outcome
from ..refilter_roles import CANONICAL_SCORE_CONVERSION
from ..schemas import ActionCandidate, DispatchRecord, FeasibilityCheck, LaunchDecision, ResultRecord
from .score_conversion_scheduling import (
    is_canonical_score_conversion_result, result_requires_score_conversion,
    score_conversion_parent_identity,
)

STATE_NAME = "evaluation_only_state.json"
SCHEMA = "binder.evaluation-only.v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read_state(root):
    path = Path(root) / STATE_NAME
    if path.is_symlink():
        raise ValueError("evaluation-only state must not be a symlink")
    if not path.exists():
        return None
    if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("invalid evaluation-only state file")
    value = json.loads(path.read_text(encoding="utf-8"))
    scope = value.get("scope") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or value.get("schema_version") != SCHEMA or not isinstance(scope, dict)
            or value.get("scope_sha256") != _digest(scope)
            or not isinstance(scope.get("target_id"), str)
            or not isinstance(scope.get("scope_id"), str)
            or any(not isinstance(scope.get(key), list)
                   or any(not isinstance(item, str) for item in scope[key])
                   for key in ("base_result_ids", "active_generation_candidate_ids", "suspended_candidate_ids"))):
        raise ValueError("invalid evaluation-only scope or hash")
    return value


def write_state(root, value):
    path = Path(root) / STATE_NAME
    if path.is_symlink():
        raise ValueError("evaluation-only state must not be a symlink")
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                         allow_nan=False).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=".evaluation-only-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def is_system_canonical(candidate):
    """Only the two system factories' unchanged AF2 action contract is allowed."""
    return bool(candidate is not None
        and candidate.method_family == "structure_refilter"
        and candidate.candidate_id.startswith("chain_")
        and candidate.refilter_role == CANONICAL_SCORE_CONVERSION
        and candidate.parent_result_id
        and not candidate.config_delta and not candidate.downstream_route_plan
        and (candidate.expected_signal or "").startswith(("auto_chain:", "lazy_rechain:"))
        and candidate.feasibility is not None and candidate.feasibility.all_ok())


def activate_scope(archive, *, target_id, pool, pending, session_id,
                   mode_already_persisted=False):
    state = read_state(archive.root)
    if state is not None:
        if state["scope"]["target_id"] != target_id:
            raise ValueError("evaluation-only scope belongs to another target")
        return state
    actions = {row.candidate_id: row for row in archive.iter_records(ActionCandidate)}
    started = {row.candidate_id for row in archive.iter_records(DispatchRecord) if row.status == "started"}
    selected = {row.candidate_id for row in archive.iter_records(LaunchDecision) if row.status == "launched"}
    native_results = [row for row in archive.iter_records(ResultRecord)
                      if row.target_id == target_id and row.backend_family != "structure_refilter"]
    native_parent_ids = {cid for row in native_results for cid in row.parent_ids}
    active_generation_ids = {slot.cand.candidate_id for slot in pool
        if slot.busy and slot.cand is not None and slot.cand.method_family != "structure_refilter"}
    # Evaluation-only recovery deliberately drops interrupted generators from
    # pending. Keep their absence visible without admitting them into the scope.
    interrupted_generation_ids = {cid for cid in (selected & started) - native_parent_ids - active_generation_ids
        if actions.get(cid) is not None and actions[cid].method_family != "structure_refilter"}
    scope = {
        "scope_id": uuid.uuid4().hex,
        "target_id": target_id,
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "activation_session_id": session_id,
        "base_result_ids": sorted(row.result_id for row in native_results),
        "active_generation_candidate_ids": sorted(active_generation_ids),
        "suspended_candidate_ids": sorted({cid for cid in set(pending) | (selected - started) | interrupted_generation_ids
                                           if not is_system_canonical(actions.get(cid))}),
        "scope_recovered_after_mode_only_crash": bool(mode_already_persisted),
    }
    state = {"schema_version": SCHEMA, "scope": scope, "scope_sha256": _digest(scope),
             "phase": "activated", "current": None}
    write_state(archive.root, state)
    return state


def _artifact_available(parent):
    artifacts = parent.artifacts or {}
    file = artifacts.get("pdb_path") or artifacts.get("cif_path")
    if file:
        return Path(file).is_file() and Path(file).stat().st_size > 0
    directory = artifacts.get("pdb_dir")
    return bool(directory and Path(directory).is_dir()
                and any(Path(directory).rglob("*.pdb")))


def _successful_canonical(result):
    return (is_canonical_score_conversion_result(result) and result.exit_status == "ok"
            and all(isinstance((result.metrics or {}).get(key), (int, float))
                    and not isinstance(result.metrics[key], bool)
                    and math.isfinite(result.metrics[key])
                    for key in ("pLDDT", "iPAE", "binder_scRMSD")))


def reconcile_scope(archive, state, *, active_ids=(), retry_pending=(),
                    recoverable_failure=lambda row: False, mint=True):
    """Refresh finite scope and return safe pending IDs, without fairness caps.

    Terminal failures are reported, never retried forever. A prior started row
    may only be resumed when the recovery path explicitly supplied its ID.
    """
    scope = state["scope"]
    results = list(archive.iter_records(ResultRecord))
    actions = list(archive.iter_records(ActionCandidate))
    dispatches = list(archive.iter_records(DispatchRecord))
    base, running = set(scope["base_result_ids"]), set(scope["active_generation_candidate_ids"])
    parents = [row for row in results if row.target_id == scope["target_id"]
               and row.backend_family != "structure_refilter"
               and (row.result_id in base or running.intersection(row.parent_ids))]
    scoped = {row.result_id: row for row in parents}
    by_parent = {}
    for candidate in actions:
        if is_system_canonical(candidate) and candidate.parent_result_id in scoped:
            by_parent.setdefault(candidate.parent_result_id, []).append(candidate)
    latest_dispatch = {row.candidate_id: row for row in dispatches}
    ever_started = {row.candidate_id for row in dispatches if row.status == "started"}
    canonical_by_parent = {}
    for result in results:
        if is_canonical_score_conversion_result(result):
            canonical_by_parent.setdefault(result.bins["refilter_source"], []).append(result)
    completed_identities = {score_conversion_parent_identity(scoped[parent_id])
        for parent_id, children in canonical_by_parent.items() if parent_id in scoped
        and any(_successful_canonical(child) for child in children)}
    active_ids, retry_pending = set(active_ids), set(retry_pending)
    eligible, remaining, completed, excluded, aliases, minted = [], [], [], [], [], []
    remaining.extend({"parent_result_id": pid, "candidate_id": None, "reason": "scoped_result_missing"}
                     for pid in sorted(base - set(scoped)))
    represented = {}
    capability = default_registry().get("structure_refilter")
    for parent in sorted(parents, key=lambda row: row.result_id):
        pid = parent.result_id
        if not result_requires_score_conversion(parent.backend_family, parent):
            excluded.append({"parent_result_id": pid, "reason": "unsupported_or_non_successful_native_record"})
            continue
        identity = score_conversion_parent_identity(parent)
        if identity in completed_identities:
            completed.append(pid)
            continue
        if identity in represented:
            aliases.append({"parent_result_id": pid, "represented_by": represented[identity]})
            continue
        represented[identity] = pid
        row = {"parent_result_id": pid, "candidate_id": None, "reason": "pending"}
        if canonical_by_parent.get(pid):
            row["reason"] = "prior_canonical_failure_or_incomplete_metrics"
        elif not _artifact_available(parent):
            row["reason"] = "missing_artifact"
        elif capability is None or capability.availability != "available":
            row["reason"] = "canonical_backend_unavailable"
        else:
            candidates = by_parent.get(pid, [])
            # A single prior failed system evaluation is terminal for this mode;
            # operators inspect/fix it rather than silently minting a retry loop.
            terminal = [latest_dispatch[c.candidate_id] for c in candidates
                        if c.candidate_id in latest_dispatch
                        and (latest_dispatch[c.candidate_id].status == "parse_failed"
                             or (latest_dispatch[c.candidate_id].status == "dispatch_failed"
                                 and not recoverable_failure(latest_dispatch[c.candidate_id])
                                 and not (c.candidate_id not in ever_started
                                     and classify_dispatch_outcome(latest_dispatch[c.candidate_id]) == "cancelled_before_start")))]
            if terminal:
                row.update(candidate_id=terminal[-1].candidate_id, reason="prior_terminal_dispatch_failure")
            else:
                if not candidates and mint:
                    cid = "chain_drain_" + scope["scope_id"][:12] + "_" + hashlib.sha256(pid.encode()).hexdigest()[:20]
                    candidate = ActionCandidate(candidate_id=cid, hypothesis_ids=[], parent_result_id=pid,
                        method_family="structure_refilter", operator_id=capability.default_operator_id,
                        lane_id=capability.default_lane_id, config_delta={}, downstream_route_plan=[],
                        estimated_cost_class=capability.default_cost_class,
                        expected_signal=f"lazy_rechain:{parent.backend_family}->structure_refilter parent={pid} role={CANONICAL_SCORE_CONVERSION}",
                        evidence_refs=[pid], feasibility=FeasibilityCheck(backend_healthy=True,
                            runtime_bucket_id="rb_v7", compiler_ok=True, verifier_ok=True, route_cap_ok=True, cost_ok=True),
                        baseline_result_id=pid, refilter_role=CANONICAL_SCORE_CONVERSION)
                    archive.append(candidate)
                    candidates = [candidate]
                    minted.append(cid)
                if candidates:
                    candidate = next((c for c in candidates if c.candidate_id in active_ids), candidates[0])
                    cid = candidate.candidate_id
                    row["candidate_id"] = cid
                    dispatch = latest_dispatch.get(cid)
                    if cid in active_ids:
                        row["reason"] = "running"
                    elif dispatch is not None and dispatch.status == "started" and cid not in retry_pending:
                        row["reason"] = "started_worker_not_recovered"
                    else:
                        eligible.append(cid)
                else:
                    row["reason"] = "not_queued"
        remaining.append(row)
    return {
        "scoped_native_result_ids": sorted(scoped),
        "completed_parent_ids": completed, "remaining": remaining,
        "eligible_candidate_ids": eligible, "excluded_native_records": excluded,
        "duplicate_parent_aliases": aliases, "newly_queued_candidate_ids": minted,
        "active_worker_candidate_ids": sorted(active_ids),
    }


def update_state(archive, state, current, *, phase, reason=None, exit_code=None):
    state.update(current=current, phase=phase, reason=reason, exit_code=exit_code,
                 updated_at=datetime.now(timezone.utc).isoformat())
    write_state(archive.root, state)
    return state


def finalize_evidence(archive, state, *, target, config, round_id):
    """Refresh strict/cluster totals directly, without planning or evidence rows."""
    from ..evidence.attribution import is_canonical_su_record
    from ..output_identity import prepare_archive_results
    from ..tick.archive_snapshot import load_tick_archive_snapshot
    from ..tick.evidence_clustering import ResultClusteringRequest, cluster_evidence_results

    snapshot = load_tick_archive_snapshot(archive, target.target_id)
    current = state.get("current") or reconcile_scope(archive, state, mint=False)
    scoped = set(current["scoped_native_result_ids"])
    results = prepare_archive_results([row for row in snapshot.target_results
        if is_canonical_score_conversion_result(row)
        and row.bins.get("refilter_source") in scoped], archive.root)
    spawning = dict(snapshot.spawning_action_by_result_id)
    strict = [row for row in results if is_canonical_su_record(row, spawning)]
    if not strict:
        return {"canonical_result_count": len(results), "strict_result_ids": [], "strict_count": 0,
                "structure_cluster_count": 0, "sequence_cluster_count": 0,
                "foldseek_status": "no_strict", "sequence_status": "no_strict", "complete": True}
    clustered = cluster_evidence_results(ResultClusteringRequest(
        results=results, target=target, spawning_actions=spawning, previous_evidence=None,
        tick_id_int=max(round_id, 1), window_size=max(config.window_size, len(results)),
        foldseek_config=config.foldseek, sequence_config=config.sequence_dedup))
    structure_ok = clustered.foldseek_su_status == "ok" and clustered.foldseek_su_coverage == 1.0
    sequence_ok = (not config.sequence_dedup.enabled or
                   (clustered.sequence_dedup_status == "ok" and clustered.sequence_dedup_coverage == 1.0))
    return {"canonical_result_count": len(results), "strict_result_ids": sorted(row.result_id for row in strict),
            "strict_count": len(strict),
            "structure_cluster_count": len({row.bins["foldseek_su"] for row in strict if row.bins.get("foldseek_su")}) if structure_ok else None,
            "sequence_cluster_count": len({row.bins["sequence_su"] for row in strict if row.bins.get("sequence_su")}) if sequence_ok and config.sequence_dedup.enabled else None,
            "foldseek_status": clustered.foldseek_su_status, "foldseek_coverage": clustered.foldseek_su_coverage,
            "sequence_status": clustered.sequence_dedup_status, "sequence_coverage": clustered.sequence_dedup_coverage,
            "complete": bool(structure_ok and sequence_ok),
            "scope": "current canonical results in frozen evaluation-only scope; no scientific decision appended"}
