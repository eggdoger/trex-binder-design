"""Inspect, validate, and audit a T-REX append-only campaign archive.

The commands in this module operate on archived records only.  They never
modify generated structures, scores, or EvidenceSummary rows.
"""

from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
from dataclasses import fields
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from .archive import Archive, RECORD_FILES
from .archive_schema import archive_layout
from .dispatch_outcomes import classify_dispatch_outcome, count_dispatch_outcomes
from .execution.progress_checkpoint import read_runtime_metrics
from .provenance import (
    PROVENANCE_SCHEMA,
    SUPPORTED_PROVENANCE_SCHEMAS,
)
from .success_criteria import STRICT_SUCCESS, valid_qualification_measurement
from .schemas import ResultRecord


SUMMARY_SCHEMA_VERSION = "binder_runtime.analysis-summary.v1"
VALIDATION_SCHEMA_VERSION = "binder_runtime.archive-validation.v1"
TRACE_ENTRY_SCHEMA_VERSION = "binder_runtime.decision-trace-entry.v1"
TRACE_OUTPUT_SCHEMA_VERSION = "binder_runtime.decision-trace.v1"
CODEX_PROVENANCE_SCHEMA = "binder.codex-provenance.v1"
CODEX_OUTCOME_SCHEMA = "binder.codex-outcome.v1"

# These values identify deterministic system-origin candidates rather than
# archived HypothesisCard rows. Keep the vocabulary explicit so a typo or a
# genuinely broken card reference still produces a validation warning.
SYSTEM_CANDIDATE_ORIGIN_IDS = frozenset(
    {
        "cross_family_escape",
        "diagnostic_i4_mcts",
        "evidence_fallback",
        "evidence_fallback_dead_probe",
        "route_value_replay",
        "warmstart",
    }
)


JSONL_FILES = tuple(RECORD_FILES.values())


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return rows, errors
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_no}: invalid JSON: {exc.msg}")
                continue
            if not isinstance(value, dict):
                errors.append(f"{path.name}:{line_no}: expected a JSON object")
                continue
            rows.append(value)
    return rows, errors


def _records(root: Path, name: str) -> list[dict[str, Any]]:
    rows, errors = _read_jsonl(root / name)
    if errors:
        raise ValueError("archive stream contains malformed rows; run validate: " + "; ".join(errors[:5]))
    return rows


def _require_archive_root(root: Path | str) -> Path:
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"archive root is not a directory: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_paths(root: Path) -> list[Path]:
    """Recognize embedded-runtime provenance and legacy read-only archives."""
    return sorted(set(root.glob("run_provenance*.json")) |
                  set(root.glob("codex_provenance_*.json")))


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _metadata(path: Path) -> dict[str, Any]:
    if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("metadata is a symlink or exceeds 4 MiB")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metadata root must be a JSON object")
    return value


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.fromisoformat(value).tzinfo is not None
    except ValueError:
        return False


def _validate_codex_provenance(root: Path, path: Path, value: dict[str, Any],
                               errors: list[str], warnings: list[str]) -> None:
    def invalid(field: str):
        errors.append(f"{path.name}: missing/invalid {field}")

    session = value.get("session_id")
    if (not isinstance(session, str) or re.fullmatch(r"[0-9a-f]{32}", session) is None
            or path.name != f"codex_provenance_{session}.json"):
        invalid("session_id/filename binding")
    if not _valid_timestamp(value.get("started_at")):
        invalid("started_at")
    if value.get("provider") != "external Codex conversation":
        invalid("provider")
    label = value.get("model_label")
    if not isinstance(label, str) or not label.startswith("codex-external:") or not label[15:]:
        invalid("model_label")
    for field in ("model_content_sha256", "token_usage"):
        if field not in value or value[field] is not None:
            invalid(field + " (external conversation value must remain unknown)")
    if value.get("publication_model_parity") is not False:
        invalid("publication_model_parity")
    for section, fields in {
        "runtime_bundle": ("manifest_sha256",),
        "transport": ("runner_sha256", "mailbox_sha256"),
        "inputs": ("campaign_sha256", "target_pdb_sha256", "target_constraint_sha256", "resolved_sha256"),
    }.items():
        payload = value.get(section)
        for field in fields:
            if not isinstance(payload, dict) or not _valid_sha256(payload.get(field)):
                invalid(f"{section}.{field}")
    inputs = value.get("inputs")
    if isinstance(inputs, dict):
        for field, digest_field in (("input_path", "campaign_sha256"),
                                    ("resolved_path", "resolved_sha256")):
            declared = inputs.get(field)
            if not isinstance(declared, str) or not declared:
                invalid("inputs." + field)
                continue
            # Saved absolute paths may belong to the original execution host;
            # immutable input artifacts are verified within this archive only.
            artifact = root / Path(declared).name
            try:
                if artifact.is_symlink() or not artifact.is_file():
                    errors.append(f"{path.name}: linked artifact not found or unsafe: {artifact.name}")
                elif _sha256_file(artifact) != inputs.get(digest_field):
                    errors.append(f"{path.name}: linked artifact SHA256 mismatch: {artifact.name}")
            except OSError as exc:
                errors.append(f"{path.name}: cannot read linked artifact {artifact.name}: {exc}")
    workers = value.get("worker_gpus")
    if not isinstance(workers, list) or not workers or any(not isinstance(item, str) or not item for item in workers):
        invalid("worker_gpus")
    for field in ("max_wall_hours", "decision_timeout_seconds"):
        number = _finite(value.get(field))
        if isinstance(value.get(field), bool) or number is None or number <= 0:
            invalid(field)
    backend = value.get("backend_verification")
    if not isinstance(backend, dict):
        warnings.append(f"{path.name}: backend verification metadata unavailable")
    else:
        if backend.get("gpu_execution_tested") is not False:
            invalid("backend_verification.gpu_execution_tested (preflight is not GPU execution)")
        for field in ("preflight", "resolved_paths"):
            if not isinstance(backend.get(field), dict):
                invalid("backend_verification." + field)
    environment = value.get("runtime_environment")
    if not isinstance(environment, dict):
        warnings.append(f"{path.name}: controller runtime environment metadata unavailable")
    else:
        for field in ("python", "packages", "package_metadata_errors"):
            if not isinstance(environment.get(field), dict):
                invalid("runtime_environment." + field)
        if isinstance(environment.get("packages"), dict):
            for name, version in environment["packages"].items():
                if version is not None and (not isinstance(version, str) or not version):
                    invalid(f"runtime_environment.packages.{name}")
                elif version is None:
                    warnings.append(f"{path.name}: controller package version unknown: {name}")


def codex_outcome_summaries(root: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Read controller exits without treating missing outcomes as running jobs."""
    rows, errors, warnings = [], [], []
    sessions = {path.name[len("codex_provenance_"):-5]
                for path in root.glob("codex_provenance_*.json")}
    outcomes = {path.name[len("codex_outcome_"):-5]: path
                for path in root.glob("codex_outcome_*.json")}
    for session in sorted(sessions | set(outcomes)):
        path = outcomes.get(session)
        if path is None:
            rows.append({"session_id": session, "status": "outcome_unrecorded", "exit_code": None})
            warnings.append(f"session {session}: controller outcome is unrecorded; current process state is unknown")
            continue
        if session not in sessions:
            errors.append(f"{path.name}: no matching standalone provenance")
        try:
            value = _metadata(path)
        except (OSError, ValueError, RecursionError) as exc:
            errors.append(f"{path.name}: invalid outcome JSON: {exc}")
            rows.append({"session_id": session, "status": "invalid_outcome", "exit_code": None})
            continue
        local_errors = []
        if value.get("schema_version") != CODEX_OUTCOME_SCHEMA:
            local_errors.append("schema_version")
        if value.get("session_id") != session or re.fullmatch(r"[0-9a-f]{32}", session) is None:
            local_errors.append("session_id/filename binding")
        if not _valid_timestamp(value.get("ended_at")):
            local_errors.append("ended_at")
        code = value.get("exit_code")
        if "exit_code" not in value or (code is not None and type(code) is not int):
            local_errors.append("exit_code")
        decisions = value.get("decisions")
        if not isinstance(decisions, list):
            local_errors.append("decisions")
        else:
            for index, decision in enumerate(decisions):
                if not isinstance(decision, dict):
                    local_errors.append(f"decisions[{index}]")
                    continue
                if decision.get("audit_error"):
                    if not isinstance(decision["audit_error"], str) or decision.get("session_verified") is not False:
                        local_errors.append(f"decisions[{index}].audit_error")
                    warnings.append(f"{path.name}: decision audit incomplete: {decision.get('audit_error')}")
                    continue
                request_id = decision.get("request_id")
                if (not isinstance(request_id, str) or re.fullmatch(r"[0-9a-f]{32}", request_id) is None
                        or decision.get("file") != f"{request_id}.request.json"):
                    local_errors.append(f"decisions[{index}].request_id/file binding")
                if not _valid_sha256(decision.get("request_sha256")):
                    local_errors.append(f"decisions[{index}].request_sha256")
                response_sha = decision.get("response_sha256")
                if "response_sha256" not in decision or (response_sha is not None and not _valid_sha256(response_sha)):
                    local_errors.append(f"decisions[{index}].response_sha256")
        errors.extend(f"{path.name}: missing/invalid {field}" for field in local_errors)
        status = ("invalid_outcome" if local_errors else "controller_exit_unknown" if code is None
                  else "controller_exited" if code == 0 else "decision_stopped" if code == 75
                  else "controller_failed")
        rows.append({"session_id": session, "status": status, "exit_code": code,
                     "ended_at": value.get("ended_at"),
                     "decision_records": len(decisions) if isinstance(decisions, list) else None})
    return rows, errors, warnings


def _counter(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    values = collections.Counter(str(row.get(key, "<missing>")) for row in rows)
    return dict(sorted(values.items()))


def _string_values(rows: Iterable[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value:
            values.append(value)
    return values


def _list_string_values(rows: Iterable[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            values.extend(item for item in value if isinstance(item, str) and item)
    return values


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _matches_json_field_shape(value: Any, declared_type: str) -> bool:
    """Check the top-level JSON shape represented by a dataclass annotation."""

    if value is None:
        return "None" in declared_type or declared_type == "Any"
    base_type = declared_type.split("|", 1)[0].strip()
    if base_type == "str":
        return isinstance(value, str)
    if base_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if base_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if base_type == "bool":
        return isinstance(value, bool)
    if base_type.startswith("list["):
        return isinstance(value, list)
    if base_type.startswith("tuple["):
        return isinstance(value, list)
    if base_type.startswith("dict["):
        return isinstance(value, dict)
    return True


def _token_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Historical external-Codex records used zero as a missing-value default.
    # Those zeros cannot be reconstructed into genuine token measurements.
    known = all(
        not str(row.get("model", "")).startswith("codex-external")
        and all(type(row.get(field)) is int and row[field] >= 0
                for field in ("tokens_in", "tokens_out"))
        for row in rows
    )
    return {"token_usage_known": known,
            "tokens_in": sum(row["tokens_in"] for row in rows) if known else None,
            "tokens_out": sum(row["tokens_out"] for row in rows) if known else None}


def current_result_summary(
    root: Path, *, results: list[dict[str, Any]] | None = None,
    latest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute qualification from one result snapshot without altering history.

    Clustering still belongs to its dated evidence snapshot or a fresh export;
    this function never infers structural uniqueness from a record count.
    """
    from .evidence_reducer import build_qualification_summary

    if results is None:
        results = _records(root, "result_records.jsonl")
    if latest is None:
        evidence = _records(root, "evidence_summaries.jsonl")
        latest = evidence[-1] if evidence else {}
    names = {item.name for item in fields(ResultRecord)}
    typed = []
    seen = set()
    for row in results:
        rid = row.get("result_id")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("invalid or duplicate result_id; run analyze validate")
        seen.add(rid)
        try:
            record = ResultRecord(**{key: value for key, value in row.items() if key in names})
            if (not isinstance(record.bins, dict) or not isinstance(record.metrics, dict)
                    or not isinstance(record.parent_ids, list) or not isinstance(record.artifacts, dict)):
                raise ValueError("invalid result field shape")
            typed.append(record)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid result {rid}; run analyze validate: {exc}") from exc
    summary = build_qualification_summary(typed)
    scope = [row for row in results if row.get("target_id") == latest.get("target_id")]
    previous_count = latest.get("completed_children")
    freshness = "unavailable"
    if type(previous_count) is int and latest.get("target_id"):
        freshness = ("result_count_matches" if previous_count == len(scope)
                     else "older_than_results" if previous_count < len(scope)
                     else "inconsistent_result_count")
    return {
        **summary,
        "source": "result_records.jsonl",
        "record_scope": "all_targets_in_archive",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "stored result records only; canonical AF2 gate and native filtering are separate; no current PDB revalidation, new scoring or clustering",
        "latest_evidence_comparison": {
            "status": freshness,
            "tick_id": latest.get("tick_id"),
            "target_id": latest.get("target_id"),
            "snapshot_result_count": previous_count,
            "current_target_result_count": len(scope) if latest.get("target_id") else None,
            "snapshot_strict_count": latest.get("strict_count"),
        },
    }


def campaign_summary(root: Path) -> dict[str, Any]:
    root = _require_archive_root(root)
    results = _records(root, "result_records.jsonl")
    evidence = _records(root, "evidence_summaries.jsonl")
    calls = _records(root, "llm_call_records.jsonl")
    launches = _records(root, "launch_decisions.jsonl")
    dispatches = _records(root, "dispatch_records.jsonl")
    candidates = _records(root, "action_candidates.jsonl")
    panels = _records(root, "panel_selections.jsonl")
    latest = evidence[-1] if evidence else {}
    runtime_metrics = read_runtime_metrics(
        Archive(root), target_id=latest.get("target_id"),
        minimum_elapsed_wall_h=_finite(latest.get("elapsed_wall_h")) or 0.0,
    )
    current_budget = runtime_metrics or latest

    by_role: dict[str, dict[str, Any]] = {}
    for role, rows in _group_by(calls, "role").items():
        by_role[role] = {
            "calls": len(rows),
            **_token_usage(rows),
            "latency_s": (
                sum(_finite(row.get("latency_s")) for row in rows)
                if all(_finite(row.get("latency_s")) is not None for row in rows)
                else None
            ),
            "parse_status": _counter(rows, "parse_status"),
            "fallback_calls": sum(bool(row.get("fallback_triggered")) for row in rows),
        }

    started = [row for row in dispatches if row.get("status") == "started"]
    target_ids = sorted(
        {
            str(row.get("target_id"))
            for row in results + evidence
            if row.get("target_id") is not None
        }
    )
    worker_wall = _finite(latest.get("worker_wall_gpu_h_total"))
    su_count = latest.get("run_su_count")
    su_rate = _finite(latest.get("run_su_per_worker_wall_gpu_h_total"))
    if su_rate is None and worker_wall and _finite(su_count) is not None and worker_wall > 0:
        su_rate = _finite(su_count) / worker_wall
    provenance_files = [path.name for path in provenance_paths(root)]
    controller_outcomes, _, _ = codex_outcome_summaries(root)

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "archive_root": str(root.resolve()),
        "target_ids": target_ids,
        "latest_tick": latest.get("tick_id"),
        "state_label": latest.get("state_label"),
        "runtime_metrics": runtime_metrics,
        "current_results": current_result_summary(root, results=results, latest=latest),
        "budget": {
            "source": "controller_checkpoint.json" if runtime_metrics else "latest_evidence",
            "elapsed_wall_h": _finite(current_budget.get("elapsed_wall_h")),
            "remaining_wall_h": _finite(current_budget.get("remaining_wall_h")),
            "worker_wall_gpu_count": _finite(current_budget.get("worker_wall_gpu_count")),
            "worker_wall_gpu_h": _finite(current_budget.get("worker_wall_gpu_h_total")),
            "completed_result_gpu_h": _finite(latest.get("worker_gpu_h_total")),
        },
        "endpoint": {
            "snapshot_scope": "latest scientific evidence; counts and rates are not recomputed from current runtime metrics",
            "strict_count": latest.get("strict_count"),
            "structure_unique_successes_tm_live": su_count,
            "su_per_worker_wall_gpu_h": su_rate,
            "foldseek_su_status": latest.get("foldseek_su_status"),
            "foldseek_su_coverage": _finite(latest.get("foldseek_su_coverage")),
            "structure_dedup_scope": latest.get("structure_dedup_scope"),
            "sequence_dedup_status": latest.get("sequence_dedup_status"),
            "sequence_dedup_coverage": _finite(latest.get("sequence_dedup_coverage")),
            "sequence_unique_strict_count": latest.get("seq_unique_strict_count"),
        },
        "records": {
            "results": len(results),
            "evidence_summaries": len(evidence),
            "action_candidates": len(candidates),
            "launch_decisions": len(launches),
            "dispatch_records": len(dispatches),
            "panel_selections": len(panels),
        },
        "results_by_family": _counter(results, "backend_family"),
        "result_exit_status": _counter(results, "exit_status"),
        "launch_status": _counter(launches, "status"),
        "dispatch_status": _counter(dispatches, "status"),
        "dispatch_outcome": count_dispatch_outcomes(dispatches),
        "started_by_family": _counter(started, "method_family"),
        "started_by_mode": _counter(started, "supervisor_mode"),
        "llm_usage": {
            "calls": len(calls),
            **_token_usage(calls),
            "by_role": by_role,
        },
        "provenance_files": provenance_files,
        "controller_outcomes": controller_outcomes,
    }


def _group_by(
    rows: Iterable[dict[str, Any]], key: str
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "<missing>"))].append(row)
    return dict(grouped)


def _dispatch_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "candidate_id",
        "status",
        "method_family",
        "operator_id",
        "supervisor_mode",
        "worker_slot",
        "gpu_id",
        "score_credit_basis",
    )
    projected = {field: row.get(field) for field in fields}
    projected["outcome"] = classify_dispatch_outcome(row)
    return projected


def validate_archive(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    streams: dict[str, list[dict[str, Any]]] = {}
    if not root.is_dir():
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "ok": False,
            "errors": [f"archive root is not a directory: {root}"],
            "warnings": [],
            "record_counts": {},
            "typed_record_counts": {},
            "skipped_records": {},
            "target_ids": [],
            "provenance_files": [],
            "lineage_references": {
                "total": 0,
                "candidate_only": 0,
                "result_only": 0,
                "ambiguous": 0,
                "external_or_unknown": 0,
                "external_or_unknown_examples": [],
            },
        }

    for name in JSONL_FILES:
        rows, stream_errors = _read_jsonl(root / name)
        streams[name] = rows
        counts[name] = len(rows)
        errors.extend(stream_errors)

    typed_archive = Archive(root)
    layout_by_file = {
        stream["file_name"]: stream for stream in archive_layout()["streams"]
    }
    typed_counts: dict[str, int] = {}
    for record_type, file_name in RECORD_FILES.items():
        try:
            typed_counts[file_name] = sum(
                1 for _ in typed_archive.iter_records(record_type)
            )
        except (OSError, TypeError, ValueError) as exc:
            typed_counts[file_name] = 0
            errors.append(
                f"{file_name}: typed schema validation failed: "
                f"{type(exc).__name__}: {exc}"
            )
        stream_schema = layout_by_file[file_name]
        fields_by_name = {field["name"]: field for field in stream_schema["fields"]}
        for record_number, row in enumerate(streams[file_name], start=1):
            missing = sorted(
                field_name
                for field_name, field_schema in fields_by_name.items()
                if field_schema["required"] and field_name not in row
            )
            if missing:
                errors.append(
                    f"{file_name}:record {record_number}: missing required fields: {missing}"
                )
            unknown = sorted(set(row) - set(fields_by_name))
            if unknown:
                errors.append(
                    f"{file_name}:record {record_number}: unknown fields: {unknown}"
                )
            for field_name, field_schema in fields_by_name.items():
                if field_name in row and not _matches_json_field_shape(
                    row[field_name],
                    field_schema["type"],
                ):
                    errors.append(
                        f"{file_name}:record {record_number}: field {field_name!r} "
                        f"expected {field_schema['type']}; got {type(row[field_name]).__name__}"
                    )
    typed_skips = typed_archive.read_skip_counts()
    for key, count in sorted(typed_skips.items()):
        if key.endswith(":schema_drift"):
            errors.append(f"{key}: {count} record(s) failed typed reconstruction")

    results = streams["result_records.jsonl"]
    for index, row in enumerate(results, start=1):
        for field in ("metrics", "metrics_calibrated"):
            values = row.get(field)
            if isinstance(values, dict):
                for name, value in values.items():
                    if value is not None and (
                        isinstance(value, bool) or not isinstance(value, (int, float))
                        or _finite(value) is None
                    ):
                        errors.append(
                            f"result_records.jsonl:record {index}: invalid numeric {field}.{name}: {value!r}"
                        )
        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            for axis in STRICT_SUCCESS:
                value = metrics.get(axis)
                # Missing measurements are valid for failed/diagnostic-only
                # records, but a supplied invalid value must not mint credit.
                if value is not None and not valid_qualification_measurement(axis, value):
                    errors.append(
                        f"result_records.jsonl:record {index}: invalid canonical metric {axis}: {value!r}"
                    )
    result_ids = _string_values(results, "result_id")
    duplicate_results = sorted(
        value for value, count in collections.Counter(result_ids).items() if count > 1
    )
    if duplicate_results:
        errors.append(f"duplicate result_id values: {duplicate_results[:20]}")
    for stream_name, identifier in (
        ("llm_call_records.jsonl", "call_id"),
        ("launch_decisions.jsonl", "launch_id"),
        ("dispatch_records.jsonl", "dispatch_id"),
        ("route_records.jsonl", "route_id"),
        ("panel_selections.jsonl", "panel_id"),
    ):
        duplicates = sorted(
            value
            for value, count in collections.Counter(
                _string_values(streams[stream_name], identifier)
            ).items()
            if count > 1
        )
        if duplicates:
            errors.append(f"duplicate {identifier} values: {duplicates[:20]}")

    candidates = streams["action_candidates.jsonl"]
    candidate_ids = set(_string_values(candidates, "candidate_id"))
    result_id_set = set(result_ids)
    duplicate_candidates = sorted(
        value
        for value, count in collections.Counter(
            _string_values(candidates, "candidate_id")
        ).items()
        if count > 1
    )
    if duplicate_candidates:
        warnings.append(
            "candidate_id values recur across ticks/resumes; joins use the archived "
            f"candidate rows: {duplicate_candidates[:20]}"
        )
    parent_tokens = _list_string_values(results, "parent_ids")
    candidate_only = [
        value
        for value in parent_tokens
        if value in candidate_ids and value not in result_id_set
    ]
    result_only = [
        value
        for value in parent_tokens
        if value in result_id_set and value not in candidate_ids
    ]
    ambiguous = [
        value
        for value in parent_tokens
        if value in candidate_ids and value in result_id_set
    ]
    external_or_unknown = [
        value
        for value in parent_tokens
        if value not in candidate_ids and value not in result_id_set
    ]
    lineage_references = {
        "total": len(parent_tokens),
        "candidate_only": len(candidate_only),
        "result_only": len(result_only),
        "ambiguous": len(ambiguous),
        "external_or_unknown": len(external_or_unknown),
        "external_or_unknown_examples": sorted(set(external_or_unknown))[:20],
    }

    for stream_name in ("launch_decisions.jsonl", "dispatch_records.jsonl"):
        unknown = sorted(
            set(_string_values(streams[stream_name], "candidate_id")) - candidate_ids
        )
        if unknown:
            warnings.append(
                f"{stream_name} references {len(unknown)} candidate(s) absent from "
                f"action_candidates.jsonl; first={unknown[:5]}"
            )

    launch_ids = set(_string_values(streams["launch_decisions.jsonl"], "launch_id"))
    unknown_launch_ids = sorted(
        set(_string_values(streams["dispatch_records.jsonl"], "launch_id")) - launch_ids
    )
    if unknown_launch_ids:
        warnings.append(
            f"dispatch_records.jsonl references unknown launch_id values: "
            f"{unknown_launch_ids[:20]}"
        )

    for field_name in ("parent_result_id", "baseline_result_id"):
        unknown_result_ids = sorted(
            set(_string_values(candidates, field_name)) - result_id_set
        )
        if unknown_result_ids:
            warnings.append(
                f"action_candidates.jsonl references unknown {field_name} values: "
                f"{unknown_result_ids[:20]}"
            )

    hypothesis_ids = set(
        _string_values(streams["hypothesis_cards.jsonl"], "hypothesis_id")
    )
    unknown_hypothesis_ids = sorted(
        set(_list_string_values(candidates, "hypothesis_ids"))
        - hypothesis_ids
        - SYSTEM_CANDIDATE_ORIGIN_IDS
    )
    if unknown_hypothesis_ids:
        warnings.append(
            "action_candidates.jsonl references unknown hypothesis_id values: "
            f"{unknown_hypothesis_ids[:20]}"
        )

    unknown_panel_result_ids = sorted(
        set(_list_string_values(streams["panel_selections.jsonl"], "selected_ids"))
        - result_id_set
    )
    if unknown_panel_result_ids:
        errors.append(
            "panel_selections.jsonl references unknown result_id values: "
            f"{unknown_panel_result_ids[:20]}"
        )

    target_ids = {
        str(row["target_id"])
        for name in (
            "result_records.jsonl",
            "evidence_summaries.jsonl",
            "target_constraints.jsonl",
        )
        for row in streams[name]
        if row.get("target_id") is not None
    }
    if len(target_ids) > 1:
        errors.append(f"archive contains multiple target_ids: {sorted(target_ids)}")
    if not streams["evidence_summaries.jsonl"]:
        warnings.append(
            "no EvidenceSummary rows; endpoint and controller state are unavailable"
        )
    if not streams["dispatch_records.jsonl"]:
        warnings.append(
            "no DispatchRecord rows; selected-to-started realization cannot be audited"
        )
    provenance = provenance_paths(root)
    if not provenance:
        warnings.append(
            "no codex_provenance_*.json or legacy run_provenance*.json; execution provenance is unavailable"
        )
    else:
        for path in provenance:
            try:
                value = _metadata(path)
            except (OSError, ValueError, RecursionError) as exc:
                errors.append(f"{path.name}: invalid provenance JSON: {exc}")
                continue
            if not isinstance(value, dict):
                errors.append(f"{path.name}: provenance root must be a JSON object")
                continue
            if path.name.startswith("codex_provenance_"):
                if value.get("schema_version") != CODEX_PROVENANCE_SCHEMA:
                    errors.append(f"{path.name}: unsupported provenance schema: {value.get('schema_version')!r}")
                else:
                    _validate_codex_provenance(root, path, value, errors, warnings)
                continue
            for section, digest_key in (
                ("source", "tree_sha256"),
                ("target", "pdb_sha256"),
                ("model", "content_sha256"),
            ):
                component = value.get(section)
                if not isinstance(component, dict) or not component.get(digest_key):
                    errors.append(f"{path.name}: missing {section}.{digest_key}")
            schema_version = value.get("schema_version")
            if (
                schema_version is not None
                and schema_version not in SUPPORTED_PROVENANCE_SCHEMAS
            ):
                errors.append(
                    f"{path.name}: unsupported provenance schema: {schema_version!r}"
                )
            elif schema_version == PROVENANCE_SCHEMA:
                prompts = value.get("prompts")
                prompt_roles = (
                    prompts.get("roles", {}) if isinstance(prompts, dict) else {}
                )
                if not isinstance(prompt_roles, dict):
                    prompt_roles = {}
                for role in ("planner", "supervisor", "critic"):
                    role_payload = prompt_roles.get(role)
                    prompt_sha = (
                        role_payload.get("system_prompt_sha256")
                        if isinstance(role_payload, dict)
                        else None
                    )
                    if (
                        not isinstance(prompt_sha, str)
                        or len(prompt_sha) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in prompt_sha
                        )
                    ):
                        errors.append(
                            f"{path.name}: missing/invalid "
                            f"prompts.roles.{role}.system_prompt_sha256"
                        )
                campaign = value.get("campaign")
                if campaign is not None:
                    if not isinstance(campaign, dict):
                        errors.append(
                            f"{path.name}: campaign provenance must be an object"
                        )
                    else:
                        for artifact_key in (
                            "input_artifact",
                            "resolved_artifact",
                        ):
                            declared = campaign.get(artifact_key)
                            expected_sha = campaign.get(f"{artifact_key}_sha256")
                            if not declared or not expected_sha:
                                errors.append(
                                    f"{path.name}: missing "
                                    f"campaign.{artifact_key} linkage"
                                )
                                continue
                            artifact_path = root / Path(str(declared)).name
                            if not artifact_path.is_file():
                                errors.append(
                                    f"{path.name}: linked artifact not found: "
                                    f"{artifact_path.name}"
                                )
                            elif _sha256_file(artifact_path) != expected_sha:
                                errors.append(
                                    f"{path.name}: linked artifact SHA256 "
                                    f"mismatch: {artifact_path.name}"
                                )
                        source_sha = campaign.get("source_sha256")
                        input_sha = campaign.get("input_artifact_sha256")
                        if source_sha and input_sha and source_sha != input_sha:
                            errors.append(
                                f"{path.name}: campaign source/input SHA256 " "mismatch"
                            )

    controller_outcomes, outcome_errors, outcome_warnings = codex_outcome_summaries(root)
    errors.extend(outcome_errors)
    warnings.extend(outcome_warnings)
    latest = (
        streams["evidence_summaries.jsonl"][-1]
        if streams["evidence_summaries.jsonl"]
        else {}
    )
    if latest:
        if latest.get("foldseek_su_status") != "ok":
            warnings.append(
                f"latest Foldseek SU status is {latest.get('foldseek_su_status')!r}"
            )
        coverage = _finite(latest.get("foldseek_su_coverage"))
        if coverage is not None and coverage < 0.999:
            warnings.append(
                f"latest Foldseek SU coverage is {coverage:.3f}, below 0.999"
            )
        scope = latest.get("structure_dedup_scope")
        if scope not in (None, "binder_chain"):
            warnings.append(
                f"latest structure dedup scope is {scope!r}, not 'binder_chain'"
            )
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "record_counts": counts,
        "typed_record_counts": typed_counts,
        "skipped_records": typed_skips,
        "target_ids": sorted(target_ids),
        "provenance_files": [path.name for path in provenance],
        "controller_outcomes": controller_outcomes,
        "lineage_references": lineage_references,
    }


def decision_trace(root: Path, *, limit: int = 10) -> list[dict[str, Any]]:
    root = _require_archive_root(root)
    if limit < 1:
        raise ValueError("trace limit must be positive")
    evidence = _records(root, "evidence_summaries.jsonl")
    hypotheses = _records(root, "hypothesis_cards.jsonl")
    candidates = _records(root, "action_candidates.jsonl")
    supervisors = _records(root, "supervisor_decisions.jsonl")
    launches = _records(root, "launch_decisions.jsonl")
    dispatches = _records(root, "dispatch_records.jsonl")

    ticks = [str(row.get("tick_id")) for row in evidence if row.get("tick_id")]
    selected_ticks = ticks[-max(1, limit) :]
    evidence_by_tick = {str(row.get("tick_id")): row for row in evidence}
    supervisor_by_tick = {str(row.get("tick_id")): row for row in supervisors}
    hypotheses_by_tick = {
        tick: list({str(row.get("hypothesis_id")): row for row in rows}.values())
        for tick, rows in _group_by(hypotheses, "tick_created").items()
    }
    candidate_rows_by_id = {
        candidate_id: rows
        for candidate_id, rows in _group_by(candidates, "candidate_id").items()
        if candidate_id != "<missing>"
    }
    candidate_by_id = {
        candidate_id: rows[0]
        for candidate_id, rows in candidate_rows_by_id.items()
        if len(rows) == 1
    }
    launches_by_tick = _group_by(launches, "tick_id")
    dispatch_by_tick = _group_by(dispatches, "tick_id")

    hypothesis_ticks: list[str] = []
    action_ticks: list[str] = []
    for tick in ticks:
        trailing_tick = re.search(r"(\d+)$", tick)
        hypothesis_key = str(int(trailing_tick.group(1))) if trailing_tick else tick
        if hypotheses_by_tick.get(hypothesis_key):
            hypothesis_ticks.append(tick)
        if launches_by_tick.get(tick) or dispatch_by_tick.get(tick):
            action_ticks.append(tick)
    selected_ticks = (hypothesis_ticks or action_ticks or ticks)[-max(1, limit) :]

    output: list[dict[str, Any]] = []
    for tick in selected_ticks:
        ev = evidence_by_tick.get(tick, {})
        supervisor = supervisor_by_tick.get(tick, {})
        trailing_tick = re.search(r"(\d+)$", tick)
        hypothesis_key = str(int(trailing_tick.group(1))) if trailing_tick else tick
        cards = hypotheses_by_tick.get(hypothesis_key, [])
        launch_rows = launches_by_tick.get(tick, [])
        dispatch_rows = dispatch_by_tick.get(tick, [])
        launched_candidates = []
        for launch in launch_rows:
            candidate_id = str(launch.get("candidate_id"))
            candidate_matches = candidate_rows_by_id.get(candidate_id, [])
            candidate = candidate_by_id.get(candidate_id, {})
            launched_candidates.append(
                {
                    "candidate_id": launch.get("candidate_id"),
                    "candidate_join_status": (
                        "unique"
                        if len(candidate_matches) == 1
                        else "missing"
                        if not candidate_matches
                        else "ambiguous"
                    ),
                    "candidate_match_count": len(candidate_matches),
                    "launch_status": launch.get("status"),
                    "family": candidate.get("method_family"),
                    "operator": candidate.get("operator_id"),
                    "mode": candidate.get("supervisor_mode"),
                    "config_delta": candidate.get("config_delta", {}),
                    "expected_signal": candidate.get("expected_signal"),
                    "why": launch.get("why"),
                }
            )
        output.append(
            {
                "schema_version": TRACE_ENTRY_SCHEMA_VERSION,
                "tick_id": tick,
                "state_label": ev.get("state_label"),
                "diagnostic_driver": ev.get("diagnostic_driver_tldr"),
                "strict_count": ev.get("strict_count"),
                "run_su_count": ev.get("run_su_count"),
                "su_per_worker_wall_gpu_h": ev.get(
                    "run_su_per_worker_wall_gpu_h_total"
                ),
                "hypotheses": [
                    {
                        "hypothesis_id": card.get("hypothesis_id"),
                        "claim": card.get("claim"),
                        "mode_affinity": card.get("mode_affinity"),
                        "evidence_refs": card.get("evidence_refs"),
                        "recommended_action_families": card.get(
                            "recommended_action_families"
                        ),
                        "reasoning_trace": card.get("reasoning_trace"),
                    }
                    for card in cards
                ],
                "supervisor": {
                    "mode_mixture": supervisor.get("mode_mixture"),
                    "fallback_used": supervisor.get("fallback_used"),
                    "clamps_applied": supervisor.get("clamps_applied", []),
                    "rationale": supervisor.get("rationale"),
                },
                "launches": launched_candidates,
                "dispatches": [_dispatch_trace_row(row) for row in dispatch_rows],
            }
        )
    return output


def _print_summary(payload: dict[str, Any]) -> None:
    endpoint = payload["endpoint"]
    budget = payload["budget"]
    print(f"Archive: {payload['archive_root']}")
    print(f"Target:  {', '.join(payload['target_ids']) or '<unknown>'}")
    print(
        f"Tick/state: {payload['latest_tick'] or '<none>'} / "
        f"{payload['state_label'] or '<unknown>'}"
    )
    print(
        "Current results (canonical AF2 only): "
        f"evaluated={payload['current_results']['canonical_af2_evaluated_count']} "
        f"passed={payload['current_results']['canonical_af2_pass_count']} "
        f"native status among passes={payload['current_results']['native_status_among_canonical_pass']}"
    )
    print(
        "Endpoint (latest scientific snapshot): "
        f"strict={endpoint['strict_count']} "
        f"SU={endpoint['structure_unique_successes_tm_live']} "
        f"SU/worker-wall-GPU-h={endpoint['su_per_worker_wall_gpu_h']}"
    )
    print(
        f"Budget ({budget['source']}): "
        f"elapsed_h={budget['elapsed_wall_h']} "
        f"remaining_h={budget['remaining_wall_h']} "
        f"worker-wall GPU-h={budget['worker_wall_gpu_h']}"
    )
    print(
        "Dedup:   "
        f"Foldseek={endpoint['foldseek_su_status']} "
        f"coverage={endpoint['foldseek_su_coverage']} "
        f"scope={endpoint['structure_dedup_scope']}"
    )
    print(
        "LLM records: "
        f"count={payload['llm_usage']['calls']} "
        f"input_tokens={payload['llm_usage']['tokens_in']} "
        f"output_tokens={payload['llm_usage']['tokens_out']}"
    )
    print(
        "  Counts are archive records, not API invocation counts "
        "(includes deterministic guards and skipped decisions)."
    )
    print(
        "Started by family: " + json.dumps(payload["started_by_family"], sort_keys=True)
    )


def _print_trace(payload: list[dict[str, Any]]) -> None:
    if not payload:
        print("No evidence-backed decision ticks found.")
        return
    for index, tick in enumerate(payload):
        if index:
            print()
        print(
            f"{tick['tick_id']}: state={tick['state_label']} "
            f"strict={tick['strict_count']} SU={tick['run_su_count']}"
        )
        print(f"  diagnostic: {tick['diagnostic_driver'] or '<not recorded>'}")
        print(
            "  hypotheses: "
            + (
                ", ".join(str(card.get("hypothesis_id")) for card in tick["hypotheses"])
                if tick["hypotheses"]
                else "none"
            )
        )
        for label, rows, status_key in (
            ("launches", tick["launches"], "launch_status"),
            ("dispatches", tick["dispatches"], "outcome"),
        ):
            rendered = ", ".join(
                f"{row.get('candidate_id')}:{row.get(status_key)}" for row in rows
            )
            print(f"  {label}: {rendered or 'none'}")


def _print_archive_layout(payload: dict[str, Any]) -> None:
    print(f"Archive layout: {payload['schema_version']}")
    print(f"Lineage: {payload['lineage_semantics']}")
    for stream in payload["streams"]:
        primary_key = ", ".join(stream["primary_key"]) or "<none>"
        join_keys = ", ".join(stream["join_keys"]) or "<none>"
        print(
            f"  {stream['file_name']:<30} {stream['record_type']:<22} "
            f"primary={primary_key}; joins={join_keys}"
        )
        print(f"    {stream['description']}")
    print("Relationships (explode list-valued source fields before joining):")
    for relationship in payload["relationships"]:
        print(
            "  "
            f"{relationship['source_stream']}.{relationship['source_field']} -> "
            f"{relationship['target_stream']}.{relationship['target_field']}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("summary", "validate"):
        command = sub.add_parser(name)
        command.add_argument("--archive-root", type=Path, required=True)
        command.add_argument("--json", action="store_true", dest="as_json")
    trace = sub.add_parser("trace", help="join recent evidence-to-dispatch decisions")
    trace.add_argument("--archive-root", type=Path, required=True)
    trace.add_argument("--limit", type=int, default=10)
    trace.add_argument("--json", action="store_true", dest="as_json")
    schema = sub.add_parser(
        "schema", help="describe archive streams, fields, and join keys"
    )
    schema.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "summary":
            payload = campaign_summary(args.archive_root)
            if args.as_json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_summary(payload)
            return 0
        if args.command == "validate":
            payload = validate_archive(args.archive_root)
            if args.as_json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print("PASS" if payload["ok"] else "FAIL")
                for item in payload["errors"]:
                    print(f"ERROR: {item}")
                for item in payload["warnings"]:
                    print(f"WARN:  {item}")
                print(
                    "Records: " + json.dumps(payload["record_counts"], sort_keys=True)
                )
            return 0 if payload["ok"] else 1
        if args.command == "trace":
            payload = decision_trace(args.archive_root, limit=args.limit)
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "schema_version": TRACE_OUTPUT_SCHEMA_VERSION,
                            "archive_root": str(
                                args.archive_root.expanduser().resolve()
                            ),
                            "limit": args.limit,
                            "entries": payload,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                _print_trace(payload)
            return 0
        if args.command == "schema":
            payload = archive_layout()
            if args.as_json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_archive_layout(payload)
            return 0
    except (OSError, ValueError) as exc:
        print(f"trex-analyze: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
