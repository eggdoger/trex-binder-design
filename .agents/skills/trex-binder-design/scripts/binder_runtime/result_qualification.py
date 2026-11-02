"""Report canonical AF2 qualification separately from native backend filtering.

Stored scores alone do not prove canonical evaluation. Resolve archive lineage
without consulting a mutable backend installation or borrowing an ancestor's
filter result after a sequence redesign. This module never changes score gates.
"""

from __future__ import annotations

import json
import math
import operator
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .capability_registry import default_registry
from .native_filters import ASSESSMENT_SCHEMA
from .refilter_roles import CANONICAL_SCORE_CONVERSION, PARENT_MODEL_REFOLD
from .schemas import ResultRecord
from .success_criteria import is_strict_success

ResultLike = ResultRecord | Mapping[str, Any]
ResultIndex = dict[str, list[ResultLike]]
_NATIVE_FAMILIES = frozenset(
    family for family, capability in default_registry().capabilities.items()
    if capability.role in {"generator", "seq_redesign"}
)
_COMPARATORS = {"<": operator.lt, "<=": operator.le, ">": operator.gt,
                ">=": operator.ge, "==": operator.eq, "!=": operator.ne}


def _get(record: ResultLike, key: str, default: Any = None) -> Any:
    return record.get(key, default) if isinstance(record, Mapping) else getattr(record, key, default)


def build_result_index(records: Iterable[ResultLike]) -> ResultIndex:
    """Keep duplicate IDs visible, rather than silently choosing one record."""
    index: ResultIndex = defaultdict(list)
    for record in records:
        index[str(_get(record, "result_id", ""))].append(record)
    return dict(index)


def _validate_complete_native_source(assessment: dict[str, Any], source: dict[str, Any]) -> None:
    """Require archived evidence references before accepting a complete claim.

    This validates the stored evidence contract; it does not reread remote files
    or claim that a stored hash independently authenticates its scientific data.
    """
    if assessment.get("schema_version") != ASSESSMENT_SCHEMA:
        raise ValueError("complete native assessment has unknown or missing schema")
    if "failed_conditions" not in assessment:
        raise ValueError("complete native assessment lacks failure condition list")
    for name in ("filters", "csv", "snapshot_provenance", "original_filters"):
        reference = source.get(name)
        if (not isinstance(reference, dict)
                or not isinstance(reference.get("path"), str) or not reference["path"].strip()
                or not isinstance(reference.get("sha256"), str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", reference["sha256"]) is None):
            raise ValueError(f"complete native assessment lacks valid {name} path/hash")
    if source["filters"]["sha256"].lower() != source["original_filters"]["sha256"].lower():
        raise ValueError("complete native assessment filter source hashes disagree")
    row_hash = source.get("csv_row_sha256")
    if not isinstance(row_hash, str) or re.fullmatch(r"[0-9a-fA-F]{64}", row_hash) is None:
        raise ValueError("complete native assessment lacks valid CSV row hash")
    evaluated = source.get("evaluated_condition_count")
    if type(evaluated) is not int or evaluated < 1 or evaluated < len(assessment["failed_conditions"]):
        raise ValueError("complete native assessment has invalid evaluated condition count")
    skipped = source.get("skipped_conditions")
    if not isinstance(skipped, list) or (assessment["status"] == "accepted" and skipped):
        raise ValueError("complete native acceptance has missing or skipped condition evidence")


def _native_assessment(record: ResultLike) -> dict[str, Any]:
    bins = _get(record, "bins", {}) or {}
    result = {
        "native_filter_status": "unknown",
        "native_filter_assessment_status": "unavailable",
        "native_filter_failures": [],
        "native_filter_source": {},
        "native_filter_unavailable_reason": "native_filter_assessment_not_recorded",
    }
    family = _get(record, "backend_family", "")
    if family not in _NATIVE_FAMILIES:
        result["native_filter_unavailable_reason"] = "native_source_family_unknown"
        return result
    legacy = str(bins.get("bindcraft_filter_status", "")).lower() if family == "bindcraft" else ""
    if legacy in {"accepted", "rejected"}:
        result["native_filter_status"] = legacy
        result["native_filter_unavailable_reason"] = "legacy_status_only; filter criteria and failure details were not archived"
    raw = bins.get("native_filter_assessment")
    if raw is None:
        return result
    try:
        assessment = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(assessment, dict):
            raise ValueError("assessment is not an object")
        json.dumps(assessment, allow_nan=False)
        if (str(assessment.get("schema_version", "")).startswith("binder.bindcraft-")
                and family != "bindcraft"):
            raise ValueError("BindCraft assessment belongs to a different backend family")
        status = assessment.get("status")
        completeness = assessment.get("assessment_status")
        failures = assessment.get("failed_conditions", [])
        source = assessment.get("source", {})
        if status not in {"accepted", "rejected", "unknown"}:
            raise ValueError("invalid native status")
        if completeness not in {"complete", "unavailable", "inconsistent"}:
            raise ValueError("invalid native assessment status")
        if not isinstance(failures, list) or any(not isinstance(item, dict) for item in failures):
            raise ValueError("invalid failure conditions")
        for item in failures:
            if (not isinstance(item.get("metric"), str) or not item["metric"]
                    or item.get("operator") not in _COMPARATORS
                    or any(isinstance(item.get(key), bool)
                           or not isinstance(item.get(key), (float, int))
                           or not math.isfinite(item[key]) for key in ("value", "threshold"))):
                raise ValueError("invalid failure condition fields")
            if _COMPARATORS[item["operator"]](item["value"], item["threshold"]):
                raise ValueError("reported failure satisfies its passing condition")
        if not isinstance(source, dict):
            raise ValueError("invalid native evidence source")
        if completeness == "complete":
            _validate_complete_native_source(assessment, source)
        inconsistent = (
            completeness == "inconsistent"
            or (legacy in {"accepted", "rejected"} and status != legacy)
            or (status == "accepted" and bool(failures))
            or (completeness == "complete" and status == "rejected" and not failures)
            or (completeness == "complete" and status == "unknown")
        )
        result.update({
            "native_filter_status": "unknown" if inconsistent else status,
            "native_filter_assessment_status": "inconsistent" if inconsistent else completeness,
            "native_filter_failures": failures,
            "native_filter_source": source,
            "native_filter_unavailable_reason": (
                "native_filter_evidence_inconsistent" if inconsistent
                else str(assessment.get("unavailable_reason") or (
                    "native_filter_evidence_not_available" if completeness != "complete" else ""
                ))
            ),
        })
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        result.update({
            "native_filter_status": "unknown",
            "native_filter_assessment_status": "inconsistent",
            "native_filter_unavailable_reason": f"invalid_native_filter_assessment: {exc}",
        })
    return result


def _parent(record: ResultLike, index: ResultIndex) -> tuple[ResultLike | None, str]:
    """Resolve explicit source first; candidate IDs are not result parents."""
    bins = _get(record, "bins", {}) or {}
    source = bins.get("refilter_source")
    known_parents = {
        str(parent) for parent in (_get(record, "parent_ids", []) or [])
        if str(parent) in index
    }
    if source:
        source = str(source)
        if source in index:
            candidates = index[source]
        elif "/" in source or "\\" in source:
            # Older AF2 records stored the complete input path. Never match a
            # basename: identically named outputs from different workers collide.
            candidates = [
                parent for records in index.values() for parent in records
                if (_get(parent, "artifacts", {}) or {}).get("pdb_path") == source
            ]
        else:
            candidates = []
        if not candidates:
            return None, "missing_refilter_source"
        if len(candidates) != 1:
            return None, "ambiguous_refilter_source"
        parent = candidates[0]
        if known_parents - {str(_get(parent, "result_id", ""))}:
            return None, "conflicting_parent_links"
    else:
        if len(known_parents) != 1:
            return None, "ambiguous_parent" if known_parents else "missing_parent"
        candidates = index[next(iter(known_parents))]
        if len(candidates) != 1:
            return None, "ambiguous_parent"
        parent = candidates[0]
    if _get(parent, "target_id") != _get(record, "target_id"):
        return None, "target_mismatch"
    hinted_family = bins.get("refilter_source_family")
    if hinted_family and hinted_family != _get(parent, "backend_family"):
        return None, "source_family_mismatch"
    return parent, "resolved"


def assess_result_qualification(record: ResultLike, index: ResultIndex) -> dict[str, Any]:
    """Assess one record using explicit canonical provenance and archive lineage.

    ``canonical_af2_pass=None`` means canonical provenance cannot be established;
    ``False`` means a noncanonical record or a verified canonical gate failure.
    Native statuses concern the directly scored generated/redesigned structure,
    not an additional Rosetta assessment of the canonical prediction.
    """
    rid = str(_get(record, "result_id", ""))
    bins = _get(record, "bins", {}) or {}
    result: dict[str, Any] = {
        "canonical_af2_pass": None,
        "canonical_assessment_status": "unknown",
        "canonical_unavailable_reason": "",
        "parent_result_id": "",
        "source_family": "",
        "lineage_status": "unresolved",
        "lineage_path": [rid],
        **_native_assessment({}),
    }
    if len(index.get(rid, [])) > 1:
        result.update({
            "lineage_status": "ambiguous_result_id",
            "canonical_unavailable_reason": "ambiguous_result_id",
            "native_filter_unavailable_reason": "ambiguous_result_id",
        })
        return result
    if _get(record, "backend_family") != "structure_refilter":
        result.update({
            "canonical_af2_pass": False,
            "canonical_assessment_status": "not_canonical",
            "canonical_unavailable_reason": "native_backend_result",
            "parent_result_id": rid,
            "source_family": _get(record, "backend_family", ""),
            "lineage_status": "native_record",
            **_native_assessment(record),
        })
        return result

    basis, role = bins.get("af2_strict_basis"), bins.get("refilter_role")
    model_count = bins.get("af2_model_count", 1)
    try:
        ensemble = isinstance(model_count, bool) or float(model_count) != 1.0
    except (TypeError, ValueError, OverflowError):
        ensemble = True
    if role == PARENT_MODEL_REFOLD or basis == "advisory_refold" or ensemble:
        result.update({
            "canonical_af2_pass": False,
            "canonical_assessment_status": "not_canonical",
            "canonical_unavailable_reason": "advisory_or_ensemble_evaluation",
        })
    elif (basis not in {None, "", "canonical"}
          or role not in {None, "", CANONICAL_SCORE_CONVERSION}
          or not (basis == "canonical" or role == CANONICAL_SCORE_CONVERSION)):
        result["canonical_unavailable_reason"] = "canonical_provenance_not_established"

    current = record
    seen = {rid}
    while True:
        parent, status = _parent(current, index)
        if parent is None:
            result["lineage_status"] = status
            result["native_filter_unavailable_reason"] = status
            if result["canonical_assessment_status"] == "unknown":
                result["canonical_unavailable_reason"] = status
            return result
        parent_id = str(_get(parent, "result_id", ""))
        result["lineage_path"].append(parent_id)
        if parent_id in seen:
            result["lineage_status"] = "cycle"
            result["native_filter_unavailable_reason"] = "lineage_cycle"
            if result["canonical_assessment_status"] == "unknown":
                result["canonical_unavailable_reason"] = "lineage_cycle"
            return result
        seen.add(parent_id)
        if _get(parent, "backend_family") != "structure_refilter":
            if _get(parent, "backend_family") not in _NATIVE_FAMILIES:
                result.update({
                    "lineage_status": "unknown_source_family",
                    "native_filter_unavailable_reason": "native_source_family_unknown",
                })
                if result["canonical_assessment_status"] == "unknown":
                    result["canonical_unavailable_reason"] = "native_source_family_unknown"
                return result
            result.update({
                "parent_result_id": parent_id,
                "source_family": _get(parent, "backend_family", ""),
                "lineage_status": "resolved",
                **_native_assessment(parent),
            })
            break
        current = parent

    if not result["canonical_unavailable_reason"]:
        result.update({
            "canonical_assessment_status": "complete",
            "canonical_af2_pass": (
                _get(record, "exit_status") == "ok"
                and bins.get("output_chain_identity") != "unresolved"
                and is_strict_success(_get(record, "metrics", {}))
            ),
        })
    return result
