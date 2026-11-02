"""BindCraft-native filter evidence, separate from canonical qualification.

Reproduce the pinned backend's inclusive threshold comparisons on its CSV
values. Missing values are recorded as skipped, as in BindCraft check_filters;
they are never evidence that a condition passed. No aggregate failure counter
is used to attribute a failure to an individual design.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


ASSESSMENT_SCHEMA = "binder.bindcraft-native-filter-assessment.v1"
SNAPSHOT_SCHEMA = "binder.bindcraft-filter-snapshot.v1"
_INTERFACE_AAS = {"Average_InterfaceAAs", *(f"{n}_InterfaceAAs" for n in range(1, 6))}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_stable(path: Path) -> bytes:
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed during read: {path}")
    return raw


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _json(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object)


def _number(value: Any) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("measurement/threshold is not a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("measurement/threshold is not finite")
    return result


def _rules(filters: Any) -> list[tuple[str, str | None, float | int | None, str]]:
    if not isinstance(filters, dict) or not filters:
        raise ValueError("filters must be a nonempty JSON object")
    rules = []
    for label, definition in filters.items():
        nested = label in _INTERFACE_AAS
        if not isinstance(definition, dict) or not definition:
            raise ValueError(f"invalid filter definition: {label}")
        entries = definition.items() if nested else [(None, definition)]
        for amino_acid, condition in entries:
            if not isinstance(condition, dict) or set(condition) != {"threshold", "higher"}:
                raise ValueError(f"invalid threshold definition: {label}/{amino_acid}")
            if not isinstance(condition["higher"], bool):
                raise ValueError(f"higher must be boolean: {label}/{amino_acid}")
            threshold = condition["threshold"]
            if threshold is not None:
                # JSON numeric strings are not valid backend threshold values.
                if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                    raise ValueError(f"threshold must be numeric or null: {label}/{amino_acid}")
                threshold = _number(threshold)
            rules.append((label, amino_acid, threshold, ">=" if condition["higher"] else "<="))
    return rules


def snapshot_bindcraft_filters(source_path: Path, output_dir: Path) -> Path:
    """Pin exact bytes before launch; keep the upstream filter protocol basename.

    An existing identical snapshot is reusable. A partial or changed snapshot is
    refused instead of overwriting evidence from a prior launch.
    """
    source_path, output_dir = Path(source_path), Path(output_dir)
    raw = _read_stable(source_path)
    _rules(_json(raw))
    directory = output_dir / "native_filters"
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("native filter snapshot directory must not be a symlink")
    snapshot = directory / source_path.name
    provenance = directory / "provenance.json"
    if snapshot == provenance:
        raise ValueError("filter basename conflicts with snapshot provenance")
    metadata = {
        "schema_version": SNAPSHOT_SCHEMA,
        "source": {"path": str(source_path.resolve()), "sha256": _sha(raw)},
        "snapshot": {"filename": snapshot.name, "sha256": _sha(raw)},
    }
    if snapshot.exists() or provenance.exists():
        if snapshot.is_symlink() or provenance.is_symlink():
            raise ValueError("native filter evidence must not be a symlink")
        if not snapshot.is_file() or not provenance.is_file():
            raise ValueError("incomplete native filter snapshot")
        if _read_stable(snapshot) != raw or _json(_read_stable(provenance)) != metadata:
            raise ValueError("refusing to replace existing native filter evidence")
        return snapshot
    with snapshot.open("xb") as handle:
        handle.write(raw)
    with provenance.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return snapshot


def _nested_value(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw, object_pairs_hook=_unique_object)
        except json.JSONDecodeError:
            # BindCraft CSV serializes a Python dictionary; never execute it.
            raw = ast.literal_eval(raw)
    if not isinstance(raw, dict):
        raise ValueError("InterfaceAAs must be a dictionary")
    return raw


def assess_bindcraft_native_filters(
    output_dir: Path,
    *,
    design_name: str,
    native_status: str,
    row: Mapping[str, Any] | None,
    csv_path: Path | None,
) -> dict[str, Any]:
    """Return provenance-bound, per-design native evidence without minting strict.

    ``status`` retains the observed native directory status. ``complete`` means
    that the original comparisons explain that status, within the explicitly
    recorded evaluated/skipped scope; it is not canonical or wet-lab acceptance.
    """
    status = native_status if native_status in {"accepted", "rejected"} else "unknown"
    source: dict[str, Any] = {"design_name": design_name}
    result: dict[str, Any] = {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": status,
        "assessment_status": "unavailable",
        "failed_conditions": [],
        "source": source,
    }
    try:
        directory = Path(output_dir) / "native_filters"
        provenance = directory / "provenance.json"
        metadata_raw = _read_stable(provenance)
        metadata = _json(metadata_raw)
        if not isinstance(metadata, dict) or metadata.get("schema_version") != SNAPSHOT_SCHEMA:
            raise ValueError("unknown native filter snapshot schema")
        filename = metadata["snapshot"]["filename"]
        if not isinstance(filename, str) or not filename or Path(filename).name != filename or "\\" in filename:
            raise ValueError("invalid filter snapshot filename")
        snapshot = directory / filename
        if directory.is_symlink() or snapshot.is_symlink() or provenance.is_symlink():
            raise ValueError("native filter evidence must not be a symlink")
        raw = _read_stable(snapshot)
        source["filters"] = {"path": str(snapshot.resolve()), "sha256": _sha(raw)}
        source["snapshot_provenance"] = {"path": str(provenance.resolve()), "sha256": _sha(metadata_raw)}
        source["original_filters"] = metadata["source"]
        if _sha(raw) != metadata["snapshot"]["sha256"] or _sha(raw) != metadata["source"]["sha256"]:
            raise ValueError("native filter snapshot hash mismatch")
        rules = _rules(_json(raw))
        if row is None or csv_path is None:
            raise ValueError("no per-design CSV row available")
        csv_path = Path(csv_path)
        csv_raw = _read_stable(csv_path)
        source["csv"] = {"path": str(csv_path.resolve()), "sha256": _sha(csv_raw)}
        matches = [r for r in csv.DictReader(io.StringIO(csv_raw.decode("utf-8-sig")))
                   if (r.get("Design") or "").strip() == design_name]
        if len(matches) != 1 or matches[0] != dict(row):
            raise ValueError("per-design CSV row changed, is duplicated, or does not match")
        source["csv_row_sha256"] = _sha(json.dumps(dict(row), sort_keys=True, ensure_ascii=False,
                                                   allow_nan=False).encode("utf-8"))
        source["model_scope"] = {
            "basis": "nonempty per-model CSV measurements; no inference of absent models",
            "observed_model_indices": sorted({int(m.group(1)) for k, v in row.items()
                if v not in (None, "") and (m := re.match(r"^([1-5])_", k))}),
        }
        skipped: list[dict[str, str]] = []
        evaluated = 0
        failed = result["failed_conditions"]
        for label, amino_acid, threshold, operator in rules:
            if threshold is None:
                continue
            metric = f"{label}_{amino_acid}" if amino_acid is not None else label
            value = row.get(label)
            if value in (None, ""):
                skipped.append({"metric": metric, "reason": "missing_value_skipped_by_native_rule"})
                continue
            if amino_acid is not None:
                value = _nested_value(value).get(amino_acid)
                if value is None:
                    skipped.append({"metric": metric, "reason": "missing_amino_acid_skipped_by_native_rule"})
                    continue
            value = _number(value)
            evaluated += 1
            if (operator == ">=" and value < threshold) or (operator == "<=" and value > threshold):
                failed.append({"metric": metric, "value": value, "threshold": threshold, "operator": operator})
        source["evaluated_condition_count"] = evaluated
        source["skipped_conditions"] = skipped
        source["scope"] = "Per-design native CSV comparisons only; aggregate failure_csv is not an attribution source."
        if not evaluated:
            raise ValueError("no native filter condition could be evaluated")
        if status == "unknown":
            raise ValueError("native artifact status is unknown")
        if (status == "rejected" and not failed) or (status == "accepted" and failed):
            result["assessment_status"] = "inconsistent"
            result["unavailable_reason"] = "native artifact status disagrees with evaluated per-design conditions"
        elif status == "accepted" and skipped:
            # Original None-skipping is preserved, but missing measurements are
            # insufficient for this assessment to affirm a native pass.
            result["unavailable_reason"] = "native accepted status observed, but active conditions lack measurements"
        else:
            result["assessment_status"] = "complete"
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, SyntaxError, OverflowError, RecursionError) as exc:
        result["unavailable_reason"] = str(exc)
    return result


__all__ = ["assess_bindcraft_native_filters", "snapshot_bindcraft_filters"]
