"""Complexa output parser.

Reads:
    <output_dir>/rewards_<config>_<seed>.csv

CSV fields:
  Required columns:
    pdb_path                    — PDB output path on disk
    sample_type                 — "final" or "lookahead". Filter to "final".
  Native diagnostic columns (optional):
    af2folding_plddt_log        — native pLDDT in 0-1 (use ×100)
    af2folding_i_pae            — native normalized iPAE [0-1]
    af2folding_rmsd             — native design-vs-prediction Cα RMSD in Å
  Optional (diagnostic):
    af2folding_i_ptm_log        — interface pTM
    af2folding_min_ipae         — min iPAE across models
    af2folding_avg_ipsae        — averaged interface pSAE
    af2folding_max_ipsae        — max iPAE-sub-error
    af2folding_con              — contact-density score
    af2folding_i_con            — interface contact density
    af2folding_fape             — FAPE loss

Filter rules:
  - Only emit ResultRecord for rows with `sample_type == "final"`
  - All rows are diagnostic; only rows with usable structure artifacts are
    marked for independent canonical scoring.
  - If `rewards_*.csv` missing → ParseError
  - If empty (no final samples) → return []

ResultRecord mapping:
  Native AF2 reward and the independent verifier share several settings, but
  differ in template sequence masking and numerical/runtime options. Preserve
  native rewards as diagnostics without treating them as canonical strict/SU
  evidence. Usable final structures enter independent AF2 score conversion.
  result_id        := hash(target_id, candidate_id, launch namespace,
                           sample_index, pdb basename)
  parent_ids       := [ctx.candidate_id] (worker spawn point)
  backend_family   := from ctx.method_family (complexa_beam, complexa_best_of_n,
  complexa_fk_steering, complexa_mcts)
  target_id        := ctx.target_id
  metrics["complexa_native_pLDDT"]         := af2folding_plddt_log × 100
  metrics["complexa_native_iPAE"]          := af2folding_i_pae
  metrics["complexa_native_binder_scRMSD"]  := af2folding_rmsd
  + diagnostic: i_ptm, min_ipae, ipsae, contact density when present
  gpu_h := parser placeholder; the controller assigns measured job compute after
  parsing.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any

from ..schemas import ResultRecord
from .types import ParseError, ParserContext


# Parser fallback cost; the controller replaces it with measured job compute.
COMPLEXA_GPU_H_PER_FINAL = 0.5


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _result_id(
    target_id: str,
    candidate_id: str,
    launch_namespace: str,
    sample_index: int,
    pdb_path: str,
) -> str:
    pdb_name = Path(pdb_path).name if pdb_path else ""
    blob = (
        f"{target_id}::{candidate_id}::{launch_namespace}::"
        f"{sample_index}::{pdb_name}"
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _find_rewards_csv(output_dir: Path) -> Path | None:
    """Find the rewards CSV (filename pattern: rewards_*.csv)."""
    candidates = list(output_dir.glob("rewards_*.csv"))
    # Prefer non-"all_rewards" file (which is the per-step accumulator)
    main_csvs = [p for p in candidates if not p.name.startswith("all_rewards")]
    return main_csvs[0] if main_csvs else (candidates[0] if candidates else None)


def _resolve_structure_path(raw_path: str, output_dir: Path) -> str | None:
    """Return an absolute existing structure path, or None if unusable."""
    raw_path = (raw_path or "").strip()
    if not raw_path:
        return None
    p = Path(raw_path)
    candidates = [p] if p.is_absolute() else [output_dir / p, p]
    for cand in candidates:
        try:
            if cand.is_file() and cand.suffix.lower() in {".pdb", ".cif", ".mmcif"}:
                return str(cand.resolve())
        except OSError:
            continue
    return None


def parse_complexa_output(
    output_dir: Path, ctx: ParserContext
) -> list[ResultRecord]:
    """Convert a Complexa inference output directory into T-REX ResultRecords."""
    if not output_dir.exists():
        raise ParseError(f"Complexa output dir missing: {output_dir}")

    csv_path = _find_rewards_csv(output_dir)
    if csv_path is None:
        raise ParseError(f"Complexa output dir has no rewards_*.csv: {output_dir}")

    records: list[ResultRecord] = []
    # Prefer the spawning candidate family, then the candidate-ID family, then the
    # parser default.
    if ctx.method_family:
        family = ctx.method_family
    elif ":" in ctx.candidate_id:
        family = ctx.candidate_id.split(":")[-1]
    else:
        family = "complexa_beam"

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if (row.get("sample_type") or "").strip() != "final":
                continue
            plddt_log = _safe_float(row.get("af2folding_plddt_log"))
            i_pae     = _safe_float(row.get("af2folding_i_pae"))
            rmsd      = _safe_float(row.get("af2folding_rmsd"))
            metrics: dict[str, float] = {}
            # Native reward scores inform search and conversion priority. Only
            # the independent AF2 runner supplies canonical qualification axes.
            plddt = (plddt_log * 100.0) if plddt_log is not None else None
            has_core = (
                plddt is not None and i_pae is not None and rmsd is not None
            )
            _maybe(metrics, "complexa_native_pLDDT", plddt)
            _maybe(metrics, "complexa_native_iPAE", i_pae)
            _maybe(metrics, "complexa_native_binder_scRMSD", rmsd)

            # Diagnostic axes (Tier 1 — Complexa provides ipTM, ipSAE, min_ipae natively)
            _maybe(metrics, "ipTM",         _safe_float(row.get("af2folding_i_ptm_log")))
            _maybe(metrics, "min_ipae",     _safe_float(row.get("af2folding_min_ipae")))
            _maybe(metrics, "avg_ipsae",    _safe_float(row.get("af2folding_avg_ipsae")))
            _maybe(metrics, "max_ipsae",    _safe_float(row.get("af2folding_max_ipsae")))
            _maybe(metrics, "contact_density",   _safe_float(row.get("af2folding_con")))
            _maybe(metrics, "interface_contact_density",
                                                _safe_float(row.get("af2folding_i_con")))
            _maybe(metrics, "fape",         _safe_float(row.get("af2folding_fape")))

            pdb_path_raw = (row.get("pdb_path") or "").strip()
            pdb_path = _resolve_structure_path(pdb_path_raw, output_dir)
            # `candidate_id` can intentionally repeat for warm-start Complexa
            # launches across ticks. Include a launch namespace so each
            # physical output row has a unique ResultRecord identity.
            launch_namespace = "::".join(
                part for part in (ctx.tick_id, output_dir.name) if part
            )
            rid = _result_id(
                ctx.target_id, ctx.candidate_id, launch_namespace, idx,
                pdb_path or pdb_path_raw,
            )

            bins = {"sample_index": str(idx)}
            bins["complexa_af2folding_status"] = (
                "native_diagnostic" if has_core else "missing_core_metrics"
            )
            if pdb_path:
                bins["needs_canonical_score_conversion"] = "1"
            else:
                bins["score_conversion_status"] = "missing_structure_artifact"
            if pdb_path_raw and not pdb_path:
                bins["artifact_status"] = "missing_or_unusable_pdb_path"

            records.append(ResultRecord(
                result_id=rid,
                parent_ids=list(ctx.parent_ids),
                backend_family=family,
                runtime_bucket_id=ctx.runtime_bucket_id,
                target_id=ctx.target_id,
                metrics=metrics,
                metrics_calibrated={},
                route_lineage=[family],
                gpu_h=COMPLEXA_GPU_H_PER_FINAL,
                exit_status="ok",
                bins=bins,
                artifacts={"pdb_path": pdb_path} if pdb_path else {},
                panel_ready=False,
                tick_id=(ctx.tick_id or None),
            ))
    return records


def _maybe(d: dict, key: str, value: float | None) -> None:
    if value is not None:
        d[key] = value
