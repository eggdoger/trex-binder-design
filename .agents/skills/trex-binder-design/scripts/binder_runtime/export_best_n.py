"""Export archive-linked canonical AF2 candidates for further review.

This is a SEPARATE artifact from ``finalize_panel`` / ``select_production_panel``:
that selector returns a structurally-deduped DIVERSITY panel (at most one
candidate per Foldseek SU bin), so ``--panel-size 100`` there yields only as many designs
as there are distinct structures. This "best-N by score" list instead
ranks every strict success by ``production_quality`` and takes the top N,
allowing structural near-duplicates (you may want several variants of a strong
basin). It then COPIES each binder PDB into a single, stable directory so the
panel survives even if the per-run scratch output dirs are later cleaned, and
writes a manifest (CSV + JSON) joining each design to its score, strict metrics,
and structural/sequence dedup bins. Native backend filtering is reported
separately, including rejections and unavailable evidence. It does not establish
synthesis readiness or independently reproduce the stored scientific scores.

Usage:
    python -m binder_runtime.export_best_n \
        --archive-root <ARCHIVE> --target-id 02_PDL1 --n 100 \
        --out-dir <ARCHIVE>/wetlab_panel_best100
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import tempfile
import json
import re
import shutil
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .archive import Archive
from .analysis import validate_archive
from .output_identity import prepare_archive_results
from .foldseek_clusterer import apply_clusters_to_bins, cluster_archive_pdbs
from .panel import _has_structure_artifact, production_bins, production_quality
from .result_qualification import assess_result_qualification, build_result_index
from .schemas import ResultRecord
from .sequence_clusterer import (
    apply_sequence_clusters_to_bins,
    cluster_archive_sequences,
)
from .success_criteria import STRICT_SUCCESS, is_strict_success

# Strict axes carried into the manifest for operator review (order = report order).
_STRICT_AXES = tuple(STRICT_SUCCESS.keys())
_DIAGNOSTIC_AXES = ("ipTM", "min_ipae", "avg_ipsae", "interface_contact_density")
BEST_N_MANIFEST_SCHEMA_VERSION = "binder_runtime.best-n-manifest.v2"
BEST_N_OUTPUT_SCHEMA_VERSION = "binder_runtime.best-n-export.v2"
_OUTPUT_ENTRY_NAMES = frozenset({"manifest.json", "manifest.csv", "pdbs"})
_MANIFEST_FIELD_ORDER = (
    "rank",
    "result_id",
    "target_id",
    "backend_family",
    "parent_result_id",
    "source_family",
    "canonical_af2_pass",
    "canonical_assessment_status",
    "canonical_unavailable_reason",
    "lineage_status",
    "lineage_path",
    "native_filter_status",
    "native_filter_assessment_status",
    "native_filter_failures",
    "native_filter_source",
    "native_filter_unavailable_reason",
    "production_quality",
    *_STRICT_AXES,
    *_DIAGNOSTIC_AXES,
    "structure_bin",
    "sequence_bin",
    "binder_chain",
    "target_chains",
    "output_chain_identity",
    "src_pdb",
    "dst_pdb",
    "structure_sha256",
)


def _structure_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _filename_component(value: str) -> str:
    """Keep archive IDs as data, never as destination path components."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)[:80] or "unnamed"


def _structure_source(rec: ResultRecord) -> str:
    """PDB/CIF path for this record, or '' if none on disk."""
    art = rec.artifacts or {}
    p = art.get("pdb_path")
    if p and Path(p).is_file():
        return p
    d = art.get("pdb_dir")
    if d and Path(d).is_dir():
        for pat in ("*.pdb", "*.cif", "*.mmcif"):
            hit = sorted(Path(d).glob(pat))
            if hit:
                return str(hit[0])
    return ""


def _validate_output_directory(out_dir: Path, *, overwrite: bool) -> Path:
    """Validate a destination without changing an existing export."""

    proposed = Path(out_dir).expanduser()
    if proposed.is_symlink():
        raise ValueError(f"refusing to overwrite a symlink output directory: {proposed}")
    resolved = proposed.resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"output path is not a directory: {resolved}")
    existing = list(resolved.iterdir()) if resolved.is_dir() else []
    if existing and not overwrite:
        raise ValueError(f"output directory is not empty: {resolved} (use --overwrite)")
    unknown = sorted(
        path.name for path in existing if path.name not in _OUTPUT_ENTRY_NAMES
    )
    if unknown:
        raise ValueError(
            f"refusing to overwrite directory with unrecognized entries: {unknown}"
        )
    if existing:
        manifest = resolved / "manifest.json"
        if manifest.is_symlink() or not manifest.is_file():
            raise ValueError("refusing to overwrite a directory without a regular export manifest")
        try:
            prior = json.loads(manifest.read_text(encoding="utf-8"))
            if (not isinstance(prior, dict)
                    or prior.get("schema_version") not in {
                        "binder_runtime.best-n-manifest.v1", BEST_N_MANIFEST_SCHEMA_VERSION,
                    }
                    or not isinstance(prior.get("designs"), list)):
                raise ValueError("unrecognized export manifest")
            expected_pdbs = {Path(row["dst_pdb"]).name for row in prior["designs"] if row.get("dst_pdb")}
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"refusing to overwrite invalid export manifest: {exc}") from exc
        for entry in existing:
            if entry.is_symlink():
                raise ValueError(f"refusing to overwrite symlink export entry: {entry.name}")
            if entry.name == "pdbs":
                if not entry.is_dir() or any(
                    item.is_symlink() or not item.is_file() or item.name not in expected_pdbs
                    for item in entry.iterdir()
                ):
                    raise ValueError("refusing to overwrite unrecognized files in pdbs directory")
            elif not entry.is_file():
                raise ValueError(f"refusing to overwrite non-file export entry: {entry.name}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


@contextmanager
def _atomic_export_directory(destination: Path) -> Iterator[Path]:
    """Yield a staging root and publish it only after every write succeeds."""
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    backup: Path | None = None
    try:
        yield staging
        if destination.exists():
            backup = destination.with_name(
                f".{destination.name}.previous-{uuid.uuid4().hex}"
            )
            destination.rename(backup)
        try:
            staging.rename(destination)
        except OSError:
            if backup is not None and backup.exists():
                backup.rename(destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _write_export_artifacts(
    *,
    staging_dir: Path,
    archive_root: Path,
    target_id: str,
    diagnostics: dict,
    rows: list[dict],
    copy_pdbs: bool,
) -> None:
    """Write a complete export into an unpublished staging directory."""
    if copy_pdbs:
        staging_pdb_dir = staging_dir / "pdbs"
        staging_pdb_dir.mkdir()
        for row in rows:
            source = row["src_pdb"]
            if source:
                destination = staging_pdb_dir / Path(row["dst_pdb"]).name
                shutil.copy2(source, destination)
                if _structure_sha256(destination) != row["structure_sha256"]:
                    raise ValueError(f"structure changed during export: {source}")
    with (staging_dir / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_FIELD_ORDER)
        writer.writeheader()
        # Nested evidence remains machine-readable in CSV, rather than Python
        # repr (single quotes/None) or silently flattened failure labels.
        writer.writerows({
            key: json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
            if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        } for row in rows)
    (staging_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": BEST_N_MANIFEST_SCHEMA_VERSION,
                "archive_root": str(archive_root),
                "target_id": target_id,
                "diagnostics": diagnostics,
                "designs": rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def export_best_n(
    archive: Archive,
    *,
    target_id: str,
    n: int = 100,
    out_dir: Path,
    copy_pdbs: bool = True,
    annotate_dedup: bool = True,
    foldseek_binary: str = "foldseek",
    mmseqs_binary: str = "mmseqs",
    require_structure_artifact: bool = True,
    su_tm_score: float = 0.60,
    sequence_identity: float = 0.90,
    sequence_coverage: float = 0.80,
    overwrite: bool = False,
) -> dict:
    if n < 1:
        raise ValueError("n must be positive")
    if not target_id.strip():
        raise ValueError("target_id must be non-empty")
    for name, value in (
        ("su_tm_score", su_tm_score),
        ("sequence_identity", sequence_identity),
        ("sequence_coverage", sequence_coverage),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    validation = validate_archive(archive.root)
    if not validation["ok"]:
        raise ValueError("archive validation failed; export aborted: " + "; ".join(validation["errors"][:5]))
    if validation["target_ids"] and target_id not in validation["target_ids"]:
        raise ValueError(f"target_id {target_id!r} is absent from archive: {validation['target_ids']}")
    out_dir = _validate_output_directory(out_dir, overwrite=overwrite)

    all_results = prepare_archive_results(archive.iter_records(ResultRecord), archive.root)
    index = build_result_index(all_results)
    results = [r for r in all_results if r.target_id == target_id]
    qualifications = {r.result_id: assess_result_qualification(r, index) for r in results}
    strict = [
        r
        for r in results
        if qualifications[r.result_id]["canonical_af2_pass"] is True
        and r.bins.get("output_chain_identity") != "unresolved"
        and (not require_structure_artifact or _has_structure_artifact(r))
    ]

    diagnostics: dict = {
        "target_id": target_id,
        "n_results": len(results),
        "n_strict_with_structure": len(strict),
        "n_canonical_af2_pass": sum(q["canonical_af2_pass"] is True for q in qualifications.values()),
        "n_score_pass_excluded_noncanonical_or_unverified": sum(
            r.exit_status == "ok" and is_strict_success(r.metrics)
            and qualifications[r.result_id]["canonical_af2_pass"] is not True
            for r in results
        ),
        "qualification_exclusions": [
            {"result_id": r.result_id,
             "reason": qualifications[r.result_id]["canonical_unavailable_reason"]
                       or "canonical_gate_or_identity_failed"}
            for r in results if r.exit_status == "ok" and is_strict_success(r.metrics)
            and qualifications[r.result_id]["canonical_af2_pass"] is not True
        ],
        "n_unresolved_output_identity": sum(r.bins.get("output_chain_identity") == "unresolved" for r in results),
        "foldseek_su_status": "disabled",
        "sequence_dedup_status": "disabled",
        "archive_validation_warnings": validation["warnings"],
    }

    # Annotate strict records with strict-only structural (foldseek_su) and
    # sequence (sequence_su) bins so the manifest shows how much structural /
    # sequence redundancy is present among the top-N (the SAME bins the
    # objective's SU and the diversity panel use). Pure annotation — does not
    # change ranking or which designs are exported.
    if annotate_dedup and strict:
        strict_rids = {r.result_id for r in strict}
        struct = cluster_archive_pdbs(
            results,
            target_id=target_id,
            foldseek_binary=foldseek_binary,
            only_result_ids=strict_rids,
            min_tm_score=su_tm_score,
        )
        apply_clusters_to_bins(
            results, struct.cluster_by_result_id, bin_key="foldseek_su"
        )
        diagnostics["foldseek_su_status"] = struct.status
        seq = cluster_archive_sequences(
            results,
            target_id=target_id,
            mmseqs_binary=mmseqs_binary,
            only_result_ids=strict_rids,
            min_seq_id=sequence_identity,
            coverage=sequence_coverage,
        )
        apply_sequence_clusters_to_bins(
            results, seq.cluster_by_result_id, bin_key="sequence_su"
        )
        diagnostics["sequence_dedup_status"] = seq.status

    ranked = sorted(strict, key=lambda r: (-production_quality(r), r.result_id))[:n]

    pdb_dir = out_dir / "pdbs"

    rows: list[dict] = []
    n_copied = 0
    n_missing = 0
    family_counts: dict[str, int] = {}
    struct_bins: set[str] = set()
    for rank, rec in enumerate(ranked, start=1):
        bins = production_bins(rec)
        src = _structure_source(rec)
        ext = Path(src).suffix or ".pdb"
        dst = pdb_dir / f"rank{rank:03d}_{_filename_component(rec.backend_family)}_{_filename_component(rec.result_id)}{ext}"
        if copy_pdbs:
            if src:
                n_copied += 1
            else:
                n_missing += 1
        family_counts[rec.backend_family] = family_counts.get(rec.backend_family, 0) + 1
        struct_bins.add(bins["structure"])
        row = {
            "rank": rank,
            "result_id": rec.result_id,
            "target_id": rec.target_id,
            "backend_family": rec.backend_family,
            **qualifications[rec.result_id],
            "production_quality": round(production_quality(rec), 6),
            "structure_bin": bins["structure"],
            "sequence_bin": bins.get("sequence", ""),
            "binder_chain": rec.artifacts.get("binder_chain", ""),
            "target_chains": rec.artifacts.get("target_chains", ""),
            "output_chain_identity": rec.bins.get("output_chain_identity", "legacy_unverified"),
            "src_pdb": src,
            "dst_pdb": str(dst) if (copy_pdbs and src) else "",
            "structure_sha256": _structure_sha256(Path(src)) if src else "",
        }
        for axis in _STRICT_AXES:
            row[axis] = rec.metrics.get(axis)
        for axis in _DIAGNOSTIC_AXES:
            if axis in rec.metrics:
                row[axis] = rec.metrics.get(axis)
        rows.append(row)

    diagnostics.update(
        {
            "n_requested": n,
            "n_exported": len(rows),
            "n_pdbs_copied": n_copied,
            "n_pdbs_missing_source": n_missing,
            "distinct_structure_bins": len(struct_bins),
            "family_counts": family_counts,
            "settings": {
                "ranking": "production_quality_descending_then_result_id",
                "su_tm_score": su_tm_score,
                "sequence_identity": sequence_identity,
                "sequence_coverage": sequence_coverage,
                "annotate_dedup": annotate_dedup,
                "require_structure_artifact": require_structure_artifact,
                "copy_structures": copy_pdbs,
                "overwrite": overwrite,
            },
        }
    )
    native_counts = {
        status: sum(row["native_filter_status"] == status for row in rows)
        for status in ("accepted", "rejected", "unknown")
    }
    diagnostics["exported_native_filter_status_counts"] = native_counts
    diagnostics["n_exported_native_rejected"] = native_counts["rejected"]
    diagnostics["n_exported_native_unknown"] = native_counts["unknown"]
    diagnostics["n_exported_native_assessment_incomplete"] = sum(
        row["native_filter_assessment_status"] != "complete" for row in rows
    )
    diagnostics["warnings"] = []
    if native_counts["rejected"]:
        diagnostics["warnings"].append(
            f"{native_counts['rejected']} exported canonical AF2 passes have native backend rejection; "
            "review native_filter_failures before downstream use."
        )
    if native_counts["unknown"] or diagnostics["n_exported_native_assessment_incomplete"]:
        diagnostics["warnings"].append(
            "Some native filtering evidence or failure details are unavailable/inconsistent; "
            "canonical AF2 pass does not establish native filter pass."
        )
    if diagnostics["n_score_pass_excluded_noncanonical_or_unverified"]:
        diagnostics["warnings"].append(
            "Score-passing records lacking verified canonical AF2 lineage were excluded; "
            "see qualification_exclusions."
        )
    diagnostics["qualification_scope"] = (
        "Canonical AF2 thresholds only; native filtering describes the scored parent "
        "structure, not a new native assessment of the AF2 prediction. Synthesis readiness is not assessed."
    )

    # Publish only after every structure and manifest has been written.
    with _atomic_export_directory(out_dir) as staging_dir:
        _write_export_artifacts(
            staging_dir=staging_dir,
            archive_root=archive.root.resolve(),
            target_id=target_id,
            diagnostics=diagnostics,
            rows=rows,
            copy_pdbs=copy_pdbs,
        )
    return diagnostics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Export candidates passing stored score thresholds for review; synthesis readiness is not assessed"
    )
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--target-id", required=True)
    p.add_argument("--n", "--N", dest="n", type=int, default=100)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <archive-root>/wetlab_panel_best<n>",
    )
    p.add_argument(
        "--no-copy", action="store_true", help="manifest only; do not copy PDB files"
    )
    p.add_argument(
        "--no-dedup", action="store_true", help="skip foldseek/mmseqs bin annotation"
    )
    p.add_argument("--foldseek-binary", default="foldseek")
    p.add_argument("--mmseqs-binary", default="mmseqs")
    p.add_argument("--su-tm-score", type=float, default=0.60)
    p.add_argument("--sequence-identity", type=float, default=0.90)
    p.add_argument("--sequence-coverage", type=float, default=0.80)
    p.add_argument("--allow-missing-structure", action="store_true")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only a prior T-REX export; reject unrelated files",
    )
    args = p.parse_args(argv)

    try:
        archive_root = args.archive_root.expanduser().resolve()
        if not archive_root.is_dir():
            raise ValueError(f"archive root is not a directory: {archive_root}")
        if args.out_dir is not None and args.out_dir.expanduser().is_symlink():
            raise ValueError(f"refusing a symlink output directory: {args.out_dir}")
        out_dir = (
            args.out_dir.expanduser().resolve()
            if args.out_dir is not None
            else archive_root / f"wetlab_panel_best{args.n}"
        )
        diagnostics = export_best_n(
            Archive(archive_root),
            target_id=args.target_id,
            n=args.n,
            out_dir=out_dir,
            copy_pdbs=not args.no_copy,
            annotate_dedup=not args.no_dedup,
            foldseek_binary=args.foldseek_binary,
            mmseqs_binary=args.mmseqs_binary,
            require_structure_artifact=not args.allow_missing_structure,
            su_tm_score=args.su_tm_score,
            sequence_identity=args.sequence_identity,
            sequence_coverage=args.sequence_coverage,
            overwrite=args.overwrite,
        )
        manifest_csv = out_dir / "manifest.csv"
        print(
            json.dumps(
                {
                    "schema_version": BEST_N_OUTPUT_SCHEMA_VERSION,
                    "archive_root": str(archive_root),
                    "out_dir": str(out_dir),
                    "manifest_json": str(out_dir / "manifest.json"),
                    "manifest_csv": str(manifest_csv),
                    "diagnostics": diagnostics,
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError) as exc:
        print(f"trex-export: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
