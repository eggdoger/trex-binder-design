"""Create an explicit, editable campaign YAML template."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..validation import DEFAULT_FAMILIES
from .models import CAMPAIGN_SCHEMA_VERSION, PolicyConfig, RunConfig


def campaign_template(
    *,
    name: str,
    target: str,
    archive_root: str,
    asset_root: str | None,
    target_constraint: str | None,
    target_pdb: str | None,
    max_wall_hours: float = RunConfig.max_wall_hours,
    total_gpus: int = 1,
) -> dict[str, Any]:
    """Return a complete template with every user-facing policy visible."""

    if total_gpus < 1:
        raise ValueError("total_gpus must be at least 1")
    if max_wall_hours <= 0:
        raise ValueError("max_wall_hours must be greater than zero")

    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "name": name,
        "target": {
            "name": target,
            "asset_root": str(asset_root) if asset_root is not None else None,
            "constraint": str(target_constraint) if target_constraint is not None else None,
            "pdb": str(target_pdb) if target_pdb is not None else None,
        },
        "run": {
            "archive_root": str(archive_root) if archive_root is not None else None,
            "max_wall_hours": max_wall_hours,
            "seed": 0,
            "worker_gpus": [str(index) for index in range(total_gpus)],
            "enabled_families": list(DEFAULT_FAMILIES),
        },
        "llm": {
            "base_url": "codex://mailbox",
            "model": "codex-external:unknown",
        },
        "policy": {
            "critic": True,
            "evidence_skip": False,
            "exemplars": True,
            "foldseek_su_tm_score": 0.60,
            "foldseek_collapse_tm_score": 0.60,
            "selector_quota_realization": "fractional_carry",
            "selector_mode_window_k": PolicyConfig.selector_mode_window_k,
            "selector_adaptive_mode_window_k": PolicyConfig.selector_adaptive_mode_window_k,
        },
        "memory": {
            "cross_campaign_path": None,
        },
        "backends": {
            "external_root": None,
            "complexa_repo": None,
            "legacy_complexa_repo": None,
            "complexa_python": None,
            "bindcraft_repo": None,
            "bindcraft_env": None,
            "boltzgen_repo": None,
            "boltzgen_binary": None,
            "boltzgen_cache": None,
            "foldseek_binary": None,
            "mmseqs_binary": None,
            "community_models_path": None,
            "af2_data_path": None,
            "af2_python": None,
            "proteinmpnn_path": None,
            "proteinmpnn_weights_path": None,
            "proteinmpnn_python": None,
        },
    }


def write_campaign_template(
    out: Path | str,
    *,
    name: str,
    target: str,
    archive_root: str,
    asset_root: str | None = None,
    target_constraint: str | None = None,
    target_pdb: str | None = None,
    max_wall_hours: float = RunConfig.max_wall_hours,
    total_gpus: int = 1,
    force: bool = False,
) -> Path:
    """Write one template, refusing accidental overwrite by default."""

    path = Path(out).expanduser().resolve()
    if path.exists() and not force:
        raise ValueError(f"refusing to overwrite existing file: {path} (use --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = campaign_template(
        name=name,
        target=target,
        archive_root=archive_root,
        asset_root=asset_root,
        target_constraint=target_constraint,
        target_pdb=target_pdb,
        max_wall_hours=max_wall_hours,
        total_gpus=total_gpus,
    )
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
