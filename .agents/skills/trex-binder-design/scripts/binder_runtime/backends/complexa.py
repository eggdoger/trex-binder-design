"""Deterministic command construction for built-in Complexa families."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Mapping

import yaml


ALGORITHM_BY_FAMILY = {
    "complexa_beam": "beam-search",
    "complexa_best_of_n": "best-of-n",
    "complexa_fk_steering": "fk-steering",
    "complexa_mcts": "mcts",
    "complexa_single_pass": "single-pass",
}

NAMESPACE_BY_ALGORITHM = {
    "beam-search": "beam_search",
    "fk-steering": "fk_steering",
    "mcts": "mcts",
    "best-of-n": "best_of_n",
    "single-pass": None,
}


def refinement_overrides(config_delta: Mapping[str, object]) -> list[str]:
    """Return internally consistent sequence-refinement Hydra overrides."""

    overrides: list[str] = []
    greedy_requested = (
        int(config_delta.get("n_greedy_iters", 0) or 0) > 0
        or bool(config_delta.get("enable_greedy_optimization", False))
        or "greedy_percentage" in config_delta
    )
    refinement_algorithm = str(config_delta.get("refinement_algorithm", "") or "")
    if greedy_requested and refinement_algorithm != "sequence_hallucination":
        refinement_algorithm = "sequence_hallucination"
        print(
            "  [complexa] enabling sequence_hallucination because greedy "
            "knobs were requested",
            flush=True,
        )
    if "refinement_algorithm" in config_delta or greedy_requested:
        overrides.append(
            "++generation.refinement.algorithm=" f"{refinement_algorithm or 'null'}"
        )
    if refinement_algorithm == "sequence_hallucination":
        if "n_greedy_iters" in config_delta:
            overrides.append(
                "++generation.refinement.n_greedy_iters="
                f"{int(config_delta['n_greedy_iters'])}"
            )
        if "enable_greedy_optimization" in config_delta:
            enabled = bool(config_delta["enable_greedy_optimization"])
            overrides.append(
                "++generation.refinement.enable_greedy_optimization="
                f"{str(enabled).lower()}"
            )
        if "greedy_percentage" in config_delta:
            overrides.append(
                "++generation.refinement.greedy_percentage="
                f"{float(config_delta['greedy_percentage'])}"
            )
    return overrides


def reward_filter_overrides(config_delta: Mapping[str, object]) -> list[str]:
    """Return reward overrides, rejecting unsupported post-generation filtering."""

    if "filter_samples_limit" in config_delta:
        raise ValueError(
            "unsupported_param:filter_samples_limit:generate_does_not_run_filter"
        )
    overrides: list[str] = []
    reward_keys = {
        "reward_i_pae_weight": "i_pae",
        "reward_plddt_weight": "plddt",
        "reward_min_ipae_weight": "min_ipae",
        "reward_min_ipsae_weight": "min_ipsae",
        "reward_avg_ipsae_weight": "avg_ipsae",
        "reward_max_ipsae_weight": "max_ipsae",
        "reward_i_con_weight": "i_con",
        "reward_i_ptm_weight": "i_ptm",
    }
    for config_key, reward_key in reward_keys.items():
        if config_key in config_delta:
            overrides.append(
                "++generation.reward_model.reward_models.af2folding."
                f"reward_weights.{reward_key}={float(config_delta[config_key])}"
            )
    return overrides


def canonical_af2_overrides() -> list[str]:
    """Partially align native AF2 rewards while preserving the existing search.

    The API name is retained for compatibility. Template sequence masking and
    numerical/runtime options differ from the independent AF2 verifier, so
    these overrides do not make native reward measurements canonical.
    """

    return [
        "++generation.reward_model.reward_models.af2folding.model_nums=0",
        "++generation.reward_model.reward_models.af2folding.num_recycles=3",
        "++generation.reward_model.reward_models.af2folding.use_initial_guess=True",
        "++generation.reward_model.reward_models.af2folding.use_multimer=True",
    ]


def _launch_seed(candidate_id: str, run_name: str, campaign_seed: int) -> int:
    material = f"{candidate_id}|{run_name}|s{campaign_seed}".encode()
    return int(hashlib.sha256(material).hexdigest()[:8], 16) % 1_000_000


def _hydra_value(value: object) -> str:
    """Quote strings using Hydra's rules, which differ from JSON escapes."""
    if isinstance(value, str):
        # Hydra only unescapes backslashes immediately before the enclosing
        # quote character. Preserve other backslashes and Unicode literally.
        escaped = re.sub(r'\\+(?="|$)', lambda match: match.group() * 2, value)
        return '"' + escaped.replace('"', '\\"') + '"'
    return json.dumps(value, ensure_ascii=False)


def inspect_complexa_target(
    *, repo: Path, target_id: str, target_pdb: Path,
    chain_ids: tuple[str, ...], hotspots: tuple[str, ...],
    config_delta: Mapping[str, object] | None = None,
    require_binder_length: bool = False,
) -> dict[str, object]:
    """Validate and describe target binding without modifying the backend or PDB.

    Preflight can leave a new target's length pending for the planner. Dispatch
    must require an explicit length range. Registered crops are never expanded
    or silently applied to a different set of campaign target residues.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", target_id):
        raise ValueError("Complexa target_id must contain only letters, digits, _ or -")
    if (not chain_ids or len(set(chain_ids)) != len(chain_ids)
            or any(not re.fullmatch(r"[A-Za-z]", c) for c in chain_ids)):
        raise ValueError("Complexa requires unique one-letter target chains")
    pdb = Path(target_pdb).expanduser().resolve()
    if "${" in str(pdb):
        raise ValueError("Complexa target path cannot contain Hydra interpolation syntax")
    residues: dict[str, list[int]] = {c: [] for c in chain_ids}
    seen: set[tuple[str, int]] = set()
    for line in pdb.read_text(encoding="utf-8").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        chain = line[21:22]
        if chain not in residues:
            continue
        if line[26:27].strip():
            raise ValueError("Complexa does not support target residue insertion codes")
        try:
            number = int(line[22:26])
        except ValueError as exc:
            raise ValueError("Complexa target has an invalid PDB residue number") from exc
        if number <= 0:
            raise ValueError("Complexa target residue numbers must be positive")
        if (chain, number) not in seen:
            seen.add((chain, number))
            residues[chain].append(number)
    for chain, numbers in residues.items():
        if not numbers:
            raise ValueError(f"Complexa target chain {chain!r} is absent or has no CA atoms")
    hotspot_ids = {f"{chain}{number}" for chain, number in seen}
    if any(h not in hotspot_ids for h in hotspots):
        raise ValueError("Complexa hotspot is absent from the selected target residues")
    segments: list[str] = []
    for chain, numbers in residues.items():
        ordered = sorted(numbers)
        first = last = ordered[0]
        for number in ordered[1:]:
            if number == last + 1:
                last = number
            else:
                segments.append(f"{chain}{first}-{last}")
                first = last = number
        segments.append(f"{chain}{first}-{last}")
    registry_path = Path(repo) / "configs/targets/targets_dict.yaml"
    try:
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("Complexa target registry is not valid YAML") from exc
    if not isinstance(registry, dict) or not isinstance(registry.get("target_dict_cfg"), dict):
        raise ValueError("Complexa target registry is malformed")
    registered = registry["target_dict_cfg"].get(target_id)
    if registered is not None and not isinstance(registered, dict):
        raise ValueError("Complexa registered target entry is malformed")
    task = dict(registered or {})
    if registered is not None:
        crop: set[tuple[str, int]] = set()
        for segment in str(task.get("target_input", "")).split(","):
            match = re.fullmatch(r"([A-Za-z])(\d+)-(\d+)", segment.strip())
            if match is None:
                raise ValueError("Complexa registered crop uses unsupported target_input syntax")
            chain, first, last = match.groups()
            if int(first) <= 0 or int(last) < int(first):
                raise ValueError("Complexa registered crop has an invalid residue range")
            crop.update((chain, n) for n in range(int(first), int(last) + 1))
        if crop != seen:
            raise ValueError("Complexa registered crop differs from the campaign target residues; provide an explicitly cropped target with matching constraints or a distinct custom target_id")
    else:
        task.update(target_input=",".join(segments), pdb_id=None,
                    source="campaign", target_filename=pdb.stem)
    task.update(target_path=str(pdb), hotspot_residues=list(hotspots))
    delta = config_delta or {}
    low, high = delta.get("target_length_min"), delta.get("target_length_max")
    if low is not None or high is not None:
        if (type(low) is not int or type(high) is not int
                or not 40 <= low <= 120 or not 60 <= high <= 200 or low > high):
            raise ValueError("Complexa binder length requires integer target_length_min [40,120] and target_length_max [60,200], with min <= max")
        task["binder_length"] = [low, high]
    length = task.get("binder_length")
    if length is not None and (
        not isinstance(length, (list, tuple)) or len(length) != 2
        or any(type(n) is not int or n < 1 for n in length) or length[0] > length[1]
    ):
        raise ValueError("Complexa registered binder length range is invalid")
    if require_binder_length and length is None:
        raise ValueError("Complexa custom target requires explicit target_length_min and target_length_max in config_delta")
    return {
        "target_id": target_id, "registered_task": registered is not None,
        "target_residue_count": len(seen), "chain_ids": list(chain_ids),
        "requires_explicit_binder_length": length is None,
        "required_action_config_params": (["target_length_min", "target_length_max"]
                                          if registered is None else []),
        "action_config_requirement": (
            "Every Complexa generation action for this custom target must supply BOTH "
            "target_length_min and target_length_max as legal integer config_delta values. "
            "Preflight does not choose these scientific parameters."
            if registered is None else "Registered target crop and binder length are preserved."
        ),
        "task": task,
    }


def build_complexa_overrides(
    *,
    family: str,
    candidate_id: str,
    config_delta: Mapping[str, object] | None,
    target_id: str,
    run_name: str,
    repo: Path,
    campaign_seed: int,
    seed_run_name: str | None = None,
    target_pdb: Path | None = None,
    chain_ids: tuple[str, ...] = (),
    hotspots: tuple[str, ...] = (),
) -> list[str]:
    """Compile one validated candidate into the complete Hydra override list."""

    algorithm = ALGORITHM_BY_FAMILY.get(family, "beam-search")
    namespace = NAMESPACE_BY_ALGORITHM.get(algorithm)
    seed = _launch_seed(
        candidate_id,
        seed_run_name if seed_run_name is not None else run_name,
        campaign_seed,
    )
    overrides = [
        f"++generation.task_name={_hydra_value(target_id)}",
        f"++ckpt_path={repo}/ckpts",
        "++ckpt_name=complexa.ckpt",
        f"++autoencoder_ckpt_path={repo}/ckpts/complexa_ae.ckpt",
        f"++run_name={run_name}",
        "++gen_njobs=1",
        "++job_id=0",
        f"++seed={seed}",
        "++base_config_name=search_binder_local_pipeline",
        f"++generation.search.algorithm={algorithm}",
    ]
    delta = config_delta or {}
    if target_pdb is not None:
        detail = inspect_complexa_target(
            repo=repo, target_id=target_id, target_pdb=target_pdb,
            chain_ids=chain_ids, hotspots=hotspots, config_delta=delta,
            require_binder_length=True,
        )
        task = detail["task"]
        for key in ("target_path", "target_input", "hotspot_residues", "binder_length",
                    "pdb_id", "source", "target_filename"):
            if key in task:
                overrides.append(f"++generation.target_dict_cfg.{target_id}.{key}="
                                 + _hydra_value(task[key]))
    elif "target_length_min" in delta or "target_length_max" in delta:
        raise ValueError("Complexa binder length override requires explicit target PDB and chains")
    if "nsteps" in delta:
        nsteps = int(delta["nsteps"])
        checkpoints = sorted(
            {
                0,
                nsteps // 4,
                nsteps // 2,
                3 * nsteps // 4,
                nsteps,
            }
        )
        overrides.extend(
            [
                f"++generation.args.nsteps={nsteps}",
                "++generation.search.step_checkpoints="
                f"[{','.join(str(value) for value in checkpoints)}]",
            ]
        )
    if "nsamples" in delta:
        overrides.append(
            f"++generation.dataloader.dataset.nres.nsamples={int(delta['nsamples'])}"
        )
    if "batch_size" in delta:
        overrides.append(
            f"++generation.dataloader.batch_size={int(delta['batch_size'])}"
        )
    if "sc_scale_noise" in delta:
        overrides.append(
            "++generation.model.bb_ca.simulation_step_params.sc_scale_noise="
            f"{float(delta['sc_scale_noise'])}"
        )
    overrides.extend(refinement_overrides(delta))
    overrides.extend(canonical_af2_overrides())
    overrides.extend(reward_filter_overrides(delta))

    if namespace in {"beam_search", "fk_steering"}:
        prefix = f"generation.search.{namespace}"
        if "beam_width" in delta:
            overrides.append(f"++{prefix}.beam_width={int(delta['beam_width'])}")
        if "n_branch" in delta:
            overrides.append(f"++{prefix}.n_branch={int(delta['n_branch'])}")
        if "temperature" in delta and namespace == "fk_steering":
            overrides.append(f"++{prefix}.temperature={float(delta['temperature'])}")
    elif namespace == "best_of_n" and "replicas" in delta:
        overrides.append(
            f"++generation.search.best_of_n.replicas={int(delta['replicas'])}"
        )
    elif namespace == "mcts":
        prefix = "generation.search.mcts"
        if "n_simulations" in delta:
            overrides.append(f"++{prefix}.n_simulations={int(delta['n_simulations'])}")
        if "exploration_prob" in delta:
            overrides.append(
                f"++{prefix}.exploration_prob={float(delta['exploration_prob'])}"
            )
        if "exploration_constant" in delta:
            overrides.append(
                "++generation.search.mcts.exploration_constant="
                f"{float(delta['exploration_constant'])}"
            )
    return overrides


def build_complexa_shell_command(
    *,
    repo: Path,
    python: Path,
    overrides: list[str],
) -> str:
    """Build the reviewed shell boundary required to source Complexa env.sh."""

    override_text = " ".join(shlex.quote(value) for value in overrides)
    return (
        f"cd {shlex.quote(str(repo))} && "
        "source env.sh > /dev/null 2>&1 && "
        f"{shlex.quote(str(python))} -m proteinfoundation.generate "
        f"--config-path {shlex.quote(str(repo / 'configs'))} "
        "--config-name search_binder_local_pipeline "
        f"{override_text}"
    )


__all__ = [
    "ALGORITHM_BY_FAMILY",
    "NAMESPACE_BY_ALGORITHM",
    "build_complexa_overrides",
    "build_complexa_shell_command",
    "canonical_af2_overrides",
    "inspect_complexa_target",
    "refinement_overrides",
    "reward_filter_overrides",
]
