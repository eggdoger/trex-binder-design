"""Small subprocess entry point for ColabDesign AF2-multimer refiltering."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from .af2_chain_identity import ChainIdentityError, resolve_prediction_chains


def _default_community_root() -> Path:
    external = Path(os.environ.get("TREX_EXTERNAL_ROOT") or Path.home() / ".cache" / "trex-binder-design" / "external")
    complexa = Path(os.environ.get("TREX_LEGACY_COMPLEXA_REPO") or os.environ.get("TREX_COMPLEXA_REPO") or external / "Proteina-Complexa")
    return Path(os.environ.get("TREX_COMMUNITY_ROOT") or complexa / "community_models").expanduser().resolve()


DEFAULT_COMMUNITY_ROOT = _default_community_root()


def main() -> None:
    args = _args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(args.community_root))

    from colabdesign import mk_af_model  # type: ignore

    model_names = tuple(item.strip() for item in args.model_names.split(",") if item.strip())
    use_ig = bool(args.use_initial_guess)
    model = mk_af_model(
        protocol="binder",
        use_multimer=args.use_multimer,
        data_dir=str(args.af2_data_dir),
        model_names=list(model_names) if model_names else None,
        num_seq=1,
        use_bfloat16=False,
        use_remat=False,
        use_initial_guess=use_ig,
    )
    # With initial guessing enabled, use the designed binder as a template and
    # remove inter-chain template information, matching the qualification protocol.
    prep_kwargs = dict(
        target_chain=args.target_chain,
        binder_chain=args.binder_chain,
        rm_binder_seq=False,
    )
    if use_ig:
        prep_kwargs["use_binder_template"] = True
        prep_kwargs["rm_template_ic"] = True
    model.prep_inputs(str(args.input_pdb), **prep_kwargs)
    # Initialize from the input PDB sequence; predict() otherwise uses a uniform
    # sequence.
    model.set_seq(mode="wildtype")
    aux = model.predict(
        num_models=len(model._model_names),
        num_recycles=args.num_recycles,
        sample_models=False,
        dropout=False,
        return_aux=True,
        verbose=False,
        seed=args.seed,
    )
    predicted_pdb = (args.out_dir / "af2_refilter_prediction.pdb").resolve()
    model.save_pdb(str(predicted_pdb), get_best=False)
    losses = _float_dict(aux.get("losses", {}))
    metrics = {
        "i_pae": losses.get("i_pae"),
        "plddt": 1.0 - losses["plddt"] if "plddt" in losses else None,
        "binder_scrmsd_ca": losses.get("rmsd"),
        "iptm": _iptm_score(aux.get("i_ptm")),
        "raw_losses": losses,
        "model_names": list(model._model_names),
        "predicted_pdb": str(predicted_pdb),
    }
    report = {
        "schema_version": "v5_af2_refilter_result.v1",
        "input_pdb": str(args.input_pdb.resolve()),
        "target_chain": args.target_chain,
        "binder_chain": args.binder_chain,
        "metrics": metrics,
    }
    # Retain input labels as provenance; prediction labels can be different.
    try:
        report["prediction_chains"] = resolve_prediction_chains(
            report, predicted_pdb, report_dir=args.out_dir,
        )
    except ChainIdentityError as exc:
        # Keep measurements, but never guess a chain for diversity credit.
        report["prediction_chain_error"] = str(exc)
    (args.out_dir / "af2_refilter_result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pdb", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--community-root", type=Path, default=DEFAULT_COMMUNITY_ROOT)
    parser.add_argument("--af2-data-dir", type=Path, default=None)
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--binder-chain", default="B")
    parser.add_argument("--model-names", default="model_1_multimer_v3")
    parser.add_argument("--num-recycles", type=int, default=3)  # AF2 multimer requires recycling for convergence.
    parser.add_argument("--use-initial-guess", type=int, choices=(0, 1), default=1,
                        help="1=use designed structure as AF2 template/initial guess "
                             "(matches Complexa/BindCraft validation); 0=blind prediction")
    parser.add_argument("--use-multimer", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.community_root = args.community_root.expanduser().resolve()
    args.af2_data_dir = Path(args.af2_data_dir or os.environ.get("TREX_AF2_DATA_DIR") or args.community_root / "ckpts" / "AF2").expanduser().resolve()
    return args


def _iptm_score(value: Any) -> float | None:
    """Read a scalar confidence; missing or invalid values remain unmeasured."""
    if (isinstance(value, bool)
            or getattr(getattr(value, "dtype", None), "kind", None) == "b"
            or getattr(value, "shape", ()) != ()):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


def _float_dict(values: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


if __name__ == "__main__":
    main()
