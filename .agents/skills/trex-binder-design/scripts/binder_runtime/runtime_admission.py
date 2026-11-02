"""Shared wall-clock admission for planning and independently bounded workers.

Accepted-design quotas and relaxed-trajectory quotas are not runtime bounds. Only
the worker-enforced wall-clock cap can replace the conservative family estimate.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

BINDCRAFT_MIN_RUNTIME_SECONDS = 300
BINDCRAFT_MAX_RUNTIME_SECONDS = 7200
BINDCRAFT_CLEANUP_RESERVE_SECONDS = 20
CANONICAL_AF2_RESERVE_SECONDS = 900
LEGACY_DRAIN_RESERVE_SECONDS = 360

FAMILY_RUNTIME_H: dict[str, float] = {
    "bindcraft": 9000 / 3600,
    "complexa_beam": 1800 / 3600,
    "complexa_best_of_n": 1800 / 3600,
    "complexa_fk_steering": 1800 / 3600,
    "complexa_mcts": 3600 / 3600,
    "structure_refilter": 900 / 3600,
    "proteinmpnn_redesign": 300 / 3600,
    "boltzgen": 7200 / 3600,
}


def bindcraft_runtime_cap_seconds(config_delta: Mapping[str, Any] | None) -> int | None:
    """Return an explicit cap, rejecting malformed values rather than dropping it."""
    if not config_delta or "max_runtime_seconds" not in config_delta:
        return None
    value = config_delta["max_runtime_seconds"]
    if type(value) is not int or not (
        BINDCRAFT_MIN_RUNTIME_SECONDS <= value <= BINDCRAFT_MAX_RUNTIME_SECONDS
    ):
        raise ValueError(
            "max_runtime_seconds must be an integer in "
            f"[{BINDCRAFT_MIN_RUNTIME_SECONDS}, {BINDCRAFT_MAX_RUNTIME_SECONDS}]"
        )
    return value


def family_required_wall_seconds(
    family: str, config_delta: Mapping[str, Any] | None = None
) -> float:
    """Required admission budget, including cleanup and score-conversion reserve."""
    if family == "bindcraft":
        cap = bindcraft_runtime_cap_seconds(config_delta)
        if cap is not None:
            return cap + BINDCRAFT_CLEANUP_RESERVE_SECONDS + CANONICAL_AF2_RESERVE_SECONDS
    runtime_h = FAMILY_RUNTIME_H.get(family)
    return (runtime_h * 3600 + LEGACY_DRAIN_RESERVE_SECONDS) if runtime_h is not None else 0.0


def family_minimum_wall_seconds(family: str) -> float:
    """Minimum legal job budget; bounded BC is possible only with an explicit cap."""
    delta = {"max_runtime_seconds": BINDCRAFT_MIN_RUNTIME_SECONDS} if family == "bindcraft" else None
    return family_required_wall_seconds(family, delta)


def budget_admissible_families(families: Iterable[str], remaining_wall_h: float) -> list[str]:
    seconds = float(remaining_wall_h) * 3600
    if not math.isfinite(seconds) or seconds <= 0:
        return []
    return [family for family in families if family_minimum_wall_seconds(family) <= seconds]


def runtime_admission_for_prompt(
    families: Iterable[str], remaining_wall_h: float
) -> dict[str, dict[str, Any]]:
    """Show legal runtime choices without inventing scientific settings."""
    seconds = max(0.0, float(remaining_wall_h) * 3600)
    out: dict[str, dict[str, Any]] = {}
    for family in families:
        required = family_required_wall_seconds(family)
        row: dict[str, Any] = {
            "remaining_wall_seconds": math.floor(seconds),
            "unbounded_required_wall_seconds": required,
            "unbounded_fits": required <= seconds,
            "minimum_required_wall_seconds": family_minimum_wall_seconds(family),
        }
        if family == "bindcraft":
            maximum = min(BINDCRAFT_MAX_RUNTIME_SECONDS, math.floor(
                seconds - BINDCRAFT_CLEANUP_RESERVE_SECONDS - CANONICAL_AF2_RESERVE_SECONDS
            ))
            row.update({
                "max_runtime_seconds_integer_range": (
                    [BINDCRAFT_MIN_RUNTIME_SECONDS, maximum]
                    if maximum >= BINDCRAFT_MIN_RUNTIME_SECONDS else None
                ),
                "cleanup_reserve_seconds": BINDCRAFT_CLEANUP_RESERVE_SECONDS,
                "canonical_af2_reserve_seconds": CANONICAL_AF2_RESERVE_SECONDS,
                "explicit_runtime_cap_required": required > seconds,
                "semantics": (
                    "max_runtime_seconds is an independently enforced per-batch wall-clock cap. "
                    "Budget exhaustion is a bounded completion; partial artifacts are retained. "
                    "The AF2 reserve is a planning allowance, not a guarantee of evaluation for every output. "
                    "max_trajectories is the accepted-design quota; max_relaxed_trajectories counts "
                    "upstream Relaxed trajectories. Neither limits all attempts or bounds wall time."
                ),
            })
        out[family] = row
    return out
