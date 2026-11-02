"""Critic configuration and reference prompts for the deterministic advisory guard.

The embedded runtime uses critic_guard; it contains no model-backed critic caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .schemas import EvidenceSummary


CRITIC_SYSTEM = (
    "You are a critic reviewing a Planner's recommendation. Flag ONLY if "
    "you can identify SPECIFIC, EVIDENCE-CITED issues. Do not invent target "
    "characteristics you don't have evidence for.\n\n"
    "Categories (flag ONLY with specific evidence):\n"
    "  (a) Target-prior contradicted by evidence: ONLY flag if "
    "evidence.recipes contains strict_success entries from a family "
    "DIFFERENT from the Planner's choice AND those successes are RECENT "
    "(last 3 ticks). Do NOT speculate about whether the target 'should' "
    "prefer a family you cannot verify from evidence.\n"
    "  (b) Deadline violation: cite remaining_wall_h vs expected method "
    "runtime.\n"
    "  (c) Ignored recent failure: cite the matching joint_fail recipe.\n"
    "  (d) Default-bias: ONLY flag if the SAME family was proposed in "
    "the last 3 ticks across 3+ different target_classes with no "
    "rationale shift.\n\n"
    "If you cannot identify a SPECIFIC issue cited from the evidence "
    "above, respond 'no_flags'. Novel targets, exploratory probes, and "
    "reasoned first-attempts are NOT issues. When in doubt, do not flag."
)


@dataclass(frozen=True)
class CriticCallConfig:
    model: str = "deterministic-guard"
    base_url: str = "none"
    max_tokens: int = 256
    temperature: float = 0.3
    enable_thinking: bool = False
    timeout_s: float = 30.0
    enabled: bool = False


def build_critic_payload(
    evidence: EvidenceSummary,
    planner_output: Any,  # PlannerOutput; avoid circular import
) -> dict:
    """Build the compact evidence view used by the advisory Critic."""
    alt = getattr(evidence, "diagnostic_alt_model_scores", None) or {}
    return {
        "target_id": evidence.target_id,
        "target_class": getattr(evidence, "target_class", None),
        "remaining_wall_h": evidence.remaining_wall_h,
        "state_label": evidence.state_label,
        "axis_stats": {k: {"pass": v.pass_count, "near_pass": v.near_pass_count,
                             "fail": v.fail_count, "median": v.median_raw}
                         for k, v in evidence.axis_stats.items()},
        "diagnostic_axis_stats": {k: {
            "pass": v.pass_count, "near_pass": v.near_pass_count,
            "fail": v.fail_count, "below_accept": v.below_accept_count,
            "median": v.median_raw, "quality_threshold": v.quality_threshold,
            "pass_threshold": v.pass_threshold,
        } for k, v in evidence.diagnostic_axis_stats.items()},
        "diagnostic_alt_model_scores": {
            fam: {sk: sv.get("median") for sk, sv in scores.items()}
            for fam, scores in alt.items()
        },
        "exemplar_diagnostic_blockers": [
            {"result_id": e.result_id, "kind": e.kind, "family": e.family,
             "diagnostic_blocking_axis": e.diagnostic_blocking_axis}
            for e in (getattr(evidence, "exemplars", None) or [])
            if e.diagnostic_blocking_axis
        ][:6],
        "recipes": [{
            "family": r.method_family, "class": r.recipe_class,
            "config_delta": r.config_delta, "recency_tick": r.recency_tick,
            "descendant_count": r.descendant_count,
        } for r in evidence.recipes[:8]],
        "planner_proposal": {
            "valid": planner_output.valid,
            "abstain": planner_output.abstain,
            "rationale": (planner_output.rationale or "")[:500],
            "cards": [{
                "claim": (c.claim or "")[:300],
                "mode_affinity": c.mode_affinity,
                "recommended_action_families": c.recommended_action_families,
            } for c in (planner_output.cards or [])[:4]],
        },
    }


def build_critic_user_prompt(
    evidence: EvidenceSummary,
    planner_output: Any,
) -> str:
    """Build the exact user message sent to the publication critic."""

    payload = build_critic_payload(evidence, planner_output)
    return (
        "Review the Planner's recommendation against the evidence. "
        "Flag specific, evidence-cited issues per the categories above, or "
        "'no_flags'.\n\n"
        + json.dumps(payload, default=str, indent=2)
    )

