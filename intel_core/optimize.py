"""Pure-Python MMM budget simulation & diminishing-returns allocation consuming `response_curves.json`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def hill_saturation(spend_usd: float, ec50_usd: float, slope: float = 1.5) -> float:
    """Computes Hill saturation response in [0, 1) for weekly channel spend in USD."""
    x = max(0.0, float(spend_usd))
    if x <= 0.0:
        return 0.0
    k = max(1e-6, float(ec50_usd))
    s = max(0.1, float(slope))
    xs = x**s
    ks = k**s
    return xs / (xs + ks)


def channel_contribution(spend_usd: float, curve: dict[str, Any]) -> float:
    """Computes expected weekly incremental outcome contribution for one channel."""
    beta = float(curve.get("beta", 0.0))
    ec50 = float(curve.get("ec50_usd", 1.0))
    slope = float(curve.get("slope", 1.5))
    return beta * hill_saturation(spend_usd, ec50_usd=ec50, slope=slope)


def marginal_response(spend_usd: float, curve: dict[str, Any], delta_usd: float = 100.0) -> float:
    """Numerical marginal contribution per $1 incremental spend at `spend_usd`."""
    step = max(1.0, float(delta_usd))
    c0 = channel_contribution(spend_usd, curve)
    c1 = channel_contribution(spend_usd + step, curve)
    return (c1 - c0) / step


def load_response_curves(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Loads `response_curves.json` from a file path or returns a copy of a dict payload."""
    if isinstance(source, dict):
        return dict(source)
    path = Path(source)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _optimize_greedy_waterfill(
    curves_by_ch: dict[str, dict[str, Any]],
    total_budget_usd: float,
    baseline_spend: dict[str, float],
    min_share_ratio: float = 0.5,
    max_share_ratio: float = 1.75,
    n_steps: int = 200,
) -> dict[str, float]:
    """Pure-Python constrained budget allocator maximizing total Hill response via discrete marginal ascent."""
    channels = list(curves_by_ch.keys())
    if not channels or total_budget_usd <= 0:
        return {ch: 0.0 for ch in channels}

    floors: dict[str, float] = {}
    ceilings: dict[str, float] = {}
    for ch in channels:
        base = max(0.0, float(baseline_spend.get(ch, 0.0)))
        if base > 0:
            floors[ch] = base * min_share_ratio
            ceilings[ch] = max(base * max_share_ratio, total_budget_usd / len(channels))
        else:
            floors[ch] = 0.0
            ceilings[ch] = total_budget_usd

    floor_sum = sum(floors.values())
    if floor_sum > total_budget_usd:
        scale = total_budget_usd / floor_sum
        floors = {ch: v * scale for ch, v in floors.items()}

    alloc = dict(floors)
    remaining = max(0.0, total_budget_usd - sum(alloc.values()))
    if remaining <= 1e-6:
        return {ch: round(v, 2) for ch, v in alloc.items()}

    step = remaining / float(max(1, n_steps))
    for _ in range(n_steps):
        eligible = [ch for ch in channels if alloc[ch] + step <= ceilings[ch] + 1e-6]
        if not eligible:
            eligible = channels
        best_ch = max(
            eligible,
            key=lambda ch: marginal_response(alloc[ch], curves_by_ch[ch], delta_usd=step),
        )
        alloc[best_ch] += step

    # Absorb tiny float rounding drift on highest-marginal channel
    drift = total_budget_usd - sum(alloc.values())
    if channels and abs(drift) > 1e-6:
        alloc[channels[0]] = max(0.0, alloc[channels[0]] + drift)
    return {ch: round(v, 2) for ch, v in alloc.items()}


def simulate_budget_reallocation(
    response_curves: str | Path | dict[str, Any],
    *,
    proposed_spend_by_channel: dict[str, float] | None = None,
    total_budget_usd: float | None = None,
    optimize: bool = False,
    min_share_ratio: float = 0.5,
    max_share_ratio: float = 1.75,
) -> dict[str, Any]:
    """Evaluates or optimizes weekly channel budget allocation using `response_curves.json`.

    Never requires PyMC/PySpark, keeping FastAPI/ADK lean while preserving ROI/proxy disclaimers.
    """
    payload = load_response_curves(response_curves)
    raw_channels = payload.get("channels") or {}
    if isinstance(raw_channels, list):
        curves_by_ch = {str(item["channel"]): dict(item) for item in raw_channels}
    else:
        curves_by_ch = {str(k): dict(v) for k, v in raw_channels.items()}

    baseline_spend = {
        ch: max(0.0, float(curve.get("baseline_weekly_spend_usd", 0.0)))
        for ch, curve in curves_by_ch.items()
    }
    baseline_total = sum(baseline_spend.values())

    if proposed_spend_by_channel is not None and not optimize:
        target_spend = {
            ch: max(0.0, float(proposed_spend_by_channel.get(ch, baseline_spend[ch])))
            for ch in curves_by_ch
        }
    else:
        budget = float(total_budget_usd) if total_budget_usd is not None else baseline_total
        target_spend = _optimize_greedy_waterfill(
            curves_by_ch,
            total_budget_usd=budget,
            baseline_spend=baseline_spend,
            min_share_ratio=min_share_ratio,
            max_share_ratio=max_share_ratio,
        )

    channel_rows: list[dict[str, Any]] = []
    base_outcome_total = 0.0
    sim_outcome_total = 0.0
    for ch, curve in curves_by_ch.items():
        b_spend = baseline_spend[ch]
        s_spend = target_spend[ch]
        b_contrib = channel_contribution(b_spend, curve)
        s_contrib = channel_contribution(s_spend, curve)
        base_outcome_total += b_contrib
        sim_outcome_total += s_contrib
        ec50 = float(curve.get("ec50_usd", 1.0))
        slope = float(curve.get("slope", 1.5))
        channel_rows.append(
            {
                "channel": ch,
                "baseline_weekly_spend_usd": round(b_spend, 2),
                "simulated_weekly_spend_usd": round(s_spend, 2),
                "spend_delta_usd": round(s_spend - b_spend, 2),
                "baseline_contribution": round(b_contrib, 4),
                "simulated_contribution": round(s_contrib, 4),
                "contribution_delta": round(s_contrib - b_contrib, 4),
                "saturation_pct": round(hill_saturation(s_spend, ec50_usd=ec50, slope=slope) * 100.0, 2),
                "marginal_response_per_usd": round(marginal_response(s_spend, curve), 6),
                "adstock_lambda": float(curve.get("adstock_lambda", 0.0)),
                "roas_mean": curve.get("roas_mean"),
                "roas_hdi_90": curve.get("roas_hdi_90"),
            }
        )

    lift_pct = (
        ((sim_outcome_total - base_outcome_total) / base_outcome_total) * 100.0
        if base_outcome_total > 1e-9
        else 0.0
    )
    is_proxy = bool(payload.get("is_proxy_outcome", False))
    is_spend_modeled = bool(payload.get("is_spend_modeled", False))
    roi_label = payload.get(
        "roi_label",
        "proxy_index_only"
        if is_proxy
        else ("modeled_reach_cpm_proxy" if is_spend_modeled else "incremental_roas"),
    )
    disclaimer = payload.get("disclaimer", "")
    if not disclaimer:
        if is_proxy:
            disclaimer = (
                "Proxy index outcome (e.g. Google Trends): directional elasticity & index lift only, NOT financial ROAS."
            )
        elif is_spend_modeled:
            disclaimer = (
                "Spend is modeled from reach/CPM benchmarks (EU DSA); treat marginal returns as directional efficiency estimates."
            )

    return {
        "brand_id": payload.get("brand_id"),
        "country_iso": payload.get("country_iso"),
        "outcome_metric_name": payload.get("outcome_metric_name", "conversions"),
        "roi_label": roi_label,
        "is_proxy_outcome": is_proxy,
        "is_spend_modeled": is_spend_modeled,
        "disclaimer": disclaimer,
        "baseline_total_spend_usd": round(baseline_total, 2),
        "simulated_total_spend_usd": round(sum(target_spend.values()), 2),
        "baseline_total_contribution": round(base_outcome_total, 4),
        "simulated_total_contribution": round(sim_outcome_total, 4),
        "expected_outcome_lift_pct": round(lift_pct, 2),
        "channels": channel_rows,
    }
