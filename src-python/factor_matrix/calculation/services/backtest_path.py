"""Versioned path functionals, participation bounds and supported-policy OPE.

Functions accept audited observations; they do not load market data, loosen a
research gate, fit impact parameters, or select a profitable policy/window.
"""
from __future__ import annotations

from decimal import Decimal

from .accounting import decimal, instant, quantity

D = Decimal


def _path_functionals(rows: list[dict], *, initial_nav: str, initial_time: str, flow_timing="period_end") -> dict:
    """Daily-observation path, end-of-period external cash-flow convention.

    Flow-neutral wealth W[t] = W[t-1] * (NAV[t]-flow[t])/NAV[t-1].
    No intraday high/low drawdown is inferred from daily NAV observations.
    """
    if flow_timing not in ("period_end", "period_start"):
        raise ValueError("INVALID_FLOW_TIMING")
    prior_nav, wealth, peak = decimal(initial_nav), D(1), D(1)
    if prior_nav <= 0 or not rows:
        raise ValueError("PATH_REQUIRES_POSITIVE_INITIAL_NAV_AND_OBSERVATIONS")
    prior_time = instant(initial_time)
    peak_time, peak_index = initial_time, -1
    maximum, area, quadratic_variation = D(0), D(0), D(0)
    peak_of_max, trough_of_max, recovery = initial_time, initial_time, None
    max_peak_value, max_trough_index = D(1), -1
    longest_underwater, current_underwater = 0, 0
    curve = []
    for index, row in enumerate(rows):
        now = instant(row["time"])
        nav, flow = decimal(row["nav"]), decimal(row.get("external_flow", "0"))
        denominator = prior_nav + flow if flow_timing == "period_start" else prior_nav
        numerator = nav if flow_timing == "period_start" else nav-flow
        if now <= prior_time or nav <= 0 or numerator <= 0 or denominator <= 0:
            raise ValueError("PATH_CLOCK_OR_CAPITAL_INVALID")
        gross = numerator / denominator
        wealth *= gross
        quadratic_variation += gross.ln() ** 2
        if wealth >= peak:
            peak, peak_time, peak_index = wealth, row["time"], index
            current_underwater = 0
        else:
            current_underwater = index - peak_index
        longest_underwater = max(longest_underwater, current_underwater)
        dd = D(1) - wealth / peak
        if dd > maximum:
            maximum, peak_of_max, trough_of_max = dd, peak_time, row["time"]
            max_peak_value, max_trough_index, recovery = peak, index, None
        elif maximum > 0 and recovery is None and index > max_trough_index and wealth >= max_peak_value:
            recovery = row["time"]
        area += dd
        curve.append({"time": row["time"], "nav": nav, "external_flow": flow,
                      "period_return": gross - 1, "unit_wealth": wealth,
                      "running_peak": peak, "drawdown": dd, "underwater_observations": current_underwater})
        prior_nav, prior_time = nav, now
    return {"contract_id": "backtest_path_v1", "sampling": "supplied_nav_observations", "initial_nav": initial_nav, "initial_time": initial_time,
            "flow_convention": "external_flow_at_"+flow_timing, "total_return": wealth - 1,
            "max_drawdown": maximum, "max_drawdown_peak_time": peak_of_max if maximum else None,
            "max_drawdown_trough_time": trough_of_max if maximum else None,
            "max_drawdown_recovery_time": recovery, "longest_underwater_observations": longest_underwater,
            "terminal_underwater_observations": current_underwater, "drawdown_area_observations": area,
            "mean_drawdown": area / len(rows), "log_quadratic_variation": quadratic_variation, "curve": curve}


def path_functionals(rows: list[dict], *, initial_nav: str, initial_time: str) -> dict:
    """Frozen v1: flows arrive after period return and before reported NAV."""
    return _path_functionals(rows, initial_nav=initial_nav, initial_time=initial_time)


def path_functionals_v3(rows: list[dict], *, initial_nav: str, initial_time: str, flow_timing: str) -> dict:
    """Explicit initial/end flow timing; v1/v2 default arithmetic is unchanged."""
    result = _path_functionals(rows, initial_nav=initial_nav, initial_time=initial_time, flow_timing=flow_timing)
    result['contract_id'] = 'backtest_path_v3'
    result['right_censored'] = result['terminal_underwater_observations'] > 0
    return result


def path_functionals_v2(rows: list[dict], *, initial_nav: str, initial_time: str) -> dict:
    """v1 arithmetic plus explicit sampled-observation drawdown durations.

    A duration counts intervals, includes the recovery observation, and is null
    when unrecovered. This is not elapsed calendar time or an intraday estimate.
    """
    result = path_functionals(rows, initial_nav=initial_nav, initial_time=initial_time)
    indices = {initial_time: -1, **{r['time']: i for i, r in enumerate(rows)}}
    peak, trough, recovery = (result[f'max_drawdown_{key}_time'] for key in ('peak', 'trough', 'recovery'))
    result.update(contract_id='backtest_path_v2', duration_unit='supplied_observation_intervals',
        max_drawdown_peak_to_trough_observations=indices[trough]-indices[peak] if peak is not None else 0,
        max_drawdown_trough_to_recovery_observations=indices[recovery]-indices[trough] if recovery is not None else None,
        max_drawdown_peak_to_recovery_observations=indices[recovery]-indices[peak] if recovery is not None else None)
    return result


def capacity_diagnostics(rows: list[dict], *, participation: str, lot_size: int) -> dict:
    """Prior-ADV participation bound; not a calibrated market-impact capacity.

    ADV must be in SHARES (not vendor lots). The estimated days-to-trade assumes
    unchanged ADV; it is not a promise about future liquidity or realizable PnL.
    """
    rate = decimal(participation)
    quantity(lot_size)
    if not D(0) < rate <= 1:
        raise ValueError("INVALID_PARTICIPATION")
    output = []
    for r in rows:
        requested = quantity(r["requested_quantity"], allow_zero=True)
        price = decimal(r["reference_price"])
        decision = instant(r["decision_time"])
        if price <= 0:
            raise ValueError("INVALID_CAPACITY_PRICE")
        status, adv = "available", r.get("adv_shares")
        if adv is None or r.get("adv_observations", 0) < r["required_observations"]:
            status = "insufficient_liquidity_history"
        elif (instant(r["adv_available_at"]) > decision
              or instant(r["adv_last_observation"]) >= decision):
            raise ValueError("FUTURE_LIQUIDITY_INPUT")
        elif decimal(adv) <= 0:
            status = "no_observed_liquidity"
        cap = int(decimal(adv) * rate / lot_size) * lot_size if status == "available" else None
        output.append({"asset_id": r["asset_id"], "decision_time": r["decision_time"], "status": status,
            "requested_quantity": requested, "adv_shares": adv, "participation": rate,
            "one_session_capacity_shares": cap, "one_session_capacity_notional": cap * price if cap is not None else None,
            "participation_demand": D(requested) / decimal(adv) if status == "available" else None,
            "estimated_sessions_at_constant_adv": ((requested + cap - 1) // cap if cap else (0 if not requested and cap == 0 else None)),
            "quantity_shortfall": max(0, requested - cap) if cap is not None else None})
    return {"contract_id": "participation_capacity_v1", "volume_unit": "shares", "rows": output,
            "impact_calibration_status": "unavailable", "economic_capacity_cny": None,
            "limitation": "ADV participation bound only; no uncalibrated impact assumption is treated as zero"}


def off_policy_evaluation(episodes: list[list[dict]], *, assumptions_verified: bool = False, discount: str = "1") -> dict:
    """Finite-action episodic IS/PDIS/WIS; no weight clipping or support repair.

    Each step supplies full behavior and target action probability vectors, the
    observed action index, and its reward. Unverified logging/causal assumptions
    or unsupported target actions fail closed, rather than fabricating an OPE.
    WIS is self-normalized and generally biased in finite samples.
    """
    unavailable = {"contract_id": "episodic_ope_v1", "status": "not_estimable", "estimate": None}
    if not assumptions_verified:
        return {**unavailable, "reason": "LOGGING_AND_SEQUENTIAL_IDENTIFICATION_NOT_VERIFIED"}
    gamma = decimal(discount)
    if not 0 <= gamma <= 1 or not episodes or any(not episode for episode in episodes):
        raise ValueError("INVALID_OPE_INPUT")
    estimates, weights, pdis = [], [], []
    for episode in episodes:
        weight, total, step_total = D(1), D(0), D(0)
        for t, step in enumerate(episode):
            if not {"behavior_probabilities", "target_probabilities", "action", "reward"} <= set(step):
                return {**unavailable, "reason": "MISSING_LOGGED_PROPENSITIES"}
            b, p = [decimal(x) for x in step["behavior_probabilities"]], [decimal(x) for x in step["target_probabilities"]]
            action = step["action"]
            if (not b or len(b) != len(p) or type(action) is not int or not 0 <= action < len(b)
                or any(x < 0 or x > 1 for x in b + p) or abs(sum(b) - 1) > D("1e-12") or abs(sum(p) - 1) > D("1e-12")):
                raise ValueError("INVALID_POLICY_PROBABILITIES")
            if any(pi > 0 and bi == 0 for pi, bi in zip(p, b)) or b[action] == 0:
                return {**unavailable, "reason": "TARGET_POLICY_OUTSIDE_BEHAVIOR_SUPPORT"}
            weight *= p[action] / b[action]
            reward = decimal(step["reward"]) * gamma ** t
            total += reward
            step_total += weight * reward
        estimates.append(weight * total)
        weights.append(weight)
        pdis.append(step_total)
    denominator = sum(weights)
    squared = sum(w * w for w in weights)
    if not denominator:
        return {**unavailable, "reason": "NO_OBSERVED_TARGET_WEIGHT"}
    return {"contract_id": "episodic_ope_v1", "status": "estimated_under_declared_assumptions",
            "trajectory_is": sum(estimates) / len(episodes), "per_decision_is": sum(pdis) / len(episodes),
            "weighted_is": sum(estimates) / denominator, "effective_sample_size": denominator ** 2 / squared,
            "max_trajectory_weight": max(weights), "episodes": len(episodes), "weight_clipping": False,
            "confidence_interval": None, "policy_selection_authorized": False,
            "limitations": ["Sequential identification and environment consistency must hold",
                            "WIS finite-sample bias; no confidence guarantee from ESS alone",
                            "Single deterministic backtest is not a supported behavior-policy log"]}
