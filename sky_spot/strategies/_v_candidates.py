"""Closed-form V candidates for appendix A.7's progress-value ablation.

The six the appendix tabulates, and how they are wired into the utility pipeline,
are listed in unified_cost_model_f_ablation.py. Run them with
artifact/v_ablation.py.

Each candidate has signature:
    v_<name>(t, T, p, P, c_od, *, c_spot_min=None, phi=None) -> float

Inputs are SI units: t / T in seconds, p / P in seconds-of-work, c_od / c_spot_min in $/hour.
Output is V in $/hour.

Candidates decompose as V = time_factor(t, T) * shape(p, P) * c_od, except for ramp_v
which follows the piecewise-linear form used by the existing strategy.
"""

from __future__ import annotations
import math
import os
from typing import Callable, Dict

EPS = 1e-9
BIG = 1e9  # sentinel for "essentially infinite"; keeps CSVs float-parseable


def _on_schedule_progress(t, T, p, P) -> float:
    """`p`, or the on-schedule progress P*t/T when no progress exists yet.

    Several candidates divide by p, and at the first tick p is exactly zero, so
    the quotient diverges and the policy would pay any price for its first
    instance. But a job that has just started is by definition *on schedule*,
    and the equilibrium-anchoring property these forms are built around says
    V = c_od there. Substituting the on-schedule progress P*t/T is what makes
    that anchor hold at the start: for the log-barrier form it gives
    c_od * t/(T-t) * (P - Pt/T)/(Pt/T) = c_od * T/(T-t), which tends to c_od.

    This is also what `unified_cost_model_rate_past` does, so the closed form
    and the strategy that carries the same formula agree everywhere.
    """
    if p > EPS:
        return p
    if t <= EPS or T <= EPS:
        return EPS
    return max(P * t / T, EPS)


def _safe(x: float) -> float:
    if not math.isfinite(x):
        return BIG if x > 0 else -BIG
    if x > BIG:
        return BIG
    return x


# ---------- canonical surrogate family: time_factor × shape × c_od ----------

def v_s1t1(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Canonical form of the surrogate family; equals rate_past.
    V = c_od * t/(T-t) * (P-p)/p
    """
    rem_t = T - t
    if rem_t <= EPS:
        return BIG
    p_eff = _on_schedule_progress(t, T, p, P)
    return _safe(c_od * (t / rem_t) * ((P - p_eff) / p_eff))


def v_s2t1(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Quadratic shape: V = c_od * t/(T-t) * (P-p)/P"""
    rem_t = T - t
    if rem_t <= EPS:
        return BIG
    return _safe(c_od * (t / rem_t) * ((P - p) / P))


def v_s3t1(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """KL-style shape: V = c_od * t/(T-t) * log(P/p)"""
    rem_t = T - t
    if rem_t <= EPS:
        return BIG
    p_eff = _on_schedule_progress(t, T, p, P)
    return _safe(c_od * (t / rem_t) * math.log(P / p_eff))


def v_s4t1_a05(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Power-alpha=0.5: V = c_od * t/(T-t) * sqrt((P-p)/p)"""
    rem_t = T - t
    if rem_t <= EPS:
        return BIG
    p_eff = _on_schedule_progress(t, T, p, P)
    r = (P - p_eff) / p_eff
    if r < 0:
        return 0.0
    return _safe(c_od * (t / rem_t) * math.sqrt(r))


def v_s4t1_a20(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Power-alpha=2: V = c_od * t/(T-t) * ((P-p)/p)^2"""
    rem_t = T - t
    if rem_t <= EPS:
        return BIG
    p_eff = _on_schedule_progress(t, T, p, P)
    r = (P - p_eff) / p_eff
    return _safe(c_od * (t / rem_t) * r * r)


def v_s1t3(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Quadratic-in-ratio time factor: V = c_od * (t/(T-t))^2 * (P-p)/p.
    Sharper deadline panic than S1T1 while remaining dimensionless.
    """
    rem_t = T - t
    if rem_t <= EPS:
        return BIG
    p_eff = _on_schedule_progress(t, T, p, P)
    tf = t / rem_t
    return _safe(c_od * (tf * tf) * ((P - p_eff) / p_eff))


def v_s1t4(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Log time factor: V = c_od * (-log(1 - t/T)) * (P-p)/p"""
    if T <= EPS:
        return BIG
    frac = t / T
    if frac >= 1 - EPS:
        return BIG
    if frac <= 0:
        return 0.0
    p_eff = _on_schedule_progress(t, T, p, P)
    return _safe(c_od * (-math.log(1 - frac)) * ((P - p_eff) / p_eff))


def v_s5t1(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Time-only (no progress pressure): V = c_od * t/(T-t)"""
    rem_t = T - t
    if rem_t <= EPS:
        return BIG
    return _safe(c_od * (t / rem_t))


def v_s1t5(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Progress-only (no deadline pressure): V = c_od * (P-p)/p"""
    p_eff = _on_schedule_progress(t, T, p, P)
    return _safe(c_od * ((P - p_eff) / p_eff))


# ---------- existing in-codebase variants ----------

def v_rate_ratio(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Existing strategy: V = c_od * T/(T-t) * (P-p)/P,
    i.e., required_rate / planned_rate D/T.
    """
    rem_t = T - t
    if rem_t <= EPS or T <= EPS:
        return BIG
    v = c_od * (T / rem_t) * ((P - p) / P)
    if c_spot_min is not None:
        v = max(v, c_spot_min)
    return _safe(v)


def v_rate_past(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Existing strategy: V = c_od * (required_rate) / (past_rate p/t).
    Identical to S1T1 algebraically; kept for explicit labeling.
    """
    rem_t = T - t
    if rem_t <= EPS:
        return BIG
    if p <= EPS or t <= EPS:
        # Early stage: past_rate undefined; fall back to planned rate D/T
        # (matching the real strategy's early-stage behavior).
        if T <= EPS:
            return BIG
        past_rate = P / T
    else:
        past_rate = p / t
    if past_rate <= EPS:
        return BIG
    req_rate = (P - p) / rem_t
    v = c_od * (req_rate / past_rate)
    if c_spot_min is not None:
        v = max(v, c_spot_min)
    return _safe(v)


def v_rate_avail(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Existing strategy: V = c_od * required_rate / avg_availability."""
    rem_t = T - t
    if rem_t <= EPS:
        return BIG
    if phi is None or phi <= 0.01:
        phi = 0.01
    req_rate = (P - p) / rem_t
    v = c_od * (req_rate / phi)
    if c_spot_min is not None:
        v = max(v, c_spot_min)
    return _safe(v)


def v_ramp_v(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Existing strategy: piecewise-linear ramp from c_spot to c_od as V_req→phi."""
    rem_t = T - t
    if rem_t <= EPS:
        return BIG
    V_req = (P - p) / rem_t
    if phi is None or phi <= 0.01:
        phi = 0.1
    c_s = c_spot_min if c_spot_min is not None else 0.0
    if V_req >= phi:
        v = c_od + 0.1
    else:
        v = c_s + (V_req / phi) * (c_od - c_s) + 0.1
    return _safe(v)


# ---------- appendix A.7 forms not in the family above ----------
# The appendix writes alpha-power, exponential and Neely against the schedule
# ratio theta/theta_bar as a whole, where the S-by-T family above varies the
# shape and time factors separately. Same pipeline, different parameterisation.


def _schedule_ratio(t, T, p, P) -> float:
    """theta/theta_bar = [(P-p)/(T-t)] / (p/t), the dimensionless schedule ratio.

    Before any progress exists there is no past rate, so this falls back to the
    planned rate P/T, matching what the strategies do at t=0.
    """
    rem_t = T - t
    if rem_t <= EPS:
        return BIG
    theta = (P - p) / rem_t
    theta_bar = _on_schedule_progress(t, T, p, P) / t if t > EPS else (
        (P / T) if T > EPS else 0.0)
    if theta_bar <= EPS:
        return BIG
    return theta / theta_bar


def _planned_rate_ratio(t, T, p, P) -> float:
    """theta/theta_tilde with theta_tilde the *planned* rate P/T: T(P-p)/((T-t)P).

    The appendix's log-barrier row is V = c_od * theta/theta_tilde, which in closed
    form is v_rate_ratio, c_od * T/(T-t) * (P-p)/P. Dividing out c_od leaves
    exactly this expression, so theta_tilde here is the rate the job was planned to
    run at, P/T -- not the rate it has achieved, p/t. The whole alpha-power family
    and the exponential form below are built on this ratio; _schedule_ratio is the
    achieved-rate reading, which is singular at t=0.
    """
    rem_t = T - t
    if rem_t <= EPS or P <= EPS:
        return BIG
    return (T * (P - p)) / (rem_t * P)


def v_alpha_power(t, T, p, P, c_od, *, c_spot_min=None, phi=None, alpha=0.5):
    """Power family on the whole ratio: V = c_od * (theta/theta_tilde)^alpha.

    alpha=1 recovers the log-barrier default (v_rate_ratio).
    """
    r = _planned_rate_ratio(t, T, p, P)
    if r >= BIG:
        return BIG
    return _safe(c_od * (max(r, 0.0) ** alpha))


def v_alpha_05(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Sublinear member of the power family, alpha=0.5."""
    return v_alpha_power(t, T, p, P, c_od, c_spot_min=c_spot_min, phi=phi, alpha=0.5)


def v_alpha_15(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Superlinear member of the power family, alpha=1.5."""
    return v_alpha_power(t, T, p, P, c_od, c_spot_min=c_spot_min, phi=phi, alpha=1.5)


def v_hjb_exp(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Exponential urgency anchored at c_od on schedule:
    V = c_od * exp(theta/theta_tilde - 1).

    The exponent is clipped to [-20, 20]. Where the clip binds it is only a guard
    against overflow: the pipeline treats every V far above c_od alike.
    """
    r = _planned_rate_ratio(t, T, p, P)
    if r >= BIG:
        return BIG
    return _safe(c_od * math.exp(max(-20.0, min(20.0, r - 1.0))))


def v_neely_dpp(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Neely drift-plus-penalty weight from the virtual backlog queue.

    A job that should have finished P*t/T by now but has only done p carries a
    backlog q = (P*t/T - p)+, and the weight is c_od * (1 + q/(P-p)): on schedule
    the queue is empty and V anchors at c_od, and it climbs as the backlog grows
    relative to the work left. Ignores the deadline except through the backlog,
    which is what the ablation contrasts it for.
    """
    if T <= EPS:
        return BIG
    q = max(0.0, (P / T) * t - p)
    rem_work = P - p
    if rem_work <= EPS:
        return 0.0
    return _safe(c_od * (1.0 + q / rem_work))


# LLF divides by laxity, so how it behaves near the deadline is set entirely by
# the floor under that divisor. The appendix writes `max(laxity, eps)` without
# fixing eps, and the choice is not cosmetic: at 1e-9 the value diverges to the
# sentinel and the policy buys on-demand for the rest of the job. Set
# V_ABLATION_LLF_EPS to a number of seconds to floor it at a physical scale
# instead (one tick, one restart, ...).
LLF_EPS = float(os.environ.get("V_ABLATION_LLF_EPS", "0") or 0)


def v_llf(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Least-Laxity-First: V = c_od * (P-p) / max((T-t)-(P-p), eps).

    Laxity is the slack left after the remaining work; as it goes to zero the
    value diverges, which is the deadline panic LLF is built around.
    """
    laxity = (T - t) - (P - p)
    floor = LLF_EPS if LLF_EPS > 0 else EPS
    if laxity <= floor:
        laxity = floor
    if laxity <= EPS:
        return BIG
    return _safe(c_od * (P - p) / laxity)


def v_s2t1_floored(t, T, p, P, c_od, *, c_spot_min=None, phi=None):
    """Quadratic deficit: V = c_od * (P-p)/P * max(t/(T-t), 1).

    v_s2t1 with an equilibrium-anchoring floor. Bare S2T1 gives V=0 at t=0,
    which breaks the anchor V(theta/theta_bar = 1) = c_od and idles early in a
    job; flooring at c_od*(P-p)/P keeps the late-phase aggression and the anchor.
    """
    v_original = v_s2t1(t, T, p, P, c_od, c_spot_min=c_spot_min, phi=phi)
    v_floor = c_od * max(P - p, 0.0) / P if P > EPS else 0.0
    return _safe(max(v_original, v_floor))


# ---------- registry ----------

ALL_CANDIDATES: Dict[str, Callable[..., float]] = {
    "V_S1T1": v_s1t1,
    "V_S2T1": v_s2t1,
    "V_S3T1": v_s3t1,
    "V_S4T1_a05": v_s4t1_a05,
    "V_S4T1_a20": v_s4t1_a20,
    "V_S1T3": v_s1t3,
    "V_S1T4": v_s1t4,
    "V_S5T1": v_s5t1,
    "V_S1T5": v_s1t5,
    "V_rate_ratio": v_rate_ratio,
    "V_rate_past": v_rate_past,
    "V_rate_avail": v_rate_avail,
    "V_ramp_v": v_ramp_v,
    "V_alpha_05": v_alpha_05,
    "V_alpha_15": v_alpha_15,
    "V_hjb_exp": v_hjb_exp,
    "V_neely_dpp": v_neely_dpp,
    "V_llf": v_llf,
    "V_s2t1_floored": v_s2t1_floored,
}
