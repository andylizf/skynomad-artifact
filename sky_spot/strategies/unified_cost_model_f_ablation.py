"""Closed-form V-shape ablation strategies.

Each strategy routes `_compute_time_value()` through a candidate from
`sky_spot.strategies._v_candidates`, keeping the rest of the UCM pipeline
identical. Only the `V = c_od * f(theta/theta_tilde)` shape varies; the lifetime
predictor and gate behaviour inherit from UCM defaults, so the comparison
isolates V.

These are the arms behind the progress-value ablation of the appendix. The six it
tabulates, with the closed form each one scores:

    Log-barrier (default)   ucm_v_s1t1          c_od * theta/theta_bar
    alpha = 0.5             ucm_v_alpha_05      c_od * (theta/theta_bar)^0.5
    Quadratic surrogate     ucm_v_s2t1          c_od * t/(T-t) * (P-p)/P
    HJB exponential         ucm_v_hjb_exp       c_od * exp(theta/theta_bar - 1)
    Time-only               ucm_v_s5t1          c_od * t/(T-t)
    Neely DPP               ucm_v_neely_dpp     c_od * (1 + q/(P-p)), q the backlog

`ucm_v_s1t1` and `unified_cost_model_rate_past` are the same closed form under two
registered names.

Four cells: V100 4-region and H100 8-region, each at T/P in {1.5, 2.0}, twelve
stratified windows apiece.

Three further shapes the appendix does not tabulate -- an unfloored quadratic
deficit, alpha=1.5, and least-laxity-first -- are registered in
ABLATION_EXTRA_STRATEGIES.

Run the ablation with artifact/v_ablation.py.
"""
from __future__ import annotations

import os
import typing

from sky_spot.strategies import _v_candidates as vcand
from sky_spot.strategies.unified_cost_model import UnifiedCostModelStrategy
from sky_spot.strategies.unified_cost_model_v_variants import OracleDurationMixin
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env as env_lib


class _VCandidateMixin:
    """Mixin that wires a `_v_candidates` function into UCM's V slot.

    Concrete subclasses set V_FUNC; the mixin comes before UnifiedCostModelStrategy
    in the MRO so it does not trigger Strategy's __init_subclass__ registration.
    """

    V_FUNC: typing.Callable = staticmethod(vcand.v_s1t1)  # overridden per subclass

    # The ablation isolates V, so every arm shares the estimator the rest of the
    # evaluation uses. Set V_ABLATION_ORACLE_DURATION=1 to swap in the oracle
    # that reads true remaining lifetime off the trace.
    USE_ORACLE_DURATION = os.environ.get("V_ABLATION_ORACLE_DURATION", "0") == "1"

    def _compute_time_value(self) -> float:
        D = self.task_duration
        T = self.deadline
        p = sum(self.task_done_time)
        t = self.env.elapsed_seconds
        if T - t <= 0:
            return float("inf")
        if D - p <= 0:
            return 0.0

        env_t = typing.cast("env_lib.MultiTraceEnv", self.env)
        od_prices = [
            env_t.envs[r].get_price()[ClusterType.ON_DEMAND]
            for r in range(env_t.num_regions)
        ]
        spot_prices = [
            env_t.envs[r].get_price()[ClusterType.SPOT]
            for r in range(env_t.num_regions)
        ]
        c_od = min(od_prices)
        c_spot_min = min(spot_prices)

        V = self.__class__.V_FUNC(t, T, p, D, c_od, c_spot_min=c_spot_min)
        floor = os.environ.get("V_ABLATION_FLOOR", "spot")
        if floor == "none":
            return V
        if floor == "od":
            # Floor at the on-demand price rather than the cheapest spot price.
            # Not the default: on the H100 cells c_od dominates every candidate's
            # V, so all six collapse onto one trajectory and the ablation stops
            # separating them at all.
            return max(V, c_od)
        return max(V, c_spot_min)

    def _log_time_value(self, V: float) -> None:  # pragma: no cover - debug only
        pass


# Every arm lists its bases directly rather than sharing an intermediate class:
# an intermediate would inherit UnifiedCostModelStrategy's NAME and collide with
# it in the registry the moment it is defined.


# ---------- the S-by-T family ----------

class UCMVs1t1(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    """V = c_od * t/(T-t) * (P-p)/p, the achieved-rate ratio theta/theta_bar."""
    NAME = "ucm_v_s1t1"
    V_FUNC = staticmethod(vcand.v_s1t1)


class UCMVLogBarrier(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    """Log-barrier costate: V = c_od * T/(T-t) * (P-p)/P.

    Divides remaining work by P and scales time by T, where `ucm_v_s1t1` divides
    by p and scales by t. The two agree only when the job is exactly on schedule;
    off schedule they diverge, by 33% a fifth of the way into a job that has
    fallen behind.
    """
    NAME = "ucm_v_log_barrier"
    V_FUNC = staticmethod(vcand.v_rate_ratio)


class UCMVs2t1(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    """Quadratic surrogate: V = c_od * t/(T-t) * (P-p)/P."""
    NAME = "ucm_v_s2t1"
    V_FUNC = staticmethod(vcand.v_s2t1)


class UCMVs2t1Floored(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    """Quadratic deficit: S2T1 with the equilibrium-anchoring floor."""
    NAME = "ucm_v_s2t1_floored"
    V_FUNC = staticmethod(vcand.v_s2t1_floored)


class UCMVs3t1(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    NAME = "ucm_v_s3t1"
    V_FUNC = staticmethod(vcand.v_s3t1)


class UCMVs4t1A05(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    NAME = "ucm_v_s4t1_a05"
    V_FUNC = staticmethod(vcand.v_s4t1_a05)


class UCMVs4t1A20(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    NAME = "ucm_v_s4t1_a20"
    V_FUNC = staticmethod(vcand.v_s4t1_a20)


class UCMVs1t3(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    NAME = "ucm_v_s1t3"
    V_FUNC = staticmethod(vcand.v_s1t3)


class UCMVs1t4(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    NAME = "ucm_v_s1t4"
    V_FUNC = staticmethod(vcand.v_s1t4)


class UCMVs5t1(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    """Time-only: V = c_od * t/(T-t)."""
    NAME = "ucm_v_s5t1"
    V_FUNC = staticmethod(vcand.v_s5t1)


class UCMVs1t5(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    NAME = "ucm_v_s1t5"
    V_FUNC = staticmethod(vcand.v_s1t5)


# ---------- forms written against the whole schedule ratio ----------

class UCMVAlpha05(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    """Power family, alpha=0.5: V = c_od * (theta/theta_bar)^0.5."""
    NAME = "ucm_v_alpha_05"
    V_FUNC = staticmethod(vcand.v_alpha_05)


class UCMVAlpha15(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    """Power family, alpha=1.5."""
    NAME = "ucm_v_alpha_15"
    V_FUNC = staticmethod(vcand.v_alpha_15)


class UCMVHJBExp(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    """HJB exponential: V = c_od * exp(theta/theta_bar - 1)."""
    NAME = "ucm_v_hjb_exp"
    V_FUNC = staticmethod(vcand.v_hjb_exp)


class UCMVNeelyDPP(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    """Neely drift-plus-penalty on the virtual backlog q = (P*t/T - p)+:
    V = c_od * (1 + q/(P-p)), anchored at c_od when the job is on schedule."""
    NAME = "ucm_v_neely_dpp"
    V_FUNC = staticmethod(vcand.v_neely_dpp)


class UCMVLLF(_VCandidateMixin, OracleDurationMixin, UnifiedCostModelStrategy):
    """Least-laxity-first: V = c_od * (P-p) / max((T-t)-(P-p), eps)."""
    NAME = "ucm_v_llf"
    V_FUNC = staticmethod(vcand.v_llf)


# ---------- lambda-grid: scalar multiplier on the rate-ratio V ----------

class _LambdaScaledMixin:
    LAM: float = 1.0

    def _compute_time_value(self) -> float:
        D = self.task_duration
        T = self.deadline
        p = sum(self.task_done_time)
        t = self.env.elapsed_seconds
        if T - t <= 0:
            return float("inf")
        if D - p <= 0:
            return 0.0

        env_t = typing.cast("env_lib.MultiTraceEnv", self.env)
        od_prices = [
            env_t.envs[r].get_price()[ClusterType.ON_DEMAND]
            for r in range(env_t.num_regions)
        ]
        spot_prices = [
            env_t.envs[r].get_price()[ClusterType.SPOT]
            for r in range(env_t.num_regions)
        ]
        c_od = min(od_prices)
        c_spot_min = min(spot_prices)

        ratio = ((D - p) * T) / ((T - t) * D)
        V = self.__class__.LAM * c_od * ratio
        return max(V, c_spot_min)


class UCMLambda05(_LambdaScaledMixin, UnifiedCostModelStrategy):
    NAME = "ucm_lambda_0_5"
    LAM = 0.5


class UCMLambda08(_LambdaScaledMixin, UnifiedCostModelStrategy):
    NAME = "ucm_lambda_0_8"
    LAM = 0.8


class UCMLambda12(_LambdaScaledMixin, UnifiedCostModelStrategy):
    NAME = "ucm_lambda_1_2"
    LAM = 1.2


class UCMLambda15(_LambdaScaledMixin, UnifiedCostModelStrategy):
    NAME = "ucm_lambda_1_5"
    LAM = 1.5


# The six arms the appendix tabulates, in its display order.
ABLATION_STRATEGIES = [
    UCMVLogBarrier.NAME,  # Log-barrier (default)
    UCMVAlpha05.NAME,    # alpha = 0.5
    UCMVs2t1.NAME,       # Quadratic surrogate
    UCMVHJBExp.NAME,     # HJB exponential
    UCMVs5t1.NAME,       # Time-only
    UCMVNeelyDPP.NAME,   # Neely DPP
]

# Further shapes the appendix does not tabulate, reachable with
# `artifact/v_ablation.py --extra`.
ABLATION_EXTRA_STRATEGIES = [
    UCMVs2t1Floored.NAME,  # quadratic deficit, the floored quadratic
    UCMVAlpha15.NAME,      # alpha = 1.5
    UCMVLLF.NAME,          # least laxity first
    UCMVs1t1.NAME,         # the achieved-rate ratio
]
