"""The closed-form progress values compute what they say they compute.

Each candidate is a one-line formula, which is exactly the kind of code that
drifts silently: swap a p for a P and the ablation still runs, still produces a
table, and still looks plausible. Each is checked against its closed form
evaluated independently here, at a mid-job state where none of the guards fire.
"""

import math

import pytest

from sky_spot.strategies import _v_candidates as vc
from sky_spot.strategies.strategy import Strategy
from sky_spot.strategies.unified_cost_model_f_ablation import (
    ABLATION_EXTRA_STRATEGIES,
    ABLATION_STRATEGIES,
)

# Mid-job: 30 h elapsed of a 100 h deadline, 20 h done of 60 h of work. Behind
# schedule, so the schedule ratio is above 1 and the candidates spread out.
T = 100 * 3600.0
t = 30 * 3600.0
P = 60 * 3600.0
p = 20 * 3600.0
C_OD = 3.06

THETA = (P - p) / (T - t)
# Two readings of theta_tilde, kept apart because they are not interchangeable:
# PAST is the rate the job has achieved, p/t; PLANNED is the rate it was scheduled
# to run at, P/T. Dividing c_od out of v_rate_ratio leaves exactly THETA/PLANNED.
THETA_PAST = p / t
THETA_PLANNED = P / T
RATIO_PAST = THETA / THETA_PAST
RATIO = THETA / THETA_PLANNED


def test_schedule_ratio_matches_its_expansion():
    # theta/theta_bar expands to t(P-p) / ((T-t)p).
    assert vc._schedule_ratio(t, T, p, P) == pytest.approx(
        (t * (P - p)) / ((T - t) * p)
    )


def test_planned_rate_ratio_matches_its_expansion():
    # theta/theta_tilde == T(P-p) / ((T-t)P) when theta_tilde is the planned rate.
    assert vc._planned_rate_ratio(t, T, p, P) == pytest.approx(
        (T * (P - p)) / ((T - t) * P)
    )


def test_rate_ratio_is_c_od_times_the_planned_ratio():
    """v_rate_ratio is c_od * theta/theta_tilde on the planned reading."""
    assert vc.v_rate_ratio(t, T, p, P, C_OD) == pytest.approx(C_OD * RATIO)


@pytest.mark.parametrize(
    "func, expected",
    [
        (vc.v_s1t1, C_OD * (t / (T - t)) * ((P - p) / p)),
        (vc.v_s2t1, C_OD * (t / (T - t)) * ((P - p) / P)),
        (vc.v_s3t1, C_OD * (t / (T - t)) * math.log(P / p)),
        (vc.v_s5t1, C_OD * (t / (T - t))),
        (vc.v_s1t5, C_OD * ((P - p) / p)),
        (vc.v_alpha_05, C_OD * RATIO ** 0.5),
        (vc.v_alpha_15, C_OD * RATIO ** 1.5),
        (vc.v_hjb_exp, C_OD * math.exp(RATIO - 1.0)),
        # Drift-plus-penalty on the virtual backlog; see v_neely_dpp's docstring.
        (vc.v_neely_dpp, C_OD * (1.0 + max(0.0, (P / T) * t - p) / (P - p))),
        (vc.v_llf, C_OD * (P - p) / ((T - t) - (P - p))),
        (vc.v_rate_ratio, C_OD * (T / (T - t)) * ((P - p) / P)),
        (vc.v_rate_past, C_OD * ((P - p) / (T - t)) / (p / t)),
    ],
)
def test_candidate_matches_closed_form(func, expected):
    assert func(t, T, p, P, C_OD) == pytest.approx(expected)


def test_s1t1_and_rate_past_are_the_same_formula():
    """Both compute the achieved-rate costate, so they must agree exactly."""
    assert vc.v_s1t1(t, T, p, P, C_OD) == pytest.approx(
        vc.v_rate_past(t, T, p, P, C_OD)
    )


def test_rate_ratio_and_rate_past_are_not_the_same_formula():
    """One divides by the planned rate P/T, the other by the achieved rate p/t.

    Pinned so a future edit cannot quietly merge two distinct closed forms.
    """
    assert vc.v_rate_ratio(t, T, p, P, C_OD) != pytest.approx(
        vc.v_rate_past(t, T, p, P, C_OD)
    )


@pytest.mark.parametrize(
    "func", [vc.v_s1t1, vc.v_alpha_05, vc.v_hjb_exp, vc.v_rate_ratio, vc.v_neely_dpp]
)
def test_anchored_candidates_equal_c_od_on_schedule(func):
    """The anchor these forms are built around: on schedule, V = C_od.

    On schedule means half the time gone and half the work done, which makes both
    readings of theta_tilde equal theta. The ratio-only forms then land on 1**0.5
    and exp(0), and Neely's backlog queue is empty, so all of them sit at C_od.
    """
    assert func(T / 2, T, P / 2, P, C_OD) == pytest.approx(C_OD)


def test_quadratic_deficit_floors_the_bare_quadratic():
    """s2t1_floored never drops below c_od*(P-p)/P, which bare s2t1 does at t=0."""
    early_t = 1.0  # one second in
    bare = vc.v_s2t1(early_t, T, p, P, C_OD)
    floored = vc.v_s2t1_floored(early_t, T, p, P, C_OD)
    assert bare < floored
    assert floored == pytest.approx(C_OD * (P - p) / P)


def test_every_candidate_in_the_registry_is_callable():
    for name, func in vc.ALL_CANDIDATES.items():
        value = func(t, T, p, P, C_OD)
        assert isinstance(value, float), name
        assert value >= 0.0, name


def test_ablation_strategies_are_registered():
    for name in ABLATION_STRATEGIES + ABLATION_EXTRA_STRATEGIES:
        assert name in Strategy.SUBCLASSES, name
    assert len(ABLATION_STRATEGIES) == 6
