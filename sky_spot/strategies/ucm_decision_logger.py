"""
UCM Decision Logger - detailed per-tick decision logging.

Usage:
    Set environment variable UCM_DECISION_LOG=1 to enable logging.
    Logs are written to stdout (terminal).
"""

import os
import sys
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Check if logging is enabled
UCM_LOG_ENABLED = os.environ.get("UCM_DECISION_LOG", "0") == "1"

# Setup logger (outputs to stderr to avoid capture by subprocess)
_logger: Optional[logging.Logger] = None

def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("ucm_decision")
        _logger.setLevel(logging.DEBUG)
        _logger.handlers.clear()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(handler)
        _logger.propagate = False
    return _logger


def log(msg: str) -> None:
    """Log a message if UCM_DECISION_LOG is enabled."""
    if UCM_LOG_ENABLED:
        _get_logger().debug(msg)


def log_tick_header(tick: int, elapsed_h: float, progress_pct: float, remaining_h: float) -> None:
    """Log tick header."""
    log("=" * 80)
    log(f"[UCM] TICK {tick} | elapsed={elapsed_h:.2f}h | progress={progress_pct:.1f}% | remaining={remaining_h:.2f}h")
    log("=" * 80)


def log_preemption(region: Optional[int], status: str) -> None:
    """Log preemption check result."""
    if region is None:
        log(f"\n>>> PREEMPTION: no active SPOT")
    else:
        log(f"\n>>> PREEMPTION: R{region} {status}")


def log_burst_header() -> None:
    """Log burst section header."""
    log("\n>>> BURST/JITTER:")
    log("# events=(tick, risk, failures), find suffix with risk<=1 maximizing fail/risk ratio, g=max(1,ratio)")


def log_burst_region(
    region: int,
    events: List[Dict[str, Any]],
    suffix_len: int,
    suffix_risk: float,
    suffix_fail: float,
    g_hat: float,
) -> None:
    """Log burst info for a region."""
    if not events:
        log(f"    R{region}: events=[] -> g=1.00")
        return

    # Format events (last 5)
    events_str = ",".join(f"(t{e.get('tick', '?')},r={e.get('risk', 0):.2f},f={e.get('failures', 0)})" for e in events[-5:])
    if len(events) > 5:
        events_str = f"...{events_str}"

    # Cap g_hat display to reasonable range
    g_display = min(g_hat, 99.99)

    if suffix_risk > 0.001 and suffix_fail > 0:
        ratio = suffix_fail / suffix_risk
        ratio_display = min(ratio, 99.99)
        log(f"    R{region}: events=[{events_str}] suffix(len={suffix_len},r={suffix_risk:.2f},f={suffix_fail:.0f}) -> ratio={ratio_display:.2f} -> g={g_display:.2f}")
    else:
        log(f"    R{region}: events=[{events_str}] suffix(len={suffix_len},r={suffix_risk:.2f},f={suffix_fail:.0f}) -> g={g_display:.2f}")


def log_duration_header() -> None:
    """Log duration section header."""
    log("\n>>> DURATION:")
    log("# knots=(failure_time, hazard_step), E=∫S(t)dt with S=exp(-g*H0)")


def log_duration_region(
    region: int,
    knots: List[tuple],
    tail: float,
    g_hat: float,
    age: float,
    e_dur: float,
    avail: float,
    calc_steps: str = "",
) -> None:
    """Log duration prediction for a region."""
    knots_str = ",".join(f"({t:.1f}h,Δ={h:.2f})" for t, h in knots[:3]) if knots else "[]"
    if len(knots) > 3:
        knots_str += ",..."

    age_str = f", age={age:.1f}h" if age > 0 else ""

    # Cap g_hat display to reasonable range
    g_display = min(g_hat, 99.99)

    if calc_steps:
        log(f"    R{region}: knots={knots_str} tail={tail:.2f}, g={g_display:.2f}{age_str}")
        log(f"        E = {calc_steps} = {e_dur:.1f}h, avail={avail*100:.0f}%")
    else:
        log(f"    R{region}: knots={knots_str} tail={tail:.2f}, g={g_display:.2f}{age_str} -> E={e_dur:.1f}h, avail={avail*100:.0f}%")


def log_time_value_header() -> None:
    """Log time value section header."""
    log("\n>>> TIME VALUE:")
    log("# V = c_ref × 1.2, c_ref = weighted avg spot cost by availability")


def log_time_value(
    od_weight: float,
    rate: float,
    intersect: Optional[float],
    c_ref: float,
    egress_per_h: float,
    V: float,
    region_contributions: str = "",
) -> None:
    """Log time value computation."""
    if intersect is not None and intersect > 100:
        log(f"    OD weight: rate={rate:.2f}, intersect={intersect:.0f}h>deadline -> od_w={od_weight:.2f}")
    else:
        log(f"    OD weight: rate={rate:.2f}, intersect={intersect:.1f}h -> od_w={od_weight:.2f}" if intersect else f"    OD weight: {od_weight:.2f}")

    if region_contributions:
        log(f"    c_ref: {region_contributions} = ${c_ref:.3f}/h + egress${egress_per_h:.3f}/h")
    else:
        log(f"    c_ref: ${c_ref:.3f}/h + egress${egress_per_h:.3f}/h")

    log(f"    V = ${c_ref + egress_per_h:.3f} × 1.2 = ${V:.2f}/h")


def log_candidates_header(V: float, overhead: float, from_region: Optional[int]) -> None:
    """Log candidates section header."""
    from_str = f"R{from_region}" if from_region is not None else "none"
    log(f"\n>>> CANDIDATES (V=${V:.2f}/h, overhead={overhead:.1f}h, from={from_str}):")
    log("# net/h = V × (1 - overhead/E) - spot_price - egress/E")


def log_candidate(
    region: int,
    e_dur: float,
    spot_price: float,
    egress: float,
    V: float,
    overhead: float,
    net: float,
    is_best: bool = False,
    skip_reason: str = "",
) -> None:
    """Log candidate evaluation."""
    if skip_reason:
        log(f"    R{region}: {skip_reason}")
        return

    eff_ratio = 1 - overhead / e_dur if e_dur > 0 else 0

    calc = f"{V:.2f}×{eff_ratio:.2f} - {spot_price:.2f} - {egress:.2f}/{e_dur:.1f}"
    suffix = " ← BEST" if is_best else ""

    if egress > 0:
        log(f"    R{region}: E={e_dur:.1f}h, spot=${spot_price:.2f}/h, egress=${egress:.2f}")
        log(f"        net = {calc} = ${net:.2f}/h{suffix}")
    else:
        log(f"    R{region}: E={e_dur:.1f}h, spot=${spot_price:.2f}/h, egress=$0 (same)")
        log(f"        net = {V:.2f}×{eff_ratio:.2f} - {spot_price:.2f} = ${net:.2f}/h{suffix}")


def log_launch_attempt(attempt: int, region: int, cluster_type: str, result: str) -> None:
    """Log launch attempt."""
    if attempt == 1:
        log("\n>>> LAUNCH ATTEMPTS:")
    log(f"    #{attempt} TRY R{region} {cluster_type} -> {result}")


def log_probes(launched: List[int], skipped: Dict[int, str]) -> None:
    """Log probe scheduling."""
    if not launched and not skipped:
        return
    log("\n>>> PROBES:")
    if launched:
        log(f"    Launched: {', '.join(f'R{r}' for r in launched)}")
    # Only show a few skipped reasons if needed
