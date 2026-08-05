"""Structured logging system for strategy decision tracking."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RegionHistory:
    """Historical segments for a region."""
    segments: List[Tuple[float, float, str]] = field(default_factory=list)
    
    def add_segment(self, start: float, duration: float, event_type: str):
        """Add a segment to history."""
        self.segments.append((start, duration, event_type))
    
    def format(self, limit: int = 5) -> str:
        """Format history as compact string."""
        if not self.segments:
            return "[]"
        
        items = []
        for start, dur, evt in self.segments[-limit:]:
            evt_char = evt[0].upper()  # F for failure, C for censor, L for live
            items.append(f"{evt_char}({start:.1f},{dur:.1f})")
        
        if len(self.segments) > limit:
            items.append(f"...+{len(self.segments)-limit}")
        
        return f"[{' '.join(items)}]"


@dataclass
class JitterState:
    """Adaptive jitter summary for a region."""
    subject: str = "idle"
    g: float = 1.0
    g_raw: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    n_seg: float = 0.0
    f_seg: float = 0.0
    delta_lambda: float = 0.0
    failures: int = 0
    mode: str = "legacy"
    suffix_failures: float = 0.0
    suffix_risk: float = 0.0
    suffix_len: int = 0
    events: list = field(default_factory=list)


@dataclass
class DurationPrediction:
    """Duration prediction details."""
    region: int
    knots: List[Tuple[float, float]]  # (time, hazard_step) pairs
    tail_hazard: float
    g_multiplier: float
    expected_hours: float
    mode: str = "jitter"  # jitter, km, fallback etc
    gap_hours: float = 0.0  # actual scheduling tick length in hours (filled by caller)
    
    def format_knots(self, limit: int = 3) -> str:
        """Format knots compactly."""
        if not self.knots:
            return "[]"
        
        items = [f"{t:.1f}→{h:.2f}" for t, h in self.knots[:limit]]
        if len(self.knots) > limit:
            items.append(f"...+{len(self.knots)-limit}")
        return f"[{' '.join(items)}]"


@dataclass
class DecisionOption:
    """A decision option with value calculation."""
    region: Optional[int]
    cluster_type: str  # SPOT, ON_DEMAND, NONE
    value: float
    duration_hours: Optional[float] = None
    spot_price: Optional[float] = None
    effective_hours: Optional[float] = None
    penalty: Optional[float] = None
    is_current: bool = False
    status: str = "candidate"  # candidate, current, failed, blocked
    
    def format_calc(self, V: float) -> str:
        """Format value calculation."""
        parts = []
        
        if self.cluster_type == "ON_DEMAND":
            od_cost = self.spot_price or 0
            parts.append(f"V({V:.1f})-c_od({od_cost:.2f})")
            if self.penalty:
                # For OD, penalty is already in USD total (not per hour)
                parts.append(f"-mig({self.penalty:.1f})")
        elif self.cluster_type == "SPOT":
            if self.effective_hours is not None and self.duration_hours and self.duration_hours > 0:
                # For SPOT, distribute the division: val = V*eff/dur - c - pen/dur
                eff_per_hour = self.effective_hours / self.duration_hours
                # Show how effective rate is calculated: (dur-restart)/dur
                restart_hours = self.duration_hours - self.effective_hours
                if restart_hours > 0.01:  # Only show if restart is significant
                    parts.append(f"V({V:.1f})*(dur({self.duration_hours:.1f})-restart({restart_hours:.1f}))/dur({self.duration_hours:.1f})")
                else:
                    parts.append(f"V({V:.1f})*{eff_per_hour:.3f}")
                if self.spot_price is not None:
                    parts.append(f"-c({self.spot_price:.2f})")
                if self.penalty and not self.is_current:
                    # Show penalty calculation: migration_penalty/duration
                    parts.append(f"-mig({self.penalty:.1f})/dur({self.duration_hours:.1f})")
            elif self.penalty and not self.is_current:
                parts.append(f"-pen({self.penalty:.1f})")
        
        if not parts:
            return ""
        
        return f"[{' '.join(parts)}]"


class StructuredLogger:
    """Collects and formats structured logs for strategy decisions."""
    
    def __init__(self, log_level: str = "INFO"):
        self.tick: int = 0
        self.now_hours: float = 0.0
        self.events: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.region_histories: Dict[int, RegionHistory] = defaultdict(RegionHistory)
        self.jitter_states: Dict[int, JitterState] = defaultdict(JitterState)
        self.duration_predictions: Dict[int, DurationPrediction] = {}
        self.decision_options: List[DecisionOption] = []
        self.selected_action: Optional[str] = None
        self.V_value: Optional[float] = None
        self.log_level = getattr(logging, log_level.upper())
        
    def set_tick(self, tick: int, elapsed_seconds: float):
        """Set current tick context."""
        self.tick = tick
        self.now_hours = elapsed_seconds / 3600.0
        # Don't clear here - will be cleared after emit_all()
    
    def clear_tick_data(self):
        """Clear per-tick data."""
        self.events.clear()
        self.jitter_states.clear()
        self.duration_predictions.clear()
        self.decision_options.clear()
        self.selected_action = None
        self.V_value = None
    
    def add_event(self, region: int, event_type: str, **kwargs):
        """Record an event."""
        event = {"type": event_type, **kwargs}
        self.events[region].append(event)
    
    def add_history_segment(self, region: int, start: float, duration: float, event_type: str):
        """Add historical segment."""
        self.region_histories[region].add_segment(start, duration, event_type)

    def clear_history(self, region: int):
        """Reset stored history for a region to avoid duplicated segments."""
        self.region_histories[region] = RegionHistory()

    def set_jitter_state(
        self,
        region: int,
        *,
        subject: str,
        g: float,
        g_raw: float,
        alpha: float,
        beta: float,
        n_seg: float,
        f_seg: float,
        delta_lambda: float,
        failures: int,
        mode: str = "legacy",
        suffix_failures: float = 0.0,
        suffix_risk: float = 0.0,
        suffix_len: int = 0,
        events: list = None,
    ):
        """Set jitter state summary for a region."""
        self.jitter_states[region] = JitterState(
            subject=subject,
            g=g,
            g_raw=g_raw,
            alpha=alpha,
            beta=beta,
            n_seg=n_seg,
            f_seg=f_seg,
            delta_lambda=delta_lambda,
            failures=failures,
            mode=mode,
            suffix_failures=suffix_failures,
            suffix_risk=suffix_risk,
            suffix_len=suffix_len,
        )
        # Store events separately for detailed logging
        if events is not None:
            self.jitter_states[region].events = events
    
    def add_duration_prediction(self, region: int, knots: List[Tuple[float, float]],
                                tail: float, g: float, expected: float, mode: str = "ph",
                                gap_hours: float = 1.0):
        """Add duration prediction."""
        self.duration_predictions[region] = DurationPrediction(
            region=region, knots=knots, tail_hazard=tail,
            g_multiplier=g, expected_hours=expected, mode=mode,
            gap_hours=gap_hours
        )
    
    def add_decision_option(self, region: Optional[int], cluster_type: str,
                            value: float, **kwargs):
        """Add a decision option."""
        self.decision_options.append(DecisionOption(
            region=region, cluster_type=cluster_type, value=value, **kwargs
        ))
    
    def set_decision(self, action: str, V: float):
        """Set final decision."""
        self.selected_action = action
        self.V_value = V
    
    def emit_events_log(self):
        """Emit events log if there are events."""
        if not self.events:
            return
        
        parts = [f"[PLAN] step=events tick={self.tick}"]
        for region, region_events in sorted(self.events.items()):
            event_strs = []
            for evt in region_events:
                evt_type = evt.get("type", "unknown")
                if "duration_h" in evt:
                    event_strs.append(f"{evt_type}:{evt['duration_h']:.1f}h")
                else:
                    event_strs.append(evt_type)
            parts.append(f"  R{region}: {' '.join(event_strs)}")
        
        logger.log(self.log_level, "\n".join(parts))
    
    def emit_jitter_log(self):
        """Emit jitter log with clear format."""
        if not self.jitter_states:
            return

        lines = [f"[PLAN] step=jitter tick={self.tick} now_h={self.now_hours:.1f}"]

        for region in sorted(self.jitter_states.keys()):
            js = self.jitter_states[region]
            hist = self.region_histories[region]

            # Main summary line
            line = f"  R{region}: "
            line += f"state={js.subject} "
            line += f"g={js.g:.1f} dens={js.g_raw:.2f} "

            # Show window summary
            line += f"window[n={js.suffix_len}, risk={js.suffix_risk:.2f}, fails={js.suffix_failures:.0f}]"

            # Show current tick update if any
            if js.delta_lambda > 0 or js.failures > 0:
                updates = []
                if js.delta_lambda > 0:
                    updates.append(f"risk+{js.delta_lambda:.3f}")
                if js.failures > 0:
                    updates.append(f"fail+{js.failures}")
                line += f" tick[{', '.join(updates)}]"

            lines.append(line)

            # Show detailed event accumulation if available
            if hasattr(js, 'events') and js.events:
                event_details = []
                cumulative_risk = 0.0
                cumulative_fails = 0
                for event in js.events:
                    risk = event.get('risk', 0)
                    fails = event.get('failures', 0)
                    tick = event.get('tick', -1)
                    cumulative_risk += risk
                    cumulative_fails += fails
                    if risk > 0 or fails > 0:
                        event_str = f"t{tick}[r={risk:.3f},f={fails}]"
                        event_details.append(event_str)

                if event_details:
                    # Show first few events and summary
                    shown = event_details[:6]
                    if len(event_details) > 6:
                        shown.append(f"...+{len(event_details)-6}")
                    lines.append(f"       events: {' '.join(shown)} => total[r={cumulative_risk:.2f},f={cumulative_fails}]")

        logger.log(self.log_level, "\n".join(lines))

    def emit_duration_log(self):
        """Emit duration prediction log."""
        if not self.duration_predictions:
            return
        
        lines = [f"[PLAN] step=duration tick={self.tick}"]
        
        # Sort by expected duration (longest first)
        sorted_preds = sorted(self.duration_predictions.items(), 
                             key=lambda x: -x[1].expected_hours)
        
        for region, pred in sorted_preds:
            # First line: summary
            parts = [f"  R{region}:"]
            parts.append(f"E[T]={pred.expected_hours:.1f}h")
            
            state = "jitter" if pred.g_multiplier > 1 else "calm"
            parts.append(f"({state}, g={pred.g_multiplier:.1f})")
            
            # Show history segments (from structured logger history)
            if region in self.region_histories:
                hist = self.region_histories[region].format(limit=4)
                parts.append(f"hist={hist}")
            
            # Format tail hazard (use scientific notation for very small values)
            if pred.tail_hazard < 1e-6:
                parts.append(f"tail={pred.tail_hazard:.2e}")
            else:
                parts.append(f"tail={pred.tail_hazard:.3f}")
            lines.append(" ".join(parts))
            
            # Second line: calculation details (faithful: first K intervals + Σrest + tail = total)
            calc_parts = ["       calc:"]
            K = 3
            gap = pred.gap_hours if pred.gap_hours > 0 else 1.0
            if pred.knots:
                # Compute full contributions once for correctness
                prev_t = 0.0
                S = 1.0
                interval_contribs = []
                S_vals = []  # S at start of each interval
                for (t, step) in pred.knots:
                    interval_contribs.append(S * (t - prev_t))
                    S_vals.append((prev_t, S, t - prev_t))
                    S *= math.exp(-pred.g_multiplier * step)
                    prev_t = t
                if pred.tail_hazard > 0:
                    exp_term = math.exp(-pred.g_multiplier * pred.tail_hazard)
                    denom = 1.0 - exp_term
                    if denom > 1e-9:
                        tail_contrib = S * pred.gap_hours / denom
                    else:
                        effective_rate = max(pred.g_multiplier * pred.tail_hazard, 1e-9)
                        tail_contrib = S * pred.gap_hours / effective_rate
                else:
                    tail_contrib = 0.0
                total_intervals = sum(interval_contribs)
                total = total_intervals + tail_contrib
                # First K terms
                shown_terms = []
                for i, (t0, S0, dt) in enumerate(S_vals[:K]):
                    shown_terms.append(f"S({t0:.1f})={S0:.2f}×{dt:.1f}h")
                shown_sum = sum(interval_contribs[:K])
                rest_sum = max(0.0, total_intervals - shown_sum)
                if shown_terms:
                    calc_parts.append(" + ".join(shown_terms))
                if rest_sum > 1e-3:
                    calc_parts.append(f"+ Σrest={rest_sum:.1f}h")
                # Tail
                if pred.tail_hazard > 0:
                    # Show g with appropriate precision (integer if close to 1, decimal otherwise)
                    g_str = f"{pred.g_multiplier:.0f}" if abs(pred.g_multiplier - round(pred.g_multiplier)) < 0.01 else f"{pred.g_multiplier:.2f}"
                    calc_parts.append(
                        f"+ tail={S:.2f}×{gap:.2f}/(1-exp(-{g_str}×{pred.tail_hazard:.2f}))"
                    )
                # Equality to total
                calc_parts.append(f"= {total:.1f}h")
            else:
                # No failures: pure exponential tail
                g_str = f"{pred.g_multiplier:.0f}" if abs(pred.g_multiplier - round(pred.g_multiplier)) < 0.01 else f"{pred.g_multiplier:.2f}"
                if pred.tail_hazard < 1e-6:
                    calc_parts.append(f"{gap:.2f}/(1-exp(-{g_str}×{pred.tail_hazard:.2e})) = {pred.expected_hours:.1f}h")
                else:
                    calc_parts.append(f"{gap:.2f}/(1-exp(-{g_str}×{pred.tail_hazard:.3f})) = {pred.expected_hours:.1f}h")
            lines.append(" ".join(calc_parts))
        
        logger.log(self.log_level, "\n".join(lines))
    
    def emit_decision_log(self):
        """Emit decision log."""
        if not self.decision_options:
            return
        
        header = f"[PLAN] step=decision tick={self.tick}"
        if self.V_value:
            header += f" V={self.V_value:.1f}"
        if self.selected_action:
            header += f" selected={self.selected_action}"
        
        lines = [header]
        
        # Sort options by value
        sorted_options = sorted(self.decision_options, key=lambda x: -x.value)
        
        for opt in sorted_options:
            parts = []
            
            # Format type and region clearly
            if opt.cluster_type == "SPOT":
                parts.append(f"  SPOT@R{opt.region}:")
            elif opt.cluster_type == "ON_DEMAND":
                if opt.region is not None:
                    parts.append(f"  OD@R{opt.region}:")
                else:
                    parts.append(f"  OD:")
            else:
                parts.append(f"  {opt.cluster_type}:")
            
            parts.append(f"val={opt.value:.1f}")
            
            # Add duration if available
            if opt.duration_hours is not None:
                parts.append(f"dur={opt.duration_hours:.1f}h")
            
            # Add calculation
            if self.V_value:
                calc = opt.format_calc(self.V_value)
                if calc:
                    parts.append(f"calc={calc}")
            
            # Add status - make current more prominent
            if opt.is_current:
                parts.append("**CURRENT**")
            elif opt.status == "launch_success":
                parts.append("(launched)")
            elif opt.status != "candidate":
                parts.append(f"({opt.status})")
            
            lines.append(" ".join(parts))
        
        logger.log(self.log_level, "\n".join(lines))
    
    def emit_all(self):
        """Emit all logs for current tick in order."""
        self.emit_events_log()
        self.emit_jitter_log()
        self.emit_duration_log()
        self.emit_decision_log()
        # Clear data after emitting
        self.clear_tick_data()
