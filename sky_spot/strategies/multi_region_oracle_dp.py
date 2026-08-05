import typing
import math
import logging
import time
from dataclasses import dataclass

from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.multi_region_types import TryLaunch, Terminate, Action, LaunchResult
from sky_spot.utils import ClusterType, COST_K
from sky_spot import env as env_lib
from sky_spot.migration_model import get_transfer_time_hours, get_transfer_cost_usd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Mode:
    kind: str  # 'NONE', 'OD', 'SPOT'
    region: int  # -1 for NONE, otherwise region index


class MultiRegionOracleDPStrategy(MultiRegionStrategy):
    """
    Oracle upper-bound strategy using per-region dynamic programming.

    Assumptions/notes:
    - Uses full future knowledge of availability and (if present) per-tick spot prices.
    - Respects single-instance-at-a-time, launch billing, and restart/migration downtime.
    - Approximates downtime by rounding to whole ticks (ceil(hours/gap_seconds)); no partial-tick credit.
      This is slightly pessimistic but yields a valid and simple upper-bound schedule.
    - Objective includes compute cost and one-time S3 transfer $ when switching regions.
    """

    NAME = 'multi_region_oracle_dp'
    # Oracle of full future; bypass safety net like ILP/clairvoyant solvers
    IGNORE_SAFETY_NET = True

    def __init__(self, args):
        super().__init__(args)
        self._plan: list[Mode] = []
        # No overlapped handoff: keep executor simple, follow per-tick plan
        self._handoff_timers: dict[int, int] = {}
        # Cached fields (may be unused when not doing overlapped handoff)
        self._region_names: list[str] = []
        self._mig_ticks: list[list[int]] = []
        self._H_base: int = 0
        # Flag to track if plan has been generated
        self._plan_generated: bool = False

    def reset(self, env: 'env_lib.Env', task: 'typing.Any'):
        super().reset(env, task)
        # Reset the plan generation flag for new trace
        self._plan_generated = False
        # Clear the plan for new trace
        self._plan = []
    
    def _ensure_plan_generated(self, env: 'env_lib.Env'):
        """Generate plan if not already generated for this trace."""
        if not self._plan_generated:
            env_m = typing.cast('env_lib.MultiTraceEnv', env)
            self._generate_plan(env_m)
            self._plan_generated = True
    
    def _generate_plan(self, env_m: 'env_lib.MultiTraceEnv'):
        """Generate the optimal plan using optimized dynamic programming."""
        gap_s = env_m.gap_seconds
        T = int(math.floor(self.deadline / gap_s))
        K = int(math.ceil(self.task_duration / gap_s))
        
        logger.info(
            f"[OracleDP] _generate_plan(): gap_s={gap_s:.3f}s, deadline={self.deadline:.1f}s, T={T}, "
            f"task_duration={self.task_duration:.1f}s, K={K}"
        )
        
        R = env_m.num_regions
        region_names = [env_m.get_region_name(i) for i in range(R)]
        self._region_names = region_names
        
        # Build availability and price matrices
        avail = [[0] * T for _ in range(R)]
        spot_price_per_tick = [[0.0] * T for _ in range(R)]
        # Get per-region OD prices (they can differ across regions)
        od_prices = [float(env_m.envs[r].get_price()[ClusterType.ON_DEMAND]) for r in range(R)]
        
        t0 = time.perf_counter()
        for r in range(R):
            sub = env_m.envs[r]
            start = sub._start_index
            n = len(sub.trace)
            for t in range(T):
                idx = start + t
                if 0 <= idx < n:
                    avail[r][t] = 1 if (not sub.trace[idx]) else 0
                    price = sub.trace.get_price(idx)
                    if price is None:
                        spot_price_per_tick[r][t] = float(
                            sub._spot_price if sub._spot_price is not None else sub._base_price / COST_K
                        )
                    else:
                        spot_price_per_tick[r][t] = float(price)
                else:
                    avail[r][t] = 0
                    spot_price_per_tick[r][t] = float(
                        sub._spot_price if sub._spot_price is not None else sub._base_price / COST_K
                    )
        
        t1 = time.perf_counter()
        logger.info(f"[OracleDP] built matrices: elapsed={(t1 - t0):.3f}s")
        
        # Compute migration ticks
        base_restart_s = float(self.restart_overhead)
        H_base = int(math.ceil(base_restart_s / gap_s))
        self._H_base = H_base
        
        mig_ticks = [[0] * R for _ in range(R)]
        for i in range(R):
            for j in range(R):
                if i == j:
                    mig_ticks[i][j] = H_base
                else:
                    transfer_h = get_transfer_time_hours(
                        region_names[i], region_names[j], 
                        getattr(self.task, 'checkpoint_size_gb', 50.0)
                    )
                    mig_ticks[i][j] = int(math.ceil(
                        ((base_restart_s / 3600.0) + transfer_h) * 3600.0 / gap_s
                    ))
        self._mig_ticks = mig_ticks
        
        # Build modes
        modes = [Mode('NONE', -1)]  # NONE_START
        for r in range(R):
            modes.append(Mode('NONE', r))
        for r in range(R):
            modes.append(Mode('OD', r))
        for r in range(R):
            modes.append(Mode('SPOT', r))
        
        M = len(modes)
        
        logger.info(
            f"[OracleDP] modes ready: M={M} (1+3R: NONE_START + {R}xNONE + {R}xOD + {R}xSPOT)"
        )
        
        # PRE-COMPUTATION OPTIMIZATION
        # Pre-compute all tick costs for each mode at each time
        tick_costs = [[0.0] * T for _ in range(M)]
        # Per-region OD price per tick
        od_price_per_tick = [od_prices[r] * (gap_s / 3600.0) for r in range(R)]

        for m_idx, m in enumerate(modes):
            if m.kind == 'NONE':
                # All zeros, already initialized
                pass
            elif m.kind == 'OD':
                for t in range(T):
                    tick_costs[m_idx][t] = od_price_per_tick[m.region]
            else:  # SPOT
                for t in range(T):
                    tick_costs[m_idx][t] = spot_price_per_tick[m.region][t] * (gap_s / 3600.0)
        
        # Pre-compute feasibility for each mode at each time
        feasible_matrix = [[True] * T for _ in range(M)]
        for m_idx, m in enumerate(modes):
            if m.kind == 'SPOT':
                for t in range(T):
                    feasible_matrix[m_idx][t] = (avail[m.region][t] == 1)
        
        # Pre-compute switching costs (H values) between all mode pairs
        switch_H_matrix = [[0] * M for _ in range(M)]
        for i, prev in enumerate(modes):
            for j, nxt in enumerate(modes):
                # NONE_START (-1) -> active
                if prev.kind == 'NONE' and prev.region == -1:
                    if nxt.kind == 'NONE':
                        switch_H_matrix[i][j] = 0
                    else:
                        switch_H_matrix[i][j] = H_base
                # active -> NONE (same region)
                elif nxt.kind == 'NONE' and prev.kind in ('OD', 'SPOT'):
                    switch_H_matrix[i][j] = 0
                # NONE(i) -> active(j)
                elif prev.kind == 'NONE' and prev.region >= 0 and nxt.kind in ('OD', 'SPOT'):
                    if prev.region == nxt.region:
                        switch_H_matrix[i][j] = H_base
                    else:
                        switch_H_matrix[i][j] = mig_ticks[prev.region][nxt.region]
                # active(i) -> active(j)
                elif prev.kind in ('OD', 'SPOT') and nxt.kind in ('OD', 'SPOT'):
                    if prev.region == nxt.region:
                        switch_H_matrix[i][j] = H_base
                    else:
                        switch_H_matrix[i][j] = mig_ticks[prev.region][nxt.region]
                else:
                    switch_H_matrix[i][j] = 0
        
        # Pre-compute transfer costs between mode pairs
        # Charge transfer $ when the effective source region (where the latest
        # checkpoint resides) differs from the destination region at launch.
        # This applies to:
        #   - active(i) -> active(j), i != j
        #   - NONE(i)   -> active(j), i != j   (waiting anchored at region i)
        # We do NOT charge for NONE_START (-1) -> active(j) because the first
        # launch has no prior source.
        transfer_costs = [[0.0] * M for _ in range(M)]
        for i, prev in enumerate(modes):
            for j, nxt in enumerate(modes):
                try:
                    if nxt.kind in ('OD', 'SPOT'):
                        # From an active source region
                        if prev.kind in ('OD', 'SPOT') and prev.region != nxt.region:
                            src_name = region_names[prev.region]
                            dst_name = region_names[nxt.region]
                            ckpt_gb = float(getattr(self.task, 'checkpoint_size_gb', 50.0))
                            transfer_costs[i][j] = float(
                                get_transfer_cost_usd(src_name, dst_name, ckpt_gb)
                            )
                        # From a remembered NONE(i) source region (not NONE_START)
                        elif prev.kind == 'NONE' and prev.region >= 0 and prev.region != nxt.region:
                            src_name = region_names[prev.region]
                            dst_name = region_names[nxt.region]
                            ckpt_gb = float(getattr(self.task, 'checkpoint_size_gb', 50.0))
                            transfer_costs[i][j] = float(
                                get_transfer_cost_usd(src_name, dst_name, ckpt_gb)
                            )
                except Exception:
                    # Keep as 0.0 on parsing/lookup errors (e.g., synthetic region names in tests)
                    pass
        
        # Pre-compute valid transitions
        valid_transitions = [[False] * M for _ in range(M)]
        for i, prev in enumerate(modes):
            for j, nxt in enumerate(modes):
                # Same mode is always valid
                if i == j:
                    valid_transitions[i][j] = True
                    continue
                
                # Check transition validity
                # From active(i) to NONE(j): only allow j==i
                if nxt.kind == 'NONE' and nxt.region >= 0 and prev.kind in ('OD', 'SPOT') and nxt.region != prev.region:
                    continue
                # From NONE_START to NONE(r): disallow
                if prev.kind == 'NONE' and prev.region == -1 and nxt.kind == 'NONE' and nxt.region != -1:
                    continue
                # From NONE(i) to NONE(j!=i): disallow
                if prev.kind == 'NONE' and prev.region >= 0 and nxt.kind == 'NONE' and nxt.region != prev.region:
                    continue
                
                valid_transitions[i][j] = True
        
        t_precompute = time.perf_counter()
        logger.info(
            f"[OracleDP] pre-computation done: elapsed={(t_precompute - t1):.3f}s"
        )
        
        # Cost upper bound (use min OD price - cheapest guaranteed completion)
        min_od_price = min(od_prices)
        upper_bound_cost = min_od_price * (gap_s / 3600.0) * (H_base + K)
        logger.info(f"[OracleDP] Cost upper bound: ${upper_bound_cost:.2f} (min OD price: ${min_od_price:.2f}/h)")
        
        # ROLLING ARRAY OPTIMIZATION
        # Only keep current and next layer
        from math import inf
        
        # Initialize current layer
        cur = {}
        cur[(0, 0, 0)] = 0.0  # (mode_idx, h, k) -> cost
        
        # Store minimal backtracking info
        backtrack_info = []  # List of backpointers for each time step
        
        logger.info(
            f"[OracleDP] Starting DP with optimizations: T={T}, K={K}, M={M}"
        )
        t_dp0 = time.perf_counter()
        log_every = max(1, T // 20)
        
        for t in range(T):
            # Pre-allocate next layer
            nxt = {}
            states_in = len(cur)
            
            # Track backpointers for this time step
            backptr = {}
            
            for (m_idx, h, k), cost in cur.items():
                # Cost upper bound prune
                if cost > upper_bound_cost:
                    continue
                
                # Option 1: Stay in same mode
                if feasible_matrix[m_idx][t]:
                    new_h = max(0, h - 1)
                    m = modes[m_idx]
                    gain = 1 if (h == 0 and m.kind in ('OD', 'SPOT')) else 0
                    new_k = min(K, k + gain)
                    new_cost = cost + tick_costs[m_idx][t]
                    
                    # Reachability prune
                    if new_k + (T - (t + 1)) >= K:
                        key = (m_idx, new_h, new_k)
                        if key not in nxt or new_cost < nxt[key]:
                            nxt[key] = new_cost
                            backptr[key] = (m_idx, h, k)
                
                # Option 2: Switch to another mode
                for m2_idx in range(M):
                    if m2_idx == m_idx:
                        continue
                    
                    # Use pre-computed validity and feasibility
                    if not valid_transitions[m_idx][m2_idx]:
                        continue
                    if not feasible_matrix[m2_idx][t]:
                        continue
                    
                    # Special case: NONE_START not allowed after t=0
                    if t > 0 and m2_idx == 0:  # NONE_START
                        continue
                    
                    H = switch_H_matrix[m_idx][m2_idx]
                    new_h = max(0, H - 1)
                    # Timing semantics: if switch has zero overhead, the new
                    # instance makes progress in this very tick (decision at
                    # tick start, progress realized over [t, t+1]). If there is
                    # overhead (H > 0), no progress this tick.
                    gain_on_switch = 1 if (H == 0 and modes[m2_idx].kind in ('OD', 'SPOT')) else 0
                    new_k = min(K, k + gain_on_switch)
                    new_cost = cost + tick_costs[m2_idx][t] + transfer_costs[m_idx][m2_idx]
                    
                    # Pruning
                    if new_cost > upper_bound_cost:
                        continue
                    if new_k + (T - (t + 1)) < K:
                        continue
                    
                    key = (m2_idx, new_h, new_k)
                    if key not in nxt or new_cost < nxt[key]:
                        nxt[key] = new_cost
                        backptr[key] = (m_idx, h, k)
            
            states_out = len(nxt)
            if (t + 1) % log_every == 0 or t == 0 or t == T - 1:
                logger.info(
                    f"[OracleDP] DP t={t+1}/{T}: states_in={states_in}, states_out={states_out}"
                )
            
            # Save backtracking info
            backtrack_info.append(backptr)
            
            # Rolling array: current becomes next
            cur = nxt
        
        t_dp1 = time.perf_counter()
        logger.info(f"[OracleDP] DP completed in {t_dp1 - t_dp0:.3f}s")
        
        # Find best terminal state with k>=K
        best_cost = inf
        best_state = None
        for (m_idx, h, k), cost in cur.items():
            if k >= K and cost < best_cost:
                best_cost = cost
                best_state = (m_idx, h, k)
        
        if best_state is None:
            logger.warning("Oracle-DP could not find solution; falling back to ALL-OD")
            self._plan = [Mode('OD', 0) for _ in range(T)]
            return
        
        # Backtrack to recover plan
        plan_modes = [0] * T
        state = best_state
        for t_idx in range(T - 1, -1, -1):
            if state is None:
                break
            m_idx = state[0]
            plan_modes[t_idx] = m_idx
            if t_idx < len(backtrack_info):
                state = backtrack_info[t_idx].get(state)

        self._plan = [modes[m_idx] for m_idx in plan_modes]
        logger.info(f"[OracleDP] Found plan with cost≈${best_cost:.2f}")

    def _step_multi(self) -> typing.Generator[Action, typing.Optional[LaunchResult], None]:
        env = typing.cast('env_lib.MultiTraceEnv', self.env)
        
        # Ensure plan is generated (lazy generation on first step)
        self._ensure_plan_generated(env)
        
        t = env.tick
        # If done, clean up
        remaining_task = self.task_duration - sum(self.task_done_time)
        if remaining_task <= 1e-3:
            logger.info(f"[OracleDP] Tick {t}: task finished; terminating all active instances")
            for region in list(env.get_active_instances().keys()):
                yield Terminate(region=region)
            return

        # Guard out-of-range
        if t >= len(self._plan):
            # No plan beyond T: default to NONE/terminate
            active = env.get_active_instances()
            if active:
                logger.info(f"[OracleDP] Tick {t}: beyond plan horizon; terminating all active instances")
                for region in list(active.keys()):
                    yield Terminate(region=region)
            return

        desired = self._plan[t]
        active = env.get_active_instances()

        # No overlapped handoff timers when using pure per-tick planning

        if desired.kind == 'NONE':
            if active:
                logger.info(f"[OracleDP] Tick {t}: desired=NONE; terminating {list(active.keys())}")
                for region in list(active.keys()):
                    yield Terminate(region=region)
            return

        # Map desired Mode to (region, cluster_type)
        target_region = desired.region
        target_type = ClusterType.ON_DEMAND if desired.kind == 'OD' else ClusterType.SPOT

        # If already running in target region with target type, take no action
        if any((r == target_region and ct == target_type) for r, ct in active.items()):
            logger.debug(
                f"[OracleDP] Tick {t}: already running {target_type.name} in r{target_region}; no action"
            )
            return

        # Otherwise, launch target and close old instances within this tick (no overlapping handover)
        pre_launch_active = set(active.keys())
        logger.info(
            f"[OracleDP] Tick {t}: launching {target_type.name} in r{target_region}; pre_active={sorted(pre_launch_active)}"
        )
        result = yield TryLaunch(region=target_region, cluster_type=target_type)
        assert result is not None
        if result.success:
            # Terminate any previously active regions immediately (single instance policy)
            if pre_launch_active:
                for r in pre_launch_active:
                    # Always terminate the old instance, even if it's in the same region
                    # This handles both cross-region migration and same-region type switching
                    logger.info(f"[OracleDP] Tick {t}: terminating previous instance in r{r}")
                    yield Terminate(region=r)
            return
        # If SPOT launch failed (shouldn't happen for oracle plan), fall back to OD in same region
        if target_type == ClusterType.SPOT:
            logger.warning(
                f"[OracleDP] Tick {t}: SPOT launch failed in r{target_region}; trying OD fallback"
            )
            result2 = yield TryLaunch(region=target_region, cluster_type=ClusterType.ON_DEMAND)
            assert result2 is not None
            if result2.success:
                for r in pre_launch_active:
                    # Always terminate the old instance after OD fallback
                    logger.info(f"[OracleDP] Tick {t}: terminating previous instance in r{r} after OD fallback")
                    yield Terminate(region=r)
                return
        # Otherwise keep current
        logger.warning(
            f"[OracleDP] Tick {t}: launch failed for both desired and OD fallback; keeping current active={sorted(env.get_active_instances().keys())}"
        )
        return

    @classmethod
    def _from_args(cls, parser):
        # No specific flags for now; re-use base args
        return cls(parser.parse_args())
