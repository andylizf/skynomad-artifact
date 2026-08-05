# Multi-Region Progress Model (Single-Instance Design)

This document defines the semantics for multi-region progress under the **single-instance invariant**. It serves as the contract for implementations and tests in this repository.

## Overview

The simulator enforces a strict single-instance invariant:
- At most **one instance** (Spot or On-Demand) can be running at any time across all regions
- No parallel instances, no multi-instance strategies  
- Progress comes from the single running instance after deducting its overhead

Key concepts:
- Each instance carries its own overhead state based on its launch context
- When launching a new instance, it inherits progress from the previous instance and incurs restart + transfer overhead
- Migration (switching regions) requires terminating the old instance before or after launching the new one

## Terminology

- `tick`: Discrete time slot of length `gap_seconds`.
- `instance`: The single running instance at `(region, instance_type)` like `(us-east-1a, SPOT)`
- `Active(t)`: Whether an instance is running during tick `t` (not preempted, not terminated)
- `Launch(region, type, t)`: Instance is launched during tick `t` and becomes active for tick `[t, t+1)` (immediate billing)
- `O(t)`: Remaining overhead (seconds) for the instance at the start of tick `t`. Must be "consumed" before contributing compute
- `C(t)`: Effective compute time during tick `t`, defined as `C(t) = max(gap_seconds − O(t), 0)` if active, else 0
- `P(t)`: Cumulative task progress at the end of tick `t`
- `active_instances`: Dict mapping `(region, ClusterType)` → launch_tick for tick `[t, t+1)` (what runs during current tick)

## Rules

### 1) Single-Instance Invariant and Progress

- **At most one instance** can be active at any time (enforced by the environment)
- The single active instance computes `C(t)` each tick based on its current overhead
- Global progress is simply: `P(t) = P(t-1) + C(t)`
- No aggregation needed since only one instance exists

### 2) Instance Launch and Migration (launch at tick `t`)

When launching a new instance at tick `t`:

- The new instance inherits all previous progress: `P_new = P(t-1)`
- Assign overhead based on launch context:
  - **Cold start** (no previous instance): `O = restart_overhead_seconds`
  - **Same-region restart**: `O = restart_overhead_seconds` (no transfer)
  - **Cross-region migration**: `O = restart_overhead_seconds + transfer_seconds(src→dst, checkpoint_size)`
- **Billing starts immediately** at tick `t` (new semantics)

Notes:

- **Transfer Time Semantics (Important):**

  - **Same-region transfers (same-zone or cross-AZ):** Transfer time is **zero**.
    The baseline S3 download time within the same region is already included
    in `restart_overhead` to avoid double-counting. Both same-zone and cross-AZ
    have the same S3 speed (≈9.72 Gbps) and are treated equally.
  - **Cross-region/continent transfers:** Return the **additional** transfer time
    beyond the baseline. We compute the actual transfer time based on measured speeds:

    - Cross-region (same continent): ≈8.20 Gbps
    - Cross-continent: ≈3.59 Gbps

    Then subtract the baseline same-region time (9.72 Gbps) to get only the
    additional overhead: `additional_hours = max(0, actual_time - baseline_time)`.

  - **Rationale:** This design ensures backward compatibility while avoiding
    double-counting. The `restart_overhead` parameter represents the total time
    for a cold start including the baseline S3 download. The `transfer_time`
    represents only the additional network latency for cross-region transfers.

### 3) Instance Replacement Within a Tick

**New semantics**: Can launch a new instance and terminate the old one within the same tick:
- Order matters: Must `TryLaunch` first, then `Terminate`
- Cannot terminate the newly launched instance in the same tick (minimum 1-tick lifecycle)
- Billing for the tick reflects the **new instance** (what runs during `[t, t+1)`)
- The old instance's final progress is still counted before termination

**Probes** (`ProbeLaunch`): Special 1-minute billing for availability testing:
- Billed for 60 seconds instead of full `gap_seconds`
- Does not contribute progress or become leader
- Independent of the main instance (can probe while main instance runs)

### 4) Overhead Decay

- After each tick's compute is accounted, decay the instance's overhead:
  `O(t+1) = max(O(t) − gap_seconds, 0)`
- Overhead is tracked per region in `strategy._per_region_overhead` dict
- When an instance is preempted, its overhead is cleared

### 5) Terminate and Same-Region Relaunch

- Terminating an instance ends its active status
- Relaunching in the same region: inherits progress, applies only `restart_overhead` (no transfer)

## Examples (assume `gap_seconds = 600` and `restart_overhead = 720s`)

### Example A: Cold start

- Tick 0 (execute phase): Launch R0; billing starts immediately for tick 0
- Tick 1 (update phase): Start with `O_R0(1) = 720`
  - Compute `C_R0(1) = max(600 − 720, 0) = 0` → no progress this tick
  - End of tick (decay): `O_R0(2) = max(720 − 600, 0) = 120`
- Tick 2 (update phase): Start with `O_R0(2) = 120`
  - Compute `C_R0(2) = 600 − 120 = 480` → progress this tick is 480
  - End of tick (decay): `O_R0(3) = 0`
- Tick 3+: `O_R0 = 0`, so `C_R0 = 600` per tick

### Example B: Migration Between Regions

- Suppose R0 already cleared its overhead. At tick `t`, migrate to R1:
  - First: `TryLaunch(R1, SPOT)` - succeeds, R1 starts billing
  - Then: `Terminate(R0)` - R0 stops
- R1 initialization: Inherits progress; `O_R1 = 720 + transfer(R0 → R1)`
- Tick `t`: Bills for R1 (not R0), progress still counted from R0's work
- Tick `t+1`: R1 contributes `max(600−O_R1, 0)` progress

### Example C: Preemption Timing

- R0 running, gets preempted at tick 2
- Tick 2 execution order:
  1. Calculate tick 1→2 progress (normal overhead/progress rules apply)
  2. Handle preemption, remove R0
- The key is the ORDER - progress first, then preemption

### Example D: Same-Region Relaunch

- Terminate R0 and relaunch in the same region
- Initialization: inherits progress, transfer time is **zero** (baseline S3 already in restart_overhead)
- Total overhead is just `restart_overhead`

### Example E: Same-Tick Replacement

- At tick `t`: Instance running in R0 (SPOT)
- Strategy decides to switch to ON_DEMAND:
  - `TryLaunch(R0, ON_DEMAND)` - succeeds
  - `Terminate(R0, SPOT)` - terminates old SPOT
- Billing for tick `t`: ON_DEMAND (the instance running during `[t, t+1)`)
- Progress for tick `t`: Still uses work done before replacement

## Safety Net Mechanism

The multi-region environment includes a "safety net" to ensure deadline guarantees:

### When Safety Net Triggers

The safety net activates at the **boundary tick** - the first time when:
`needed_time >= remaining_time`, where:

- `needed_time = ceil((remaining_task + restart_overhead) / gap_seconds) * gap_seconds`
- `remaining_time = floor((deadline - elapsed) / gap_seconds) * gap_seconds`

### Safety Net Actions

When triggered, the safety net:

1. **Consolidates to single instance** (enforces single-instance invariant)
2. **Evaluates current state:**
   - If ON_DEMAND running → keep it, terminate others if any
   - If SPOT with overhead ≥ 1e-3 → switch to ON_DEMAND
   - If no instance → launch ON_DEMAND
   - If SPOT with no overhead → keep it (may re-evaluate if preempted)

3. **After latching:** Refuse all strategy actions, maintain status quo

### Key Design Principle

Always use the **current or most recent region** to avoid transfer overhead. Since data/checkpoint is already there, this ensures no additional transfer time that could cause deadline miss.

## Implementation Notes

- Track the single instance's overhead in `strategy._per_region_overhead` dict
- Enforce single-instance invariant: `len(active_instances) ≤ 1` with assertions
- Key structure:
  - `(region, cluster_type)` allows precise instance identification
  - `_launched_this_tick` set prevents same-tick termination of new instances
  - No automatic replacement - strategies must explicitly manage instances
- Every tick:
  1. `observe()`: Check invariant, finalize previous tick's costs
  2. `update_strategy_progress()`: Calculate progress, handle preemptions, check invariant
  3. `execute_multi_strategy()`: Strategy actions, enforce invariant at end
  4. Decay overhead for the active instance
- Tests should assert:
  - Cold start overhead is applied and clears across ticks
  - Migrations add `restart + transfer` overhead
  - Same-tick replacement bills for the new instance
  - Preemption at tick start doesn't lose previous tick's progress

### Billing vs Progress Timing

- **Immediate billing**: Launches at tick `t` are billed for tick `t`, not `t+1`
- **active_instances**: Dict keyed by `(region, ClusterType)` with launch_tick as value
  - Represents what runs during `[t, t+1)` after strategy actions
  - The key structure is what makes instance identification and termination precise
- Billing reads `active_instances` directly at observe time

**Execution order per tick**:
1. `observe()`: 
   - Checks single-instance invariant
   - Finalizes tick `t-1` costs using current active_instances
   - Clears `_launched_this_tick` for new tick
2. `update_strategy_progress()`: 
   - Calculates tick `t-1` progress using current active_instances (before preemption)
   - Then handles tick `t` preemptions (updates active_instances)
   - Checks single-instance invariant after preemptions
3. `execute_multi_strategy()`: 
   - Strategy sees post-preemption state, makes decisions
   - Enforces single-instance invariant at end
4. `tick++`

**Key insight**: Progress calculation happens BEFORE preemption handling, so work done in tick `t-1` is counted even if the instance is preempted at tick `t`.

### Call Order & Invariants (Strict Contract)

- Call order: `update_strategy_progress()` must be called after `observe()` for the same tick
- Single-instance invariant: At most one instance can be active at any time
  - Enforced via assertions at multiple points
  - Temporarily violated during strategy execution, enforced at completion
- New launches inherit progress from the previous instance
- Minimum lifecycle: Cannot terminate an instance in the same tick it was launched
  - Enforced by `_launched_this_tick` set check in `_terminate_internal`
