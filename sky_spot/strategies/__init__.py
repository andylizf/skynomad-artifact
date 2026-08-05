from sky_spot.strategies import strategy

# Import a minimal, widely used subset (keeps runtime light for worker environments)
from sky_spot.strategies import on_demand
from sky_spot.strategies import rc_cr_threshold
from sky_spot.strategies import multi_region_rc_cr_threshold
from sky_spot.strategies import multi_region_rc_cr_threshold_eager_failover
from sky_spot.strategies import unified_cost_model
from sky_spot.strategies import unified_cost_model_oracle
from sky_spot.strategies import unified_cost_model_v_variants
from sky_spot.strategies import unified_cost_model_f_ablation
from sky_spot.strategies import multi_region_oracle_dp
from sky_spot.strategies import unified_cost_model_risk_legacy

# Best-effort optional imports: ignore if unavailable (heavy/rare dependencies)
_OPTIONAL = [
    'rc_threshold',
    'multi_region_availability_probe_simple', 'multi_region_probe_cost_ratio'
]

for _m in _OPTIONAL:
    try:
        __import__(f'sky_spot.strategies.{_m}')
    except Exception:
        pass
