"""
Experiment configuration modules.

Each config file defines:
- STRATEGIES: list of strategy names to test
- SCENARIOS: list of scenario dicts
- PARAMS: dict of default parameters (deadline_ratios, checkpoint_sizes, etc.)

Usage:
    python benchmark_multi_region_modular.py --config v_ablation
    python benchmark_multi_region_modular.py --config region_scaling
"""

import importlib
from pathlib import Path
from typing import Dict, List, Any

# Available configs (auto-discovered from this directory)
_CONFIG_DIR = Path(__file__).parent


def list_configs() -> List[str]:
    """List all available config names."""
    configs = []
    for f in _CONFIG_DIR.glob("*.py"):
        if f.name.startswith("_"):
            continue
        configs.append(f.stem)
    return sorted(configs)


def load_config(name: str) -> Dict[str, Any]:
    """Load a config module by name and return its settings."""
    try:
        module = importlib.import_module(f".{name}", package=__package__)
    except ModuleNotFoundError:
        available = ", ".join(list_configs())
        raise ValueError(f"Config '{name}' not found. Available: {available}")

    # Extract required attributes
    config = {
        "STRATEGIES": getattr(module, "STRATEGIES", []),
        "SCENARIOS": getattr(module, "SCENARIOS", []),
        "PARAMS": getattr(module, "PARAMS", {}),
        "DESCRIPTION": getattr(module, "DESCRIPTION", ""),
        "SINGLE_REGION_STRATEGIES": getattr(module, "SINGLE_REGION_STRATEGIES", []),
    }
    return config
