#!/usr/bin/env python3
"""
Reconstruct Figure 13 (model_size_vs_cost): egress cost per migration vs hourly
spot compute cost for open-weight models across three GPU generations.

The original plotting script was never committed to the repo. This reconstruction
follows the specification given verbatim in the paper's Appendix caption:

    "Both quantities derive from total parameter count: compute from memory-based
     GPU scaling (18 bytes/param for mixed-precision training with Adam) priced at
     current AWS spot rates, and egress from the full training checkpoint
     (10 bytes/param) at $0.02/GB inter-region."

Derivation, and how it was validated:

    egress(P)      = P * 10 bytes / 1e9 * $0.02/GB          = 0.2 * P_in_billions
    compute(P, g)  = (P * 18 bytes / VRAM_g) * (node_price_g / 8)

  where VRAM_g is per-GPU memory and node_price_g is the 8-GPU node hourly spot
  price. The slope of each diagonal is egress/compute, which is independent of P.

  Cross-check against the three ratios stated in the paper text:
      H100 (80 GB,  $17/hr node): 0.2 / (18/80 * 17/8 * 1e-9 * 1e9) = 41.8%  [paper: 41.8%]
      H200 (141 GB, $18/hr node): ...                               = 69.6%  [paper: 69.6%]
      B200 (179 GB, $31/hr node): ...                               = 51.3%  [paper: 51.3%]
  All three match, which is what pins down the "$17/hr is a node price, not a
  per-GPU price" reading of the caption.

  Model parameter counts are recovered from the egress labels printed in the
  original figure (egress = 0.2 * P_in_billions), e.g. Llama 3.1 405B -> $81.

Usage:
    python scripts/plot_model_size_vs_cost.py --output figs/model_size_vs_cost.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

# --- Specification from the paper ---------------------------------------------
BYTES_PER_PARAM_TRAINING = 18   # mixed-precision weights + Adam optimizer state
BYTES_PER_PARAM_CHECKPOINT = 10  # full training checkpoint
EGRESS_USD_PER_GB = 0.02         # inter-region transfer

# GPU generations: (label, per-GPU VRAM in GB, 8-GPU node spot price USD/hr)
GPUS = [
    ("H100", 80, 17),
    ("H200", 141, 18),
    ("B200", 179, 31),
]
GPUS_PER_NODE = 8

# Open-weight models, total parameter count in billions.
# Recovered from the egress labels in the paper figure (egress = 0.2 * P_B).
MODELS = [
    ("Qwen3-32B", 32),
    ("Llama 4 Scout", 110),
    ("Qwen3-235B", 235),
    ("Llama 3.1 405B", 405),
    ("DeepSeek-V3", 685),
    ("GLM-5", 745),
    ("Kimi K2.5", 1000),
]


def egress_cost(params_b: float) -> float:
    """Egress cost per migration (USD) for a model of P billion parameters."""
    ckpt_gb = params_b * 1e9 * BYTES_PER_PARAM_CHECKPOINT / 1e9
    return ckpt_gb * EGRESS_USD_PER_GB


def compute_cost_per_hour(params_b: float, vram_gb: float, node_price: float) -> float:
    """Ideal-packing hourly spot compute cost (USD) on a given GPU generation."""
    gpus_needed = params_b * 1e9 * BYTES_PER_PARAM_TRAINING / (vram_gb * 1e9)
    return gpus_needed * (node_price / GPUS_PER_NODE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/figures/model_size_vs_cost.pdf")
    ap.add_argument("--xmax", type=float, default=500.0)
    ap.add_argument("--ymax", type=float, default=250.0)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 5))

    # Diagonals: ideal-packing cost per GPU generation (slope = egress/compute)
    for label, vram, price in GPUS:
        # slope is P-independent; evaluate at any P
        slope = egress_cost(100) / compute_cost_per_hour(100, vram, price)
        xs = [0, args.xmax]
        ax.plot(xs, [0, slope * args.xmax], linewidth=1.8,
                label=f"{label} ideal ({vram} GB, ${price}/hr)")

    # Horizontal dashed line per model: egress is fixed regardless of GPU choice.
    # Only the first one carries a legend label, matching the paper's legend.
    for i, (name, params_b) in enumerate(MODELS):
        e = egress_cost(params_b)
        ax.axhline(e, color="gray", linestyle=":", linewidth=0.7, zorder=0,
                   label="Model egress per migration" if i == 0 else None)

    # y = x reference
    ax.plot([0, args.xmax], [0, args.xmax], "k--", linewidth=1.0, label="y=x")

    for name, params_b in MODELS:
        e = egress_cost(params_b)
        ax.annotate(f"{name} (${e:.0f})",
                    xy=(args.xmax * 0.62, e), xytext=(args.xmax * 0.62, e + 4),
                    fontsize=9)

    ax.set_xlabel("Hourly spot compute cost (USD)")
    ax.set_ylabel("Egress cost per migration (USD)")
    ax.set_xlim(0, args.xmax)
    ax.set_ylim(0, args.ymax)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved: {out}")

    # Report the ratios so they can be checked against the paper text
    print("\nSlope (egress/compute) per GPU generation:")
    for label, vram, price in GPUS:
        slope = egress_cost(100) / compute_cost_per_hour(100, vram, price)
        print(f"  {label:5s} {slope*100:5.1f}%")
    print("\nEgress per migration:")
    for name, params_b in MODELS:
        print(f"  {name:18s} {params_b:5.0f}B -> ${egress_cost(params_b):6.1f}")


if __name__ == "__main__":
    main()
