#!/usr/bin/env python3
"""AWS inter-region data transfer pricing, behind Figure 4b.

Queries the AWS Pricing API for real egress rates and checks them against the
model in sky_spot/migration_model.py. `--summary-only` reports from the shipped
table and needs no credentials, which is how artifact/reproduce_all.sh runs it.

Usage:
    python research/migration_costs.py --summary-only
    python research/migration_costs.py --validate-model     # needs AWS creds
    python research/migration_costs.py --export-csv pricing_data.csv
"""

import subprocess
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import pickle
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np


# Standard AWS regions (excludes local zones, wavelength zones, AND GOV REGIONS)
STANDARD_REGIONS = {
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "ca-central-1",
    "ca-west-1",
    "eu-central-1",
    "eu-central-2",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-north-1",
    "eu-south-1",
    "eu-south-2",
    "ap-east-1",
    "ap-east-2",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-southeast-3",
    "ap-southeast-4",
    "ap-southeast-5",
    "ap-southeast-6",
    "ap-southeast-7",
    "ap-south-1",
    "ap-south-2",
    "sa-east-1",
    "af-south-1",
    "me-south-1",
    "me-central-1",
    "il-central-1",
    "mx-central-1",
    # Excluded: "us-gov-east-1", "us-gov-west-1" - government regions have special pricing
}


###############

RESEARCH = 1  # For research presentation.
# RESEARCH = 0  # For paper pdf.

# font_size = {0: 10, 1: 8}[RESEARCH]
# font_size = 6.5  # Manual fix for 1-full-col figures.
font_size = 12  # Manual fix for 0.5-full-col figures.
TEXT_USETEX = bool(1 - RESEARCH)

# Get from LaTeX using "The column width is: \the\columnwidth"
vldb_col_width_pt = 240
icml_col_width_pt = 234.8775
sigmod20_col_width_pt = 241.14749
vldb21_col_width_pt = 241.14749
nsdi23_col_width_pt = 241.02039


def InitMatplotlib(font_size, title_size=9):
    print("use_tex", TEXT_USETEX, "\nfont_size", font_size, "\ntitle_size", title_size)
    # https://matplotlib.org/3.2.1/tutorials/introductory/customizing.html
    params = {
        "backend": "ps",
        "text.usetex": TEXT_USETEX,
        # 'font.family': 'serif',
        # 'font.serif': ['Times'],
        # 'font.family': 'sans-serif',
        # "font.sans-serif": [
        #     # 'Lato',
        #     # 'DejaVu Sans', 'Bitstream Vera Sans',
        #     # 'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva',
        #     # 'Lucid',
        #     # 'Arial',
        #     "Helvetica",
        #     "Avant Garde",
        #     "sans-serif",
        # ],
        # Make math fonts (e.g., tick labels) sans-serif.
        # https://stackoverflow.com/a/20709149/1165051
        "text.latex.preamble": "\n".join(
            [
                r"\usepackage{siunitx}",  # i need upright \micro symbols, but you need...
                r"\sisetup{detect-all}",  # ...this to force siunitx to actually use your fonts
                r"\usepackage{helvet}",  # set the normal font here
                r"\usepackage{sansmath}",  # load up the sansmath so that math -> helvet
                r"\sansmath",  # <- tricky! -- gotta actually tell tex to use!
            ]
        ),
        # axes.titlesize      : large   # fontsize of the axes title
        # 'axes.titlesize': font_size,
        "axes.titlesize": title_size,  # For plt.title().
        # 'axes.labelsize': 7,
        # 'legend.fontsize': 7,
        # 'font.size': 7,
        # 'xtick.labelsize': 7,
        # 'ytick.labelsize': 7,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "font.size": font_size,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "0.1",  # ~black border.
        "legend.shadow": False,
        "legend.frameon": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # http://phyletica.org/matplotlib-fonts/
        # Important for cam-ready (otherwise some fonts are not embedded):
        "pdf.fonttype": 42,
        "lines.linewidth": 1,
        # Styling.
        # 'grid.color': '#dedddd',
        # 'grid.linewidth': .5,
        # 'axes.grid.axis': 'y',
        "xtick.bottom": False,
        "xtick.top": False,
        "ytick.left": False,
        "ytick.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.axisbelow": True,
    }

    plt.style.use("seaborn-v0_8-colorblind")
    plt.rcParams.update(params)


def get_cache_path() -> Path:
    """Get the path for the cache file."""
    cache_dir = Path.home() / ".cache" / "aws_pricing"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "aws_transfer_pricing_cache.pkl"


def load_cached_data() -> Optional[Dict]:
    """Load cached pricing data if available and not expired."""
    cache_path = get_cache_path()
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)

        # Check if cache is expired (7 days)
        cache_time = datetime.fromtimestamp(cached['timestamp'])
        if datetime.now() - cache_time > timedelta(days=7):
            print("Cache is older than 7 days, will refresh...")
            return None

        print(f"Using cached data from {cache_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return cached['data']
    except Exception as e:
        print(f"Error loading cache: {e}", file=sys.stderr)
        return None


def save_to_cache(data: Dict) -> None:
    """Save pricing data to cache."""
    cache_path = get_cache_path()
    try:
        cached = {
            'timestamp': datetime.now().timestamp(),
            'data': data
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cached, f)
        print(f"Cached pricing data to {cache_path}")
    except Exception as e:
        print(f"Error saving cache: {e}", file=sys.stderr)


def get_aws_pricing_data(
    service_code: str, filters: List[str], max_items: Optional[int] = None, use_cache: bool = True
) -> Optional[Dict]:
    """
    Fetch ALL pricing data from AWS Pricing API using AWS CLI with pagination.

    IMPORTANT: AWS provides incomplete inter-region pricing data. Many region pairs
    (especially EU <-> AP) don't have direct pricing, possibly because traffic
    routes through other regions. This function now:
    1. Caches data locally (7 day expiry) to avoid repeated API calls
    2. Paginates through ALL available data (no limit unless specified)
    3. Automatically adds reverse direction pricing (A->B implies B->A)
    4. Filters out government regions which have special pricing

    Args:
        service_code: AWS service code (e.g., 'AWSDataTransfer', 'AmazonS3')
        filters: List of filter strings (e.g., 'Type=TERM_MATCH,Field=transferType,Value=InterRegion Outbound')
        max_items: Maximum total number of items to fetch (None = get all)
        use_cache: Whether to use cached data if available

    Returns:
        Combined parsed JSON data with all items or None if error
    """
    # Try to load from cache first
    if use_cache:
        cached_data = load_cached_data()
        if cached_data:
            return cached_data
    all_items = []
    next_token = None
    page_count = 0
    batch_size = 100  # Use smaller batches for more reliable pagination

    print(f"Fetching AWS pricing data (unlimited pagination)...")

    # If max_items is None, fetch everything
    limit = max_items if max_items is not None else float('inf')

    while len(all_items) < limit:
        page_count += 1
        cmd = [
            "aws",
            "pricing",
            "get-products",
            "--service-code",
            service_code,
            "--region",
            "us-east-1",
            "--max-items",
            str(batch_size),
        ]

        if next_token:
            cmd.extend(["--starting-token", next_token])

        for filter_str in filters:
            cmd.extend(["--filters", filter_str])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            # Add items from this page
            price_list = data.get("PriceList", [])
            all_items.extend(price_list)

            print(f"  Page {page_count}: fetched {len(price_list)} items (total: {len(all_items)})")

            # Check if there's more data
            next_token = data.get("NextToken")
            if not next_token or len(price_list) == 0:
                print(f"  Reached end of data at page {page_count}")
                break

        except subprocess.CalledProcessError as e:
            print(f"Error on page {page_count}: {e}", file=sys.stderr)
            if page_count == 1:
                return None  # Failed on first page, return None
            break  # Failed on later page, return what we have
        except json.JSONDecodeError as e:
            print(f"JSON decode error on page {page_count}: {e}", file=sys.stderr)
            if page_count == 1:
                return None
            break

    print(f"Fetched total of {len(all_items)} items across {page_count} pages")

    result = {"PriceList": all_items}

    # Save to cache for future use
    if use_cache and len(all_items) > 0:
        save_to_cache(result)

    return result


def parse_pricing_item(item_str: str) -> Optional[Dict]:
    """Parse a single pricing item and extract relevant information."""
    try:
        item = json.loads(item_str)
        product = item.get("product", {})
        attributes = product.get("attributes", {})

        # Extract key information
        info = {
            "from_region": attributes.get("fromRegionCode", ""),
            "to_region": attributes.get("toRegionCode", ""),
            "from_location": attributes.get("fromLocation", ""),
            "to_location": attributes.get("toLocation", ""),
            "transfer_type": attributes.get("transferType", ""),
            "usage_type": attributes.get("usagetype", ""),
        }

        # Extract pricing
        terms = item.get("terms", {}).get("OnDemand", {})
        for term_data in terms.values():
            price_dims = term_data.get("priceDimensions", {})
            for dim_data in price_dims.values():
                price_info = dim_data.get("pricePerUnit", {})
                if "USD" in price_info:
                    info["price_usd"] = float(price_info["USD"])
                    info["description"] = dim_data.get("description", "")
                    return info

        return None
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def analyze_connectivity(price_groups: Dict[float, List[Tuple[str, str, float]]]) -> None:
    """Analyze and report connectivity statistics."""
    all_regions = set()
    connections = {}

    # Collect all regions and connections
    for price_list in price_groups.values():
        for src, dst, price in price_list:
            all_regions.add(src)
            all_regions.add(dst)
            connections[(src, dst)] = price

    # Check connectivity between region groups
    us_regions = {r for r in all_regions if r.startswith('us-')}
    eu_regions = {r for r in all_regions if r.startswith('eu-')}
    ap_regions = {r for r in all_regions if r.startswith('ap-')}

    print("\n=== Region Connectivity Analysis ===")
    print(f"Total regions: {len(all_regions)}")
    print(f"US regions: {len(us_regions)}")
    print(f"EU regions: {len(eu_regions)}")
    print(f"AP regions: {len(ap_regions)}")
    print(f"Total connections: {len(connections)}")

    # Check inter-region connectivity
    eu_ap_connections = 0
    us_eu_connections = 0
    us_ap_connections = 0

    for (src, dst), _ in connections.items():
        if (src in eu_regions and dst in ap_regions) or (src in ap_regions and dst in eu_regions):
            eu_ap_connections += 1
        elif (src in us_regions and dst in eu_regions) or (src in eu_regions and dst in us_regions):
            us_eu_connections += 1
        elif (src in us_regions and dst in ap_regions) or (src in ap_regions and dst in us_regions):
            us_ap_connections += 1

    print(f"\nInter-region connections:")
    print(f"  US <-> EU: {us_eu_connections}")
    print(f"  US <-> AP: {us_ap_connections}")
    print(f"  EU <-> AP: {eu_ap_connections}")

    # Show missing connections
    print(f"\nPotential missing connections:")
    expected_connections = len(all_regions) * (len(all_regions) - 1)
    print(f"  Expected (if fully connected): {expected_connections}")
    print(f"  Actual: {len(connections)}")
    print(f"  Missing: {expected_connections - len(connections)}")

    # Show example missing EU-AP connections
    missing_examples = []
    for eu_reg in list(eu_regions)[:2]:  # Sample 2 EU regions
        for ap_reg in list(ap_regions)[:2]:  # Sample 2 AP regions
            if (eu_reg, ap_reg) not in connections and (ap_reg, eu_reg) not in connections:
                missing_examples.append(f"{eu_reg} <-> {ap_reg}")

    if missing_examples:
        print(f"\nExample missing EU-AP connections:")
        for example in missing_examples[:5]:
            print(f"  {example}")


# --- Offline pricing table (added for artifact evaluation) -------------------
# The derived region-pair egress table ships with the artifact so Figure 4b can
# be produced without AWS credentials. Pass --refresh-pricing to re-fetch from
# the AWS Pricing API instead (requires a configured AWS CLI).
SHIPPED_PRICING_CSV = (
    Path(__file__).resolve().parent.parent / "research" / "data" / "aws_egress_pricing.csv"
)


def load_shipped_pricing() -> Optional[Dict[float, List[Tuple[str, str, float]]]]:
    """Rebuild price_groups from the shipped CSV, or None if it is absent."""
    if not SHIPPED_PRICING_CSV.exists():
        return None
    import csv as _csv
    groups: Dict[float, List[Tuple[str, str, float]]] = {}
    with open(SHIPPED_PRICING_CSV, newline="") as f:
        for row in _csv.DictReader(f):
            try:
                price = float(row["price_usd_per_gb"])
            except (KeyError, ValueError):
                continue
            groups.setdefault(price, []).append(
                (row["from_region"], row["to_region"], price)
            )
    return groups or None


def analyze_standard_region_pricing(max_items: Optional[int] = None, use_cache: bool = True, refresh_pricing: bool = False) -> Dict[float, List[Tuple[str, str, float]]]:
    """
    Analyze pricing for standard AWS regions only.

    Args:
        max_items: Maximum number of items to fetch (None = get all)
        use_cache: Whether to use cached data if available

    Returns:
        Dictionary mapping price levels to lists of (from_region, to_region, price) tuples
    """
    print("Starting AWS Data Transfer pricing analysis...")

    if not refresh_pricing:
        shipped = load_shipped_pricing()
        if shipped is not None:
            print(
                f"Using shipped pricing table ({SHIPPED_PRICING_CSV.name}, "
                f"{sum(len(v) for v in shipped.values())} region pairs). "
                "Pass --refresh-pricing to fetch live AWS prices instead."
            )
            return shipped

    # Fetch data (unlimited by default)
    data = get_aws_pricing_data(
        "AWSDataTransfer",
        ["Type=TERM_MATCH,Field=transferType,Value=InterRegion Outbound"],
        max_items=max_items,
        use_cache=use_cache
    )

    if not data:
        return {}

    region_prices = {}
    processed = 0
    unique_pairs = set()  # Track unique region pairs

    for item_str in data.get("PriceList", []):
        parsed = parse_pricing_item(item_str)
        if not parsed:
            continue

        from_region = parsed["from_region"]
        to_region = parsed["to_region"]
        price = parsed.get("price_usd", 0)

        # Only include standard regions with valid pricing
        if (
            from_region in STANDARD_REGIONS
            and to_region in STANDARD_REGIONS
            and price > 0  # Changed to > 0 to exclude free transfers
        ):
            # Store the original direction
            key = f"{from_region} -> {to_region}"
            region_prices[key] = price
            unique_pairs.add((from_region, to_region))
            processed += 1

            # Also add reverse direction if not already present
            # AWS transfer pricing is typically symmetric
            reverse_key = f"{to_region} -> {from_region}"
            if reverse_key not in region_prices:
                region_prices[reverse_key] = price
                processed += 1

    print(f"Processed {len(unique_pairs)} unique region pairs")
    print(f"Total directional prices: {processed} (including reverse directions)")

    # Group by price level
    price_groups = {}
    for pair, price in region_prices.items():
        if price not in price_groups:
            price_groups[price] = []
        from_reg, to_reg = pair.split(" -> ")
        price_groups[price].append((from_reg, to_reg, price))

    return price_groups


def validate_migration_model(price_groups: Dict[float, List[Tuple]]) -> None:
    """
    Validate current migration_model.py assumptions against real AWS pricing.
    """
    print("\n=== Validating migration_model.py assumptions ===")

    # Flatten price data for lookup
    pricing_lookup = {}
    for price_list in price_groups.values():
        for from_reg, to_reg, price in price_list:
            pricing_lookup[(from_reg, to_reg)] = price

    # Test cases from migration_model.py
    test_cases = [
        ("us-east-1", "us-east-2", 0.01, "US East special rate"),
        ("us-east-1", "us-west-1", 0.02, "US standard rate"),
        ("ca-central-1", "eu-west-1", 0.02, "CA-Central to EU (should be standard)"),
        ("ca-west-1", "eu-west-1", 0.05, "CA-West to EU"),
        ("ca-west-1", "me-south-1", 0.14, "CA-West to Middle East"),
        ("ap-east-1", "ap-southeast-1", 0.08, "Asia inter-region"),
        ("me-south-1", "us-east-1", 0.085, "Middle East to US"),
        ("af-south-1", "eu-west-1", 0.147, "Africa to EU"),
        ("sa-east-1", "us-east-1", 0.138, "South America to US"),
    ]

    for from_reg, to_reg, expected_price, description in test_cases:
        # Try both directions
        actual_price = None
        direction = None

        if (from_reg, to_reg) in pricing_lookup:
            actual_price = pricing_lookup[(from_reg, to_reg)]
            direction = "->"
        elif (to_reg, from_reg) in pricing_lookup:
            actual_price = pricing_lookup[(to_reg, from_reg)]
            direction = "<-"

        if actual_price is not None:
            diff = abs(actual_price - expected_price)
            status = "✅ MATCH" if diff < 0.005 else "❌ MISMATCH"
            print(
                f"{description}: Expected ${expected_price:.3f}, Got ${actual_price:.3f} ({direction}) {status}"
            )
            if diff >= 0.005:
                print(f"  Difference: ${diff:.3f}")
        else:
            print(
                f"{description}: Expected ${expected_price:.3f}, NOT FOUND in AWS data ❌"
            )


def export_pricing_data(price_groups: Dict, output_file: str) -> None:
    """Export pricing data to CSV file."""
    import csv

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["from_region", "to_region", "price_usd_per_gb"])

        for price_list in price_groups.values():
            for from_reg, to_reg, price in price_list:
                writer.writerow([from_reg, to_reg, price])

    print(f"Exported pricing data to {output_file}")


def display_pricing_summary(price_groups: Dict) -> None:
    """Display a summary of pricing patterns."""
    print("\n=== AWS Inter-Region Data Transfer Pricing Summary ===")

    total_pairs = sum(len(pairs) for pairs in price_groups.values())
    print(f"Total standard region pairs: {total_pairs}")

    for price in sorted(price_groups.keys()):
        pairs = price_groups[price]
        if price == 0:
            continue  # Skip free transfers

        print(f"\n${price:.4f}/GB ({len(pairs)} pairs):")

        # Show key examples
        shown = 0
        for from_reg, to_reg, _ in sorted(pairs)[:15]:
            print(f"  {from_reg} -> {to_reg}")
            shown += 1

        if len(pairs) > 15:
            print(f"  ... and {len(pairs) - 15} more")


def _flatten_price_groups(
    price_groups: Dict[float, List[Tuple[str, str, float]]]
) -> Dict[Tuple[str, str], float]:
    """Flatten grouped pricing data into a simple lookup.

    Returns:
        Dictionary mapping (from_region, to_region) -> price_usd_per_gb
    """
    pricing_lookup: Dict[Tuple[str, str], float] = {}
    for price_list in price_groups.values():
        for from_reg, to_reg, price in price_list:
            pricing_lookup[(from_reg, to_reg)] = price
    return pricing_lookup


def build_chord_matrix(
    price_groups: Dict[float, List[Tuple[str, str, float]]],
    selected_regions: Optional[Iterable[str]] = None,
    min_price: float = 0.0,
) -> Tuple[List[str], List[List[float]]]:
    """Build a symmetric adjacency matrix for chord diagram visualization.

    Args:
        price_groups: Output from analyze_standard_region_pricing().
        selected_regions: Optional iterable of region codes to include.
            If None, all regions present in the pricing data are used.
        min_price: Minimum $/GB value to visualize. Pairs below this
            threshold are treated as zero.

    Returns:
        (regions, matrix) where:
            regions: Ordered list of region codes
            matrix: NxN symmetric matrix of prices (float)
    """
    pricing_lookup = _flatten_price_groups(price_groups)

    # Determine region set
    region_set = set()
    for (from_reg, to_reg), _ in pricing_lookup.items():
        region_set.add(from_reg)
        region_set.add(to_reg)

    if selected_regions is not None:
        # Only keep regions explicitly requested (and present in data)
        requested = {r for r in selected_regions if r in region_set}
        if not requested:
            print(
                "Warning: No selected regions found in pricing data; "
                "falling back to all regions.",
                file=sys.stderr,
            )
        else:
            region_set = requested

    regions = sorted(region_set)

    if not regions:
        return [], []

    n = len(regions)
    matrix: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]

    # Build symmetric matrix using the maximum price between each pair.
    # AWS pricing is often one-directional, so we need to check both directions.
    # IMPORTANT: Treat each unordered pair {src, dst} exactly once to avoid
    # order-dependent overrides when prices differ by direction.
    for i, src in enumerate(regions):
        for j in range(i + 1, n):
            dst = regions[j]

            # Check both directions
            price_ab = pricing_lookup.get((src, dst))
            price_ba = pricing_lookup.get((dst, src))

            # Use the maximum price among the available directions
            candidates: List[float] = []
            if price_ab is not None:
                candidates.append(price_ab)
            if price_ba is not None:
                candidates.append(price_ba)

            if not candidates:
                continue

            price = max(candidates)

            # Apply minimum-price threshold at the pair level
            if price >= min_price:
                matrix[i][j] = price
                matrix[j][i] = price  # Enforce symmetry explicitly

    return regions, matrix


def _plot_chord_matplotlib_static(
    df, regions, output_path: Path
) -> None:
    """Render a static chord diagram using plain matplotlib.

    This avoids HoloViews' matplotlib backend and any selenium/Bokeh
    requirements, and draws exactly one curve per (source, target) pair.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path
        import numpy as np
    except ImportError:
        print(
            "Error: matplotlib is not installed.\n"
            "Install with: uv pip add matplotlib",
            file=sys.stderr,
        )
        return

    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw={"aspect": "equal"},
    )
    ax.axis("off")

    # Place regions evenly on a circle; each region occupies an arc segment.
    n = len(regions)
    if n == 0:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Chord diagram saved to {output_path} (no regions)")
        return

    radius = 1.0
    # Center angle for each region
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    region_to_index = {region: idx for idx, region in enumerate(regions)}

    # Region labels (at center of each arc)
    for region, angle in zip(regions, base_angles):
        label_radius = radius * 1.08
        lx = label_radius * np.cos(angle)
        ly = label_radius * np.sin(angle)
        ha = "left" if lx >= 0 else "right"
        ax.text(
            lx,
            ly,
            region,
            ha=ha,
            va="center",
            fontsize=7,
        )

    # Outer circle for context
    outer = plt.Circle(
        (0.0, 0.0),
        radius,
        edgecolor="lightgray",
        facecolor="none",
        linewidth=0.5,
        zorder=1,
    )
    ax.add_patch(outer)

    if df.empty:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Chord diagram saved to {output_path} (no edges)")
        return

    # For each region, collect incident edges to spread attachment points
    # along that region's arc instead of a single point.
    adjacency: Dict[str, List[int]] = {r: [] for r in regions}
    for edge_idx, row in df.reset_index().iterrows():
        src = row["source"]
        dst = row["target"]
        if src in adjacency:
            adjacency[src].append(edge_idx)
        if dst in adjacency:
            adjacency[dst].append(edge_idx)

    # Compute per-endpoint attachment angle on the circle
    # (edge_idx, region) -> angle
    endpoint_angle: Dict[Tuple[int, str], float] = {}

    # Fraction of the gap between neighboring region centers that is used
    # as the region's own arc (rest is blank gap).
    gap = 2 * np.pi / n
    arc_fraction = 0.7
    arc_span = gap * arc_fraction

    for region in regions:
        edges_for_region = adjacency.get(region, [])
        deg = len(edges_for_region)
        idx = region_to_index[region]
        center_angle = base_angles[idx]

        if deg == 0:
            continue
        if deg == 1:
            endpoint_angle[(edges_for_region[0], region)] = center_angle
            continue

        # Evenly spread edges across the arc_span around center_angle
        start = center_angle - arc_span / 2.0
        for pos, edge_idx in enumerate(sorted(edges_for_region)):
            t = (pos + 0.5) / deg  # in (0,1)
            angle = start + t * arc_span
            endpoint_angle[(edge_idx, region)] = angle

    # Color and width mapping based on value
    cmap = cm.get_cmap("RdYlGn_r")
    vmin = float(df["value"].min())
    vmax = float(df["value"].max())
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Thicker lines for more expensive (darker) edges
    min_lw = 0.3
    max_lw = 3.0

    # Draw one quadratic Bezier curve per edge
    for edge_idx, row in df.reset_index().iterrows():
        src = row["source"]
        dst = row["target"]
        value = float(row["value"])
        if src == dst:
            continue

        # Attachment points for each endpoint
        angle_src = endpoint_angle.get((edge_idx, src))
        angle_dst = endpoint_angle.get((edge_idx, dst))
        if angle_src is None or angle_dst is None:
            continue

        x0 = radius * np.cos(angle_src)
        y0 = radius * np.sin(angle_src)
        x1 = radius * np.cos(angle_dst)
        y1 = radius * np.sin(angle_dst)

        color = cmap(norm(value))
        t = norm(value)
        width = min_lw + (max_lw - min_lw) * t

        # Use the origin as a simple control point so chords bow inward.
        verts = [(x0, y0), (0.0, 0.0), (x1, y1)]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(verts, codes)
        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=width,
            alpha=0.4,
            zorder=2,
        )
        ax.add_patch(patch)

    # Fix axis limits so the full circle is visible and centered
    margin = 1.2 * radius
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Transfer Cost ($/GB)")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Chord diagram saved to {output_path} "
        f"(static {output_path.suffix.upper()} via matplotlib)"
    )


def plot_migration_cost_lollipop(
    price_groups: Dict[float, List[Tuple[str, str, float]]],
    selected_regions: Optional[Iterable[str]] = None,
    output_file: str = "migration_costs_lollipop.pdf",
    figsize: Tuple[float, float] = (10, 5),
) -> None:
    """Generate a lollipop chart of inter-region S3 egress costs by source region.

    Groups regions by major area (US, CA, EU, AP, etc.) on x-axis,
    with different colors for each specific region within the group.

    Args:
        price_groups: Output from analyze_standard_region_pricing()
        selected_regions: Optional list of regions to include (default: all)
        output_file: Output file path (.pdf/.png/.svg)
        figsize: Figure size in inches
    """

    InitMatplotlib(font_size=12, title_size=14)

    # Flatten pricing data and aggregate by source region
    # For each source region, compute average transfer cost
    pricing_lookup = _flatten_price_groups(price_groups)

    # Filter to selected regions if specified
    selected_set = set(selected_regions) if selected_regions else None

    # Group prices by source region
    src_prices: Dict[str, List[float]] = {}
    for (from_reg, to_reg), price in pricing_lookup.items():
        # Only include selected source regions
        if selected_set and from_reg not in selected_set:
            continue
        if from_reg not in src_prices:
            src_prices[from_reg] = []
        src_prices[from_reg].append(price)

    if not src_prices:
        print("No pricing data available for lollipop chart.", file=sys.stderr)
        return

    # Compute average price per source region
    src_avg_prices: Dict[str, float] = {
        src: float(np.mean(prices)) for src, prices in src_prices.items()
    }

    # Define major region mapping
    def get_major_region(region: str) -> str:
        prefixes = [
            ('us-', 'US'),
            ('ca-', 'CA'),
            ('eu-', 'EU'),
            ('ap-', 'AP'),
            ('sa-', 'SA'),
            ('af-', 'AF'),
            ('me-', 'ME'),
            ('il-', 'IL'),
            ('mx-', 'MX'),
        ]
        for prefix, label in prefixes:
            if region.startswith(prefix):
                return label
        return 'Other'

    # Group regions by major area
    major_groups: Dict[str, List[Tuple[str, float]]] = {}
    for src, avg_price in src_avg_prices.items():
        major = get_major_region(src)
        if major not in major_groups:
            major_groups[major] = []
        major_groups[major].append((src, avg_price))

    # Sort regions within each group by price
    for major in major_groups:
        major_groups[major].sort(key=lambda x: x[1])

    # Sort major regions by their average cost
    major_avg = {}
    for major, regions_list in major_groups.items():
        major_avg[major] = float(np.mean([price for _, price in regions_list]))
    major_order = sorted(major_groups.keys(), key=lambda m: major_avg[m])

    # Define base colors for each major region (one color per major region)
    base_colors = {
        'US': '#bcbd22',   # Yellow-green
        'CA': '#1f77b4',   # Blue
        'EU': '#2ca02c',   # Green
        'AP': '#9467bd',   # Purple
        'SA': '#d62728',   # Red
        'AF': '#8c564b',   # Brown
        'ME': '#e377c2',   # Pink
        'IL': '#7f7f7f',   # Gray
        'MX': '#ff7f0e',   # Orange
        'Other': '#17becf', # Cyan
    }

    def get_gradient_color(base_hex: str, intensity: float) -> tuple:
        """Generate gradient color. intensity: 0=light, 1=dark."""
        import matplotlib.colors as mcolors
        rgb = mcolors.hex2color(base_hex)
        # Blend with white for lighter shades
        light_factor = 0.3 + 0.7 * intensity  # Range from 0.3 (light) to 1.0 (full color)
        return tuple(1 - light_factor * (1 - c) for c in rgb)

    # # Create figure - adjust width based on number of regions
    # total_regions = sum(len(regions_in_group) for regions_in_group in major_groups.values())
    # fig_width = max(4, total_regions * 0.15 + len(major_groups) * 0.25)
    # fig, ax = plt.subplots(figsize=(fig_width, 5))
    fig, ax = plt.subplots(figsize=(3.2, 4.0))

    # Plot lollipops
    x_positions = []
    x_labels = []
    current_x = 0

    # Minimum gap between major regions (last zone of A to first zone of B)
    # Base gap = 1 (last zone increment) + 1.2 (group spacing) = 2.2
    # If min_group_gap > 2.2, we add extra padding
    min_group_gap = 3.0
    group_gap = 1.0
    base_gap = 1 + group_gap

    for idx, major in enumerate(major_order):
        regions_in_group = major_groups[major]
        group_start = current_x
        n_regions = len(regions_in_group)
        base_color = base_colors.get(major, '#333333')

        for i, (src, avg_price) in enumerate(regions_in_group):
            # Calculate intensity for gradient (0=lightest, 1=darkest)
            intensity = (i + 0.5) / max(n_regions - 0.5, 2) if n_regions > 1 else 0.5
            color = get_gradient_color(base_color, intensity)

            # Draw the stem (line)
            ax.plot([current_x, current_x], [0, avg_price], color=color, linewidth=2, zorder=1)

            # Draw the lollipop head (circle)
            ax.scatter(current_x, avg_price, color=color, s=75, zorder=2, edgecolors='white', linewidths=1.5)

            x_positions.append(current_x)
            current_x += 1

        # Add major region label at center of group
        group_center = (group_start + current_x - 1) / 2
        x_labels.append((group_center, major))

        # Add spacing between major groups
        # Only add extra gap when both adjacent groups are small (few zones)
        if idx < len(major_order) - 1:
            n_next = len(major_groups[major_order[idx + 1]])
            # Only enforce minimum gap if both groups have <= 2 zones
            if n_regions <= 2 and n_next <= 2:
                extra_gap = max(0, min_group_gap - base_gap)
            else:
                extra_gap = 0
            current_x += group_gap + extra_gap
        else:
            current_x += group_gap

    # Set x-axis labels (major regions only)
    ax.set_xticks([pos for pos, _ in x_labels])
    ax.set_xticklabels([label for _, label in x_labels], fontsize=12)

    # Add grid and styling
    ax.set_ylabel('Avg Transfer Cost ($/GB)', fontsize=14, labelpad=3)
    ax.set_xlabel('Source Region', fontsize=14)

    ax.set_ylim(0, None)
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # No legend
    fig.subplots_adjust(top=0.98, right=0.99, left=0.225, bottom=0.125)

    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in ['.pdf', '.svg']:
        fig.savefig(output_path, dpi=300, format=output_path.suffix[1:])
    else:
        fig.savefig(output_path, dpi=200)
        fig.savefig(output_path.with_suffix('.pdf'), dpi=300)

    plt.close(fig)
    print(f"Lollipop chart saved to {output_path}")


def plot_migration_cost_heatmap(
    price_groups: Dict[float, List[Tuple[str, str, float]]],
    selected_regions: Optional[Iterable[str]] = None,
    output_file: str = "migration_costs_heatmap.pdf",
    figsize: Tuple[float, float] = (12, 10),
    annotate: bool = True,
    cmap: str = 'RdYlGn_r',  # Red=expensive, Green=cheap
    min_connections: int = 5,  # Minimum connections required for a region
) -> None:
    """Generate a heatmap of inter-region S3 egress costs.

    Perfect for academic papers - shows all costs in a clear matrix format.

    Args:
        price_groups: Output from analyze_standard_region_pricing()
        selected_regions: Optional list of regions to include
        output_file: Output file path (.pdf/.png/.svg)
        figsize: Figure size in inches
        annotate: Whether to show values in cells
        cmap: Colormap to use
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
    except ImportError:
        print(
            "Error: matplotlib/seaborn not installed.\n"
            "Install with: uv add matplotlib seaborn",
            file=sys.stderr,
        )
        return

    # Get the pricing matrix
    regions, matrix = build_chord_matrix(
        price_groups, selected_regions=selected_regions, min_price=0.0
    )

    if not regions:
        print("No regions available for heatmap; skipping.", file=sys.stderr)
        return

    # Filter out regions with too few connections
    if min_connections > 0:
        # Count non-zero connections for each region
        connection_counts = {}
        for i, region in enumerate(regions):
            count = sum(1 for j in range(len(regions)) if i != j and matrix[i][j] > 0)
            connection_counts[region] = count

        # Keep only well-connected regions
        well_connected = [r for r in regions if connection_counts[r] >= min_connections]

        if len(well_connected) < len(regions):
            print(f"Filtering from {len(regions)} to {len(well_connected)} regions (min connections: {min_connections})")

            # Rebuild matrix with only well-connected regions
            regions_filtered, matrix_filtered = build_chord_matrix(
                price_groups, selected_regions=well_connected, min_price=0.0
            )
            regions = regions_filtered
            matrix = matrix_filtered

    # Convert to numpy array for better handling
    matrix_np = np.array(matrix)

    # Create mask for missing data (but keep diagonal as 0)
    mask = np.zeros_like(matrix_np, dtype=bool)
    for i in range(len(regions)):
        for j in range(len(regions)):
            if i != j and matrix_np[i][j] == 0:
                mask[i][j] = True  # Mark missing data

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        matrix_np,
        mask=mask,  # Mask missing data
        annot=annotate,
        fmt='.3f',
        cmap=cmap,
        xticklabels=regions,
        yticklabels=regions,
        square=True,
        cbar_kws={'label': 'Transfer Cost ($/GB)'},
        linewidths=0.5,
        linecolor='gray',
        vmin=0.0,
        vmax=0.15,  # Max typical cost
        ax=ax,
        annot_kws={'size': 8 if len(regions) > 10 else 10}
    )

    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Set title and labels
    ax.set_title('AWS Inter-Region S3 Data Transfer Costs ($/GB)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Destination Region', fontsize=12)
    ax.set_ylabel('Source Region', fontsize=12)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = Path(output_file)
    extension = output_path.suffix.lower()

    if extension in ['.pdf', '.svg']:
        # Vector format for papers
        fig.savefig(output_path, dpi=300, bbox_inches='tight', format=extension[1:])
    else:
        # Raster format
        fig.savefig(output_path, dpi=200, bbox_inches='tight')

    plt.close(fig)
    print(f"Heatmap saved to {output_path}")


def plot_migration_cost_chord(
    price_groups: Dict[float, List[Tuple[str, str, float]]],
    selected_regions: Optional[Iterable[str]] = None,
    min_price: float = 0.0,  # Default: show all connections
    output_file: str = "migration_costs_chord.png",  # Default: PNG for papers
    top_n: Optional[int] = None,
) -> None:
    """Generate a chord diagram of inter-region S3 egress costs.

    This uses a symmetric matrix where each pair's value is the maximum
    $/GB price observed between the two directions. This highlights
    regions connected by particularly expensive migration paths.

    Line thickness and color represent the transfer cost:
    - Thicker lines = more expensive transfers
    - Red = expensive, Yellow = medium, Green = cheap

    Args:
        price_groups: Output from analyze_standard_region_pricing()
        selected_regions: Optional list of regions to include
        min_price: Minimum price threshold to display (default: 0.0 = show all)
        output_file: Output file path (default: migration_costs_chord.png)
        top_n: If specified, only show the N most expensive connections

    Output format is determined by file extension:
    - .html: Interactive Bokeh visualization (default)
    - .png: Static PNG image
    - .svg: Vector SVG image
    - .pdf: PDF for academic papers
    """
    try:
        import holoviews as hv
        import pandas as pd
        hv.extension('bokeh')
    except ImportError:
        print(
            "Error: holoviews is not installed.\n"
            "Install it in this project with:\n"
            "  uv add holoviews bokeh",
            file=sys.stderr,
        )
        return

    regions, matrix = build_chord_matrix(
        price_groups, selected_regions=selected_regions, min_price=min_price
    )

    if not regions:
        print("No regions available for chord diagram; skipping.", file=sys.stderr)
        return

    total_weight = sum(sum(row) for row in matrix)
    if total_weight == 0:
        print(
            "All migration costs are below the specified threshold; "
            "no chords to draw.",
            file=sys.stderr,
        )
        return

    # Convert matrix to edge list format for HoloViews
    edges = []
    max_price = 0
    min_price_found = float('inf')

    for i, src in enumerate(regions):
        for j, dst in enumerate(regions):
            if i < j and matrix[i][j] > 0:  # Only process upper triangle
                edges.append({
                    'source': src,
                    'target': dst,
                    'value': matrix[i][j]
                })
                max_price = max(max_price, matrix[i][j])
                min_price_found = min(min_price_found, matrix[i][j])

    if not edges:
        print("No edges to display after filtering.", file=sys.stderr)
        return

    # Create DataFrame from edges
    df = pd.DataFrame(edges)

    # Normalize values for better visualization
    # Use VERY subtle line widths to avoid visual confusion
    if max_price > min_price_found:
        # Use square root scaling for more balanced visualization
        import numpy as np
        normalized_values = (df['value'] - min_price_found) / (max_price - min_price_found)
        # Map to 0.3-1.0 range (very subtle differences)
        df['normalized_width'] = 0.3 + np.sqrt(normalized_values) * 0.7
    else:
        df['normalized_width'] = 0.5  # Default width if all prices are the same

    # Sort by value to draw expensive connections on top
    df = df.sort_values('value', ascending=True)

    # If top_n is specified, keep only the N most expensive connections
    if top_n is not None and len(df) > top_n:
        df = df.nlargest(top_n, 'value')
        print(f"Showing top {top_n} most expensive connections (out of {len(edges)} total)")

    # Create the chord diagram
    chord = hv.Chord(df)

    # Style the chord diagram with very thin lines to avoid visual confusion
    chord.opts(
        width=800,
        height=800,
        title="AWS Inter-Region S3 Data Transfer Costs ($/GB)",
        labels='index',
        cmap='RdYlGn_r',  # Red = expensive, Green = cheap
        edge_color='value',
        edge_cmap='RdYlGn_r',  # Color edges by cost
        edge_line_width=hv.dim('normalized_width'),  # Use normalized width
        edge_alpha=0.4,  # More transparency
        node_color='#2b2b2b',  # Dark gray nodes
        node_size=8,  # Smaller nodes
        node_line_color='white',
        node_line_width=1,
        colorbar=True,
        colorbar_opts={'title': '$/GB'}
    )

    # Save the diagram based on file extension.
    output_path = Path(output_file)
    extension = output_path.suffix.lower()

    if extension == '.html':
        hv.save(chord, output_path)
        print(f"Chord diagram saved to {output_path} (interactive HTML)")
    elif extension in ['.png', '.svg', '.pdf']:
        # Plain matplotlib PDF/PNG/SVG without selenium or Bokeh export.
        _plot_chord_matplotlib_static(df, regions, output_path)
    else:
        # Unknown or no extension: default to HTML
        if extension:
            print(
                f"Warning: Unsupported chord output extension '{extension}', using .html instead.",
                file=sys.stderr,
            )
        html_path = output_path.with_suffix('.html')
        hv.save(chord, html_path)
        print(f"Chord diagram saved to {html_path} (interactive HTML)")


def main():
    # Default representative regions for cleaner visualizations
    DEFAULT_REGIONS = [
        "us-east-1",
        "us-west-1",
        "ca-central-1",
        "ca-west-1",
        "eu-central-1",
        "eu-west-1",
        "eu-north-1",
        "eu-south-1",
        "ap-east-1",
        "ap-northeast-1",
        "ap-southeast-1",
        "ap-south-1",
        "sa-east-1",
        "af-south-1",
        "me-south-1",
        "me-central-1",
        "il-central-1",
        "mx-central-1",
    ]

    parser = argparse.ArgumentParser(
        description="Analyze AWS Data Transfer pricing and generate visualizations. "
                    "By default, generates both chord diagram and heatmap visualizations."
    )
    parser.add_argument(
        "--validate-model",
        action="store_true",
        help="Validate migration_model.py assumptions",
    )
    parser.add_argument(
        "--export-csv", metavar="FILE", help="Export pricing data to CSV file"
    )
    parser.add_argument(
        "--refresh-pricing",
        action="store_true",
        help="Re-fetch prices from the AWS Pricing API instead of using "
             "the shipped table (requires a configured AWS CLI)",
    )
    parser.add_argument(
        "--summary-only", action="store_true", help="Show only pricing summary"
    )
    parser.add_argument(
        "--analyze-connectivity", action="store_true", help="Analyze region connectivity"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of pricing items to fetch from AWS API (default: None = fetch all)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached data, fetch fresh from AWS API",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the local cache before fetching",
    )
    parser.add_argument(
        "--chord-diagram",
        action="store_true",
        default=None,
        help="Generate a chord diagram of inter-region S3 egress costs (default: True if no specific option)",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        default=None,
        help="Generate a heatmap of inter-region S3 egress costs (default: True if no specific option)",
    )
    parser.add_argument(
        "--lollipop",
        action="store_true",
        default=None,
        help="Generate a lollipop chart of S3 egress costs grouped by source region",
    )
    parser.add_argument(
        "--lollipop-output",
        metavar="FILE",
        default="outputs/migration_costs/migration_costs_lollipop.png",
        help="Output path for lollipop chart. Default: outputs/migration_costs/migration_costs_lollipop.pdf",
    )
    parser.add_argument(
        "--lollipop-regions",
        metavar="REGION",
        nargs="+",
        default=DEFAULT_REGIONS,
        help="List of AWS region codes to include in lollipop chart. Default: DEFAULT_REGIONS",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualizations",
    )
    parser.add_argument(
        "--chord-output",
        metavar="FILE",
        default=None,
        help="Output path for chord diagram. Format determined by extension: .html (interactive), .png/.svg/.pdf (static, for papers). Default: migration_costs_chord.png",
    )
    parser.add_argument(
        "--chord-regions",
        metavar="REGION",
        nargs="+",
        default=DEFAULT_REGIONS,
        help=(
            "Optional list of AWS region codes to include in the chord diagram "
            "(e.g., us-east-1 us-west-2 eu-west-1). "
            "By default, uses representative regions: 3 US, 3 EU, 3 AP, 3 Others."
        ),
    )
    parser.add_argument(
        "--chord-min-price",
        type=float,
        default=0.0,
        help=(
            "Minimum $/GB transfer price to visualize in the chord diagram. "
            "Pairs below this threshold are omitted. Default: 0.0 (shows all connections)"
        ),
    )
    parser.add_argument(
        "--chord-top-n",
        type=int,
        default=None,
        help=(
            "Show only the N most expensive connections in the chord diagram. "
            "Useful for creating cleaner visualizations for papers."
        ),
    )
    parser.add_argument(
        "--heatmap-output",
        metavar="FILE",
        default=None,
        help="Output path for heatmap. Default: migration_costs_heatmap.pdf (best for papers)",
    )
    parser.add_argument(
        "--heatmap-regions",
        metavar="REGION",
        nargs="+",
        default=DEFAULT_REGIONS,
        help=(
            "Optional list of AWS region codes to include in the heatmap "
            "(e.g., us-east-1 us-west-2 eu-west-1). "
            "By default, uses representative regions: 3 US, 3 EU, 3 AP, 3 Others."
        ),
    )
    parser.add_argument(
        "--heatmap-annotate",
        action="store_true",
        default=True,
        help="Show price values inside heatmap cells (default: True)",
    )
    parser.add_argument(
        "--heatmap-no-annotate",
        action="store_false",
        dest="heatmap_annotate",
        help="Hide price values inside heatmap cells",
    )
    parser.add_argument(
        "--heatmap-min-connections",
        type=int,
        default=5,
        help="Minimum number of connections required for a region to appear in heatmap. Default: 5 (filters out poorly connected regions). Note: AWS doesn't provide pricing for all region pairs.",
    )
    parser.add_argument(
        "--all-regions",
        action="store_true",
        help="Include all available regions instead of the default representative subset (3 US, 3 EU, 3 AP, 3 Others)",
    )

    args = parser.parse_args()

    # Handle --all-regions flag
    if args.all_regions:
        args.chord_regions = None  # None means use all regions
        args.heatmap_regions = None
        args.lollipop_regions = None
        # Also disable min_connections filter when using all regions
        args.heatmap_min_connections = 0

    # Ensure we're in the right directory
    if not Path("sky_spot/migration_model.py").exists():
        print("Error: Please run from the project root directory", file=sys.stderr)
        sys.exit(1)

    # Clear cache if requested
    if args.clear_cache:
        cache_path = get_cache_path()
        if cache_path.exists():
            cache_path.unlink()
            print(f"Cleared cache at {cache_path}")

    # Analyze pricing data
    price_groups = analyze_standard_region_pricing(
        max_items=args.max_items,
        use_cache=not args.no_cache
    )

    if not price_groups:
        print("Error: No pricing data retrieved", file=sys.stderr)
        sys.exit(1)

    # Display summary unless only validation requested
    if not args.summary_only:
        display_pricing_summary(price_groups)

    # Analyze connectivity if requested
    if args.analyze_connectivity:
        analyze_connectivity(price_groups)

    # Determine which visualizations to generate
    # Default: generate lollipop only if no specific option is given
    generate_chord = args.chord_diagram
    generate_heatmap = args.heatmap
    generate_lollipop = args.lollipop

    if not args.no_visualizations:
        # If none is explicitly set, generate lollipop by default
        if generate_chord is None and generate_heatmap is None and generate_lollipop is None:
            generate_lollipop = True
            generate_chord = False
            generate_heatmap = False
        else:
            if generate_chord is None:
                generate_chord = False
            if generate_heatmap is None:
                generate_heatmap = False
            if generate_lollipop is None:
                generate_lollipop = False

    # Generate lollipop chart
    if generate_lollipop and not args.no_visualizations:
        print("\nGenerating lollipop chart...")
        plot_migration_cost_lollipop(
            price_groups,
            selected_regions=args.lollipop_regions,
            output_file=args.lollipop_output,
        )

    # Generate chord diagram
    if generate_chord and not args.no_visualizations:
        output_file = args.chord_output or "migration_costs_chord.png"
        print("\nGenerating chord diagram...")
        plot_migration_cost_chord(
            price_groups,
            selected_regions=args.chord_regions,
            min_price=args.chord_min_price,
            output_file=output_file,
            top_n=args.chord_top_n,
        )

    # Generate heatmap
    if generate_heatmap and not args.no_visualizations:
        output_file = args.heatmap_output or "migration_costs_heatmap.pdf"
        print("\nGenerating heatmap...")
        plot_migration_cost_heatmap(
            price_groups,
            selected_regions=args.heatmap_regions,
            output_file=output_file,
            annotate=args.heatmap_annotate,
            min_connections=args.heatmap_min_connections,
        )

    # Validate model if requested
    if args.validate_model:
        validate_migration_model(price_groups)

    # Export to CSV if requested
    if args.export_csv:
        export_pricing_data(price_groups, args.export_csv)

    print(f"\nAnalysis complete. Found {len(price_groups)} price tiers.")


if __name__ == "__main__":
    main()
