import argparse
import datetime
import json
import pathlib
from typing import Dict, List, Optional

import boto3
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Mapping from our trace names to official AWS instance types.
INSTANCE_TYPE_MAP: Dict[str, str] = {
    # "k80_1": "p2.xlarge",
    # "k80_8": "p2.8xlarge",
    # "v100_1": "p3.2xlarge",
    # "v100_8": "p3.16xlarge",
    "a100_8": "p4d.24xlarge",
    # "h100_8": "p5.48xlarge",
}

# us-east-2
# us-west-2
# us-east-1
# eu-north-1
# eu-west-1
# ca-central-1
# ap-south-1
# eu-central-1
# ap-southeast-2
# eu-west-2
# ap-northeast-1
# ap-northeast-2

# Availability zones to pull spot-price history from.
AVAILABILITY_ZONES = [
    "ap-northeast-1a",
    "ap-northeast-1c",
    "ap-south-1a",
    # "ap-southeast-2a",
    "ca-central-1a",
    "eu-central-1c",
    # "eu-north-1a",
    "eu-west-1c",
    # "eu-west-2b",
    "us-east-1a",
    # "us-east-2a",
    "us-west-2a",
]

# COLOR_LISTS = {
#     "AP": ["#3c075d", "#512f89", "#573aae"], # Hue 307
#     "CA": ["#3d6d94"], # Hue 245
#     "EU": ["#4fa069", "#4abe48"], # Hue 152 143
#     "US": ["#d4d53e", "#FDE725"], # Hue 109 102
# }
COLOR_LISTS = {
    "AP": ["#440154", "#402a7b", "#3e4788"],
    "CA": ["#4281a9"],
    "EU": ["#3CBC75", "#74D055"],
    "US": ["#d5d501", "#FDE725"],
}


def get_zone_color(zone: str, all_zones: List[str]) -> str:
    """Get color for a zone based on its prefix and alphabetical order within prefix group.

    Args:
        zone: The availability zone name (e.g., "ap-northeast-1a")
        all_zones: List of all zones to consider for ordering

    Returns:
        Hex color string
    """
    # Extract prefix (first two letters, uppercase)
    prefix = zone[:2].upper()

    # Get all zones with the same prefix, sorted alphabetically
    same_prefix_zones = sorted([z for z in all_zones if z[:2].upper() == prefix])

    # Find index of current zone within its prefix group
    try:
        idx = same_prefix_zones.index(zone)
    except ValueError:
        idx = 0

    # Get colors for this prefix
    colors = COLOR_LISTS.get(prefix, ["#888888"])

    # Return color (cycle if more zones than colors)
    return colors[idx % len(colors)]


REGION_TO_LOCATION = {
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-1": "US West (N. California)",
    "us-west-2": "US West (Oregon)",
    "eu-central-1": "EU (Frankfurt)",
    "eu-west-1": "EU (Ireland)",
    "eu-west-2": "EU (London)",
    "eu-north-1": "EU (Stockholm)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-northeast-2": "Asia Pacific (Seoul)",
    "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "ca-central-1": "Canada (Central)",
    "sa-east-1": "South America (Sao Paulo)",
    "me-south-1": "Middle East (Bahrain)",
    "af-south-1": "Africa (Cape Town)",
}


def get_on_demand_prices_cached(
    instance_type: str,
    regions: List[str],
    cache_dir: pathlib.Path,
) -> Dict[str, float]:
    """Get on-demand prices, using cache if available.

    Cache is stored as {cache_dir}/{instance_type}_on_demand_prices.json
    """
    cache_file = cache_dir / f"{instance_type}_on_demand_prices.json"

    # Try to load from cache
    if cache_file.exists():
        print(f"\nLoading On-Demand prices from cache: {cache_file}")
        with open(cache_file, "r") as f:
            prices = json.load(f)
        # Filter to requested regions
        prices = {r: p for r, p in prices.items() if r in regions}
        if prices:
            min_price = min(prices.values())
            min_region = min(prices, key=lambda k: prices[k])
            print(f"  Cheapest: {min_region} at ${min_price:.2f}/hr")
            return prices

    # Fetch from AWS
    prices = _fetch_on_demand_prices_from_aws(instance_type, regions)

    # Save to cache
    if prices:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(prices, f, indent=2)
        print(f"  Cached to: {cache_file}")

    return prices


def _fetch_on_demand_prices_from_aws(
    instance_type: str,
    regions: List[str],
) -> Dict[str, float]:
    """Fetch on-demand prices from AWS API."""
    print(f"\nFetching On-Demand prices for {instance_type} from AWS...")
    pricing_client = boto3.client("pricing", region_name="us-east-1")

    prices = {}
    for region in regions:
        location = REGION_TO_LOCATION.get(region)
        if not location:
            print(f"  {region}: Unknown region, skipping")
            continue
        try:
            paginator = pricing_client.get_paginator("get_products")
            pages = paginator.paginate(
                ServiceCode="AmazonEC2",
                Filters=[
                    {"Type": "TERM_MATCH", "Field": "location", "Value": location},
                    {
                        "Type": "TERM_MATCH",
                        "Field": "instanceType",
                        "Value": instance_type,
                    },
                    {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                    {
                        "Type": "TERM_MATCH",
                        "Field": "operatingSystem",
                        "Value": "Linux",
                    },
                    {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                    {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
                ],
            )
            for page in pages:
                for price_data in page.get("PriceList", []):
                    price_list = json.loads(price_data)
                    on_demand_terms = price_list.get("terms", {}).get("OnDemand", {})
                    if not on_demand_terms:
                        continue
                    term_code = next(iter(on_demand_terms))
                    price_dimensions = on_demand_terms[term_code].get(
                        "priceDimensions", {}
                    )
                    if not price_dimensions:
                        continue
                    dim_code = next(iter(price_dimensions))
                    price_per_unit = price_dimensions[dim_code]["pricePerUnit"].get(
                        "USD"
                    )
                    if price_per_unit:
                        value = float(price_per_unit)
                        if value > 0.0:
                            prices[region] = value
                            print(f"  {region}: ${value:.2f}/hr")
                            break
        except Exception as e:
            print(f"  {region}: Error - {e}")

    if prices:
        cheapest_region = min(prices, key=prices.get)
        print(
            f"\nCheapest region: {cheapest_region} at ${prices[cheapest_region]:.2f}/hr"
        )

    return prices


# Default time window for plots: MM.DD-MM.DD
DEFAULT_TIME_WINDOW = "10.15-11.15"


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

fig_width_pt = nsdi23_col_width_pt

inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio


def FigWidth(pt):
    return pt * inches_per_pt  # width in inches


fig_width = FigWidth(fig_width_pt)  # width in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]


def InitMatplotlib(font_size, title_size=9):
    print("use_tex", TEXT_USETEX, "\nfont_size", font_size, "\ntitle_size", title_size)
    # https://matplotlib.org/3.2.1/tutorials/introductory/customizing.html
    params = {
        "backend": "ps",
        "figure.figsize": fig_size,
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


def _parse_time_window_to_bounds(value: str) -> tuple[int, int]:
    """Parse a time window string like '10.15-11.15' into numeric bounds.

    Returns (start_md, end_md) where month/day is encoded as MMDD, for example:
    10.15 -> 1015, 11.15 -> 1115.
    """
    try:
        start_str, end_str = value.split("-")
        start_month, start_day = map(int, start_str.split("."))
        end_month, end_day = map(int, end_str.split("."))
    except ValueError as exc:
        raise ValueError(
            f"Invalid time window '{value}'. Expected format MM.DD-MM.DD, e.g., 10.15-11.15."
        ) from exc

    for month, day in ((start_month, start_day), (end_month, end_day)):
        if not (1 <= month <= 12 and 1 <= day <= 31):
            raise ValueError(
                f"Invalid date in time window '{value}': month={month}, day={day}."
            )

    start_md = start_month * 100 + start_day
    end_md = end_month * 100 + end_day
    if start_md > end_md:
        raise ValueError(
            f"Start date must not be after end date in time window '{value}'."
        )
    return start_md, end_md


def _parse_time_window_to_datetime(
    value: str, year: int
) -> tuple[datetime.datetime, datetime.datetime]:
    """Parse a time window string like '10.15-11.15' into timestamp bounds."""
    start_md, end_md = _parse_time_window_to_bounds(value)
    start_month = start_md // 100
    start_day = start_md % 100
    end_month = end_md // 100
    end_day = end_md % 100

    start_dt = datetime.datetime(
        year=year, month=start_month, day=start_day, tzinfo=datetime.timezone.utc
    )
    end_dt = datetime.datetime(
        year=year, month=end_month, day=end_day, tzinfo=datetime.timezone.utc
    )
    return start_dt, end_dt


def parse_time_window_arg(value: str) -> str:
    """argparse type for --time-window; validates format MM.DD-MM.DD."""
    try:
        _parse_time_window_to_bounds(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def fetch_on_demand_price(instance_type: str, region: str) -> Optional[float]:
    """Return the current on-demand price for the instance in the given region."""
    pricing_client = boto3.client("pricing", region_name="us-east-1")
    location = REGION_TO_LOCATION.get(region)
    if not location:
        print(f"  -> WARNING: Region {region} not mapped to a pricing location.")
        return None

    paginator = pricing_client.get_paginator("get_products")
    pages = paginator.paginate(
        ServiceCode="AmazonEC2",
        Filters=[
            {"Type": "TERM_MATCH", "Field": "location", "Value": location},
            {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
        ],
    )
    for page in pages:
        for price_data in page.get("PriceList", []):
            price_list = json.loads(price_data)
            on_demand_terms = price_list.get("terms", {}).get("OnDemand", {})
            if not on_demand_terms:
                continue
            term_code = next(iter(on_demand_terms))
            price_dimensions = on_demand_terms[term_code].get("priceDimensions", {})
            if not price_dimensions:
                continue
            dim_code = next(iter(price_dimensions))
            price_per_unit = price_dimensions[dim_code]["pricePerUnit"].get("USD")
            if not price_per_unit:
                continue
            value = float(price_per_unit)
            if value <= 0.0:
                continue
            print(f"  -> AWS On-Demand price in {region}: ${value}")
            return value
    return None


def fetch_price_history(
    client, instance_type: str, availability_zone: str, days: int
) -> List[Dict]:
    """Fetch spot price history for an instance type in one AZ."""
    print(
        f"Fetching AWS spot prices for {instance_type} in {availability_zone} for the last {days} days..."
    )
    start_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=days
    )
    paginator = client.get_paginator("describe_spot_price_history")
    pages = paginator.paginate(
        InstanceTypes=[instance_type],
        ProductDescriptions=["Linux/UNIX"],
        AvailabilityZone=availability_zone,
        StartTime=start_time,
    )
    price_history: List[Dict] = []
    try:
        for page in pages:
            for price_point in page.get("SpotPriceHistory", []):
                price_history.append(
                    {
                        "Timestamp": price_point["Timestamp"].isoformat(),
                        "SpotPrice": price_point["SpotPrice"],
                    }
                )
    except client.exceptions.ClientError as exc:
        print(f"  -> WARNING: AWS API error for {availability_zone}: {exc}. Skipping.")
        return []

    price_history.sort(key=lambda x: x["Timestamp"])
    return price_history


def plot_price_comparison(
    data_dir: pathlib.Path,
    target_name: str,
    time_window: str = DEFAULT_TIME_WINDOW,
) -> None:
    """Plot spot price history for all AZs associated with the target."""
    print(f"\n🎨 Generating plot for {target_name}...")
    price_files = sorted(data_dir.glob(f"*_{target_name}_prices.json"))
    if not price_files:
        print(f"  -> No data files found for {target_name}. Skipping plot.")
        return

    start_md, end_md = _parse_time_window_to_bounds(time_window)
    print(f"  -> Applying time window {time_window} (MM.DD).")

    aws_instance_type = INSTANCE_TYPE_MAP.get(target_name, target_name)

    # Initialize matplotlib with standard settings
    InitMatplotlib(font_size=12, title_size=14)

    # Wider figure to accommodate legend on top while keeping similar aspect ratio
    # Original with legend on right was ~5.5 x 4.5; now use same ratio with legend on top
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    fig.subplots_adjust(top=0.79, right=0.99, left=0.22, bottom=0.125)

    # Extract all zone names from files for color assignment
    all_zones = [f.name.split("_")[0] for f in price_files]

    for i, file_path in enumerate(price_files):
        az_name = file_path.name.split("_")[0]
        color = get_zone_color(az_name, all_zones)
        try:
            df = pd.read_json(file_path)
        except ValueError as exc:
            print(f"  -> Could not process {file_path}: {exc}")
            continue
        if df.empty:
            continue
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["SpotPrice"] = pd.to_numeric(df["SpotPrice"])

        # Filter by the requested time window, based on month/day.
        month_day = df["Timestamp"].dt.month * 100 + df["Timestamp"].dt.day
        mask = (month_day >= start_md) & (month_day <= end_md)
        df = df.loc[mask]
        if df.empty:
            print(
                f"  -> No data points for {az_name} in the specified time window. Skipping."
            )
            continue

        ax.plot(
            df["Timestamp"],
            df["SpotPrice"],
            label=az_name,
            color=color,
            linewidth=2,
        )

    # Get on-demand prices (cached)
    unique_regions = list(set(az[:-1] for az in all_zones))
    on_demand_prices = get_on_demand_prices_cached(
        aws_instance_type, unique_regions, data_dir
    )
    if on_demand_prices:
        min_od_price = min(on_demand_prices.values())
        min_od_region = min(on_demand_prices, key=lambda k: on_demand_prices[k])
        ax.axhline(
            y=min_od_price,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"On-Demand\nmin: {min_od_price:.2f}",
        )

    # ax.set_title(
    #     f"Historical Spot Price Comparison: {target_name} ({aws_instance_type})",
    #     pad=15,
    #     weight="bold",
    # )

    # Determine the year from the actual data (use 2025 since data is from last year)
    # Find the year from the first data file that has data
    data_year = 2025
    for file_path in price_files:
        try:
            df_check = pd.read_json(file_path)
            if not df_check.empty:
                df_check["Timestamp"] = pd.to_datetime(df_check["Timestamp"])
                data_year = df_check["Timestamp"].iloc[0].year
                break
        except Exception:
            continue

    start_dt, end_dt = _parse_time_window_to_datetime(time_window, year=data_year)
    ax.set_xlim(start_dt, end_dt)  # type: ignore

    ax.set_xlabel("Timestamp", fontsize=14)
    ax.set_ylabel("Spot Price ($ / hour)", fontsize=14, labelpad=3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    # Horizontal x-axis labels with fewer ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Every 7 days
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    # Group legend entries by region prefix
    handles, labels = ax.get_legend_handles_labels()

    # Define group order and sort handles/labels
    prefix_order = ["AP", "CA", "EU", "US"]

    def get_prefix_sort_key(label):
        prefix = label[:2].upper()
        try:
            return (prefix_order.index(prefix), label)
        except ValueError:
            return (len(prefix_order), label)

    sorted_pairs = sorted(zip(handles, labels), key=lambda x: get_prefix_sort_key(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_pairs) if sorted_pairs else ([], [])

    # Place legend at top, 3 columns (3 items per row)
    ax.legend(
        sorted_handles,
        sorted_labels,
        loc="upper center",
        bbox_to_anchor=(0.375, 1.325),
        frameon=False,
        fontsize=10,
        ncol=3,
        columnspacing=1.0,
        handletextpad=0.5,
        handlelength=1.25,
    )
    # ax.grid(True, alpha=0.5, which='major', linestyle='-', linewidth=0.5)

    if on_demand_prices:
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], max(current_ylim[1], min_od_price * 1.1))

    output_dir = pathlib.Path("outputs/price_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"spot_price_comparison_{target_name}.png"
    fig.savefig(output_filename, dpi=300)
    fig.savefig(output_filename.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  -> ✅ Plot saved successfully to {output_filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch AWS Spot Price history and save it to JSON files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/spot_prices",
        help="Directory to save pricing data.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of past days to fetch history for (max 90).",
    )
    parser.add_argument(
        "--time-window",
        type=parse_time_window_arg,
        default=DEFAULT_TIME_WINDOW,
        help=(
            "Time window for the plots, in MM.DD-MM.DD format. "
            f"Example: 10.15-11.15 (default: {DEFAULT_TIME_WINDOW})."
        ),
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching data from AWS and only generate plots from existing data.",
    )
    args = parser.parse_args()

    output_dir_path = pathlib.Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(
        "This script requires AWS credentials to be configured (e.g., via `aws configure`)."
    )

    clients: Dict[str, "boto3.client"] = {}
    if not args.skip_fetch:
        print("\n----------------------------------------")
        print("Starting data fetch from AWS...\n")
        for zone in AVAILABILITY_ZONES:
            region = zone[:-1]
            if region not in clients:
                print(f"\nCreating boto3 client for region: {region}")
                clients[region] = boto3.client("ec2", region_name=region)
            client = clients[region]

            for short_name, aws_instance_type in INSTANCE_TYPE_MAP.items():
                price_data = fetch_price_history(client, aws_instance_type, zone, args.days)
                if not price_data:
                    print(
                        f"  -> No price data found for {aws_instance_type} in {zone}. Skipping."
                    )
                    continue

                output_filename = f"{zone}_{short_name}_prices.json"
                output_path = output_dir_path / output_filename
                with open(output_path, "w", encoding="utf-8") as handle:
                    json.dump(price_data, handle, indent=2)
                print(
                    f"  -> Successfully saved {len(price_data)} price points to {output_path}"
                )

        print("\n----------------------------------------")
        print("All data fetched. Now generating plots...")

    for short_name in INSTANCE_TYPE_MAP.keys():
        plot_price_comparison(
            output_dir_path,
            short_name,
            time_window=args.time_window,
        )


if __name__ == "__main__":
    main()
