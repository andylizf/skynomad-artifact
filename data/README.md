# Data shipped with this artifact

Everything the paper's figures need is in this directory or generated from it by
`artifact/prepare_data.sh`. There is nothing to download, no remote host to
reach, and no credential to configure.

## Inventory

| Path | What it is | Size / shape |
|---|---|---|
| `h100_16_runs/` | Raw GCP probe records, one JSON per 10-minute round. Each records, per zone, how many of 16 requested `a3-highgpu-1g` (1xH100) instances actually launched. This is the artifact's main reusable asset; collecting it cost roughly $9,000 in cloud spend. | 2,104 files, 13 zones, 14 days, 27,352 zone-ticks |
| `converted_multi_region_aligned_h100_16_merged.tar.gz` | The above, converted to simulator-ready traces. `artifact/prepare_data.sh` unpacks it to `converted_multi_region_aligned_h100_16_merged/`. | 13 zones, 352 h each |
| `real/availability/2023-02-15/processed/` | Raw AWS V100 availability CSVs (`date,availability`), one per zone. | 9 zones |
| `v100_spot_prices.json` | Per-tick AWS spot price for each of the nine V100 zones, over the aligned window. Attached to the traces by `create_aligned_traces.py`. | 9 zones x 20,158 ticks, 1.2 MB |
| `v100_spot_price_archive_summary.csv` | Per-zone summary from the public AWS spot price archive, used to check the series above. | 9 rows |
| `converted_multi_region_aligned/` | Built from those CSVs by `artifact/prepare_data.sh`. Not checked in — it is generated. | 9 zones x (`full.json` + 100 windows) |
| `converted_v100_4node_600s/` | The 4-zone V100 traces the appendix's progress-value ablation runs on, 600 s cadence. | 3 zones |
| `v100_1node_prices/`, `v100_multinode_raw/`, `spot_hedge_4node_src/` | Raw inputs `convert_v100_4node_600s.py` builds the above from. | 3 dirs |
| `spot_price_cache/gcp/` | GCP Billing Catalog price entries for the 8 regions covering the 13 H100 zones, so `convert_gcp_h100_spot.py` runs without an API key. | 8 JSON files |

## Trace format

```json
{
  "metadata": {
    "gap_seconds": 600,
    "start_time": "2025-10-04T00:10:00+00:00",
    "zone": "us-central1-a",
    "price_info": {"price": 18.98, "on_demand_price": 82.53, ...}
  },
  "data": [0, 0, 1, 1, 0, ...],
  "prices": [18.98, 18.98, ...]
}
```

`data` is in **preempted** convention: `1` = preempted/unavailable, `0` =
available. This is inverted relative to the raw availability CSVs, because
`TraceEnv.spot_available()` returns `not trace[tick]`.

`metadata.price_info` is mandatory. `sky_spot/env.py` has no price fallback by
design and raises `ValueError: No on_demand_price in price_info` for a trace
without it. `tests/test_trace_price_presence.py` enforces this by constructing a
real `TraceEnv` over every generated trace.

## H100: raw probes to traces

`artifact/prepare_data.sh` unpacks the pre-converted archive, but the conversion
is reproducible from the raw probes and needs no network:

```bash
python scripts_multi/trace_sampling/convert_gcp_h100_spot.py \
    --input-dir data/h100_16_runs \
    --output-dir outputs/h100_regenerated \
    --device-name h100_16 \
    --machine-type a3-highgpu-8g
```

* A zone counts as available for a tick when **at least one** of the 16 requested
  instances launched (`--gang-threshold`, default 1). Over the full probe set
  that gives **76.7%** availability, which the command above reproduces.
* Prices come from `data/spot_price_cache/gcp/`, which ships here. `--api-key`
  (a Google Cloud Billing API key) is needed only for a region not in that
  cache. Google exposes only the *current* spot list price, so every tick of a
  trace carries the same rate, recorded in `metadata.price_info` along with the
  SKU breakdown that produced it.
* `--trace-length-hours`, `--num-traces` and `--seed` control the windowed shards
  (`0.json`, `1.json`, ...) written alongside `full.json`.

## V100: CSVs to traces

`artifact/prepare_data.sh` runs `scripts_multi/trace_sampling/create_aligned_traces.py`,
which aligns the nine `processed/*.csv` files to their common time window and
writes `full.json` plus 100 sampled 60-hour windows per zone (seeded, so a second
run is byte-identical).

The source CSVs carry availability only, so `create_aligned_traces.py` attaches
two things: a per-tick spot price series from `v100_spot_prices.json` (see below)
and a `metadata.price_info` block naming the on-demand rate ($3.06/hr for
p3.2xlarge, from `sky_spot.utils.ACTUAL_COSTS['v100_1']`).

`v100_spot_prices.json` holds one price per tick per zone, 9 x 20158 entries,
spanning $0.9180 to $1.8464, laid on the aligned timeline by `start_time`.

Two independent checks that the series is right:

* **Against the public AWS spot price archive** (Eric Pauley's, concept DOI
  `10.5281/zenodo.14198917`). The per-zone maxima agree exactly: us-east-1a
  $1.8464, us-east-1c $1.4053, us-east-1d $1.3661, us-east-1f $1.4003.
* **Against the shipped results.** Re-running `--config v100` reproduces
  `research/data/v100.csv`; `artifact/compare_sweep.py --config v100` checks it
  row by row. Under a single flat spot rate the price-sensitive policies do not
  reproduce it — the per-tick series is what closes them.

`v100_spot_price_archive_summary.csv` is the archive-side summary those maxima
were checked against.

Regenerate that summary with
`scripts_multi/trace_sampling/check_v100_archive_prices.py --archive 2023.tsv.zst`
(the archive file is 511 MB and is not downloaded automatically).
