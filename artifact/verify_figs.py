#!/usr/bin/env python3
"""
Compare reproduced figures against the paper's originals.

Two checks run per figure, and both must pass:

1. Text-layer diff. Text extracted from each PDF is normalised and diffed. This
   catches changes to axis ticks, tick ranges, legend entries and in-figure
   annotations. It does NOT catch a change in a plotted value: a bar height or a
   line vertex is vector geometry, not text. A reviewer can scale SkyNomad's cost
   by 30% in research/data/*.csv, replot, and the text layer is byte-identical.

2. Numeric sidecar diff. Each result-figure script writes the exact series it
   draws to `<figure>.values.csv` beside the PDF (see sky_spot/figure_values.py).
   Reference copies of those series live in `paper_figs/values/`. This script
   compares them cell by cell with a relative tolerance (--rtol, default 1%).
   This is the check that actually sees the numbers.

   A figure that has a reference sidecar but no reproduced one is a failure, not
   a skip -- otherwise deleting the sidecar would silently disable the check.

What the sidecar check does and does not establish: the reference sidecars were
produced by these same scripts from the shipped research/data/*.csv, which is
the data the published figures were drawn from. So it verifies that a reviewer's
rerun draws the paper's numbers, and it fails loudly if the shipped results or
the aggregation change. It is not an independent re-derivation of those results
-- that means re-running the simulation sweeps, which is the `--config` path
documented in README.md.

A third thing this script can do is carry CAVEATS: figures that reproduce exactly
but where the match needs a qualification a reviewer could not infer from the
output. Any such entry is printed under the table and counted on the summary
line, so the default review path cannot report a clean sweep without showing it.
There are none at present.

Paths default to locations inside this repository, resolved relative to this
script, so it works from any checkout.

Requires `pdftotext` (poppler-utils). Missing it is a hard error rather than a
silent pass: without it every comparison would return the same error string on
both sides and be reported as a match.

Exit status: 0 only if every paper figure was found and matched.

Usage:
    .venv/bin/python artifact/verify_figs.py
    .venv/bin/python artifact/verify_figs.py --paper-figs <dir> --repro-dir <dir>
"""

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SIDECAR_SUFFIX = ".values.csv"

# Figures that are hand-drawn schematics, or that the paper does not include.
# They have no generating script, so "missing" is expected and not a failure.
# Figures that reproduce exactly but whose content does not mean what the
# figure's own labels say. A clean MATCH on these would be misleading on its
# own, so the caveat is printed next to the result and counted in the summary
# line. The default review path is reproduce_all.sh -> verify_figs.py, and
# reproduce_all.sh redirects the plotting scripts' stdout to a log file, so a
# runtime NOTE printed by the plotting script is not enough.
CAVEATS: dict[str, str] = {
    # Populated when a figure reproduces exactly but the match needs a caveat a
    # reviewer could not infer. Empty at present.
}

NON_REPRODUCIBLE = {
    "skynomad-arch.pdf",
    "skynomad-design.pdf",
    "sky-nomad-virtual-instance.pdf",
    "region_availability_coverage.pdf",
    "traces_comparison.pdf",  # paper uses the per-accelerator variants
    "spot_price_comparison_a100_8.pdf",  # AWS API only retains 90 days; see README
}


def pdf_text(path: Path) -> str:
    out = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        capture_output=True, timeout=120, check=True,
    )
    return " ".join(out.stdout.decode("utf-8", "replace").split())


def find_repro(name: str, search_dirs) -> tuple[Path | None, str | None]:
    """Locate a reproduced figure, returning (path, ambiguity_note).

    Several scripts write into their own subdirectory of outputs/, so a recursive
    search is needed. But the rerun workflows documented in the README (which
    write to outputs/rerun/ and outputs/rerun_avail/) leave a second copy of the
    same filename, and returning whichever one the filesystem happened to yield
    first meant a freshly reproduced figure could be silently checked against a
    stale copy from an earlier run — reporting a clean pass while the figure that
    was actually just regenerated went unexamined.

    So: collect every candidate. One hit is unambiguous. When there are several,
    prefer the canonical output location (outputs/figures/), and if that does not
    single one out, report the ambiguity instead of guessing.
    """
    hits: list[Path] = []
    for d in search_dirs:
        if d.is_dir():
            hits.extend(sorted(d.rglob(name)))
    if not hits:
        return None, None
    if len(hits) == 1:
        return hits[0], None

    canonical = [p for p in hits if p.parent.name == "figures"]
    others = ", ".join(str(p) for p in hits if p not in canonical)
    if len(canonical) == 1:
        return canonical[0], f"{len(hits)} copies exist; used outputs/figures/ (ignored: {others})"
    return None, f"AMBIGUOUS: {len(hits)} copies and none is uniquely canonical: " + \
                 ", ".join(str(p) for p in hits)


def read_values(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _cell_matches(a: str, b: str, rtol: float) -> bool:
    """True if reproduced cell `a` agrees with reference cell `b`."""
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(fa) and math.isnan(fb):
        return True
    return abs(fa - fb) <= rtol * max(abs(fb), 1e-12)


def compare_values(ref: Path, got: Path, rtol: float) -> str | None:
    """Return None when the two sidecars agree, else a short description."""
    rows_ref, rows_got = read_values(ref), read_values(got)
    if len(rows_ref) != len(rows_got):
        return f"{len(rows_got)} rows vs {len(rows_ref)} in reference"

    cols_ref = list(rows_ref[0].keys()) if rows_ref else []
    cols_got = list(rows_got[0].keys()) if rows_got else []
    if cols_ref != cols_got:
        return f"columns {cols_got} vs {cols_ref}"

    diffs = []
    for i, (r, g) in enumerate(zip(rows_ref, rows_got)):
        for col in cols_ref:
            if not _cell_matches(g[col], r[col], rtol):
                diffs.append(f"row {i} {col}: {g[col]} vs {r[col]}")
    if diffs:
        head = "; ".join(diffs[:2])
        more = f" (+{len(diffs) - 2} more)" if len(diffs) > 2 else ""
        return f"{len(diffs)} value(s) differ: {head}{more}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-figs", default=str(REPO / "paper_figs"),
                    help="directory holding the paper's original figure PDFs")
    ap.add_argument("--repro-dir", action="append",
                    help="directory to search for reproduced figures (repeatable)")
    ap.add_argument("--rtol", type=float, default=0.01,
                    help="relative tolerance for the numeric sidecar comparison")
    args = ap.parse_args()

    if shutil.which("pdftotext") is None:
        print("error: pdftotext not found. Install poppler-utils "
              "(macOS: brew install poppler, Debian: apt install poppler-utils).",
              file=sys.stderr)
        return 2

    paper_figs = Path(args.paper_figs)
    if not paper_figs.is_dir():
        print(f"error: paper figure directory not found: {paper_figs}", file=sys.stderr)
        return 2

    values_dir = paper_figs / "values"
    search_dirs = [Path(d) for d in (args.repro_dir or [str(REPO / "outputs")])]

    rows, matched, mismatched, missing, errors = [], 0, 0, 0, 0
    values_checked = 0
    caveated: list[tuple[str, str]] = []
    for orig in sorted(paper_figs.glob("*.pdf")):
        if orig.name in NON_REPRODUCIBLE:
            rows.append(("SKIP", orig.name, "not reproducible by design"))
            continue
        repro, ambiguity = find_repro(orig.name, search_dirs)
        if repro is None:
            if ambiguity:
                # Several copies and no canonical one. Counted as an error rather
                # than resolved by guessing: picking the wrong copy is exactly how
                # a stale figure passes verification unnoticed.
                rows.append(("AMBIGUOUS", orig.name, ambiguity))
                errors += 1
            else:
                rows.append(("MISSING", orig.name, "-"))
                missing += 1
            continue
        if ambiguity:
            caveated.append((orig.name, ambiguity))
        try:
            a, b = pdf_text(orig), pdf_text(repro)
        except Exception as e:  # noqa: BLE001
            rows.append(("ERROR", orig.name, str(e)[:60]))
            errors += 1
            continue

        if a == b:
            status = "MATCH"
        elif sorted(a.split()) == sorted(b.split()):
            # Same tokens, different order in the PDF's content stream. Every
            # label agrees; only the drawing order differs, which happens when a
            # figure is redrawn by a different script.
            status = "MATCH (reordered)"
        else:
            sa, sb = set(a.split()), set(b.split())
            rows.append((f"DIFF(+{len(sb - sa)}/-{len(sa - sb)})", orig.name, str(repro)))
            mismatched += 1
            continue

        # Numeric check, for the figures that ship a reference sidecar.
        ref_values = values_dir / (orig.stem + SIDECAR_SUFFIX)
        if ref_values.is_file():
            got_values = repro.with_name(repro.stem + SIDECAR_SUFFIX)
            if not got_values.is_file():
                rows.append(("VALUES MISSING", orig.name, str(got_values)))
                missing += 1
                continue
            try:
                problem = compare_values(ref_values, got_values, args.rtol)
            except Exception as e:  # noqa: BLE001
                rows.append(("ERROR", orig.name, str(e)[:60]))
                errors += 1
                continue
            if problem:
                rows.append(("VALUES DIFF", orig.name, problem))
                mismatched += 1
                continue
            values_checked += 1
            status += " +values"

        if orig.name in CAVEATS:
            caveated.append((orig.name, CAVEATS[orig.name]))
            status += " (CAVEAT)"

        rows.append((status, orig.name, str(repro)))
        matched += 1

    width = max((len(r[1]) for r in rows), default=10) + 2
    for status, name, path in rows:
        print(f"{status:<28} {name:<{width}} {path}")

    if caveated:
        print()
        print("CAVEATS -- these figures match, but the match does not mean what "
              "the figure's labels say:")
        for name, note in caveated:
            print(f"  {name}: {note}")

    print()
    print(f"numeric sidecar comparisons: {values_checked} (rtol={args.rtol:g}). "
          f"Figures without a sidecar are checked on their text layer only; see "
          f"this script's docstring for what that does and does not cover.")
    print(f"matched={matched}  mismatched={mismatched}  missing={missing}  "
          f"errors={errors}  caveats={len(caveated)}")
    return 0 if (mismatched == 0 and missing == 0 and errors == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
