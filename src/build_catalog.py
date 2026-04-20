"""
build_catalog.py
----------------
Step 1: Aggregate all daily CSVs into a single clean catalog.
Performs QC checks and saves a master parquet + summary report.

Usage:
    python build_catalog.py --indir /path/to/mt_rainier_common_detections
                            --outdir /path/to/output
"""

import argparse
import ast
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── filename pattern ────────────────────────────────────────────────────────
FNAME_RE = re.compile(
    r"common_(\d{8})_(\d{4})_to_(\d{8})_(\d{4})_events\.csv"
)

EXPECTED_COLS = {
    "cluster_id", "rounded_start", "num_stations",
    "stations", "all_classes", "most_common_class",
    "mean_auc", "mean_max", "mean_prob",
}

VALID_CLASSES = {"eq", "su", "px", "noise"}


def parse_list_col(val):
    """Safely parse stringified Python lists like "['eq', 'su']"."""
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def load_one_file(fpath: Path):
    """Load a single daily CSV with basic validation."""
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        return None, f"READ_ERROR: {e}"

    # ── column check ────────────────────────────────────────────────────────
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        return None, f"MISSING_COLS: {missing}"

    if df.empty:
        return None, "EMPTY_FILE"

    # ── attach file-level date from filename ────────────────────────────────
    m = FNAME_RE.match(fpath.name)
    file_date = pd.to_datetime(m.group(1), format="%Y%m%d") if m else pd.NaT
    df["file_date"] = file_date

    return df, None


def build_catalog(indir: str, outdir: str):
    indir  = Path(indir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(indir.glob("common_*_events.csv"))
    print(f"Found {len(files):,} CSV files in {indir}")

    frames = []
    skipped = []

    for fpath in files:
        df, err = load_one_file(fpath)
        if err:
            skipped.append({"file": fpath.name, "reason": err})
            continue
        frames.append(df)

    print(f"  Loaded : {len(frames):,} files")
    print(f"  Skipped: {len(skipped):,} files")

    if not frames:
        print("ERROR: no data loaded. Check your --indir path.")
        return

    # ── concatenate ─────────────────────────────────────────────────────────
    cat = pd.concat(frames, ignore_index=True)
    print(f"\nRaw catalog size: {len(cat):,} rows")

    # ── parse timestamps ─────────────────────────────────────────────────────
    ts = pd.to_datetime(cat["rounded_start"], errors="coerce", format="mixed")
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC")
    else:
        ts = ts.dt.tz_localize("UTC")
    cat["rounded_start"] = ts
    n_bad_ts = cat["rounded_start"].isna().sum()
    if n_bad_ts:
        print(f"  WARNING: {n_bad_ts} rows with unparseable timestamps — dropped")
    cat = cat.dropna(subset=["rounded_start"])

    # ── parse list columns ───────────────────────────────────────────────────
    cat["stations"]    = cat["stations"].apply(parse_list_col)
    cat["all_classes"] = cat["all_classes"].apply(parse_list_col)

    # ── derived columns ──────────────────────────────────────────────────────
    cat["date"]   = cat["rounded_start"].dt.date
    cat["year"]   = cat["rounded_start"].dt.year
    cat["month"]  = cat["rounded_start"].dt.month
    cat["hour"]   = cat["rounded_start"].dt.hour
    cat["doy"]    = cat["rounded_start"].dt.day_of_year   # day-of-year

    # Per-cluster vote fractions
    def vote_fraction(row, cls):
        classes = row["all_classes"]
        if not classes:
            return np.nan
        return classes.count(cls) / len(classes)

    for cls in ["eq", "su", "px", "noise"]:
        cat[f"frac_{cls}"] = cat.apply(vote_fraction, cls=cls, axis=1)

    # Unanimous flag: all stations agree
    cat["unanimous"] = cat["all_classes"].apply(
        lambda x: len(set(x)) == 1 if x else False
    )

    # Vote margin: fraction for winning class
    cat["vote_margin"] = cat["all_classes"].apply(
        lambda x: max(x.count(c) for c in set(x)) / len(x) if x else np.nan
    )

    # ── QC flags ─────────────────────────────────────────────────────────────
    # mean_auc is area under smoothed prob curve, summed across stations.
    # Normalize by num_stations to get per-station average AUC.
    cat["auc_is_summed"] = cat["mean_auc"] > 1.0
    cat["mean_auc_per_station"] = cat["mean_auc"] / cat["num_stations"]

    # mean_max is the primary confidence signal: mean of per-station peak prob
    # in the detection window. Detection already guarantees >0.50 per station.
    cat["low_confidence"]  = cat["mean_max"] < 0.65
    cat["high_confidence"] = (cat["mean_max"] >= 0.75) & (cat["vote_margin"] >= 0.67)

    # Ambiguous class: bare majority or worse (critical at median 3 stations)
    cat["ambiguous_class"] = cat["vote_margin"] <= 0.5

    # ── sort ─────────────────────────────────────────────────────────────────
    cat = cat.sort_values("rounded_start").reset_index(drop=True)

    # ── global unique event id ───────────────────────────────────────────────
    cat["event_id"] = [f"MR{str(i).zfill(7)}" for i in range(len(cat))]

    # ── save master catalog ──────────────────────────────────────────────────
    out_csv = outdir / "master_catalog.csv"
    cat.to_csv(out_csv, index=False)
    print(f"\nSaved master catalog → {out_csv}")

    try:
        out_parquet = outdir / "master_catalog.parquet"
        cat.to_parquet(out_parquet, index=False)
        print(f"Saved master catalog → {out_parquet}")
    except ImportError:
        print("  (parquet skipped — install pyarrow for faster I/O on large catalogs)")

    # ── QC report ────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("CATALOG QC REPORT")
    lines.append("=" * 60)

    lines.append(f"\n--- Coverage ---")
    lines.append(f"  Date range  : {cat['rounded_start'].min()} → {cat['rounded_start'].max()}")
    lines.append(f"  Total events: {len(cat):,}")
    lines.append(f"  Unique days : {cat['date'].nunique():,}")

    # Check for date gaps
    all_dates = pd.date_range(
        cat["rounded_start"].min().date(),
        cat["rounded_start"].max().date(),
        freq="D"
    )
    present_dates = set(cat["date"])
    missing_dates = [d.date() for d in all_dates if d.date() not in present_dates]
    lines.append(f"  Missing days: {len(missing_dates):,}")
    if missing_dates[:10]:
        lines.append(f"    (first 10): {missing_dates[:10]}")

    lines.append(f"\n--- Class distribution (most_common_class) ---")
    vc = cat["most_common_class"].value_counts()
    for cls, cnt in vc.items():
        lines.append(f"  {cls:8s}: {cnt:8,}  ({100*cnt/len(cat):.1f}%)")

    lines.append(f"\n--- Station count distribution ---")
    sc = cat["num_stations"].describe()
    lines.append(f"  min={sc['min']:.0f}  median={sc['50%']:.0f}  "
                 f"max={sc['max']:.0f}  mean={sc['mean']:.1f}")

    lines.append(f"\n--- Confidence ---")
    lines.append(f"  mean_max range             : {cat['mean_max'].min():.3f} – {cat['mean_max'].max():.3f}")
    lines.append(f"  mean_prob range            : {cat['mean_prob'].min():.3f} – {cat['mean_prob'].max():.3f}")
    lines.append(f"  mean_auc_per_station range : {cat['mean_auc_per_station'].min():.3f} – {cat['mean_auc_per_station'].max():.3f}")
    lines.append(f"  Low confidence  (mean_max < 0.65)                    : {cat['low_confidence'].sum():,}  "
                 f"({100*cat['low_confidence'].mean():.1f}%)")
    lines.append(f"  High confidence (mean_max ≥ 0.75, vote_margin ≥ 0.67): {cat['high_confidence'].sum():,}  "
                 f"({100*cat['high_confidence'].mean():.1f}%)")
    lines.append(f"  Ambiguous class (vote_margin ≤ 0.50)                 : {cat['ambiguous_class'].sum():,}  "
                 f"({100*cat['ambiguous_class'].mean():.1f}%)")
    lines.append(f"  Unanimous votes                                       : {cat['unanimous'].sum():,}  "
                 f"({100*cat['unanimous'].mean():.1f}%)")

    lines.append(f"\n--- mean_auc flag ---")
    n_summed = cat["auc_is_summed"].sum()
    lines.append(f"  Rows where mean_auc > 1 (likely summed, not averaged): "
                 f"{n_summed:,}  ({100*n_summed/len(cat):.1f}%)")
    lines.append(f"  mean_auc range             : {cat['mean_auc'].min():.3f} – {cat['mean_auc'].max():.3f}")
    lines.append(f"  mean_auc_per_station range : {cat['mean_auc_per_station'].min():.3f} – {cat['mean_auc_per_station'].max():.3f}")

    lines.append(f"\n--- Skipped files ---")
    lines.append(f"  {len(skipped):,} files skipped")
    if skipped:
        for s in skipped[:20]:
            lines.append(f"    {s['file']}  →  {s['reason']}")

    lines.append(f"\n--- Events per year ---")
    yearly = cat.groupby("year").size()
    for yr, cnt in yearly.items():
        lines.append(f"  {yr}: {cnt:,}")

    report = "\n".join(lines)
    report_path = outdir / "qc_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"\nQC report saved → {report_path}")

    if skipped:
        skip_df = pd.DataFrame(skipped)
        skip_df.to_csv(outdir / "skipped_files.csv", index=False)

    return cat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build master Mt Rainier catalog")
    parser.add_argument(
        "--indir",
        default="pnw_seismic_event_detection/logs/mt_rainier_common_detections",
        help="Directory containing daily CSV files"
    )
    parser.add_argument(
        "--outdir",
        default="catalog_output",
        help="Output directory for master catalog and QC report"
    )
    args = parser.parse_args()
    build_catalog(args.indir, args.outdir)