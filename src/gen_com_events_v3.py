"""
gen_com_events_v3.py
====================
Correct multi-station event clustering.

The bug in v1: gap clustering across all stations in sorted order causes
detections from DIFFERENT events to merge if they happen to be adjacent
in the sorted list.

The bug in v2: Union-Find with overlap causes transitive over-merging —
A overlaps B, B overlaps C → A,B,C merged even if A and C are far apart.

The fix here: ANCHOR-BASED clustering.
  1. Find all detections within a rolling window centered on the earliest
     unassigned detection (the "anchor").
  2. All detections within [anchor_start - gap_s, anchor_start + WIN_S + gap_s]
     form one candidate cluster.
  3. Keep only one detection per station (strongest max_prob).
  4. The cluster is closed — move the anchor forward past the cluster end.
  5. This prevents chaining because the window is fixed to the anchor,
     not expanded by each new member.

This matches the physical reality: a seismic event has a finite duration
(~100s window), and all stations detecting it should fire within that window.

Usage (drop-in replacement)
---------------------------
    python3 gen_com_events_v3.py --start 2020-01-15T00:00:00 --end   2020-01-15T23:59:59 --input_dir  logs/mt_rainier_detections/20200115_0000_20200115_2359 --output_dir logs/mt_rainier_common_detections_v3 --min_stations 3 --gap_seconds  20
"""

import os
import argparse
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--start",        type=str, required=True)
parser.add_argument("--end",          type=str, required=True)
parser.add_argument("--input_dir",    type=str, required=True)
parser.add_argument("--output_dir",   type=str, required=True)
parser.add_argument("--min_stations", type=int,   default=3)
parser.add_argument("--gap_seconds",  type=float, default=20.0)
args = parser.parse_args()

user_start = datetime.strptime(args.start, "%Y-%m-%dT%H:%M:%S")
user_end   = datetime.strptime(args.end,   "%Y-%m-%dT%H:%M:%S")
file_start = user_start.strftime("%Y%m%d_%H%M")
file_end   = user_end.strftime("%Y%m%d_%H%M")
WIN_S      = 100.0   # QuakeXNet detection window length (fixed)
GAP_S      = args.gap_seconds


# ── LOAD FILES ────────────────────────────────────────────────────────────────
search_pattern = os.path.join(args.input_dir, "*.csv")
matched_files  = glob(search_pattern)

if not matched_files:
    raise FileNotFoundError(f"No CSV files found.\nSearched: {search_pattern}")

print(f"Found {len(matched_files)} station files")
dfs = []
for f in matched_files:
    try:
        df = pd.read_csv(f)
        if not df.empty:
            dfs.append(df)
    except Exception as e:
        print(f"  Warning: could not read {f}: {e}")

if not dfs:
    print("No detections — nothing to cluster.")
    exit(0)

df_all = pd.concat(dfs, ignore_index=True)
print(f"Total detections loaded: {len(df_all)}")


# ── PARSE TIMES ───────────────────────────────────────────────────────────────
df_all["start_time"] = pd.to_datetime(
    df_all["start_time"].astype(str).str.rstrip("Z"), utc=True
)
if "end_time" in df_all.columns:
    df_all["end_time"] = pd.to_datetime(
        df_all["end_time"].astype(str).str.rstrip("Z"), utc=True
    )
else:
    df_all["end_time"] = df_all["start_time"] + pd.Timedelta(seconds=WIN_S)

# Work in float seconds from day start for speed
t0 = df_all["start_time"].min().normalize()  # midnight
df_all["t_start_s"] = (df_all["start_time"] - t0).dt.total_seconds()
df_all["t_end_s"]   = (df_all["end_time"]   - t0).dt.total_seconds()

# Keep strongest detection per station per ~WIN_S bin upfront to reduce noise
# (prevents one chatty station from dominating)
df_all = (
    df_all
    .sort_values("max_prob", ascending=False)
    .reset_index(drop=True)
)


# ── ANCHOR-BASED CLUSTERING ───────────────────────────────────────────────────
def anchor_cluster(df, gap_s, win_s):
    """
    Sweep through detections sorted by start time.
    The first unassigned detection becomes the cluster anchor.
    All detections with start_time within [anchor_start - gap_s,
    anchor_start + win_s + gap_s] join this cluster.
    The anchor window does NOT expand as new members are added.
    """
    df_sorted = df.sort_values("t_start_s").reset_index(drop=True)
    starts = df_sorted["t_start_s"].values
    n      = len(starts)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0

    i = 0
    while i < n:
        if labels[i] != -1:
            i += 1
            continue

        # This detection is the anchor
        anchor_s = starts[i]
        lo = anchor_s - gap_s
        hi = anchor_s + win_s + gap_s

        # Find all detections within this fixed window
        # (binary search for efficiency)
        j_lo = np.searchsorted(starts, lo)
        j_hi = np.searchsorted(starts, hi, side="right")

        members = list(range(j_lo, j_hi))
        for m in members:
            labels[m] = cluster_id

        cluster_id += 1
        # Advance i past the end of this cluster window
        i = j_hi

    df_sorted["cluster_id"] = labels
    return df_sorted


print("Clustering detections (anchor-based)...")
df_all = anchor_cluster(df_all, GAP_S, WIN_S)
n_raw = df_all["cluster_id"].nunique()
print(f"  {n_raw} raw clusters found")


# ── ONE DETECTION PER STATION PER CLUSTER ────────────────────────────────────
df_all = (
    df_all
    .sort_values("max_prob", ascending=False)
    .drop_duplicates(subset=["cluster_id", "station"], keep="first")
)


# ── AGGREGATE ────────────────────────────────────────────────────────────────
def class_votes(x):
    return str(x.value_counts().to_dict())

def frac(cls):
    def _f(x):
        return (x == cls).sum() / len(x) if len(x) else 0.0
    return _f

def is_ambiguous(votes_str):
    import ast
    try:
        d = ast.literal_eval(votes_str)
        counts = sorted(d.values(), reverse=True)
        return len(counts) >= 2 and (counts[0] - counts[1]) <= 1
    except Exception:
        return False

grouped = df_all.groupby("cluster_id").agg(
    rounded_start     = ("start_time",  "min"),
    num_stations      = ("station",     "nunique"),
    stations          = ("station",     lambda x: list(sorted(set(x)))),
    all_classes       = ("class",       list),
    most_common_class = ("class",       lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown"),
    mean_auc          = ("auc",         "mean"),
    mean_max          = ("max_prob",    "mean"),
    mean_prob         = ("mean_prob",   "mean"),
    class_votes       = ("class",       class_votes),
    frac_eq           = ("class",       frac("eq")),
    frac_su           = ("class",       frac("su")),
    frac_px           = ("class",       frac("px")),
    frac_noise        = ("class",       frac("no")),
    unanimous         = ("class",       lambda x: x.nunique() == 1),
    vote_margin       = ("class",       lambda x: x.value_counts().iloc[0] / len(x) if len(x) else np.nan),
).reset_index()

grouped["ambiguous_class"] = grouped["class_votes"].apply(is_ambiguous)
grouped["high_confidence"]  = (grouped["vote_margin"] >= 0.75) & (~grouped["ambiguous_class"])
grouped["low_confidence"]   = grouped["vote_margin"] < 0.5
grouped["auc_is_summed"]    = False  # consistent with master catalog schema


# ── FILTER ────────────────────────────────────────────────────────────────────
common_events = grouped[grouped["num_stations"] >= args.min_stations].copy()
two_plus      = grouped[grouped["num_stations"] >= 2].copy()

print(f"\nAll clusters:          {len(grouped):>6}")
print(f"≥2 stations:           {len(two_plus):>6}")
print(f"≥{args.min_stations} stations:           {len(common_events):>6}")
if len(common_events):
    print("  class breakdown:")
    print(common_events["most_common_class"].value_counts().to_string())


# ── SANITY CHECK vs V1 OUTPUT ─────────────────────────────────────────────────
# Print first 5 clusters so you can visually compare with v1
print("\nFirst 5 clusters (≥3 stations):")
cols_show = ["cluster_id","rounded_start","num_stations","stations","most_common_class","vote_margin"]
print(common_events[cols_show].head().to_string(index=False))


# ── SAVE ──────────────────────────────────────────────────────────────────────
os.makedirs(args.output_dir, exist_ok=True)

out_primary = os.path.join(
    args.output_dir,
    f"common_{file_start}_to_{file_end}_events.csv"
)
out_two = os.path.join(
    args.output_dir,
    f"common2_{file_start}_to_{file_end}_events.csv"
)

common_events.to_csv(out_primary, index=False)
two_plus.to_csv(out_two, index=False)

print(f"\nSaved (≥{args.min_stations} stations) → {out_primary}")
print(f"Saved (≥2 stations)   → {out_two}")
print("Done.")