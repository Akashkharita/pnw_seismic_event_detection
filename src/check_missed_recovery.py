"""
check_missed_recovery.py
========================
Checks how many "missed" PNSN events are now recovered in the v3
clustered catalog, compared to the original v1 catalog.

This is different from check_missed_events.py — instead of scanning
raw station files, it matches against the already-clustered v3 CSVs.

Usage
-----
    python check_missed_recovery.py

Outputs
-------
    recovery_report.csv   — one row per missed PNSN event with:
                            - was it found in v1 catalog?
                            - was it found in v3 catalog?
                            - what class did v3 assign?
                            - how many stations detected it in v3?
"""

import os
import glob
import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
MISSED_CSV      = "../data/missed_su_px_events_by_model.csv"          # from earlier analysis
V3_CATALOG_DIR  = "../logs/mt_rainier_common_detections_v3"   # output of v3 bash run
V1_CATALOG_DIR  = "../logs/mt_rainier_common_detections"      # original v1 output (optional)
TOL_S           = 60     # seconds tolerance for matching
WIN_S           = 100    # detection window length
# ─────────────────────────────────────────────────────────────────────────────


def load_catalog(catalog_dir, date_range=None):
    """Load all common-event CSVs from a directory into one DataFrame."""
    pattern = os.path.join(catalog_dir, "common_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No catalog files found in: {catalog_dir}")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    cat = pd.concat(dfs, ignore_index=True)
    cat["rounded_start"] = pd.to_datetime(cat["rounded_start"], utc=True, format='mixed')
    cat["t_ns"] = cat["rounded_start"].values.astype("int64")
    return cat


def find_in_catalog(pnsn_t_ns, catalog_t_sorted, tol_ns, win_ns):
    """
    Check if a PNSN event (at pnsn_t_ns) falls within any catalog
    detection window [cat_start - tol, cat_start + win + tol].
    Returns (matched, matching_indices).
    """
    lo = pnsn_t_ns - win_ns - tol_ns
    hi = pnsn_t_ns + tol_ns
    idx_lo = np.searchsorted(catalog_t_sorted, lo)
    idx_hi = np.searchsorted(catalog_t_sorted, hi, side="right")
    return idx_hi > idx_lo, (idx_lo, idx_hi)


def get_match_details(pnsn_t_ns, catalog_df, catalog_t_sorted, tol_ns, win_ns):
    """Get details of the best matching cluster for a PNSN event."""
    lo = pnsn_t_ns - win_ns - tol_ns
    hi = pnsn_t_ns + tol_ns
    idx_lo = np.searchsorted(catalog_t_sorted, lo)
    idx_hi = np.searchsorted(catalog_t_sorted, hi, side="right")
    
    if idx_hi <= idx_lo:
        return None
    
    matches = catalog_df.iloc[idx_lo:idx_hi]
    # Pick the one with highest num_stations (most confident)
    best = matches.loc[matches["num_stations"].idxmax()]
    return best


def main():
    print(f"Loading missed events from: {MISSED_CSV}")
    missed = pd.read_csv(MISSED_CSV)
    missed["orig_dt_utc"] = pd.to_datetime(missed["orig_time_true"], format="mixed", utc=True)
    missed["t_ns"] = missed["orig_dt_utc"].values.astype("int64")  #i→ nanoseconds
    print(f"  {len(missed)} missed PNSN events")

    tol_ns = int(TOL_S * 1e9)
    win_ns = int(WIN_S * 1e9)

    # Load v3 catalog
    print(f"\nLoading v3 catalog from: {V3_CATALOG_DIR}")
    try:
        v3 = load_catalog(V3_CATALOG_DIR)
        v3_sorted = np.sort(v3["t_ns"].values)
        print(f"  {len(v3)} clusters in v3 catalog")
        print(f"  Date range: {v3['rounded_start'].min()} → {v3['rounded_start'].max()}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Run the v3 bash script first to generate the catalog.")
        return

    # Load v1 catalog (optional, for comparison)
    v1 = None
    v1_sorted = None
    if os.path.isdir(V1_CATALOG_DIR):
        print(f"\nLoading v1 catalog from: {V1_CATALOG_DIR}")
        try:
            v1 = load_catalog(V1_CATALOG_DIR)
            v1_sorted = np.sort(v1["t_ns"].values)
            print(f"  {len(v1)} clusters in v1 catalog")
        except Exception as e:
            print(f"  Warning: could not load v1 catalog: {e}")

    # ── MATCH EACH MISSED EVENT ───────────────────────────────────────────────
    print("\nMatching missed events against catalogs...")
    results = []

    for _, row in missed.iterrows():
        t_ns = int(row["t_ns"])
        evid  = row["evid"]
        etype = row["etype"]

        # Check v3
        v3_matched, v3_idx = find_in_catalog(t_ns, v3_sorted, tol_ns, win_ns)
        v3_details = get_match_details(t_ns, v3, v3_sorted, tol_ns, win_ns) if v3_matched else None

        # Check v1
        v1_matched = False
        if v1_sorted is not None:
            v1_matched, _ = find_in_catalog(t_ns, v1_sorted, tol_ns, win_ns)

        result = {
            "evid":              evid,
            "etype":             etype,
            "orig_time":         row["orig_time_true"],
            "dist_km":           row.get("dist_km", np.nan),
            "year":              row.get("year", np.nan),
            "month":             row.get("month", np.nan),
            "in_v1":             v1_matched,
            "in_v3":             v3_matched,
            "recovered_by_v3":   v3_matched and not v1_matched,
            "v3_class":          v3_details["most_common_class"] if v3_matched else "none",
            "v3_num_stations":   int(v3_details["num_stations"]) if v3_matched else 0,
            "v3_vote_margin":    float(v3_details["vote_margin"]) if v3_matched else np.nan,
            "v3_class_votes":    str(v3_details["class_votes"]) if v3_matched else "",
            "pnsn_agrees":       (v3_details["most_common_class"] == etype) if v3_matched else False,
        }
        results.append(result)

    df = pd.DataFrame(results)

    # ── PRINT SUMMARY ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RECOVERY REPORT")
    print("="*60)

    for etype in ["su", "px"]:
        sub = df[df["etype"] == etype]
        if sub.empty:
            continue
        n = len(sub)
        in_v3 = sub["in_v3"].sum()
        in_v1 = sub["in_v1"].sum()
        recovered = sub["recovered_by_v3"].sum()
        agrees = sub[sub["in_v3"]]["pnsn_agrees"].sum()
        still_missed = (~sub["in_v3"]).sum()

        print(f"\n  {etype.upper()} ({n} total missed events):")
        print(f"    Found in v1 (sanity check, should be ~0): {in_v1}")
        print(f"    Found in v3:                              {in_v3}  ({in_v3/n*100:.1f}%)")
        print(f"    Newly recovered by v3:                    {recovered}  ({recovered/n*100:.1f}%)")
        print(f"    Still missed in v3:                       {still_missed}  ({still_missed/n*100:.1f}%)")
        if in_v3 > 0:
            print(f"    V3 class agrees with PNSN:                {agrees}/{in_v3}  ({agrees/in_v3*100:.1f}%)")
        
        print(f"\n    V3 class assigned to recovered {etype.upper()} events:")
        print(sub[sub["in_v3"]]["v3_class"].value_counts().to_string())

    print("\n  Still missed in v3 (truly undetected):")
    still_missed_df = df[~df["in_v3"]]
    print(f"    Total: {len(still_missed_df)}")
    print(f"    By type: {still_missed_df['etype'].value_counts().to_dict()}")
    if not still_missed_df.empty:
        print(f"    Dist stats (km):")
        print(still_missed_df["dist_km"].describe().to_string())

    # ── SAVE ──────────────────────────────────────────────────────────────────
    out = "recovery_report.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved full report → {out}")
    print("Done.")


if __name__ == "__main__":
    main()