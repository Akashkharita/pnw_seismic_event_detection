"""
check_missed_events_v2.py
=========================
Fixed version of check_missed_events.py.

The bug in v1: the 220s search window (WIN_S + 2*TOL_S) on a busy
seismic day finds detections from OTHER events, not the missed one.
This caused 80%+ false positives in the n_stations count.

The fix: after finding candidate detections near the PNSN origin time,
verify that multiple detections are MUTUALLY CLOSE in time (within
GAP_S of each other) — i.e., they could plausibly be from the same
physical event. This mirrors what the clustering script does.

Usage
-----
    python check_missed_events_v2.py

Outputs
-------
    missed_events_station_lookup_v2.csv
    missed_events_station_detail_v2.csv
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import timedelta

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_PATH   = "../logs/mt_rainier_detections"
MISSED_CSV  = "../data/missed_su_px_events_by_model.csv"
TOL_S       = 60     # tolerance around PNSN origin time
WIN_S       = 100    # QuakeXNet detection window length
GAP_S       = 20     # max time gap between detections to be co-associated
# ─────────────────────────────────────────────────────────────────────────────


def folder_for_date(base, dt):
    date_str = dt.strftime("%Y%m%d")
    return os.path.join(base, f"{date_str}_0000_{date_str}_2359")


def parse_time(t_str):
    return pd.Timestamp(str(t_str).rstrip("Z"), tz="UTC")


def load_day_detections(base, origin_utc):
    """Load all station detections for the day of origin_utc."""
    dates_to_check = [origin_utc.date()]
    if origin_utc.hour == 0 and origin_utc.minute < 2:
        dates_to_check.append((origin_utc - timedelta(days=1)).date())

    all_dfs = []
    for date in dates_to_check:
        folder = folder_for_date(base, pd.Timestamp(date))
        if not os.path.isdir(folder):
            continue
        for fpath in glob.glob(os.path.join(folder, "*_events.csv")):
            try:
                df = pd.read_csv(fpath)
                if not df.empty and "start_time" in df.columns:
                    all_dfs.append(df)
            except Exception:
                continue

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def find_co_detections(origin_utc, day_df, tol_s=TOL_S, win_s=WIN_S, gap_s=GAP_S):
    """
    Find detections that:
    1. Fall within [origin - tol, origin + win + tol] (candidate window)
    2. Are mutually within gap_s seconds of at least one other detection
       (co-association check — filters out detections from other events)

    Returns a filtered DataFrame of detections likely from the same event
    as the PNSN origin time.
    """
    if day_df.empty:
        return pd.DataFrame()

    day_df = day_df.copy()
    day_df["_start"] = day_df["start_time"].apply(parse_time)

    lo = origin_utc - timedelta(seconds=tol_s)
    hi = origin_utc + timedelta(seconds=win_s + tol_s)

    # Step 1: narrow to candidate window
    candidates = day_df[(day_df["_start"] >= lo) & (day_df["_start"] <= hi)].copy()
    if candidates.empty:
        return pd.DataFrame()

    # Step 2: co-association filter
    # Keep only detections that have at least one OTHER detection within gap_s
    # (This removes isolated detections from unrelated events)
    starts = candidates["_start"].values
    n = len(starts)

    if n == 1:
        # Single detection — can't verify co-association, treat as 1-station
        return candidates

    keep = np.zeros(n, dtype=bool)
    starts_ns = starts.astype("datetime64[ns]").astype("int64")

    for i in range(n):
        diffs = np.abs(starts_ns - starts_ns[i]) / 1e9  # seconds
        diffs[i] = np.inf  # exclude self
        if diffs.min() <= (win_s + gap_s):
            keep[i] = True

    co_associated = candidates[keep].copy()

    # Step 3: keep best detection per station
    if not co_associated.empty and "max_prob" in co_associated.columns:
        co_associated = (
            co_associated
            .sort_values("max_prob", ascending=False)
            .drop_duplicates(subset=["station"], keep="first")
        )

    return co_associated


def summarise(det_df):
    if det_df.empty:
        return {
            "n_stations": 0, "stations": "", "ml_classes": "",
            "dominant_class": "none", "mean_max_prob": np.nan,
            "max_max_prob": np.nan, "class_votes": ""
        }
    stations  = sorted(det_df["station"].unique()) if "station" in det_df.columns else []
    classes   = det_df["class"].tolist() if "class" in det_df.columns else []
    counts    = pd.Series(classes).value_counts()
    dominant  = counts.idxmax() if len(counts) else "unknown"
    max_probs = det_df["max_prob"] if "max_prob" in det_df.columns else pd.Series(dtype=float)
    return {
        "n_stations":     len(stations),
        "stations":       "|".join(stations),
        "ml_classes":     "|".join(sorted(set(classes))),
        "dominant_class": dominant,
        "mean_max_prob":  float(max_probs.mean()) if not max_probs.empty else np.nan,
        "max_max_prob":   float(max_probs.max())  if not max_probs.empty else np.nan,
        "class_votes":    str(counts.to_dict()),
    }


def main():
    print(f"Loading missed events from: {MISSED_CSV}")
    missed = pd.read_csv(MISSED_CSV)
    missed["orig_dt_utc"] = pd.to_datetime(missed["orig_time_true"], format="mixed", utc=True)
    print(f"  {len(missed)} missed events")
    print(f"  Base path: {os.path.abspath(BASE_PATH)}")
    print(f"  Tolerance: ±{TOL_S}s | window: {WIN_S}s | co-assoc gap: {GAP_S}s\n")

    summary_rows = []
    detail_rows  = []
    _cache = {}  # cache day detections to avoid re-reading same files

    for idx, row in missed.iterrows():
        origin = row["orig_dt_utc"]
        evid   = row["evid"]
        etype  = row["etype"]
        date_key = str(origin.date())

        # Load day detections (cached)
        if date_key not in _cache:
            _cache[date_key] = load_day_detections(BASE_PATH, origin)
        day_df = _cache[date_key]

        det = find_co_detections(origin, day_df)
        summ = summarise(det)

        summary_rows.append({
            "evid": evid, "etype": etype,
            "orig_time": str(origin),
            "dist_km": row.get("dist_km", np.nan),
            "year": row.get("year", np.nan),
            "month": row.get("month", np.nan),
            "hour": row.get("hour", np.nan),
            **summ
        })

        if not det.empty:
            det = det.copy()
            det["evid"]  = evid
            det["etype"] = etype
            detail_rows.append(det)

        n_sta = summ["n_stations"]
        tag   = "✓" if n_sta >= 1 else "✗"
        if (idx + 1) % 25 == 0 or idx == 0:
            print(f"  [{idx+1:>3}/{len(missed)}] evid={evid} etype={etype} "
                  f"n_stations={n_sta} dominant={summ['dominant_class']} {tag}")

    summary_df = pd.DataFrame(summary_rows)

    # Save outputs
    out_summary = "missed_events_station_lookup_v2.csv"
    summary_df.to_csv(out_summary, index=False)
    print(f"\nSaved summary → {out_summary}")

    if detail_rows:
        detail_df = pd.concat(detail_rows, ignore_index=True)
        drop_cols = [c for c in detail_df.columns if c.startswith("_")]
        detail_df.drop(columns=drop_cols, errors="ignore").to_csv(
            "missed_events_station_detail_v2.csv", index=False)
        print(f"Saved detail  → missed_events_station_detail_v2.csv")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("RESULTS SUMMARY (with co-association filter)")
    print("="*55)

    for etype in ["su", "px"]:
        sub = summary_df[summary_df["etype"] == etype]
        if sub.empty:
            continue
        print(f"\n  {etype.upper()} missed events: {len(sub)}")
        for n in [0, 1, 2]:
            cnt = (sub["n_stations"] == n).sum()
            print(f"    {n} stations: {cnt:>4}  ({cnt/len(sub)*100:.1f}%)")
        cnt3 = (sub["n_stations"] >= 3).sum()
        print(f"    ≥3 stations: {cnt3:>4}  ({cnt3/len(sub)*100:.1f}%)")

    print("\n  Dominant ML class for co-associated SU detections (1+ stations):")
    su_any = summary_df[(summary_df["etype"]=="su") & (summary_df["n_stations"]>=1)]
    if not su_any.empty:
        print(su_any["dominant_class"].value_counts().to_string())

    print("\n  Dominant ML class for co-associated PX detections (1+ stations):")
    px_any = summary_df[(summary_df["etype"]=="px") & (summary_df["n_stations"]>=1)]
    if not px_any.empty:
        print(px_any["dominant_class"].value_counts().to_string())

    print("\n  Year breakdown of 0-station misses (SU):")
    su_0 = summary_df[(summary_df["etype"]=="su") & (summary_df["n_stations"]==0)]
    if not su_0.empty:
        print(su_0.groupby("year").size().to_string())

    print("\n  Distance breakdown of 0-station misses:")
    zero_sta = summary_df[summary_df["n_stations"]==0]
    if not zero_sta.empty:
        bins = [0, 3, 10, 20, 30, 50, 70, 100]
        zero_sta = zero_sta.copy()
        zero_sta["dist_bin"] = pd.cut(zero_sta["dist_km"], bins)
        print(zero_sta.groupby(["dist_bin","etype"]).size().unstack(fill_value=0).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()