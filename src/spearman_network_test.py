"""
spearman_network_test.py
------------------------
Tests whether the observed event rate trends in the Mt Rainier catalog
are driven by network expansion or represent real geophysical signal.

Method (following Hibert et al. 2019, GJI):
  Compute the Spearman rank correlation coefficient (rs) between monthly
  event counts and monthly active station count on an expanding window.
  If rs drops below 0.5 in the primary analysis window (2018-2026), the
  trend cannot be explained by network growth alone.

Produces:
  - spearman_network_test.png  : main figure for committee presentation
  - spearman_results.csv       : monthly rs values for all classes

Usage (run from ANY directory):
    python /home/ak287/pnw_seismic_event_detection/src/spearman_network_test.py
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR     = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
CATALOG_CSV = PROJECT_DIR / "src/catalog_output" / "master_catalog.csv"
OUTDIR      = PROJECT_DIR / "src/catalog_output"

# ── analysis boundary ─────────────────────────────────────────────────────────
ANALYSIS_START    = pd.Timestamp("2018-01-01")
ANALYSIS_START_MD = mdates.date2num(ANALYSIS_START.to_pydatetime())

# ── minimum window length for a stable correlation estimate ───────────────────
MIN_WINDOW_MONTHS = 12

# ── station list ──────────────────────────────────────────────────────────────
STATIONS = [
    {"net": "CC", "sta": "ARAT"},  {"net": "CC", "sta": "CARB"},
    {"net": "CC", "sta": "COPP"},  {"net": "CC", "sta": "CRBN"},
    {"net": "CC", "sta": "CRYS"},  {"net": "CC", "sta": "GNOB"},
    {"net": "CC", "sta": "GOBB"},  {"net": "CC", "sta": "GTWY"},
    {"net": "CC", "sta": "KAUT"},  {"net": "CC", "sta": "KAVK"},
    {"net": "CC", "sta": "LONE"},  {"net": "CC", "sta": "LONR"},
    {"net": "CC", "sta": "MILD"},  {"net": "CC", "sta": "OBSR"},
    {"net": "CC", "sta": "OPCH"},  {"net": "CC", "sta": "PANH"},
    {"net": "CC", "sta": "PARA"},  {"net": "CC", "sta": "PR01"},
    {"net": "CC", "sta": "PR02"},  {"net": "CC", "sta": "PR03"},
    {"net": "CC", "sta": "PR04"},  {"net": "CC", "sta": "PR05"},
    {"net": "CC", "sta": "RUSH"},  {"net": "CC", "sta": "SIFT"},
    {"net": "CC", "sta": "TABR"},  {"net": "CC", "sta": "TAVI"},
    {"net": "CC", "sta": "VOIT"},  {"net": "CC", "sta": "WOW"},
    {"net": "UW", "sta": "FMW"},   {"net": "UW", "sta": "LO2"},
    {"net": "UW", "sta": "LON"},   {"net": "UW", "sta": "RCM"},
    {"net": "UW", "sta": "RCS"},   {"net": "UW", "sta": "RER"},
    {"net": "UW", "sta": "STAR"},
]

CLASS_COLORS = {
    "all": "#333333",
    "su":  "#2E86AB",
    "eq":  "#E84855",
    "px":  "#F4A261",
}
CLASS_LABELS = {
    "all": "All events",
    "su":  "Surface events",
    "eq":  "Earthquakes",
    "px":  "Explosions",
}


# ── 1. fetch station availability ─────────────────────────────────────────────
def fetch_station_availability(date_min, date_max):
    print("Fetching station availability from FDSN...")
    client = Client("IRIS")
    epochs = []

    for entry in STATIONS:
        net = entry["net"]
        sta = entry["sta"]
        try:
            inv = client.get_stations(
                network=net, station=sta, level="station",
                starttime=UTCDateTime("2009-01-01"),
                endtime=UTCDateTime("2026-12-31"),
            )
            for net_obj in inv:
                for sta_obj in net_obj:
                    s = sta_obj.start_date.datetime if sta_obj.start_date \
                        else datetime(2009, 1, 1)
                    e = sta_obj.end_date.datetime if sta_obj.end_date \
                        else datetime(2026, 12, 31)
                    epochs.append({"sta": sta, "start": s, "end": e})
        except Exception as ex:
            print(f"  WARNING — {net}.{sta}: {ex}")

    months = pd.date_range(
        start=pd.Timestamp(date_min).replace(day=1),
        end=pd.Timestamp(date_max).replace(day=1),
        freq="MS"
    )
    counts = []
    for month in months:
        month_end = month + pd.offsets.MonthEnd(1)
        n = sum(1 for ep in epochs
                if ep["start"] <= month_end.to_pydatetime()
                and ep["end"]  >= month.to_pydatetime())
        counts.append(n)

    availability = pd.DataFrame({"month_ts": months, "n_active": counts})
    print(f"  Active stations range: "
          f"{availability['n_active'].min()} – {availability['n_active'].max()}")
    return availability


# ── 2. load catalog and build monthly counts ──────────────────────────────────
def load_monthly_counts(catalog_csv):
    print(f"Loading catalog: {catalog_csv}")
    cat = pd.read_csv(catalog_csv)
    cat["rounded_start"] = pd.to_datetime(
        cat["rounded_start"], utc=True, errors="coerce", format="mixed"
    )
    cat = cat.dropna(subset=["rounded_start"])
    cat["month_ts"] = (cat["rounded_start"]
                       .dt.tz_localize(None)
                       .dt.to_period("M")
                       .dt.to_timestamp())

    # total events per month
    total = (cat.groupby("month_ts")
               .size()
               .reset_index(name="n_all"))

    # per-class events per month
    for cls in ["su", "eq", "px"]:
        cls_counts = (cat[cat["most_common_class"] == cls]
                      .groupby("month_ts")
                      .size()
                      .reset_index(name=f"n_{cls}"))
        total = total.merge(cls_counts, on="month_ts", how="left")

    total = total.fillna(0)
    print(f"  Monthly series: {len(total)} months")
    return total


# ── 3. compute expanding-window Spearman rs ───────────────────────────────────
def compute_spearman_expanding(monthly, availability):
    """
    For each month t, compute Spearman rs between:
      - event counts from month 0 to t
      - active station counts from month 0 to t
    Returns a DataFrame with rs and p-value per month per class.
    """
    # merge event counts with station counts on month_ts
    merged = monthly.merge(availability, on="month_ts", how="inner")
    merged = merged.sort_values("month_ts").reset_index(drop=True)

    results = []
    n_months = len(merged)

    for i in range(MIN_WINDOW_MONTHS, n_months):
        window   = merged.iloc[:i+1]
        stations = window["n_active"].values
        row      = {"month_ts": merged.loc[i, "month_ts"]}

        for cls_key, col in [("all","n_all"),("su","n_su"),
                              ("eq","n_eq"),("px","n_px")]:
            events = window[col].values
            # need variance in both series for a meaningful correlation
            if np.std(stations) < 0.01 or np.std(events) < 0.01:
                row[f"rs_{cls_key}"]    = np.nan
                row[f"pval_{cls_key}"]  = np.nan
            else:
                rs, pval = stats.spearmanr(stations, events)
                row[f"rs_{cls_key}"]   = rs
                row[f"pval_{cls_key}"] = pval

        results.append(row)

    results_df = pd.DataFrame(results)
    print(f"  Computed rs for {len(results_df)} monthly windows")
    return results_df, merged


# ── 4. plot ───────────────────────────────────────────────────────────────────
def plot_results(results_df, merged, outpath):
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 12),
        gridspec_kw={"hspace": 0.40, "height_ratios": [2, 1.2, 1.2]}
    )

    # ── Panel A: Spearman rs over time, one line per class ────────────────────
    ax = axes[0]

    for cls_key in ["all", "su", "eq", "px"]:
        col   = f"rs_{cls_key}"
        valid = results_df.dropna(subset=[col])
        ax.plot(valid["month_ts"], valid[col],
                color=CLASS_COLORS[cls_key],
                lw=2.0 if cls_key == "all" else 1.4,
                ls="-"  if cls_key == "all" else "--",
                label=CLASS_LABELS[cls_key],
                zorder=3)

    # reference lines
    ax.axhline(0.7, color="#888", lw=0.9, ls=":", alpha=0.8)
    ax.axhline(0.5, color="#888", lw=0.9, ls=":", alpha=0.8)
    ax.axhline(0.3, color="#888", lw=0.9, ls=":", alpha=0.8)
    ax.text(results_df["month_ts"].iloc[-1] + pd.Timedelta(days=30),
            0.71, "rs = 0.7", fontsize=8, color="#888", va="bottom")
    ax.text(results_df["month_ts"].iloc[-1] + pd.Timedelta(days=30),
            0.51, "rs = 0.5", fontsize=8, color="#888", va="bottom")
    ax.text(results_df["month_ts"].iloc[-1] + pd.Timedelta(days=30),
            0.31, "rs = 0.3", fontsize=8, color="#888", va="bottom")

    # analysis boundary
    xlim = ax.get_xlim()
    ax.axvspan(xlim[0], ANALYSIS_START_MD,
               color="#dddddd", alpha=0.40, zorder=0)
    ax.axvline(ANALYSIS_START_MD, color="#333", lw=1.4, ls="--", zorder=5)
    ylim = ax.get_ylim()
    ax.text(ANALYSIS_START_MD + (xlim[1]-xlim[0])*0.005,
            ylim[0] + (ylim[1]-ylim[0])*0.05,
            "Primary analysis window →",
            fontsize=8, color="#333", style="italic")

    ax.set_ylabel("Spearman rs\n(events vs active stations)", fontsize=10)
    ax.set_title(
        "A  —  Spearman correlation: monthly event rate vs active station count\n"
        "       (expanding window from catalog start)",
        fontweight="bold", fontsize=11, loc="left")
    ax.set_ylim(-0.1, 1.05)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(results_df["month_ts"].min(),
                results_df["month_ts"].max() + pd.Timedelta(days=60))

    # annotation box explaining interpretation
    ax.text(0.01, 0.97,
            "rs > 0.7 → trend likely network-driven\n"
            "rs < 0.5 → trend likely real geophysical signal",
            transform=ax.transAxes, fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#ccc", alpha=0.9))

    # ── Panel B: raw monthly event counts (context) ───────────────────────────
    ax = axes[1]
    for cls_key, col in [("su","n_su"),("eq","n_eq"),("px","n_px")]:
        ax.plot(merged["month_ts"], merged[col],
                color=CLASS_COLORS[cls_key], lw=1.0, alpha=0.8,
                label=CLASS_LABELS[cls_key])

    xlim = ax.get_xlim()
    ax.axvspan(xlim[0], ANALYSIS_START_MD,
               color="#dddddd", alpha=0.40, zorder=0)
    ax.axvline(ANALYSIS_START_MD, color="#333", lw=1.4, ls="--", zorder=5)
    ax.set_ylabel("Events per month", fontsize=10)
    ax.set_title("B  —  Monthly event counts by class (raw)",
                 fontweight="bold", fontsize=11, loc="left")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(merged["month_ts"].min(), merged["month_ts"].max())

    # ── Panel C: active station count (context) ───────────────────────────────
    ax = axes[2]
    ax.fill_between(merged["month_ts"], merged["n_active"],
                    color="#E65100", alpha=0.25, zorder=2)
    ax.plot(merged["month_ts"], merged["n_active"],
            color="#E65100", lw=1.8, zorder=3,
            label="Active stations (FDSN)")
    ax.axhline(len(STATIONS), color="#E65100", lw=0.8,
               ls=":", alpha=0.6,
               label=f"Maximum ({len(STATIONS)} stations)")

    xlim = ax.get_xlim()
    ax.axvspan(xlim[0], ANALYSIS_START_MD,
               color="#dddddd", alpha=0.40, zorder=0)
    ax.axvline(ANALYSIS_START_MD, color="#333", lw=1.4, ls="--", zorder=5)
    ax.set_ylabel("Active stations", fontsize=10)
    ax.set_title("C  —  Active station count over time (FDSN)",
                 fontweight="bold", fontsize=11, loc="left")
    ax.set_ylim(0, len(STATIONS) * 1.15)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(merged["month_ts"].min(), merged["month_ts"].max())

    fig.suptitle(
        "Mt Rainier QuakeXNet Catalog — Network Expansion Bias Test\n"
        "Following Hibert et al. (2019, GJI) methodology",
        fontsize=12, fontweight="bold", y=1.01
    )

    fig.savefig(outpath, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Figure saved → {outpath}")


# ── 5. print summary statistics ───────────────────────────────────────────────
def print_summary(results_df):
    print("\n" + "="*60)
    print("SPEARMAN CORRELATION SUMMARY")
    print("="*60)

    for cls_key in ["all", "su", "eq", "px"]:
        col = f"rs_{cls_key}"
        full   = results_df[col].dropna()
        pre    = results_df[results_df["month_ts"] <  ANALYSIS_START][col].dropna()
        post   = results_df[results_df["month_ts"] >= ANALYSIS_START][col].dropna()

        print(f"\n{CLASS_LABELS[cls_key].upper()}")
        print(f"  Full period rs      : {full.iloc[-1]:.3f}  (final value)")
        if len(pre)  > 0: print(f"  Pre-2018 rs (mean)  : {pre.mean():.3f}")
        if len(post) > 0: print(f"  Post-2018 rs (mean) : {post.mean():.3f}")

        if len(post) > 0:
            rs_post = post.mean()
            if rs_post < 0.3:
                interp = "VERY WEAK — post-2018 trends are NOT explained by network growth"
            elif rs_post < 0.5:
                interp = "WEAK — network growth is a minor factor post-2018"
            elif rs_post < 0.7:
                interp = "MODERATE — network growth partially explains the trend"
            else:
                interp = "STRONG — trends may still be driven by network growth"
            print(f"  Interpretation      : {interp}")

    print("\n" + "="*60)


# ── 6. main ───────────────────────────────────────────────────────────────────
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not CATALOG_CSV.exists():
        print(f"ERROR: catalog not found at {CATALOG_CSV}")
        return

    monthly      = load_monthly_counts(CATALOG_CSV)
    date_min     = monthly["month_ts"].min()
    date_max     = monthly["month_ts"].max()
    availability = fetch_station_availability(date_min, date_max)

    print("\nComputing Spearman expanding-window correlations...")
    results_df, merged = compute_spearman_expanding(monthly, availability)

    print_summary(results_df)

    outpath = OUTDIR / "spearman_network_test.png"
    print(f"\nBuilding figure...")
    plot_results(results_df, merged, outpath)

    csv_out = OUTDIR / "spearman_results.csv"
    results_df.to_csv(csv_out, index=False)
    print(f"  Results saved → {csv_out}")

    print(f"\nDone. Output at:\n  {outpath}")


if __name__ == "__main__":
    main()