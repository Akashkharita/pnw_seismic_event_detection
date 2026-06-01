"""
spearman_network_test.py
------------------------
Tests whether the observed event rate trends in the Mt Rainier catalog
are driven by network expansion or represent real geophysical signal.

Method (following Hibert et al. 2019, GJI, Fig. 5a):
  Compute the Spearman rank correlation coefficient (rs) between monthly
  event counts and monthly active station count using a SHRINKING window
  with a FIXED END and a MOVING START.

  i.e.:  rs(2010→2025), rs(2011→2025), rs(2012→2025), ... rs(2024→2025)

  The x-axis represents the START year of each window. As the start moves
  later, older (network-expansion-dominated) years are dropped. If rs
  falls below 0.5 when restricting to recent years, the trend in that
  recent period is decorrelated from station count — i.e. it cannot be
  explained by network growth alone.

  This is the opposite of the original (incorrect) expanding-window
  implementation, which fixed the start and moved the end, causing rs to
  perpetually rise toward 1.0 as more correlated history was accumulated.

Produces:
  - spearman_network_test.png  : main figure (3-panel)
  - spearman_results.csv       : rs and p-value per window-start, per class

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
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# ── paths ──────────────────────────────────────────────────────────────────────
SRC_DIR     = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
CATALOG_CSV = PROJECT_DIR / "src/catalog_output" / "master_catalog.csv"
OUTDIR      = PROJECT_DIR / "src/catalog_output"

# ── analysis boundary ──────────────────────────────────────────────────────────
ANALYSIS_START = pd.Timestamp("2018-01-01")

# ── minimum window length for a stable correlation estimate ───────────────────
#    Hibert et al. use annual resolution; we use monthly so 18 months is safe.
MIN_WINDOW_MONTHS = 18

# ── station list ───────────────────────────────────────────────────────────────
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


# ── 1. fetch station availability ──────────────────────────────────────────────
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


# ── 2. load catalog and build monthly counts ───────────────────────────────────
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

    total = (cat.groupby("month_ts")
               .size()
               .reset_index(name="n_all"))

    for cls in ["su", "eq", "px"]:
        cls_counts = (cat[cat["most_common_class"] == cls]
                      .groupby("month_ts")
                      .size()
                      .reset_index(name=f"n_{cls}"))
        total = total.merge(cls_counts, on="month_ts", how="left")

    total = total.fillna(0)
    print(f"  Monthly series: {len(total)} months, "
          f"{total['month_ts'].min().date()} → {total['month_ts'].max().date()}")
    return total


# ── 3. compute SHRINKING-window Spearman rs (Hibert et al. approach) ───────────
def compute_spearman_shrinking(monthly, availability):
    """
    Fixed END, moving START — matching Hibert et al. 2019 Fig. 5a.

    For each possible window start month s, compute Spearman rs between:
      - event counts   from month s to the catalog end
      - station counts from month s to the catalog end

    The result is indexed by the START of the window (x-axis). Reading the
    plot left-to-right means progressively dropping the oldest, most
    network-expansion-dominated years. If rs falls as the start moves into
    the 2020s, the recent-period trend is NOT explained by network growth.

    Only windows with >= MIN_WINDOW_MONTHS data points are computed.
    """
    merged = monthly.merge(availability, on="month_ts", how="inner")
    merged = merged.sort_values("month_ts").reset_index(drop=True)
    n_months = len(merged)

    results = []

    # i is the index of the window START — iterate from 0 to near the end
    for i in range(n_months - MIN_WINDOW_MONTHS):
        window   = merged.iloc[i:]          # everything from i to the END
        stations = window["n_active"].values
        row      = {"window_start": merged.loc[i, "month_ts"]}

        for cls_key, col in [("all","n_all"),("su","n_su"),
                              ("eq","n_eq"),("px","n_px")]:
            events = window[col].values
            if np.std(stations) < 0.01 or np.std(events) < 0.01:
                row[f"rs_{cls_key}"]   = np.nan
                row[f"pval_{cls_key}"] = np.nan
            else:
                rs, pval = stats.spearmanr(stations, events)
                row[f"rs_{cls_key}"]   = rs
                row[f"pval_{cls_key}"] = pval

        row["window_end"]    = merged["month_ts"].iloc[-1]
        row["window_months"] = len(window)
        results.append(row)

    results_df = pd.DataFrame(results)
    print(f"  Computed rs for {len(results_df)} window-start positions")
    return results_df, merged


# ── 4. plot ────────────────────────────────────────────────────────────────────
def plot_results(results_df, merged, outpath):
    """
    Panel A: Shrinking-window rs vs window START date.
             X-axis reads left (oldest start = longest window) to right
             (newest start = shortest, most recent window).
             Vertical dashed line marks the primary analysis window start
             (2018-01-01) — windows starting to its right use only the
             clean post-expansion period.

    Panel B: Raw monthly event counts by class (context).

    Panel C: Active station count over time (context).
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 12),
        gridspec_kw={"hspace": 0.42, "height_ratios": [2, 1.2, 1.2]}
    )

    analysis_start_num = mdates.date2num(ANALYSIS_START.to_pydatetime())

    # ── Panel A ────────────────────────────────────────────────────────────────
    ax = axes[0]

    for cls_key in ["all", "su", "eq", "px"]:
        col   = f"rs_{cls_key}"
        valid = results_df.dropna(subset=[col])
        ax.plot(valid["window_start"], valid[col],
                color=CLASS_COLORS[cls_key],
                lw=2.2 if cls_key == "all" else 1.5,
                ls="-"  if cls_key == "all" else "--",
                label=CLASS_LABELS[cls_key],
                zorder=3)

    # threshold reference lines
    ax.axhline(0.0, color="#555", lw=0.9, ls="-", alpha=0.5)   # zero line
    for thresh, label in [(0.7, "rs = 0.7"), (0.5, "rs = 0.5"), (0.3, "rs = 0.3")]:
        ax.axhline(thresh, color="#999", lw=0.9, ls=":", alpha=0.8)
        ax.text(results_df["window_start"].iloc[-1] + pd.Timedelta(days=25),
                thresh + 0.01, label, fontsize=8, color="#888", va="bottom")

    # shade pre-analysis-window region and mark the 2018 boundary
    x_min = mdates.date2num(results_df["window_start"].min().to_pydatetime())
    x_max = mdates.date2num(
        (results_df["window_start"].max() + pd.Timedelta(days=60)).to_pydatetime()
    )
    ax.axvspan(x_min, analysis_start_num,
               color="#dddddd", alpha=0.40, zorder=0,
               label="Pre-2018 (excluded from primary analysis)")
    ax.axvline(analysis_start_num, color="#333", lw=1.4, ls="--", zorder=5)

    # label the 2018 line
    ylim_a = (-0.35, 1.05)
    ax.text(analysis_start_num + (x_max - x_min) * 0.005,
            ylim_a[0] + (ylim_a[1] - ylim_a[0]) * 0.04,
            "Windows starting here use\nonly the primary analysis period →",
            fontsize=7.5, color="#333", style="italic")

    ax.set_ylabel("Spearman rs\n(events vs active stations)", fontsize=10)
    ax.set_title(
        "A  —  Spearman correlation: monthly event rate vs active station count\n"
        "       (shrinking window — fixed end, moving start; "
        "following Hibert et al. 2019)",
        fontweight="bold", fontsize=10.5, loc="left")
    ax.set_ylim(ylim_a)
    ax.set_xlim(x_min, x_max)
    ax.tick_params(labelsize=8)

    # ── custom x-axis: show "catalog_end_year − window_start_year" ────────────
    # Place a tick at every even year that appears as a window start,
    # and label it as "end_year − start_year" so readers see window length.
    catalog_end_year = results_df["window_end"].iloc[0].year
    # collect every-2-year tick positions that fall inside the data range
    start_years = sorted(results_df["window_start"].dt.year.unique())
    tick_years  = [y for y in start_years if y % 2 == 0]
    tick_dates  = [pd.Timestamp(f"{y}-01-01") for y in tick_years]
    tick_nums   = mdates.date2num([t.to_pydatetime() for t in tick_dates])
    tick_labels = [f"{catalog_end_year}−{y}" for y in tick_years]
    ax.set_xticks(tick_nums)
    ax.set_xticklabels(tick_labels, fontsize=8)

    ax.legend(fontsize=8, loc="upper right", framealpha=0.90)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlabel(
        f"Window length  (end year fixed at {catalog_end_year};  "
        "label = end year − start year)",
        fontsize=9)

    # interpretation box
    ax.text(0.01, 0.97,
            "Reading direction: left = long window (more history)\n"
            "                   right = short window (recent data only)\n"
            "rs > 0.7 → trend likely network-driven\n"
            "rs < 0.5 → trend likely real geophysical signal",
            transform=ax.transAxes, fontsize=7.5,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#ccc", alpha=0.92))

    # ── Panel B: raw monthly event counts ─────────────────────────────────────
    ax = axes[1]
    for cls_key, col in [("su", "n_su"), ("eq", "n_eq"), ("px", "n_px")]:
        ax.plot(merged["month_ts"], merged[col],
                color=CLASS_COLORS[cls_key], lw=1.0, alpha=0.85,
                label=CLASS_LABELS[cls_key])

    x_min_b = mdates.date2num(merged["month_ts"].min().to_pydatetime())
    x_max_b = mdates.date2num(merged["month_ts"].max().to_pydatetime())
    ax.axvspan(x_min_b, analysis_start_num,
               color="#dddddd", alpha=0.40, zorder=0)
    ax.axvline(analysis_start_num, color="#333", lw=1.4, ls="--", zorder=5)
    ax.set_ylabel("Events per month", fontsize=10)
    ax.set_title("B  —  Monthly event counts by class (raw)",
                 fontweight="bold", fontsize=10.5, loc="left")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(x_min_b, x_max_b)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    # ── Panel C: active station count ─────────────────────────────────────────
    ax = axes[2]
    ax.fill_between(merged["month_ts"], merged["n_active"],
                    color="#E65100", alpha=0.22, zorder=2)
    ax.plot(merged["month_ts"], merged["n_active"],
            color="#E65100", lw=1.8, zorder=3,
            label="Active stations (FDSN)")
    ax.axhline(len(STATIONS), color="#E65100", lw=0.8, ls=":",
               alpha=0.6, label=f"Maximum ({len(STATIONS)} stations)")

    ax.axvspan(x_min_b, analysis_start_num,
               color="#dddddd", alpha=0.40, zorder=0)
    ax.axvline(analysis_start_num, color="#333", lw=1.4, ls="--", zorder=5)
    ax.set_ylabel("Active stations", fontsize=10)
    ax.set_title("C  —  Active station count over time (FDSN)",
                 fontweight="bold", fontsize=10.5, loc="left")
    ax.set_ylim(0, len(STATIONS) * 1.15)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(x_min_b, x_max_b)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    fig.suptitle(
        "Mt Rainier QuakeXNet Catalog — Network Expansion Bias Test\n"
        "Following Hibert et al. (2019, GJI) methodology",
        fontsize=12, fontweight="bold", y=1.01
    )

    fig.savefig(outpath, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Figure saved → {outpath}")


# ── 5. print summary statistics ────────────────────────────────────────────────
def print_summary(results_df):
    print("\n" + "=" * 65)
    print("SPEARMAN CORRELATION SUMMARY  (shrinking window, fixed end)")
    print("=" * 65)
    print(f"  Catalog end (fixed):  {results_df['window_end'].iloc[0].date()}")
    print()

    # find the rs values for specific scientifically meaningful window starts
    key_starts = {
        "Full catalog (from ~2010)":  results_df["window_start"].min(),
        "From 2015":                  pd.Timestamp("2015-01-01"),
        "From 2018 (primary window)": pd.Timestamp("2018-01-01"),
        "From 2020":                  pd.Timestamp("2020-01-01"),
        "From 2022":                  pd.Timestamp("2022-01-01"),
    }

    for cls_key in ["all", "su", "eq", "px"]:
        col = f"rs_{cls_key}"
        print(f"\n{CLASS_LABELS[cls_key].upper()}")
        for label, ts in key_starts.items():
            # find the closest available window start
            row = results_df.iloc[
                (results_df["window_start"] - ts).abs().argsort()[:1]
            ]
            if row.empty or pd.isna(row[col].values[0]):
                print(f"  {label:<38s}: n/a")
                continue
            rs_val   = row[col].values[0]
            pval     = row[f"pval_{cls_key}"].values[0]
            n_months = row["window_months"].values[0]

            if rs_val < 0.3:
                interp = "VERY WEAK — not network-driven"
            elif rs_val < 0.5:
                interp = "WEAK — trend likely real"
            elif rs_val < 0.7:
                interp = "MODERATE — partial network influence"
            else:
                interp = "STRONG — possible network bias"

            print(f"  {label:<38s}: rs={rs_val:+.3f}  "
                  f"p={pval:.3f}  n={n_months}mo  → {interp}")

    print("\n" + "=" * 65)
    print("KEY INTERPRETATION NOTE:")
    print("  Unlike an expanding window (which always drifts toward rs→1),")
    print("  the shrinking window isolates whether the RECENT period alone")
    print("  is correlated with station count. If rs falls below 0.5 for")
    print("  windows starting after ~2020, the post-2020 trend cannot be")
    print("  attributed to network expansion. This matches Hibert et al.")
    print("  (2019) who found rs < 0.5 for windows starting after 2005.")
    print("=" * 65)


# ── 6. main ────────────────────────────────────────────────────────────────────
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not CATALOG_CSV.exists():
        print(f"ERROR: catalog not found at {CATALOG_CSV}")
        return

    monthly      = load_monthly_counts(CATALOG_CSV)
    date_min     = monthly["month_ts"].min()
    date_max     = monthly["month_ts"].max()
    availability = fetch_station_availability(date_min, date_max)

    print("\nComputing Spearman shrinking-window correlations...")
    results_df, merged = compute_spearman_shrinking(monthly, availability)

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