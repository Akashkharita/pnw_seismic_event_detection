"""
make_completeness_dashboard.py
------------------------------
Figure 3: Catalog completeness dashboard — 4-panel matplotlib figure.

Panels:
  A. Events per month (full + high-confidence) with operational station
     count overlaid (derived from daily stations_*.json files).
  B. Missing days per year.
  C. mean_prob distribution by class.
  D. Monthly class proportions — stacked area.

Usage (run from src/ directory):
    python make_completeness_dashboard.py
"""

import json
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR          = Path(__file__).resolve().parent
PROJECT_DIR      = SRC_DIR.parent
CATALOG_CSV      = SRC_DIR / "catalog_output" / "master_catalog.csv"
STATION_JSON_DIR = PROJECT_DIR / "logs" / "mt_rainier_detections"
OUTDIR           = SRC_DIR / "catalog_output"

# ── analysis window ───────────────────────────────────────────────────────────
CATALOG_END = pd.Timestamp("2025-12-31 23:59:59")

# ── style ─────────────────────────────────────────────────────────────────────
CLASS_COLORS = {"su": "#2E86AB", "eq": "#E84855", "px": "#F4A261"}
CLASS_LABELS = {"su": "Surface event", "eq": "Earthquake", "px": "Explosion"}
HC_THRESHOLD = 0.5


# ── 1. load station counts from daily JSON files ──────────────────────────────
def load_station_counts(station_json_dir):
    """
    Scan all daily stations_YYYYMMDD_HHMM.json files inside
    per-day subdirectories and return a monthly DataFrame with
    columns [month_ts, n_stations_mean, n_stations_max].

    Counts unique physical station names per day, excluding the
    SY (synthetic) network which duplicates real station names.
    """
    pattern = str(Path(station_json_dir) / "*" / "stations_*.json")
    files   = sorted(glob.glob(pattern))
    print(f"Found {len(files)} station JSON files in {station_json_dir}")

    if not files:
        print("  WARNING: no station JSON files found — station count "
              "will be omitted from Panel A.")
        return None

    rows = []
    for fpath in files:
        m = re.search(r"stations_(\d{8})_\d{4}\.json", fpath)
        if not m:
            continue
        date = pd.to_datetime(m.group(1), format="%Y%m%d")

        # skip dates beyond analysis window
        if date > CATALOG_END:
            continue

        try:
            with open(fpath) as f:
                data = json.load(f)
            unique_stas = {
                entry["sta"]
                for entry in data
                if entry.get("net") != "SY"
            }
            rows.append({"date": date, "n_stations": len(unique_stas)})
        except Exception as e:
            print(f"  WARNING: could not read {fpath}: {e}")

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["month_ts"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby("month_ts").agg(
        n_stations_mean=("n_stations", "mean"),
        n_stations_max=("n_stations",  "max"),
    ).reset_index()

    print(f"  Date range   : {df['date'].min().date()} → "
          f"{df['date'].max().date()}")
    print(f"  Station range: {df['n_stations'].min()} – "
          f"{df['n_stations'].max()} unique stations/day")
    return monthly


# ── 2. load catalog ───────────────────────────────────────────────────────────
def load_catalog(path):
    print(f"\nLoading catalog: {path}")
    cat = pd.read_csv(path)
    cat["rounded_start"] = pd.to_datetime(
        cat["rounded_start"], utc=True, errors="coerce", format="mixed"
    )
    cat = cat.dropna(subset=["rounded_start"])

    # strip timezone for matplotlib compatibility
    cat["rounded_start_naive"] = cat["rounded_start"].dt.tz_localize(None)

    # cap at analysis end
    cat = cat[cat["rounded_start_naive"] <= CATALOG_END].copy()

    cat["date"]     = cat["rounded_start_naive"].dt.date
    cat["year"]     = cat["rounded_start_naive"].dt.year
    cat["month_ts"] = (cat["rounded_start_naive"]
                       .dt.to_period("M").dt.to_timestamp())
    cat["high_conf"] = cat["mean_prob"] >= HC_THRESHOLD

    print(f"  {len(cat):,} events  |  "
          f"high-conf: {cat['high_conf'].sum():,} "
          f"({100*cat['high_conf'].mean():.1f}%)")
    return cat


# ── 3. panel helpers ──────────────────────────────────────────────────────────
def panel_event_rate(ax, cat, monthly_stations):
    """Panel A: monthly event count + station count from JSON files."""

    monthly_full = cat.groupby("month_ts").agg(
        n_events=("rounded_start_naive", "count"),
    ).reset_index()

    monthly_hc = (cat[cat["high_conf"]]
                  .groupby("month_ts")
                  .agg(n_events_hc=("rounded_start_naive", "count"))
                  .reset_index())

    merged = (monthly_full
              .merge(monthly_hc, on="month_ts", how="left")
              .fillna(0))

    ax2 = ax.twinx()

    if monthly_stations is not None:
        ms = merged.merge(
            monthly_stations[["month_ts",
                               "n_stations_mean",
                               "n_stations_max"]],
            on="month_ts", how="left"
        ).fillna(0)

        ax2.plot(ms["month_ts"], ms["n_stations_mean"],
                 color="#E65100", lw=1.8, zorder=4,
                 label="Mean active stations (daily JSON)")
        ax2.fill_between(ms["month_ts"],
                         ms["n_stations_mean"],
                         ms["n_stations_max"],
                         color="#E65100", alpha=0.15, zorder=3,
                         label="Mean–Max range")
    else:
        ax2.set_visible(False)

    ax2.set_ylabel("Active stations (unique physical)",
                   color="#E65100", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#E65100", labelsize=8)
    ax2.set_ylim(0, 55)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(5))

    ax.bar(merged["month_ts"], merged["n_events"],
           width=25, color="#90CAF9", alpha=0.55,
           label="Full catalog", zorder=2)
    ax.bar(merged["month_ts"], merged["n_events_hc"],
           width=25, color="#1565C0", alpha=0.90,
           label=f"High conf. (prob ≥ {HC_THRESHOLD})", zorder=3)

    ax.set_ylabel("Events per month", fontsize=10)
    ax.set_title("A  —  Monthly event rate & station availability",
                 fontweight="bold", fontsize=11, loc="left")
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(merged["month_ts"].min(), merged["month_ts"].max())

    handles = [
        Patch(color="#90CAF9", alpha=0.55, label="Full catalog"),
        Patch(color="#1565C0", alpha=0.90,
              label=f"High conf. (prob ≥ {HC_THRESHOLD})"),
        Line2D([0], [0], color="#E65100", lw=1.8,
               label="Mean active stations"),
        Patch(color="#E65100", alpha=0.15, label="Mean–Max range"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="upper left",
              framealpha=0.85)


def panel_missing_days(ax, cat):
    """Panel B: missing days per year."""
    date_min  = cat["rounded_start_naive"].dt.date.min()
    date_max  = cat["rounded_start_naive"].dt.date.max()
    all_dates = pd.date_range(date_min, date_max, freq="D")

    present    = set(cat["date"])
    missing    = [d.date() for d in all_dates if d.date() not in present]
    missing_df = pd.DataFrame({"date": missing})

    if missing_df.empty:
        ax.text(0.5, 0.5, "No missing days", transform=ax.transAxes,
                ha="center", va="center", fontsize=12)
        ax.set_title("B  —  Data gaps per year",
                     fontweight="bold", fontsize=11, loc="left")
        return

    missing_df["year"] = pd.to_datetime(missing_df["date"]).dt.year

    per_year_missing = missing_df.groupby("year").size().reset_index(
        name="missing_days")
    per_year_total = (
        pd.DataFrame({"year": pd.Series(all_dates).dt.year.values})
        .groupby("year").size().reset_index(name="total_days")
    )

    merged = per_year_total.merge(
        per_year_missing, on="year", how="left").fillna(0)
    merged["pct_missing"] = (100 * merged["missing_days"]
                             / merged["total_days"])

    colors = ["#E53935" if p > 20 else "#FB8C00" if p > 5 else "#66BB6A"
              for p in merged["pct_missing"]]

    bars = ax.bar(merged["year"], merged["missing_days"],
                  color=colors, edgecolor="white",
                  linewidth=0.5, zorder=2)

    for bar, pct in zip(bars, merged["pct_missing"]):
        if pct > 5:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{pct:.0f}%", ha="center", va="bottom",
                    fontsize=7, color="#333")

    ax.set_ylabel("Missing days", fontsize=10)
    ax.set_title("B  —  Data gaps per year",
                 fontweight="bold", fontsize=11, loc="left")
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xticks(merged["year"])
    ax.set_xticklabels(merged["year"].astype(int),
                       rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3, zorder=1)

    handles = [
        Patch(color="#66BB6A", label="< 5% missing"),
        Patch(color="#FB8C00", label="5–20% missing"),
        Patch(color="#E53935", label="> 20% missing"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right",
              framealpha=0.85)


def panel_confidence(ax, cat):
    """Panel C: mean_prob distribution per class."""
    bins = np.linspace(0.25, 0.82, 35)
    for cls in ["su", "eq", "px"]:
        subset = cat[cat["most_common_class"] == cls]["mean_prob"].dropna()
        ax.hist(subset, bins=bins, color=CLASS_COLORS[cls], alpha=0.55,
                label=CLASS_LABELS[cls], density=True)

    ax.axvline(HC_THRESHOLD, color="black", lw=1.5, ls="--",
               label=f"High-conf. threshold ({HC_THRESHOLD})")
    ax.set_xlabel("mean_prob", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("C  —  Classification confidence by class",
                 fontweight="bold", fontsize=11, loc="left")
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)


def panel_class_proportions(ax, cat):
    """Panel D: monthly class fractions stacked area."""
    monthly = (cat.groupby(["month_ts", "most_common_class"])
               .size().unstack(fill_value=0).reset_index())

    for cls in ["su", "eq", "px"]:
        if cls not in monthly.columns:
            monthly[cls] = 0

    totals = monthly[["su", "eq", "px"]].sum(axis=1).replace(0, np.nan)
    for cls in ["su", "eq", "px"]:
        monthly[f"frac_{cls}"] = monthly[cls] / totals
        monthly[f"frac_{cls}_smooth"] = (
            monthly[f"frac_{cls}"]
            .rolling(3, center=True, min_periods=1).mean()
        )

    ax.stackplot(
        monthly["month_ts"],
        monthly["frac_su_smooth"],
        monthly["frac_eq_smooth"],
        monthly["frac_px_smooth"],
        labels=[CLASS_LABELS[c] for c in ["su", "eq", "px"]],
        colors=[CLASS_COLORS[c] for c in ["su", "eq", "px"]],
        alpha=0.85,
    )

    ax.set_ylabel("Fraction of events", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("D  —  Monthly class proportions (3-month smoothed)",
                 fontweight="bold", fontsize=11, loc="left")
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(monthly["month_ts"].min(), monthly["month_ts"].max())
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.85)


# ── 4. main ───────────────────────────────────────────────────────────────────
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not CATALOG_CSV.exists():
        print(f"ERROR: catalog not found at {CATALOG_CSV}")
        print("Run build_catalog.py first.")
        return

    cat              = load_catalog(CATALOG_CSV)
    monthly_stations = load_station_counts(STATION_JSON_DIR)

    n_total       = len(cat)
    n_sta_max     = (int(monthly_stations["n_stations_max"].max())
                     if monthly_stations is not None else "?")

    fig, axes = plt.subplots(
        2, 2, figsize=(16, 10),
        gridspec_kw={"hspace": 0.42, "wspace": 0.32}
    )

    panel_event_rate(        axes[0, 0], cat, monthly_stations)
    panel_missing_days(      axes[0, 1], cat)
    panel_confidence(        axes[1, 0], cat)
    panel_class_proportions( axes[1, 1], cat)

    fig.suptitle(
        f"Mt Rainier QuakeXNet Catalog — Completeness & Quality Dashboard\n"
        f"2010–2025  ·  {n_total:,} events  ·  "
        f"up to {n_sta_max} stations/day  ·  CC & UW networks",
        fontsize=12, fontweight="bold", y=1.01
    )

    outpath = OUTDIR / "completeness_dashboard.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"\nSaved → {outpath}")


if __name__ == "__main__":
    main()