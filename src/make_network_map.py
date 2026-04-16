"""
make_network_map.py
-------------------
Figure 1: Interactive Folium map of seismic network around Mt Rainier.

- Uses the exact 35 stations from the analysis
- Queries FDSN by explicit net.sta codes for precise coordinates
- Reads catalog from CSV (not parquet) to avoid list-column serialization bug
- Colors stations by years active in catalog (pale blue → crimson)
- Circle size scales with total detection count
- Channel code and class breakdown shown in popup

Usage (run from ANY directory):
    python /home/ak287/pnw_seismic_event_detection/src/make_network_map.py
"""

import ast
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import folium
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# ── resolve paths relative to THIS script ────────────────────────────────────
SRC_DIR     = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
CATALOG_CSV = PROJECT_DIR / "src/catalog_output" / "master_catalog.csv"
OUTDIR      = PROJECT_DIR / "src/catalog_output"

# ── Mt Rainier ────────────────────────────────────────────────────────────────
RAINIER_LAT = 46.8523
RAINIER_LON = -121.7603

# ── exact station list from the analysis ─────────────────────────────────────
STATIONS = [
    {"net": "CC", "sta": "ARAT",  "chn": "BH"},
    {"net": "CC", "sta": "CARB",  "chn": "BH"},
    {"net": "CC", "sta": "COPP",  "chn": "BH"},
    {"net": "CC", "sta": "CRBN",  "chn": "BH"},
    {"net": "CC", "sta": "CRYS",  "chn": "HH"},
    {"net": "CC", "sta": "GNOB",  "chn": "BH"},
    {"net": "CC", "sta": "GOBB",  "chn": "BH"},
    {"net": "CC", "sta": "GTWY",  "chn": "BH"},
    {"net": "CC", "sta": "KAUT",  "chn": "BH"},
    {"net": "CC", "sta": "KAVK",  "chn": "BH"},
    {"net": "CC", "sta": "LONE",  "chn": "BH"},
    {"net": "CC", "sta": "LONR",  "chn": "BH"},
    {"net": "CC", "sta": "MILD",  "chn": "BH"},
    {"net": "CC", "sta": "OBSR",  "chn": "BH"},
    {"net": "CC", "sta": "OPCH",  "chn": "BH"},
    {"net": "CC", "sta": "PANH",  "chn": "BH"},
    {"net": "CC", "sta": "PARA",  "chn": "BH"},
    {"net": "CC", "sta": "PR01",  "chn": "BH"},
    {"net": "CC", "sta": "PR02",  "chn": "BH"},
    {"net": "CC", "sta": "PR03",  "chn": "BH"},
    {"net": "CC", "sta": "PR04",  "chn": "BH"},
    {"net": "CC", "sta": "PR05",  "chn": "BH"},
    {"net": "CC", "sta": "RUSH",  "chn": "BH"},
    {"net": "CC", "sta": "SIFT",  "chn": "BH"},
    {"net": "CC", "sta": "TABR",  "chn": "BH"},
    {"net": "CC", "sta": "TAVI",  "chn": "BH"},
    {"net": "CC", "sta": "VOIT",  "chn": "BH"},
    {"net": "CC", "sta": "WOW",   "chn": "BH"},
    {"net": "UW", "sta": "FMW",   "chn": "HH"},
    {"net": "UW", "sta": "LO2",   "chn": "EH"},
    {"net": "UW", "sta": "LON",   "chn": "BH"},
    {"net": "UW", "sta": "RCM",   "chn": "HH"},
    {"net": "UW", "sta": "RCS",   "chn": "EH"},
    {"net": "UW", "sta": "RER",   "chn": "HH"},
    {"net": "UW", "sta": "STAR",  "chn": "EH"},
]

# ── approximate edifice outline (~3000 m contour) ─────────────────────────────
EDIFICE_COORDS = [
    [46.980, -121.760], [46.975, -121.720], [46.965, -121.685],
    [46.950, -121.660], [46.930, -121.645], [46.910, -121.640],
    [46.890, -121.648], [46.870, -121.660], [46.853, -121.678],
    [46.840, -121.700], [46.830, -121.725], [46.825, -121.755],
    [46.827, -121.785], [46.835, -121.812], [46.848, -121.833],
    [46.865, -121.848], [46.885, -121.855], [46.905, -121.852],
    [46.925, -121.842], [46.942, -121.825], [46.955, -121.803],
    [46.963, -121.778], [46.968, -121.752], [46.972, -121.728],
    [46.978, -121.748], [46.980, -121.760],
]


# ── 1. FDSN query — exact station codes only ──────────────────────────────────
def query_fdsn_stations():
    """Query FDSN for each station by explicit net.sta. Returns dict keyed by
    station code."""
    client = Client("IRIS")
    result = {}

    for entry in STATIONS:
        net = entry["net"]
        sta = entry["sta"]
        chn = entry["chn"]
        try:
            inv = client.get_stations(
                network=net,
                station=sta,
                level="station",
                starttime=UTCDateTime("2009-01-01"),
                endtime=UTCDateTime("2026-12-31"),
            )
            for net_obj in inv:
                for sta_obj in net_obj:
                    end   = sta_obj.end_date.datetime if sta_obj.end_date else datetime.utcnow()
                    start = sta_obj.start_date.datetime if sta_obj.start_date else None
                    result[sta] = {
                        "code":        sta,
                        "network":     net,
                        "channel":     chn,
                        "lat":         sta_obj.latitude,
                        "lon":         sta_obj.longitude,
                        "elevation_m": round(sta_obj.elevation),
                        "fdsn_start":  start,
                        "fdsn_end":    end,
                    }
                    print(f"  {net}.{sta:6s} ({chn})  "
                          f"lat={sta_obj.latitude:.3f}  "
                          f"lon={sta_obj.longitude:.3f}  "
                          f"elev={sta_obj.elevation:.0f} m")
        except Exception as e:
            print(f"  WARNING — {net}.{sta}: {e}")
            result[sta] = {
                "code": sta, "network": net, "channel": chn,
                "lat": None, "lon": None, "elevation_m": None,
                "fdsn_start": None, "fdsn_end": None,
            }

    n_ok = sum(1 for v in result.values() if v["lat"] is not None)
    print(f"\n  Coordinates retrieved: {n_ok} / {len(STATIONS)} stations")
    return result


# ── 2. Catalog activity — vectorized, reads CSV to avoid parquet list bug ─────
def get_station_activity(catalog_csv):
    """
    Reads master_catalog.csv (not parquet — parquet concatenates list columns
    into a single string on read-back). Explodes station/class list columns
    into one row per (event × station), then aggregates per station.
    """
    print(f"  Reading: {catalog_csv}")
    cat = pd.read_csv(catalog_csv)
    cat["rounded_start"] = pd.to_datetime(
        cat["rounded_start"], utc=True, errors="coerce", format="mixed"
    )
    print(f"  Rows loaded: {len(cat):,}")

    # parse stringified lists e.g. "['MILD', 'RCM', 'RCS']"
    def safe_parse(val):
        if isinstance(val, list):
            return val
        val = str(val).strip()
        if val.startswith("["):
            try:
                return ast.literal_eval(val)
            except Exception:
                pass
        return []

    cat["stations"]    = cat["stations"].apply(safe_parse)
    cat["all_classes"] = cat["all_classes"].apply(safe_parse)

    # verify parsing worked
    sample = cat["stations"].iloc[0]
    print(f"  Sample parsed station list: {sample}")
    if not sample or not isinstance(sample, list) or len(sample) == 0:
        print("  ERROR: parsing failed — station lists are empty")
        return {}

    # pad class lists to match station list length
    def pad_classes(row):
        s = row["stations"]
        c = row["all_classes"]
        if len(c) < len(s):
            c = c + ["unknown"] * (len(s) - len(c))
        return c[:len(s)]

    cat["all_classes"] = cat.apply(pad_classes, axis=1)

    # explode both list columns in parallel into a flat dataframe
    slim         = cat[["rounded_start", "stations", "all_classes"]].copy()
    stations_exp = slim.explode("stations").reset_index(drop=True)
    classes_exp  = slim.explode("all_classes").reset_index(drop=True)

    flat = pd.DataFrame({
        "timestamp": stations_exp["rounded_start"].values,
        "station":   stations_exp["stations"].values,
        "cls":       classes_exp["all_classes"].values,
    })
    flat = flat[flat["station"].notna() & (flat["station"] != "")]
    print(f"  Exploded rows      : {len(flat):,}")
    print(f"  Unique stations    : {flat['station'].nunique()}")

    # aggregate per station — use vectorized groupby, not apply()
    grp = flat.groupby("station")

    stats_df = pd.DataFrame({
        "n_detections":    grp.size(),
        "first_detection": grp["timestamp"].min(),
        "last_detection":  grp["timestamp"].max(),
        "n_eq": flat[flat["cls"] == "eq"].groupby("station").size(),
        "n_su": flat[flat["cls"] == "su"].groupby("station").size(),
        "n_px": flat[flat["cls"] == "px"].groupby("station").size(),
    }).reset_index().fillna(0)

    stats_df["n_eq"] = stats_df["n_eq"].astype(int)
    stats_df["n_su"] = stats_df["n_su"].astype(int)
    stats_df["n_px"] = stats_df["n_px"].astype(int)

    stats_df["years_in_catalog"] = (
        (stats_df["last_detection"] - stats_df["first_detection"])
        .dt.total_seconds() / (365.25 * 86400)
    ).round(1).clip(lower=0)

    # convert to dict keyed by station code
    out = {}
    for _, row in stats_df.iterrows():
        out[row["station"]] = {
            "n_detections":     int(row["n_detections"]),
            "first_detection":  row["first_detection"],
            "last_detection":   row["last_detection"],
            "n_eq":             int(row["n_eq"]),
            "n_su":             int(row["n_su"]),
            "n_px":             int(row["n_px"]),
            "years_in_catalog": float(row["years_in_catalog"]),
        }

    print("\n  Top 8 stations by detections:")
    top8 = sorted(out.items(), key=lambda x: x[1]["n_detections"], reverse=True)[:8]
    for code, s in top8:
        print(f"    {code:6s}  {s['n_detections']:7,} det  "
              f"{s['years_in_catalog']:.1f} yrs  "
              f"eq={s['n_eq']:,}  su={s['n_su']:,}  px={s['n_px']:,}")

    return out


# ── 3. Color scale ────────────────────────────────────────────────────────────
def years_to_color(years, vmin=0, vmax=15):
    t = max(0.0, min(1.0, (years - vmin) / (vmax - vmin)))
    stops = [
        (0.00, (204, 229, 255)),
        (0.25, ( 42, 157, 143)),
        (0.60, (233, 196, 106)),
        (1.00, (193,  18,  31)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            r = int(c0[0] + f * (c1[0] - c0[0]))
            g = int(c0[1] + f * (c1[1] - c0[1]))
            b = int(c0[2] + f * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#c1121f"


def make_legend_html(vmin=0, vmax=15):
    steps = 6
    items = ""
    for i in range(steps + 1):
        yr    = vmin + i * (vmax - vmin) / steps
        color = years_to_color(yr, vmin, vmax)
        label = f"{yr:.0f} yr{'s' if yr != 1 else ''}"
        items += f"""
        <div style="display:flex;align-items:center;gap:6px;margin:3px 0">
          <div style="width:16px;height:16px;border-radius:50%;
                      background:{color};border:1px solid #555;
                      flex-shrink:0"></div>
          <span style="font-size:12px">{label}</span>
        </div>"""
    return f"""
    <div style="position:fixed;bottom:40px;left:20px;z-index:1000;
                background:rgba(255,255,255,0.93);border-radius:8px;
                padding:12px 16px;box-shadow:0 2px 8px rgba(0,0,0,0.25);
                font-family:'Helvetica Neue',sans-serif;min-width:145px">
      <div style="font-weight:700;font-size:13px;margin-bottom:8px;
                  border-bottom:1px solid #ddd;padding-bottom:4px">
        Years in Catalog
      </div>
      {items}
      <div style="margin-top:10px;border-top:1px solid #ddd;padding-top:6px;
                  font-size:11px;color:#666">
        Circle size ∝ total detections
      </div>
    </div>"""


# ── 4. Build map ──────────────────────────────────────────────────────────────
def build_map(station_meta, station_stats, outpath):
    m = folium.Map(
        location=[RAINIER_LAT, RAINIER_LON],
        zoom_start=9,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # tile layers
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False, control=True,
    ).add_to(m)
    folium.TileLayer(
        "CartoDB positron", name="Light", overlay=False, control=True
    ).add_to(m)

    # edifice outline
    folium.Polygon(
        locations=EDIFICE_COORDS,
        color="#8B4513", weight=2,
        fill=True, fill_color="#cd853f", fill_opacity=0.15,
        tooltip="Mt Rainier volcanic edifice (~3000 m contour)",
        popup="Mt Rainier (4,392 m)",
    ).add_to(m)

    # summit marker
    folium.Marker(
        location=[RAINIER_LAT, RAINIER_LON],
        tooltip="Mt Rainier Summit (4,392 m)",
        icon=folium.Icon(color="red", icon="fire", prefix="fa"),
    ).add_to(m)

    # station circles
    vmin, vmax = 0, 15
    plotted    = 0

    for entry in STATIONS:
        code  = entry["sta"]
        meta  = station_meta.get(code, {})
        stats = station_stats.get(code, {})

        lat = meta.get("lat")
        lon = meta.get("lon")
        if lat is None or lon is None:
            print(f"  SKIP {code} — no coordinates")
            continue

        years = stats.get("years_in_catalog", 0.0)
        n_det = stats.get("n_detections",     0)
        n_eq  = stats.get("n_eq",             0)
        n_su  = stats.get("n_su",             0)
        n_px  = stats.get("n_px",             0)
        net   = meta.get("network",    entry["net"])
        chn   = meta.get("channel",    entry["chn"])
        elev  = meta.get("elevation_m", "?")

        color  = years_to_color(years, vmin, vmax)
        radius = max(5, min(18, 5 + np.sqrt(n_det / 500)))

        first = stats.get("first_detection", "")
        last  = stats.get("last_detection",  "")
        if hasattr(first, "strftime"):
            first = first.strftime("%Y-%m-%d")
        if hasattr(last, "strftime"):
            last = last.strftime("%Y-%m-%d")

        # dominant class badge
        if n_det > 0:
            class_counts = {"eq": n_eq, "su": n_su, "px": n_px}
            dominant     = max(class_counts, key=class_counts.get)
            badge_colors = {"eq": "#1565C0", "su": "#2E7D32", "px": "#BF360C"}
            badge_color  = badge_colors[dominant]
            pct_eq = f"{100*n_eq/n_det:.0f}%"
            pct_su = f"{100*n_su/n_det:.0f}%"
            pct_px = f"{100*n_px/n_det:.0f}%"
            class_rows = f"""
            <tr><td colspan=2 style="padding-top:6px;font-weight:600;
                                     border-top:1px solid #eee">
                Class breakdown</td></tr>
            <tr><td style="color:#666;padding:2px 8px 2px 0">Total</td>
                <td><b>{n_det:,}</b></td></tr>
            <tr><td style="color:#1565C0;padding:2px 8px 2px 0">Earthquake</td>
                <td><b>{n_eq:,} ({pct_eq})</b></td></tr>
            <tr><td style="color:#2E7D32;padding:2px 8px 2px 0">Surface event</td>
                <td><b>{n_su:,} ({pct_su})</b></td></tr>
            <tr><td style="color:#BF360C;padding:2px 8px 2px 0">Explosion</td>
                <td><b>{n_px:,} ({pct_px})</b></td></tr>"""
            badge_html = f"""
            <div style="display:inline-block;background:{badge_color};
                        color:white;font-size:11px;padding:2px 8px;
                        border-radius:10px;margin-bottom:8px">
              dominant: {dominant}
            </div>"""
        else:
            badge_html  = ""
            class_rows  = """<tr><td colspan=2 style="color:#888">
                No detections in catalog</td></tr>"""

        popup_html = f"""
        <div style="font-family:'Helvetica Neue',sans-serif;min-width:200px">
          <div style="font-size:15px;font-weight:700;margin-bottom:4px">
            {net}.{code}
            <span style="font-size:11px;font-weight:400;color:#666;
                         margin-left:6px">{chn}* channels</span>
          </div>
          {badge_html}
          <table style="font-size:12px;border-collapse:collapse;width:100%">
            <tr><td style="color:#666;padding:2px 8px 2px 0">Elevation</td>
                <td><b>{elev} m</b></td></tr>
            <tr><td style="color:#666;padding:2px 8px 2px 0">Lat / Lon</td>
                <td><b>{lat:.3f}, {lon:.3f}</b></td></tr>
            <tr><td style="color:#666;padding:2px 8px 2px 0">Years active</td>
                <td><b>{years:.1f}</b></td></tr>
            <tr><td style="color:#666;padding:2px 8px 2px 0">First detection</td>
                <td><b>{first}</b></td></tr>
            <tr><td style="color:#666;padding:2px 8px 2px 0">Last detection</td>
                <td><b>{last}</b></td></tr>
            {class_rows}
          </table>
        </div>"""

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color="white", weight=1.5,
            fill=True, fill_color=color, fill_opacity=0.9,
            tooltip=f"{net}.{code} ({chn}) — {years:.1f} yrs — {n_det:,} det",
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(m)
        plotted += 1

    print(f"  Plotted {plotted} / {len(STATIONS)} stations")

    # legend
    m.get_root().html.add_child(folium.Element(make_legend_html(vmin, vmax)))

    # title
    title_html = f"""
    <div style="position:fixed;top:15px;left:50%;transform:translateX(-50%);
                z-index:1000;background:rgba(255,255,255,0.93);
                border-radius:8px;padding:10px 22px;
                box-shadow:0 2px 8px rgba(0,0,0,0.2);
                font-family:'Helvetica Neue',sans-serif;text-align:center">
      <div style="font-size:16px;font-weight:700;color:#1a1a2e">
        Mt Rainier Seismic Network — 2010 to 2026
      </div>
      <div style="font-size:12px;color:#555;margin-top:3px">
        {len(STATIONS)} stations · CC &amp; UW networks ·
        QuakeXNet catalog · 293,457 events
      </div>
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl(position="topright").add_to(m)
    m.save(str(outpath))
    print(f"  Saved → {outpath}")


# ── 5. Main ───────────────────────────────────────────────────────────────────
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTDIR / "network_coverage_map.html"

    print(f"Catalog : {CATALOG_CSV}")
    print(f"Output  : {outpath}\n")

    if not CATALOG_CSV.exists():
        print(f"ERROR: catalog not found at {CATALOG_CSV}")
        print("Run build_catalog.py first.")
        return

    print("Step 1: Querying FDSN for station coordinates...")
    station_meta = query_fdsn_stations()

    print("\nStep 2: Computing station activity from catalog...")
    station_stats = get_station_activity(CATALOG_CSV)

    print("\nStep 3: Building map...")
    build_map(station_meta, station_stats, outpath)

    print(f"\nDone. Open in your browser:\n  {outpath}")


if __name__ == "__main__":
    main()