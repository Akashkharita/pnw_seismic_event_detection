# daily_detection_dynamic.py
#
# Dynamically fetches the station list active on the requested date within
# the Mt Rainier network bounding box, then runs QuakeXNet inference.
#
# Waveform source:
#   - pnwstore (fast local archive) for data up to end of 2022
#   - IRIS FDSN client as fallback for 2023 onward
#
# Stations already processed (CSV exists in output dir) are skipped,
# so re-running the full 2010-2025 batch only processes new stations.
#
# Usage:
#   python daily_detection_dynamic.py \
#       --start 2024-06-01T00:00:00 \
#       --end   2024-06-01T23:59:59
#
# Optional overrides:
#   --networks           *       (default: all networks within bbox)
#   --fdsn_client        IRIS    (default: IRIS)
#   --min_duration_hours 1       (skip stations with <N hours of data)
#   --save_station_list          (save per-day station JSON for auditing)
#   --pnwstore_end_year  2022    (last year served by pnwstore)

import os
import csv
import json
import argparse
import numpy as np
import pandas as pd
import torch
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

import sys

from load_model import load_quakexnet
from detect import smooth_moving_avg, detect_event_windows


# ── constants ──────────────────────────────────────────────────────────────────
# Tight rectangular bounding box from exact coordinates of all 35 stations
# plus 0.01 deg (~1 km) buffer on each edge.
# S: CC.KAUT (46.7303N)   N: CC.LONE/LONR (47.0099N)
# E: CC.CRYS (-121.5003E) W: CC.PR02 (-122.0487E)
BBOX_MIN_LAT =  46.720263   # 46.730263 - 0.01
BBOX_MAX_LAT =  47.019890   # 47.009890 + 0.01
BBOX_MIN_LON = -122.058695  # -122.048695 - 0.01
BBOX_MAX_LON = -121.490260  # -121.500260 + 0.01

# Last year fully covered by pnwstore
PNWSTORE_END_YEAR = 2022

# Infrasound and weather channel prefixes to exclude.
# Everything else (BH, HH, EH, SH, HN, EN, BN, GH, SP, DP ...) is accepted.
EXCLUDE_CHANNEL_PREFIXES = (
    "DF",   # infrasound
    "HDF",  # infrasound high-gain
    "LDF",  # infrasound low-freq
    "BDF",  # infrasound broadband
    "EDF",  # infrasound
    "LDO",  # differential pressure
    "LKO",  # tiltmeter
    "LWS",  # wind speed
    "LWD",  # wind direction
    "LRI",  # rainfall
    "LDI",  # lightning
    "VM",   # very long period / tilt
    "LD",   # pressure catch-all
)

DEFAULT_NETWORKS = "*"


# ── argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="QuakeXNet detection — dynamic stations, pnwstore/IRIS fallback."
)
parser.add_argument("--start", type=str, required=True)
parser.add_argument("--end",   type=str, required=True)
parser.add_argument("--networks", type=str, default=DEFAULT_NETWORKS)
parser.add_argument("--min_lat", type=float, default=BBOX_MIN_LAT)
parser.add_argument("--max_lat", type=float, default=BBOX_MAX_LAT)
parser.add_argument("--min_lon", type=float, default=BBOX_MIN_LON)
parser.add_argument("--max_lon", type=float, default=BBOX_MAX_LON)
parser.add_argument("--fdsn_client", type=str, default="IRIS")
parser.add_argument("--min_duration_hours", type=float, default=1.0)
parser.add_argument("--save_station_list", action="store_true")
parser.add_argument("--pnwstore_end_year", type=int, default=PNWSTORE_END_YEAR)
args = parser.parse_args()

st_time  = UTCDateTime(args.start)
et_time  = UTCDateTime(args.end)
networks = [n.strip() for n in args.networks.split(",")]

# choose waveform source based on the year of the requested window
use_pnwstore = st_time.year <= args.pnwstore_end_year

print(f"\n{'='*60}")
print(f"QuakeXNet dynamic detection")
print(f"  Window   : {st_time}  to  {et_time}")
print(f"  Box      : lat [{args.min_lat}, {args.max_lat}]  "
      f"lon [{args.min_lon}, {args.max_lon}]")
print(f"  Nets     : {'ALL' if networks == ['*'] else networks}")
# pnwstore is a UW-internal waveform archive and is optional. It is imported
# lazily so that installations without it (e.g. outside UW) can still run the
# full detection pipeline against FDSN.
pnw_client = None
if use_pnwstore:
    try:
        from pnwstore import WaveformClient
        pnw_client = WaveformClient()        # waveforms up to end of 2022
    except ImportError:
        print("  pnwstore not installed - using FDSN for all years.")
        use_pnwstore = False

print(f"  Waveforms: {'pnwstore' if use_pnwstore else 'IRIS FDSN (post-2022 fallback)'}")
print(f"{'='*60}\n")


# ── clients ────────────────────────────────────────────────────────────────────
fdsn_client = Client(args.fdsn_client)   # metadata always; waveforms post-2022



# ── waveform fetch with automatic source selection + fallback ──────────────────
def get_waveforms(net, sta, chn, starttime, endtime):
    """
    Use pnwstore for years <= pnwstore_end_year, IRIS otherwise.
    If pnwstore returns empty or raises, automatically falls back to IRIS.
    Returns (stream, source_label).
    """
    if use_pnwstore and pnw_client is not None:
        try:
            st = pnw_client.get_waveforms(
                network=net, station=sta, channel=chn + "*",
                location="*", starttime=starttime, endtime=endtime,
            )
            if st and len(st) > 0:
                return st, "pnwstore"
            print(f"  pnwstore returned empty stream, falling back to IRIS...")
        except Exception as e:
            print(f"  pnwstore failed ({e}), falling back to IRIS...")

    st = fdsn_client.get_waveforms(
        network=net, station=sta, channel=chn + "*",
        location="*", starttime=starttime, endtime=endtime,
    )
    return st, "IRIS"


# ── 1. fetch station list ──────────────────────────────────────────────────────
def fetch_active_stations(min_lat, max_lat, min_lon, max_lon,
                          starttime, endtime):
    """
    Query FDSN for all stations within the bounding box that were active
    during [starttime, endtime]. Accepts any seismic Z-component channel
    not in the infrasound/weather exclusion list.
    Returns: [{"net": .., "sta": .., "chn": ..}, ...]
    """
    print("Fetching active station list from FDSN...")
    stations_out = []
    seen = set()

    try:
        inv = fdsn_client.get_stations(
            network="*",
            minlatitude=min_lat, maxlatitude=max_lat,
            minlongitude=min_lon, maxlongitude=max_lon,
            starttime=starttime, endtime=endtime,
            level="channel",
        )
    except Exception as e:
        print(f"  ERROR: FDSN station query failed: {e}")
        return stations_out

    for net_obj in inv:
        for sta_obj in net_obj:

            sta_start = sta_obj.start_date
            sta_end   = sta_obj.end_date if sta_obj.end_date \
                        else UTCDateTime(2099, 1, 1)
            if sta_start > endtime or sta_end < starttime:
                continue

            sta_key = (net_obj.code, sta_obj.code)
            if sta_key in seen:
                continue

            # pick first usable Z channel (alphabetical sort for reproducibility)
            best_prefix = None
            for cha in sorted(sta_obj.channels, key=lambda c: c.code):
                code = cha.code
                if not code.endswith("Z"):
                    continue
                if any(code.startswith(ex) for ex in EXCLUDE_CHANNEL_PREFIXES):
                    continue
                best_prefix = code[:2]
                break

            if best_prefix is None:
                print(f"  SKIP {net_obj.code}.{sta_obj.code} "
                      f"-- no usable seismic Z channel")
                continue

            seen.add(sta_key)
            stations_out.append({
                "net": net_obj.code,
                "sta": sta_obj.code,
                "chn": best_prefix,
            })

    stations_out.sort(key=lambda x: (x["net"], x["sta"]))
    print(f"  Found {len(stations_out)} active stations:")
    for s in stations_out:
        print(f"    {s['net']}.{s['sta']:8s}  [{s['chn']}*]")
    print()
    return stations_out


stations = fetch_active_stations(
    min_lat=args.min_lat, max_lat=args.max_lat,
    min_lon=args.min_lon, max_lon=args.max_lon,
    starttime=st_time,   endtime=et_time,
)

if not stations:
    print("ERROR: No active stations found. Exiting.")
    raise SystemExit(1)


# ── 2. output paths ────────────────────────────────────────────────────────────
os.makedirs("../plots", exist_ok=True)
os.makedirs("../logs",  exist_ok=True)

start_str = st_time.strftime("%Y%m%d_%H%M")
end_str   = et_time.strftime("%Y%m%d_%H%M")
out_dir   = f"../logs/mt_rainier_detections/{start_str}_{end_str}"
os.makedirs(out_dir, exist_ok=True)

if args.save_station_list:
    sta_list_path = os.path.join(out_dir, f"stations_{start_str}.json")
    with open(sta_list_path, "w") as f:
        json.dump(stations, f, indent=2)
    print(f"  Station list saved -> {sta_list_path}")

log_file = "../logs/detections.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "station", "final_label",
                         "eq_auc", "px_auc", "su_auc"])


# ── 3. load model ──────────────────────────────────────────────────────────────
print("Loading QuakeXNet model...")
model = load_quakexnet()
print("  Model loaded.\n")


# ── 4. detection loop ──────────────────────────────────────────────────────────
class_names      = ["eq", "px", "su"]
chn_prefix       = "QuakeXNet_"
channel_map      = {cls: f"{chn_prefix}{cls}" for cls in class_names}
SECONDS_PER_STEP = 10   # stride=500 @ 50 Hz -> 10 s per step

n_skipped = n_processed = n_failed = 0

print(f"{'='*60}")
print(f"Running detection on {len(stations)} stations...")
print(f"{'='*60}\n")

for entry in stations:
    net = entry["net"]
    sta = entry["sta"]
    chn = entry["chn"]

    out_csv = os.path.join(out_dir, f"{sta}_{start_str}_to_{end_str}_events.csv")

    # ── skip already-processed stations ───────────────────────────────────────
    if os.path.exists(out_csv):
        print(f"  SKIP {net}.{sta} -- CSV already exists")
        n_skipped += 1
        continue

    print(f"Processing {net}.{sta}  [{chn}*]...")

    try:
        st, source = get_waveforms(net, sta, chn, st_time, et_time)
        print(f"  Raw stream ({source}): {st}")

        
        
        # deduplicate by location code — keep preferred location per channel
        # but keep ALL time segments (do not drop multiple traces of same channel)
        loc_preference = ["", "00", "01", "--"]
        seen_locs = {}
        for tr in st:
            chan = tr.stats.channel
            loc  = tr.stats.location
            if chan not in seen_locs:
                seen_locs[chan] = loc
            else:
                current_rank = loc_preference.index(seen_locs[chan]) \
                               if seen_locs[chan] in loc_preference else 999
                new_rank     = loc_preference.index(loc) \
                               if loc in loc_preference else 999
                if new_rank < current_rank:
                    seen_locs[chan] = loc

        # keep only traces from the preferred location per channel
        st_clean = obspy.Stream([
            tr for tr in st
            if tr.stats.location == seen_locs.get(
                tr.stats.channel, tr.stats.location)
        ])

        # resample to 100 sps before merging
        target_fs = 100.0
        for tr in st_clean:
            if tr.stats.sampling_rate != target_fs:
                print(f"  Resampling {tr.id}: "
                      f"{tr.stats.sampling_rate} -> {target_fs} sps")
                tr.resample(target_fs)

        # merge all time segments into one continuous trace, filling gaps with 0
        st_clean.sort()
        st_clean.merge(method=1, fill_value=0, interpolation_samples=0)
        print(f"  Clean stream: {st_clean}")

        # skip if too little data — use total samples across all traces
        total_samples = sum(tr.stats.npts for tr in st_clean)
        min_samples   = args.min_duration_hours * 3600 * target_fs
        if not st_clean or total_samples < min_samples:
            print(f"  SKIP -- insufficient data "
                  f"(total={total_samples/target_fs/3600:.2f}h, "
                  f"required={args.min_duration_hours}h)\n")
            n_skipped += 1
            continue
            
            
            
        """
        # deduplicate location codes
        seen_channels = {}
        for tr in sorted(st, key=lambda t: t.stats.location):
            key = tr.stats.channel
            if key not in seen_channels:
                seen_channels[key] = tr
        st_clean = obspy.Stream(list(seen_channels.values()))

        # resample to 100 sps
        target_fs = 100.0
        for tr in st_clean:
            if tr.stats.sampling_rate != target_fs:
                print(f"  Resampling {tr.id}: "
                      f"{tr.stats.sampling_rate} -> {target_fs} sps")
                tr.resample(target_fs)

        st_clean.merge(method=1, fill_value=0)
        print(f"  Clean stream: {st_clean}")

        # skip if too little data
        if not st_clean or \
           max(tr.stats.npts for tr in st_clean) < \
           args.min_duration_hours * 3600 * target_fs:
            print(f"  SKIP -- insufficient data (<{args.min_duration_hours}h)\n")
            n_skipped += 1
            continue

        """ 
            
            
            
        probs_st = model.annotate(st_clean, stride=500)
        event_records = []

        for cls in class_names:
            probs = probs_st.select(channel=channel_map[cls])
            for prob in probs:
                trace_start = prob.stats.starttime
                s_cls       = smooth_moving_avg(prob.data)
                events      = detect_event_windows(s_cls)

                for event in events:
                    start_idx = event["start"]
                    end_idx   = event["end"]
                    t0 = trace_start + start_idx * SECONDS_PER_STEP
                    t1 = trace_start + end_idx   * SECONDS_PER_STEP
                    if t1 < st_time or t0 > et_time:
                        continue
                    event_records.append({
                        "station":        sta,
                        "network":        net,
                        "class":          cls,
                        "auc":            event["area_under_curve"],
                        "mean_prob":      event["mean_prob"],
                        "max_prob":       event["max_prob"],
                        "start_index":    start_idx,
                        "end_index":      end_idx,
                        "start_time":     str(t0),
                        "end_time":       str(t1),
                        "waveform_source": source,
                    })

        df_events = pd.DataFrame(event_records) if event_records else \
                    pd.DataFrame(columns=["station","network","class","auc",
                                          "mean_prob","max_prob","start_index",
                                          "end_index","start_time","end_time",
                                          "waveform_source"])

        df_events.to_csv(out_csv, index=False)

        if event_records:
            print(df_events.to_string(index=False))
            print(f"  Saved {len(event_records)} events -> {out_csv}")
        else:
            # empty CSV written so this station is marked done on re-runs
            print(f"  No events detected. (empty CSV written)")

        n_processed += 1
        print()

    except Exception as e:
        print(f"  ERROR {net}.{sta}: {e}\n")
        n_failed += 1

print(f"\n{'='*60}")
print(f"Done.  processed={n_processed}  skipped={n_skipped}  failed={n_failed}")
print(f"Results: {out_dir}")
print(f"{'='*60}")