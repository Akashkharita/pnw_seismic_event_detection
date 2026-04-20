"""
locate_events.py
----------------
Locates seismic events from the master catalog (output of build_catalog.py)
using enveloc (XCOR). No minimum station or class purity filters applied.
SNR filtering is the only waveform quality criterion.

Architecture — producer-consumer pipeline:
  - A thread pool downloads waveforms for N events simultaneously (producers)
  - A single worker runs enveloc on completed downloads one at a time (consumer)
  - While enveloc runs on event K, the pool is already downloading events
    K+1 ... K+N, hiding most of the download latency

Waveform retrieval:
  - pnwstore (WaveformClient) for data up to end of 2022
  - IRIS FDSN for data from 2023 onward
  - pnwstore uses SQLite and is NOT thread-safe — a fresh WaveformClient()
    is created inside each thread to avoid cross-thread SQLite errors

Resuming interrupted runs:
  - Use --start_idx to resume from a specific row index
  - Checkpoints are saved every 1000 events automatically

Usage:
    conda activate enveloc
    ulimit -n 65536

    # full run
    python locate_events.py \\
        --catalog catalog_output/master_catalog.csv \\
        --outfile catalog_output/located_events.csv \\
        --download_workers 8 \\
        --prefetch 4

    # resume from row 5000 after a crash
    python locate_events.py \\
        --catalog catalog_output/master_catalog.csv \\
        --outfile catalog_output/located_events.csv \\
        --download_workers 8 \\
        --prefetch 4 \\
        --start_idx 5000

    # test on first 50 events
    python locate_events.py \\
        --catalog catalog_output/master_catalog.csv \\
        --outfile catalog_output/located_events.csv \\
        --download_workers 8 \\
        --prefetch 4 \\
        --nmax 50
"""

import argparse
import ast
import logging
import traceback
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core.util import AttribDict
from obspy.signal.filter import envelope

from pnwstore import WaveformClient

import logging
logging.getLogger("PNWstore").setLevel(logging.ERROR)
logging.getLogger("pnwstore").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SECONDS_BEFORE     = 0
SECONDS_AFTER      = 200
FREQMIN            = 1.0
FREQMAX            = 8.0
LOWPASS            = 0.2
DT                 = 10
TARGET_FS_WAVEFORM = 50.0
TARGET_FS_ENVELOPE = 5.0
SNR_THRESHOLD      = 5.0
PNWSTORE_END_YEAR  = 2022
FDSN_CLIENT        = "IRIS"
MINLAT, MAXLAT     = 46.0, 49.0
MINLON, MAXLON     = -123.0, -120.0
CHECKPOINT_EVERY   = 1000   # save a checkpoint CSV every N located events

EXCLUDE_CHANNEL_PREFIXES = (
    "DF", "HDF", "LDF", "BDF", "EDF", "LDO", "LKO",
    "LWS", "LWD", "LRI", "LDI", "VM", "LD",
)

_DONE = object()

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Waveform download helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_excluded_channel(channel: str) -> bool:
    return any(channel.upper().startswith(p) for p in EXCLUDE_CHANNEL_PREFIXES)


def resolve_net_loc_chan(fdsn_client, station, t_start, t_end,
                         loc_preference=("01", "", "--")):
    try:
        inv = fdsn_client.get_stations(
            network="*", station=station, channel="*HZ",
            starttime=t_start, endtime=t_end,
            minlatitude=MINLAT, maxlatitude=MAXLAT,
            minlongitude=MINLON, maxlongitude=MAXLON,
            level="channel",
        )
    except Exception:
        return None

    candidates = []
    for net in inv:
        for sta in net:
            for cha in sta:
                if is_excluded_channel(cha.code):
                    continue
                loc = cha.location_code or ""
                candidates.append((net.code, sta.code, loc, cha.code))

    if not candidates:
        return None

    preferred = [c for c in candidates if c[2] in loc_preference]
    pool = preferred if preferred else candidates
    loc_rank = {loc: i for i, loc in enumerate(loc_preference)}
    pool.sort(key=lambda x: (loc_rank.get(x[2], 999), x[3]))
    return pool[0]


def download_waveform_fdsn(fdsn_client, station, t_start, t_end) -> Stream:
    resolved = resolve_net_loc_chan(fdsn_client, station, t_start, t_end)
    if resolved is None:
        return Stream()
    net, sta, loc, _ = resolved
    for ch in ("BHZ", "HHZ", "EHZ"):
        if is_excluded_channel(ch):
            continue
        try:
            st = fdsn_client.get_waveforms(net, sta, loc, ch, t_start, t_end)
            if st:
                return st
        except Exception:
            pass
    return Stream()




def download_waveform_pnwstore(station, t_start, t_end) -> Stream:
    for ch in ("BHZ", "HHZ", "EHZ"):
        if is_excluded_channel(ch):
            continue
        try:
            client = WaveformClient()
            st = client.get_waveforms(
                starttime=t_start,
                endtime=t_end,
                station=station,
                channel=ch
            )
            if st:
                return st
        except Exception:
            pass
    return Stream()

def download_single_station(args):
    sta, year, t_start, t_end, fdsn_client = args
    if year <= PNWSTORE_END_YEAR:
        return download_waveform_pnwstore(sta, t_start, t_end)
    else:
        return download_waveform_fdsn(fdsn_client, sta, t_start, t_end)


def download_event(row, fdsn_client, station_pool) -> Stream:
    start_time = UTCDateTime(pd.Timestamp(row["rounded_start"]).to_pydatetime())
    t_start = start_time - SECONDS_BEFORE
    t_end   = start_time + SECONDS_AFTER
    year    = pd.Timestamp(row["rounded_start"]).year

    try:
        stations = ast.literal_eval(row["stations"])
    except Exception:
        stations = []

    if not stations:
        return Stream()

    tasks   = [(sta, year, t_start, t_end, fdsn_client) for sta in stations]
    futures = {station_pool.submit(download_single_station, t): t[0]
               for t in tasks}

    stream = Stream()
    for future in as_completed(futures):
        try:
            st = future.result()
            if st:
                stream += st
        except Exception as e:
            log.debug("  Station download error: %s", e)

    return stream


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing + enveloc
# ─────────────────────────────────────────────────────────────────────────────

def drop_stations_with_gaps(st, max_total_gap_s=0.0):
    out = Stream()
    for tr_id in sorted(set(tr.id for tr in st)):
        s = st.select(id=tr_id).copy()
        if not s:
            continue
        gaps = s.get_gaps()
        total_gap = sum(float(g[6]) for g in gaps) if gaps else 0.0
        if total_gap > max_total_gap_s:
            continue
        s.merge(method=1)
        out += s[0]
    return out


def robust_envelope_snr(tr, q=0.99, eps=1e-12):
    x = np.nan_to_num(np.asarray(tr.data, dtype=float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return np.nan
    env = np.abs(hilbert(x))
    med = np.median(env)
    if not np.isfinite(med) or med < eps:
        return np.nan
    return float(np.quantile(env, q) / med)


def filter_low_snr_traces(st, threshold=SNR_THRESHOLD):
    keep = [tr for tr in st
            if np.isfinite(robust_envelope_snr(tr))
            and robust_envelope_snr(tr) >= threshold]
    return Stream(keep)


def attach_coordinates(st_env, fdsn_client):
    for tr in st_env:
        try:
            inv = fdsn_client.get_stations(
                network=tr.stats.network,
                station=tr.stats.station,
                location=tr.stats.location,
                channel=tr.stats.channel,
                starttime=tr.stats.starttime,
                endtime=tr.stats.endtime,
            )
            tr.stats.coordinates = AttribDict({
                "latitude":  inv[0][0].latitude,
                "longitude": inv[0][0].longitude,
                "elevation": inv[0][0].elevation,
            })
        except Exception as e:
            log.debug("  Coords failed for %s: %s", tr.id, e)


def preprocess_and_locate(stream, fdsn_client):
    from enveloc.core import XCOR

    if not stream:
        return np.nan, np.nan, 0

    stream = drop_stations_with_gaps(stream)
    if not stream:
        return np.nan, np.nan, 0

    st_filt = stream.copy()
    st_filt.detrend("demean")
    st_filt.taper(max_percentage=None, max_length=5)
    st_filt.resample(TARGET_FS_WAVEFORM)
    st_filt.filter("bandpass", freqmin=FREQMIN, freqmax=FREQMAX,
                   corners=3, zerophase=True)

    st_filt = filter_low_snr_traces(st_filt)
    if not st_filt:
        return np.nan, np.nan, 0

    st_env = st_filt.copy()
    for tr in st_env:
        if tr.stats.npts % 2 == 1:
            tr.trim(tr.stats.starttime,
                    tr.stats.endtime + 1.0 / tr.stats.sampling_rate,
                    pad=True, fill_value=0)
        tr.data = envelope(tr.data)
        tr.resample(TARGET_FS_ENVELOPE)

    st_env.filter("lowpass", freq=LOWPASS)

    t1 = st_filt[0].stats.starttime + DT
    t2 = st_filt[0].stats.endtime   - DT
    st_filt.trim(t1, t2)
    st_env.trim(t1, t2)

    attach_coordinates(st_env, fdsn_client)

    st_env_ok = Stream([tr for tr in st_env
                        if hasattr(tr.stats, "coordinates")])
    if len(st_env_ok) < 3:
        return np.nan, np.nan, len(st_env_ok)

    n_traces = len(st_env_ok)
    try:
        XC = XCOR(st_env_ok, plot=False, interact=False)
        loc = XC.locate()
        lat = float(loc.latitude)  if loc.latitude  is not None else np.nan
        lon = float(loc.longitude) if loc.longitude is not None else np.nan
        return lat, lon, n_traces
    except Exception as e:
        log.debug("  enveloc failed: %s", e)
        return np.nan, np.nan, n_traces


# ─────────────────────────────────────────────────────────────────────────────
# Producer-consumer pipeline
# ─────────────────────────────────────────────────────────────────────────────

def producer(rows, fdsn_client, station_pool, result_queue, prefetch):
    row_list  = list(rows)
    n         = len(row_list)
    i         = 0
    in_flight = []

    while i < n or in_flight:
        while len(in_flight) < prefetch and i < n:
            idx, row = row_list[i]
            fut = station_pool.submit(download_event, row, fdsn_client,
                                      station_pool)
            in_flight.append((idx, row, fut))
            i += 1

        if in_flight:
            idx, row, fut = in_flight.pop(0)
            try:
                stream = fut.result()
            except Exception as e:
                log.error("  Download future error for idx=%s: %s", idx, e)
                stream = Stream()
            result_queue.put((idx, row, stream))

    result_queue.put(_DONE)


def save_output(catalog, out_cols, outfile):
    """Save current state of catalog to the output CSV."""
    cols = [c for c in out_cols if c in catalog.columns]
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    catalog[cols].to_csv(outfile, index=False)


def run_pipeline(catalog, fdsn_client, download_workers, prefetch,
                 outfile, out_cols):
    catalog["enveloc_latitude"]  = np.nan
    catalog["enveloc_longitude"] = np.nan
    catalog["n_traces_used"]     = np.nan
    catalog["location_status"]   = ""

    total       = len(catalog)
    located     = 0
    failed      = 0
    t0_run      = time.time()
    event_count = 0

    result_queue = Queue(maxsize=prefetch * 2)

    with ThreadPoolExecutor(max_workers=download_workers) as station_pool:

        prod = Thread(
            target=producer,
            args=(catalog.iterrows(), fdsn_client,
                  station_pool, result_queue, prefetch),
            daemon=True,
        )
        prod.start()

        while True:
            item = result_queue.get()
            if item is _DONE:
                break

            idx, row, stream = item
            event_count += 1

            log.info(
                "[%d/%d] event_id=%s @ %s  class=%s  stations=%d",
                event_count, total,
                row.get("event_id", idx),
                row["rounded_start"],
                row["most_common_class"],
                row["num_stations"],
            )

            if not stream:
                log.warning("  No waveforms — skipping.")
                catalog.at[idx, "location_status"] = "no_waveforms"
                failed += 1
                continue

            try:
                lat, lon, n_tr = preprocess_and_locate(stream, fdsn_client)
                catalog.at[idx, "n_traces_used"] = n_tr

                if np.isfinite(lat) and np.isfinite(lon):
                    catalog.at[idx, "enveloc_latitude"]  = lat
                    catalog.at[idx, "enveloc_longitude"] = lon
                    catalog.at[idx, "location_status"]   = "located"
                    located += 1
                    log.info("  ✓  lat=%.4f  lon=%.4f  (%d traces)",
                             lat, lon, n_tr)
                else:
                    catalog.at[idx, "location_status"] = "enveloc_failed"
                    failed += 1
                    log.warning("  ✗  enveloc nan  (%d traces)", n_tr)

            except KeyboardInterrupt:
                log.warning("Interrupted — saving partial results.")
                save_output(catalog, out_cols, outfile)
                break
            except Exception:
                log.error("  Unexpected error:\n%s", traceback.format_exc())
                catalog.at[idx, "location_status"] = "error"
                failed += 1

            # ── ETA every 50 events ───────────────────────────────────────
            if event_count % 50 == 0:
                elapsed   = time.time() - t0_run
                rate      = elapsed / event_count
                remaining = rate * (total - event_count)
                h = int(remaining // 3600)
                m = int((remaining % 3600) // 60)
                log.info(
                    "  Progress: %d/%d  |  %.1fs/event  |  ETA: %dh %dm",
                    event_count, total, rate, h, m,
                )

            # ── checkpoint every CHECKPOINT_EVERY events ──────────────────
            if event_count % CHECKPOINT_EVERY == 0:
                checkpoint_path = (
                    Path(outfile).parent /
                    f"{Path(outfile).stem}_checkpoint_{event_count}.csv"
                )
                save_output(catalog, out_cols, str(checkpoint_path))
                log.info("  Checkpoint saved → %s", checkpoint_path)

        prod.join()

    total_time = time.time() - t0_run
    log.info("─" * 60)
    log.info("Done in %.1f min.  Located: %d/%d  |  Failed: %d",
             total_time / 60, located, total, failed)

    return catalog


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Locate events from master catalog (producer-consumer pipeline)"
    )
    parser.add_argument("--catalog",
                        default="catalog_output/master_catalog.csv")
    parser.add_argument("--outfile",
                        default="catalog_output/located_events.csv")
    parser.add_argument("--nmax", type=int, default=None,
                        help="Process only the first N events after start_idx (testing)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Resume from this row index in the catalog (default: 0)")
    parser.add_argument("--download_workers", type=int, default=8,
                        help="Total threads for station downloads (default: 8)")
    parser.add_argument("--prefetch", type=int, default=4,
                        help="Events to download ahead of enveloc (default: 4)")
    parser.add_argument("--fdsn_client", default=FDSN_CLIENT)
    args = parser.parse_args()

    log.info("Loading catalog: %s", args.catalog)
    catalog = pd.read_csv(args.catalog)
    log.info("Total events in catalog: %d", len(catalog))

    # ── resume from start_idx ─────────────────────────────────────────────
    if args.start_idx > 0:
        catalog = catalog.iloc[args.start_idx:].copy()
        log.info("Resuming from row index %d  (%d events remaining)",
                 args.start_idx, len(catalog))

    # ── optional nmax cap after start_idx ─────────────────────────────────
    if args.nmax is not None:
        catalog = catalog.head(args.nmax).copy()
        log.info("Limiting to %d events (--nmax)", args.nmax)

    log.info("Pipeline config: download_workers=%d  prefetch=%d  checkpoint_every=%d",
             args.download_workers, args.prefetch, CHECKPOINT_EVERY)

    fdsn_client = Client(args.fdsn_client)

    out_cols = [
        "event_id", "cluster_id", "rounded_start", "year", "month",
        "num_stations", "most_common_class", "vote_margin",
        "mean_max", "mean_prob", "mean_auc_per_station",
        "high_confidence", "low_confidence", "ambiguous_class",
        "enveloc_latitude", "enveloc_longitude",
        "n_traces_used", "location_status",
    ]

    catalog = run_pipeline(
        catalog, fdsn_client,
        download_workers=args.download_workers,
        prefetch=args.prefetch,
        outfile=args.outfile,
        out_cols=out_cols,
    )

    save_output(catalog, out_cols, args.outfile)
    log.info("Output saved to: %s", args.outfile)


if __name__ == "__main__":
    main()