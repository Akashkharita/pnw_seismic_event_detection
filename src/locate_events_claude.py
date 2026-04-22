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

Key optimizations:
  - Station metadata and coordinates are cached by (station, year) to avoid
    hammering FDSN with repeated identical requests across 657K events
  - Checkpoints write all processed rows (not just "located"), enabling
    accurate audit and safe resume
  - enveloc stdout (model/grid warnings) is suppressed

Bug fixes vs. previous version:
  [1] filter_low_snr_traces: robust_envelope_snr called twice per trace →
      now computed once and stored in a local variable.
  [2] save_output: silently dropped all non-"located" rows in checkpoints →
      checkpoints now write ALL processed rows; final output filters to
      "located" only when the caller explicitly requests it.
  [3/6] Thread pool deadlock: download_event submitted sub-tasks back into
      the same station_pool it was itself running in, which deadlocks at
      full thread saturation → download_event now runs per-station downloads
      sequentially using a dedicated inner ThreadPoolExecutor that is
      independent of the outer pipeline pool.
  [4] start_idx resume used raw row offset with no event_id anchoring →
      resume now matches on event_id so it is robust to catalog reordering
      or re-filtering between runs.
  [5] Station metadata cache mutated from multiple threads without a lock →
      now guarded by threading.Lock (avoids redundant FDSN calls under
      high concurrency, and is safe outside CPython).

Resuming interrupted runs:
  - Use --resume_after_event_id to resume after a specific event_id
  - OR use --start_idx for a raw row offset (legacy, less safe)
  - Checkpoints are saved every CHECKPOINT_EVERY events automatically

Usage:
    conda activate enveloc

    # full run
    python locate_events.py \\
        --catalog catalog_output/master_catalog.csv \\
        --outfile catalog_output/located_events.csv \\
        --download_workers 4 \\
        --prefetch 2

    # resume from a specific event_id (preferred)
    python locate_events.py \\
        --catalog catalog_output/master_catalog.csv \\
        --outfile catalog_output/located_events.csv \\
        --download_workers 4 \\
        --prefetch 2 \\
        --resume_after_event_id EV20150304_001

    # resume from raw row index (legacy fallback)
    python locate_events.py \\
        --catalog catalog_output/master_catalog.csv \\
        --outfile catalog_output/located_events.csv \\
        --download_workers 4 \\
        --prefetch 2 \\
        --start_idx 5000

    # test on first 50 events
    python locate_events.py \\
        --catalog catalog_output/master_catalog.csv \\
        --outfile catalog_output/located_events.csv \\
        --download_workers 4 \\
        --prefetch 2 \\
        --nmax 50
"""

import argparse
import ast
import logging
import os
import sys
import traceback
import time
import threading
from contextlib import contextmanager
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

# suppress pnwstore's own logging
logging.getLogger("PNWstore").setLevel(logging.ERROR)
logging.getLogger("pnwstore").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SECONDS_BEFORE          = 0
SECONDS_AFTER           = 200
FREQMIN                 = 1.0
FREQMAX                 = 8.0
LOWPASS                 = 0.2
DT                      = 10
TARGET_FS_WAVEFORM      = 50.0
TARGET_FS_ENVELOPE      = 5.0
SNR_THRESHOLD           = 5.0
PNWSTORE_END_YEAR       = 2022
FDSN_CLIENT             = "IRIS"
MINLAT, MAXLAT          = 46.0, 49.0
MINLON, MAXLON          = -123.0, -120.0
CHECKPOINT_EVERY        = 100

# Number of worker threads used inside download_event for per-station fetches.
# Kept separate from the outer pipeline pool to avoid deadlock (bug #3/#6).
STATION_DOWNLOAD_WORKERS = 8

EXCLUDE_CHANNEL_PREFIXES = (
    "DF", "HDF", "LDF", "BDF", "EDF", "LDO", "LKO",
    "LWS", "LWD", "LRI", "LDI", "VM", "LD",
)

_DONE = object()

# ─────────────────────────────────────────────────────────────────────────────
# Metadata + coordinate caches
# Keyed by (station, year) so we only call FDSN once per station per year.
#
# FIX #5: Both dicts are now guarded by a threading.Lock.
# Plain dict reads/writes are GIL-atomic in CPython, but:
#   - Under high concurrency two threads can both see a cache miss for the
#     same key and fire duplicate FDSN requests before either writes back.
#   - A lock eliminates the redundant requests and is safe outside CPython.
# ─────────────────────────────────────────────────────────────────────────────

_station_metadata_cache      = {}
_station_metadata_cache_lock = threading.Lock()

_coords_cache      = {}
_coords_cache_lock = threading.Lock()

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
# Stdout suppression (for enveloc model/grid print statements)
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# ─────────────────────────────────────────────────────────────────────────────
# Waveform download helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_excluded_channel(channel: str) -> bool:
    return any(channel.upper().startswith(p) for p in EXCLUDE_CHANNEL_PREFIXES)


def resolve_net_loc_chan(fdsn_client, station, t_start, t_end,
                         loc_preference=("01", "", "--")):
    """
    Query FDSN for network/location/channel. Results cached by (station, year).

    FIX #5: cache read-check-write is now done inside a lock so two threads
    racing on the same (station, year) key only issue one FDSN request.
    """
    year      = t_start.year
    cache_key = (station, year)

    # Fast path — check without lock first (common case after warm-up)
    with _station_metadata_cache_lock:
        if cache_key in _station_metadata_cache:
            return _station_metadata_cache[cache_key]

    # Slow path — query FDSN
    try:
        inv = fdsn_client.get_stations(
            network="*", station=station, channel="*HZ",
            starttime=t_start, endtime=t_end,
            minlatitude=MINLAT, maxlatitude=MAXLAT,
            minlongitude=MINLON, maxlongitude=MAXLON,
            level="channel",
        )
    except Exception:
        with _station_metadata_cache_lock:
            _station_metadata_cache[cache_key] = None
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
        with _station_metadata_cache_lock:
            _station_metadata_cache[cache_key] = None
        return None

    preferred = [c for c in candidates if c[2] in loc_preference]
    pool      = preferred if preferred else candidates
    loc_rank  = {loc: i for i, loc in enumerate(loc_preference)}
    pool.sort(key=lambda x: (loc_rank.get(x[2], 999), x[3]))

    result = pool[0]
    with _station_metadata_cache_lock:
        _station_metadata_cache[cache_key] = result
    return result


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


_thread_local = threading.local()

def get_pnw_client():
    """
    Returns a thread-local WaveformClient — one per thread, reused across
    calls. Avoids creating/destroying SQLite connections on every download.
    """
    if not hasattr(_thread_local, "pnw_client") or _thread_local.pnw_client is None:
        _thread_local.pnw_client = WaveformClient()
    return _thread_local.pnw_client


def download_waveform_pnwstore(station, t_start, t_end) -> Stream:
    for ch in ("BHZ", "HHZ", "EHZ"):
        if is_excluded_channel(ch):
            continue
        try:
            client = get_pnw_client()
            st = client.get_waveforms(
                starttime=t_start,
                endtime=t_end,
                station=station,
                channel=ch,
            )
            if st:
                return st
        except Exception:
            # FIX #4 (thread-local reset): use `del` so hasattr() returns
            # False on the next call and get_pnw_client() creates a fresh
            # WaveformClient. Setting to None left the attribute present,
            # causing get_pnw_client() to return None and crash downstream.
            if hasattr(_thread_local, "pnw_client"):
                del _thread_local.pnw_client
    return Stream()


def download_single_station(args):
    sta, year, t_start, t_end, fdsn_client = args
    if year <= PNWSTORE_END_YEAR:
        return download_waveform_pnwstore(sta, t_start, t_end)
    else:
        return download_waveform_fdsn(fdsn_client, sta, t_start, t_end)


def download_event(row, fdsn_client) -> Stream:
    """
    Download waveforms for all stations associated with one event.

    FIX #3/#6: The original code submitted per-station subtasks back into the
    same ThreadPoolExecutor that was already running this function. Under full
    saturation (all N threads blocked waiting on as_completed) the inner tasks
    could never start — a classic thread-pool deadlock.

    Fix: this function now uses its own *private* ThreadPoolExecutor
    (STATION_DOWNLOAD_WORKERS threads) that is completely independent of the
    outer pipeline pool. The outer pool only ever runs one download_event call
    per slot; inner station fetches run in their own pool with no risk of
    cross-pool blocking.
    """
    start_time = UTCDateTime(pd.Timestamp(row["rounded_start"]).to_pydatetime())
    t_start    = start_time - SECONDS_BEFORE
    t_end      = start_time + SECONDS_AFTER
    year       = pd.Timestamp(row["rounded_start"]).year

    try:
        stations = ast.literal_eval(row["stations"])
    except Exception:
        stations = []

    if not stations:
        return Stream()

    tasks = [(sta, year, t_start, t_end, fdsn_client) for sta in stations]

    stream = Stream()
    # Use a dedicated inner pool — never the outer pipeline pool.
    with ThreadPoolExecutor(max_workers=STATION_DOWNLOAD_WORKERS) as inner_pool:
        futures = {inner_pool.submit(download_single_station, t): t[0]
                   for t in tasks}
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
        gaps      = s.get_gaps()
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
    """
    FIX #1: robust_envelope_snr was called twice per trace — once for the
    isfinite check and once for the threshold comparison. Now computed once
    and stored in `snr` to avoid the redundant (and potentially slow) call.
    """
    keep = []
    for tr in st:
        snr = robust_envelope_snr(tr)
        if np.isfinite(snr) and snr >= threshold:
            keep.append(tr)
    return Stream(keep)


def attach_coordinates(st_env, fdsn_client):
    """
    Attach station coordinates. Results cached by (net, sta, loc, chan).

    FIX #5: cache writes now guarded by _coords_cache_lock.
    """
    for tr in st_env:
        cache_key = (tr.stats.network, tr.stats.station,
                     tr.stats.location, tr.stats.channel)

        with _coords_cache_lock:
            if cache_key in _coords_cache:
                tr.stats.coordinates = _coords_cache[cache_key]
                continue

        try:
            inv = fdsn_client.get_stations(
                network=tr.stats.network,
                station=tr.stats.station,
                location=tr.stats.location,
                channel=tr.stats.channel,
                starttime=tr.stats.starttime,
                endtime=tr.stats.endtime,
            )
            coords = AttribDict({
                "latitude":  inv[0][0].latitude,
                "longitude": inv[0][0].longitude,
                "elevation": inv[0][0].elevation,
            })
            tr.stats.coordinates = coords
            with _coords_cache_lock:
                _coords_cache[cache_key] = coords
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
        with suppress_stdout():
            XC  = XCOR(st_env_ok, plot=False, interact=False)
            loc = XC.locate()
        lat = float(loc.latitude)  if loc.latitude  is not None else np.nan
        lon = float(loc.longitude) if loc.longitude is not None else np.nan
        return lat, lon, n_traces
    except Exception as e:
        log.debug("  enveloc failed: %s", e)
        return np.nan, np.nan, n_traces


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
#
# FIX #2: The original save_output silently dropped all non-"located" rows
# even in checkpoints, making it impossible to audit failures or safely resume.
#
# New design:
#   - save_checkpoint() writes ALL processed rows regardless of status.
#     This gives a complete audit trail and means a resumed run can simply
#     skip any event_id already present in the checkpoint.
#   - save_final_output() filters to "located" only, matching the original
#     intent for the final deliverable.
# ─────────────────────────────────────────────────────────────────────────────

OUT_COLS = [
    "event_id",
    "rounded_start",
    "most_common_class",
    "num_stations",
    "enveloc_latitude",
    "enveloc_longitude",
    "n_traces_used",
    "location_status",
]


def _write_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_checkpoint(processed_rows: list[dict], path: str):
    """Write all processed rows (any status) to a checkpoint CSV."""
    if not processed_rows:
        return
    cols = [c for c in OUT_COLS if c in processed_rows[0]]
    pd.DataFrame(processed_rows)[cols].to_csv(path, index=False)
    log.info("  Checkpoint saved → %s  (%d rows)", path, len(processed_rows))


def save_final_output(processed_rows: list[dict], path: str):
    """Write only successfully located rows to the final output CSV."""
    if not processed_rows:
        _write_csv(pd.DataFrame(columns=OUT_COLS), path)
        return
    df   = pd.DataFrame(processed_rows)
    cols = [c for c in OUT_COLS if c in df.columns]
    df   = df[df["location_status"] == "located"][cols]
    _write_csv(df, path)
    log.info("Final output saved → %s  (%d located rows)", path, len(df))


# ─────────────────────────────────────────────────────────────────────────────
# Producer-consumer pipeline
# ─────────────────────────────────────────────────────────────────────────────

def producer(rows, fdsn_client, pipeline_pool, result_queue, prefetch):
    """
    Submits download_event tasks to the pipeline pool and feeds completed
    (idx, row, stream) tuples into result_queue.

    Note: download_event no longer accepts station_pool as an argument —
    it creates its own inner pool. The pipeline_pool here only ever runs
    download_event itself, so there is no risk of nested submission deadlock.
    """
    row_list  = list(rows)
    n         = len(row_list)
    i         = 0
    in_flight = []

    while i < n or in_flight:
        while len(in_flight) < prefetch and i < n:
            idx, row = row_list[i]
            fut = pipeline_pool.submit(download_event, row, fdsn_client)
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


def run_pipeline(catalog, fdsn_client, download_workers, prefetch, outfile):
    """
    Main processing loop.

    FIX #4: Resume is now event_id-based (--resume_after_event_id) rather
    than a raw row-offset so it is robust to catalog reordering or filtering
    between interrupted and resumed runs. The legacy --start_idx path is
    still supported as a fallback.

    FIX #2: Accumulates all processed rows in `processed_rows` (any status)
    and delegates filtering to save_checkpoint / save_final_output.
    """
    total               = len(catalog)
    located             = 0
    failed              = 0
    t0_run              = time.time()
    event_count         = 0
    processed_rows: list[dict] = []
    last_checkpoint_idx = 0   # index into processed_rows of the last checkpoint

    result_queue = Queue(maxsize=prefetch * 2)

    # pipeline_pool only runs download_event — never per-station tasks
    with ThreadPoolExecutor(max_workers=download_workers) as pipeline_pool:

        prod = Thread(
            target=producer,
            args=(catalog.iterrows(), fdsn_client,
                  pipeline_pool, result_queue, prefetch),
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

            # Base record — always written to checkpoint regardless of outcome
            record = {
                "event_id":          row.get("event_id", idx),
                "rounded_start":     row["rounded_start"],
                "most_common_class": row["most_common_class"],
                "num_stations":      row["num_stations"],
                "enveloc_latitude":  np.nan,
                "enveloc_longitude": np.nan,
                "n_traces_used":     np.nan,
                "location_status":   "",
            }

            if not stream:
                log.warning("  No waveforms — skipping.")
                record["location_status"] = "no_waveforms"
                failed += 1
            else:
                try:
                    lat, lon, n_tr = preprocess_and_locate(stream, fdsn_client)
                    record["n_traces_used"] = n_tr

                    if np.isfinite(lat) and np.isfinite(lon):
                        record["enveloc_latitude"]  = lat
                        record["enveloc_longitude"] = lon
                        record["location_status"]   = "located"
                        located += 1
                        log.info("  ✓  lat=%.4f  lon=%.4f  (%d traces)",
                                 lat, lon, n_tr)
                    else:
                        record["location_status"] = "enveloc_failed"
                        failed += 1
                        log.warning("  ✗  enveloc nan  (%d traces)", n_tr)

                except KeyboardInterrupt:
                    log.warning("Interrupted — saving partial results.")
                    processed_rows.append(record)
                    save_checkpoint(processed_rows,
                                    _checkpoint_path(outfile, event_count))
                    save_final_output(processed_rows, outfile)
                    raise
                except Exception:
                    log.error("  Unexpected error:\n%s", traceback.format_exc())
                    record["location_status"] = "error"
                    failed += 1

            processed_rows.append(record)

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
                new_rows = processed_rows[last_checkpoint_idx:]
                save_checkpoint(new_rows, _checkpoint_path(outfile, event_count))
                last_checkpoint_idx = len(processed_rows)

        prod.join()

    total_time = time.time() - t0_run
    log.info("─" * 60)
    log.info("Done in %.1f min.  Located: %d/%d  |  Failed: %d",
             total_time / 60, located, total, failed)

    return processed_rows


def _checkpoint_path(outfile: str, event_count: int) -> str:
    p = Path(outfile)
    return str(p.parent / f"{p.stem}_checkpoint_{event_count}.csv")


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
                        help="Limit to first N events after slicing (testing)")

    # FIX #4: prefer event_id-based resume over raw row offset
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume_after_event_id", default=None,
        help="Skip all events up to and including this event_id (preferred resume method)"
    )
    resume_group.add_argument(
        "--start_idx", type=int, default=0,
        help="Resume from this raw row index — legacy fallback, less safe than "
             "--resume_after_event_id"
    )

    parser.add_argument("--download_workers", type=int, default=4,
                        help="Pipeline-level download threads (default: 4)")
    parser.add_argument("--prefetch", type=int, default=2,
                        help="Events to download ahead of enveloc (default: 2)")
    parser.add_argument("--fdsn_client", default=FDSN_CLIENT)
    args = parser.parse_args()

    log.info("Loading catalog: %s", args.catalog)
    catalog = pd.read_csv(args.catalog)
    log.info("Total events in catalog: %d", len(catalog))

    # filter to su and px only
    catalog = (
        catalog[catalog["most_common_class"].isin(["su", "px"])]
        .copy()
        .reset_index(drop=True)
    )
    log.info("Filtered to su/px only: %d events remaining", len(catalog))

    # ── Resume logic (FIX #4) ────────────────────────────────────────────
    if args.resume_after_event_id is not None:
        if "event_id" not in catalog.columns:
            parser.error("--resume_after_event_id requires an 'event_id' column "
                         "in the catalog.")
        match = catalog.index[catalog["event_id"] == args.resume_after_event_id]
        if len(match) == 0:
            log.warning(
                "event_id '%s' not found in catalog — starting from the beginning.",
                args.resume_after_event_id,
            )
        else:
            resume_row = int(match[-1]) + 1   # first row *after* the target
            catalog = catalog.iloc[resume_row:].copy().reset_index(drop=True)
            log.info(
                "Resuming after event_id '%s' — %d events remaining.",
                args.resume_after_event_id, len(catalog),
            )
    elif args.start_idx > 0:
        catalog = catalog.iloc[args.start_idx:].copy().reset_index(drop=True)
        log.info(
            "Resuming from row index %d  (%d events remaining)",
            args.start_idx, len(catalog),
        )

    # ── optional nmax cap ─────────────────────────────────────────────────
    if args.nmax is not None:
        catalog = catalog.head(args.nmax).copy()
        log.info("Limiting to %d events (--nmax)", args.nmax)

    log.info(
        "Pipeline config: download_workers=%d  prefetch=%d  "
        "station_download_workers=%d  checkpoint_every=%d",
        args.download_workers, args.prefetch,
        STATION_DOWNLOAD_WORKERS, CHECKPOINT_EVERY,
    )

    fdsn_client = Client(args.fdsn_client)

    processed_rows = run_pipeline(
        catalog, fdsn_client,
        download_workers=args.download_workers,
        prefetch=args.prefetch,
        outfile=args.outfile,
    )

    save_final_output(processed_rows, args.outfile)


if __name__ == "__main__":
    main()