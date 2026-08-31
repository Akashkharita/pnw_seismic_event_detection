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

══════════════════════════════════════════════════════════════════════════════
PERFORMANCE OPTIMIZATIONS (this version)
══════════════════════════════════════════════════════════════════════════════

[OPT-2] Pre-warm coordinate cache at startup
    Previous code called fdsn_client.get_stations() once per (net,sta,loc,chan)
    on first sight. That's hundreds of round-trips during the first hour of a
    run. Now we make ONE bbox-wide get_stations() call per year present in the
    catalog at startup and populate _coords_cache up-front.

    Disable with --no_prewarm_coords if FDSN is unavailable or restricted.

[OPT-3] Skip the BHZ→HHZ→EHZ loop on the resolved channel
    resolve_net_loc_chan already picks the best available channel for that
    station-year. Previously download_waveform_fdsn / download_waveform_pnwstore
    then ignored that and re-tried all three channels in order, wasting up to
    2 FDSN/pnwstore requests per station. Now we use the resolved channel
    directly and only fall back to the loop if resolution failed.

[OPT-4] Wall-clock timing instrumentation
    Per-stage timers (download / preprocess / locate) plus aggregate counters
    are printed every 50 events and at the end of the run. Compare the final
    summary line against the old script's wall-clock to see the speedup.

[OPT-1] (opt-in, OFF by default) Traveltime caching via --enable_tt_cache
    Enable a disk-backed traveltime cache keyed by (station_set, grid_hash).
    First explored as the headline optimization, but in practice:
      • Post-SNR surviving station sets vary a lot event-to-event, so the
        cache hit rate stays low on small runs.
      • The fixed grid required for cache keying is often LARGER than the
        XCOR auto-grid (which sizes itself to the station spread per event),
        making every cache MISS slower than the old per-event build, on top
        of paying disk I/O for save_traveltimes().
    Net effect on small runs (--nmax 20–50): strictly slower than baseline.
    Worth trying ONLY on long runs where the same station set genuinely
    repeats across many events. Default behavior matches the old XCOR call
    (no fixed grid, no tt_file) so locate time is at parity with the old
    script.

══════════════════════════════════════════════════════════════════════════════
BUG FIXES (carried over from previous version, unchanged)
══════════════════════════════════════════════════════════════════════════════
  [1] filter_low_snr_traces: snr computed once per trace
  [2] save_output: checkpoints contain ALL processed rows; final filters to located
  [3/6] download_event uses a private inner ThreadPoolExecutor (no deadlock)
  [4] Resume by event_id (--resume_after_event_id), legacy --start_idx kept
  [5] Metadata + coord caches guarded by threading.Lock

Resuming interrupted runs:
  - --resume_after_event_id (preferred) or --start_idx (legacy)
  - Checkpoints written every CHECKPOINT_EVERY events
  - The traveltime cache directory is reused across runs automatically

Usage:
    conda activate enveloc

    # full run
    python3 locate_events.py \\
        --catalog catalog_output/master_catalog.csv \\
        --outfile catalog_output/located_events.csv \\
        --download_workers 4 --prefetch 2

    # resume from a specific event_id (preferred)
    python locate_events.py \\
        --catalog catalog_output/master_catalog.csv \\
        --outfile catalog_output/located_events.csv \\
        --download_workers 4 --prefetch 2 \\
        --resume_after_event_id EV20150304_001

    # test on first 50 events
    python locate_events.py \\
        --catalog catalog_output/master_catalog.csv \\
        --outfile catalog_output/located_events.csv \\
        --download_workers 4 --prefetch 2 --nmax 50
"""

import argparse
import ast
import hashlib
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

# pnwstore (UW-internal waveform archive) is optional and imported lazily in
# get_pnw_client(); without it, PNWSTORE_AVAILABLE stays False and every year
# is served by FDSN instead.
try:
    from pnwstore import WaveformClient
    PNWSTORE_AVAILABLE = True
except ImportError:
    WaveformClient = None
    PNWSTORE_AVAILABLE = False

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

# OPT-1 is OFF by default. Toggled on via --enable_tt_cache. When False,
# XCOR is built exactly the same way the old script built it: no fixed
# grid, no tt_file. This guarantees parity with the old locate timings.
ENABLE_TT_CACHE = False

# Number of worker threads used inside download_event for per-station fetches.
# NOTE: no longer used in this version. The earlier per-event ThreadPoolExecutor
# leaked file descriptors (one pnwstore SQLite client per worker thread, never
# closed) and caused "Too many open files" after a few hundred events. Station
# downloads inside one event are now sequential; outer pipeline pool still runs
# `download_workers` events concurrently. Kept here in case you want to bring
# back station parallelism via a singleton module-level pool.
STATION_DOWNLOAD_WORKERS = 8

EXCLUDE_CHANNEL_PREFIXES = (
    "DF", "HDF", "LDF", "BDF", "EDF", "LDO", "LKO",
    "LWS", "LWD", "LRI", "LDI", "VM", "LD",
)

# ─── OPT-1: Fixed grid for traveltime caching ──────────────────────────────
# Defining the grid up-front (rather than letting XCOR auto-generate one
# per event from station spread) is what makes the (stations, grid) cache
# key stable. Tweak ranges/spacing to suit your science needs.
LOC_GRID = {
    "lats": np.arange(MINLAT, MAXLAT + 0.001, 0.02),  # ~150 nodes
    "lons": np.arange(MINLON, MAXLON + 0.001, 0.02),
    "deps": np.arange(0.0, 15.0 + 0.001, 1.0),        # 0–15 km, 1 km step
}

# Where pre-computed traveltime .npz files live. Reused across runs.
TT_CACHE_DIR = Path("catalog_output/tt_cache")

_DONE = object()

# ─────────────────────────────────────────────────────────────────────────────
# Caches (all guarded by locks for thread-safety)
# ─────────────────────────────────────────────────────────────────────────────

_station_metadata_cache      = {}
_station_metadata_cache_lock = threading.Lock()

_coords_cache      = {}
_coords_cache_lock = threading.Lock()

# OPT-1: maps frozenset(trace_ids) → path to the .npz traveltime file.
_tt_cache      = {}
_tt_cache_lock = threading.Lock()

# OPT-4: stage timers
_stage_times = {"download": 0.0, "preprocess": 0.0, "locate": 0.0}
_tt_hits     = 0
_tt_misses   = 0
_stage_lock  = threading.Lock()

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
# OPT-1: traveltime cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grid_hash(grid: dict) -> str:
    """Stable hash of the grid dict so the cache invalidates if grid changes."""
    parts = []
    for key in ("lats", "lons", "deps"):
        arr = np.asarray(grid[key], dtype=np.float64)
        parts.append(key.encode())
        parts.append(arr.tobytes())
    h = hashlib.sha1(b"|".join(parts)).hexdigest()[:16]
    return h


_GRID_HASH = _grid_hash(LOC_GRID)


def _tt_path_for_stations(trace_ids) -> Path:
    """Return the cache .npz path for a given set of trace IDs + the fixed grid."""
    # frozenset → sorted tuple → stable hash
    key = "|".join(sorted(trace_ids))
    sta_hash = hashlib.sha1(key.encode()).hexdigest()[:16]
    return TT_CACHE_DIR / f"tt_{_GRID_HASH}_{sta_hash}.npz"


def _build_xcor_with_cache(st_env_ok):
    """
    Build an XCOR object.

    When ENABLE_TT_CACHE is False (the default), this is exactly the same
    XCOR call the old script made: no grid_size, no tt_file. Locate timings
    therefore match the old version.

    When ENABLE_TT_CACHE is True, uses a disk-backed (station_set, grid)
    cache. See the OPT-1 note in the module header for caveats — this is
    only worth turning on for long runs where station sets genuinely repeat.
    """
    from enveloc.core import XCOR

    global _tt_hits, _tt_misses

    if not ENABLE_TT_CACHE:
        with suppress_stdout():
            XC = XCOR(st_env_ok, plot=False, interact=False)
        return XC

    trace_ids = frozenset(tr.id for tr in st_env_ok)
    tt_path   = _tt_path_for_stations(trace_ids)

    with _tt_cache_lock:
        cached_path = _tt_cache.get(trace_ids)

    if cached_path is None and tt_path.exists():
        with _tt_cache_lock:
            _tt_cache[trace_ids] = tt_path
        cached_path = tt_path

    if cached_path is not None and cached_path.exists():
        with _stage_lock:
            _tt_hits += 1
        with suppress_stdout():
            XC = XCOR(
                st_env_ok,
                grid_size=LOC_GRID,
                plot=False,
                interact=False,
                tt_file=str(cached_path),
            )
        return XC

    # Cache miss — build fresh and persist.
    with _stage_lock:
        _tt_misses += 1
    with suppress_stdout():
        XC = XCOR(
            st_env_ok,
            grid_size=LOC_GRID,
            plot=False,
            interact=False,
        )
        try:
            TT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            XC.save_traveltimes(str(tt_path))
            with _tt_cache_lock:
                _tt_cache[trace_ids] = tt_path
        except Exception as e:
            log.debug("  save_traveltimes failed for %s: %s", tt_path.name, e)
    return XC


# ─────────────────────────────────────────────────────────────────────────────
# OPT-2: Pre-warm coordinate cache from a single bbox-wide call per year
# ─────────────────────────────────────────────────────────────────────────────

def prewarm_coords_cache(fdsn_client, years):
    """
    Issue ONE get_stations() call per year covering the entire bbox and
    populate _coords_cache for every (net, sta, loc, chan) it returns.
    """
    if not years:
        return
    log.info("Pre-warming coords cache for %d year(s): %s",
             len(years), sorted(years))
    total_warmed = 0
    for year in sorted(years):
        try:
            t1 = UTCDateTime(year=int(year), month=1,  day=1)
            t2 = UTCDateTime(year=int(year), month=12, day=31, hour=23, minute=59)
            inv = fdsn_client.get_stations(
                network="*", station="*", channel="*",
                starttime=t1, endtime=t2,
                minlatitude=MINLAT, maxlatitude=MAXLAT,
                minlongitude=MINLON, maxlongitude=MAXLON,
                level="channel",
            )
        except Exception as e:
            log.warning("  Pre-warm get_stations failed for %s: %s", year, e)
            continue

        with _coords_cache_lock:
            for net in inv:
                for sta in net:
                    coords = AttribDict({
                        "latitude":  sta.latitude,
                        "longitude": sta.longitude,
                        "elevation": sta.elevation,
                    })
                    for cha in sta:
                        if is_excluded_channel(cha.code):
                            continue
                        loc = cha.location_code or ""
                        cache_key = (net.code, sta.code, loc, cha.code)
                        if cache_key not in _coords_cache:
                            _coords_cache[cache_key] = coords
                            total_warmed += 1
    log.info("  → %d (net,sta,loc,chan) entries pre-warmed", total_warmed)


# ─────────────────────────────────────────────────────────────────────────────
# Waveform download helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_excluded_channel(channel: str) -> bool:
    return any(channel.upper().startswith(p) for p in EXCLUDE_CHANNEL_PREFIXES)


def resolve_net_loc_chan(fdsn_client, station, t_start, t_end,
                         loc_preference=("01", "", "--")):
    """
    Query FDSN for network/location/channel. Results cached by (station, year).
    """
    year      = t_start.year
    cache_key = (station, year)

    with _station_metadata_cache_lock:
        if cache_key in _station_metadata_cache:
            return _station_metadata_cache[cache_key]

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


# OPT-3: helper that tries a single (net, loc, chan) combo
def _try_fdsn_one(fdsn_client, net, sta, loc, chan, t_start, t_end):
    try:
        st = fdsn_client.get_waveforms(net, sta, loc, chan, t_start, t_end)
        if st:
            return st
    except Exception:
        pass
    return None


def download_waveform_fdsn(fdsn_client, station, t_start, t_end) -> Stream:
    """
    OPT-3: Use the resolved channel directly. The previous version ignored
    `resolved` and looped BHZ → HHZ → EHZ, wasting up to 2 FDSN calls per
    station miss. Now we try the resolved channel first, and only fall back
    to the loop if (a) resolution failed entirely or (b) the resolved channel
    happens to have no data for this exact window.
    """
    resolved = resolve_net_loc_chan(fdsn_client, station, t_start, t_end)

    if resolved is not None:
        net, sta, loc, resolved_chan = resolved
        if not is_excluded_channel(resolved_chan):
            st = _try_fdsn_one(fdsn_client, net, sta, loc, resolved_chan,
                               t_start, t_end)
            if st:
                return st
        # Fall through to the loop only if the resolved channel had no data
        # for this specific window (rare — usually means a gap).
        for ch in ("BHZ", "HHZ", "EHZ"):
            if ch == resolved_chan or is_excluded_channel(ch):
                continue
            st = _try_fdsn_one(fdsn_client, net, sta, loc, ch, t_start, t_end)
            if st:
                return st
        return Stream()

    # No metadata at all — nothing to do
    return Stream()


_thread_local = threading.local()

def get_pnw_client():
    """Thread-local WaveformClient — one per thread, reused across calls."""
    if not hasattr(_thread_local, "pnw_client") or _thread_local.pnw_client is None:
        _thread_local.pnw_client = WaveformClient()
    return _thread_local.pnw_client


def download_waveform_pnwstore(station, t_start, t_end,
                               preferred_chan: str | None = None) -> Stream:
    """
    OPT-3: If we know the preferred channel from prior FDSN resolution, try
    it first; only fall back to the BHZ/HHZ/EHZ sweep if it returns nothing.
    pnwstore doesn't need net/loc, just station + channel.
    """
    candidates = []
    if preferred_chan and not is_excluded_channel(preferred_chan):
        candidates.append(preferred_chan)
    for ch in ("BHZ", "HHZ", "EHZ"):
        if ch in candidates or is_excluded_channel(ch):
            continue
        candidates.append(ch)

    for ch in candidates:
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
            if hasattr(_thread_local, "pnw_client"):
                del _thread_local.pnw_client
    return Stream()


def download_single_station(args):
    sta, year, t_start, t_end, fdsn_client = args

    # OPT-3: figure out the preferred channel up front. For pnwstore years
    # we still call resolve_net_loc_chan because it's cached and gives us
    # the channel hint to pass into download_waveform_pnwstore. The FDSN
    # call is a no-op after the first miss in that station-year.
    preferred_chan = None
    resolved = resolve_net_loc_chan(fdsn_client, sta, t_start, t_end)
    if resolved is not None:
        preferred_chan = resolved[3]

    if year <= PNWSTORE_END_YEAR and PNWSTORE_AVAILABLE:
        return download_waveform_pnwstore(sta, t_start, t_end,
                                          preferred_chan=preferred_chan)
    else:
        return download_waveform_fdsn(fdsn_client, sta, t_start, t_end)


def download_event(row, fdsn_client) -> Stream:
    """
    Download waveforms for all stations associated with one event.

    BUGFIX (file descriptor leak): an earlier version of this function
    created a fresh ThreadPoolExecutor(max_workers=STATION_DOWNLOAD_WORKERS)
    *per event*. Each worker thread held a thread-local pnwstore
    WaveformClient (= an open SQLite connection = an open file descriptor),
    and those FDs were never closed. After ~700 events on a default ulimit
    of 1024, checkpoint writes started failing with OSError [Errno 24]
    "Too many open files", which then took down the producer thread with
    "cannot schedule new futures after shutdown".

    The original motivation for the inner pool was deadlock avoidance — we
    couldn't submit station tasks back into the *outer* pipeline pool. But
    we don't actually need parallelism inside one event: the outer pipeline
    already has `download_workers` events in flight at once, so making each
    event sequential just means the outer pool sees more concurrent events
    instead of fewer events × more concurrent stations. Net throughput is
    similar and we stop leaking descriptors.

    If you find downloads are the bottleneck and want station-level
    parallelism back, do it with a *module-level* (singleton) pool created
    once in main() and reused — not a per-event pool.
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

    stream = Stream()
    for sta in stations:
        try:
            st = download_single_station(
                (sta, year, t_start, t_end, fdsn_client)
            )
            if st:
                stream += st
        except Exception as e:
            log.debug("  Station download error (%s): %s", sta, e)

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
    keep = []
    for tr in st:
        snr = robust_envelope_snr(tr)
        if np.isfinite(snr) and snr >= threshold:
            keep.append(tr)
    return Stream(keep)


def attach_coordinates(st_env, fdsn_client):
    """
    Attach station coordinates. Cache is pre-warmed at startup (OPT-2), so
    the FDSN fallback path here should rarely fire in steady state.
    """
    for tr in st_env:
        cache_key = (tr.stats.network, tr.stats.station,
                     tr.stats.location, tr.stats.channel)

        with _coords_cache_lock:
            cached = _coords_cache.get(cache_key)
        if cached is not None:
            tr.stats.coordinates = cached
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
    """
    Returns (lat, lon, n_traces_used, t_preprocess, t_locate).

    OPT-1: XCOR construction now goes through _build_xcor_with_cache, which
    reuses traveltime files on disk when the (station_set, grid) key matches.
    OPT-4: returns its own preprocess/locate timings so the caller can sum
    them into global stage timers.
    """
    if not stream:
        return np.nan, np.nan, 0, 0.0, 0.0

    t_pre0 = time.perf_counter()

    stream = drop_stations_with_gaps(stream)
    if not stream:
        return np.nan, np.nan, 0, time.perf_counter() - t_pre0, 0.0

    st_filt = stream.copy()
    st_filt.detrend("demean")
    st_filt.taper(max_percentage=None, max_length=5)
    st_filt.resample(TARGET_FS_WAVEFORM)
    st_filt.filter("bandpass", freqmin=FREQMIN, freqmax=FREQMAX,
                   corners=3, zerophase=True)

    st_filt = filter_low_snr_traces(st_filt)
    if not st_filt:
        return np.nan, np.nan, 0, time.perf_counter() - t_pre0, 0.0

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
    n_traces = len(st_env_ok)

    t_preprocess = time.perf_counter() - t_pre0

    if n_traces < 3:
        return np.nan, np.nan, n_traces, t_preprocess, 0.0

    t_loc0 = time.perf_counter()
    try:
        XC  = _build_xcor_with_cache(st_env_ok)   # OPT-1
        with suppress_stdout():
            loc = XC.locate()
        lat = float(loc.latitude)  if loc.latitude  is not None else np.nan
        lon = float(loc.longitude) if loc.longitude is not None else np.nan
        t_locate = time.perf_counter() - t_loc0
        return lat, lon, n_traces, t_preprocess, t_locate
    except Exception as e:
        log.debug("  enveloc failed: %s", e)
        return np.nan, np.nan, n_traces, t_preprocess, time.perf_counter() - t_loc0


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers (unchanged)
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
    if not processed_rows:
        return
    cols = [c for c in OUT_COLS if c in processed_rows[0]]
    pd.DataFrame(processed_rows)[cols].to_csv(path, index=False)
    log.info("  Checkpoint saved → %s  (%d rows)", path, len(processed_rows))


def save_final_output(processed_rows: list[dict], path: str):
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
    Submits download_event tasks and feeds (idx, row, stream, t_download)
    tuples into result_queue. OPT-4 adds the per-event download wall time
    so the consumer can roll it into global stage timers.
    """
    row_list  = list(rows)
    n         = len(row_list)
    i         = 0
    in_flight = []

    def _wrapped_download(row):
        t0 = time.perf_counter()
        st = download_event(row, fdsn_client)
        return st, time.perf_counter() - t0

    while i < n or in_flight:
        while len(in_flight) < prefetch and i < n:
            idx, row = row_list[i]
            try:
                fut = pipeline_pool.submit(_wrapped_download, row)
            except RuntimeError:
                # Pool was shut down (usually because the consumer raised an
                # error and exited the `with ...` block). Don't drown the
                # real error in a producer-side traceback — just stop
                # submitting and let the queue drain naturally.
                result_queue.put(_DONE)
                return
            in_flight.append((idx, row, fut))
            i += 1

        if in_flight:
            idx, row, fut = in_flight.pop(0)
            try:
                stream, t_download = fut.result()
            except Exception as e:
                log.error("  Download future error for idx=%s: %s", idx, e)
                stream, t_download = Stream(), 0.0
            result_queue.put((idx, row, stream, t_download))

    result_queue.put(_DONE)


def run_pipeline(catalog, fdsn_client, download_workers, prefetch, outfile):
    total               = len(catalog)
    located             = 0
    failed              = 0
    t0_run              = time.time()
    event_count         = 0
    processed_rows: list[dict] = []
    last_checkpoint_idx = 0

    # OPT-4: reset per-run stage timers
    global _stage_times, _tt_hits, _tt_misses
    _stage_times = {"download": 0.0, "preprocess": 0.0, "locate": 0.0}
    _tt_hits     = 0
    _tt_misses   = 0

    result_queue = Queue(maxsize=prefetch * 2)

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

            idx, row, stream, t_download = item
            event_count += 1

            with _stage_lock:
                _stage_times["download"] += t_download

            log.info(
                "[%d/%d] event_id=%s @ %s  class=%s  stations=%d",
                event_count, total,
                row.get("event_id", idx),
                row["rounded_start"],
                row["most_common_class"],
                row["num_stations"],
            )

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
                    lat, lon, n_tr, t_pre, t_loc = preprocess_and_locate(
                        stream, fdsn_client
                    )
                    record["n_traces_used"] = n_tr

                    with _stage_lock:
                        _stage_times["preprocess"] += t_pre
                        _stage_times["locate"]     += t_loc

                    if np.isfinite(lat) and np.isfinite(lon):
                        record["enveloc_latitude"]  = lat
                        record["enveloc_longitude"] = lon
                        record["location_status"]   = "located"
                        located += 1
                        log.info(
                            "  ✓  lat=%.4f  lon=%.4f  (%d traces)  "
                            "[dl=%.1fs pre=%.1fs loc=%.1fs]",
                            lat, lon, n_tr, t_download, t_pre, t_loc,
                        )
                    else:
                        record["location_status"] = "enveloc_failed"
                        failed += 1
                        log.warning(
                            "  ✗  enveloc nan  (%d traces)  "
                            "[dl=%.1fs pre=%.1fs loc=%.1fs]",
                            n_tr, t_download, t_pre, t_loc,
                        )

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

            # OPT-4: ETA + stage breakdown every 50 events
            if event_count % 50 == 0:
                elapsed   = time.time() - t0_run
                rate      = elapsed / event_count
                remaining = rate * (total - event_count)
                h = int(remaining // 3600)
                m = int((remaining % 3600) // 60)

                with _stage_lock:
                    dl  = _stage_times["download"]   / event_count
                    pre = _stage_times["preprocess"] / event_count
                    loc = _stage_times["locate"]     / event_count
                    tt_total = _tt_hits + _tt_misses
                    hit_rate = (100.0 * _tt_hits / tt_total) if tt_total else 0.0

                if ENABLE_TT_CACHE:
                    log.info(
                        "  Progress: %d/%d  |  %.2fs/event wall  |  "
                        "stages: dl=%.2fs pre=%.2fs loc=%.2fs  |  "
                        "tt_cache: %d hits / %d misses (%.1f%%)  |  "
                        "ETA: %dh %dm",
                        event_count, total, rate,
                        dl, pre, loc,
                        _tt_hits, _tt_misses, hit_rate,
                        h, m,
                    )
                else:
                    log.info(
                        "  Progress: %d/%d  |  %.2fs/event wall  |  "
                        "stages: dl=%.2fs pre=%.2fs loc=%.2fs  |  ETA: %dh %dm",
                        event_count, total, rate,
                        dl, pre, loc,
                        h, m,
                    )

            if event_count % CHECKPOINT_EVERY == 0:
                new_rows = processed_rows[last_checkpoint_idx:]
                save_checkpoint(new_rows, _checkpoint_path(outfile, event_count))
                last_checkpoint_idx = len(processed_rows)

        prod.join()

    total_time = time.time() - t0_run

    # OPT-4: end-of-run summary you can paste alongside the old version
    log.info("─" * 70)
    log.info("RUN SUMMARY")
    log.info("  Events processed : %d", event_count)
    log.info("  Located          : %d", located)
    log.info("  Failed           : %d", failed)
    log.info("  Total wall time  : %.1f min  (%.2f s/event)",
             total_time / 60,
             total_time / max(event_count, 1))
    if event_count > 0:
        with _stage_lock:
            log.info("  Avg per-event stage times:")
            log.info("    download   : %.2f s",
                     _stage_times["download"]   / event_count)
            log.info("    preprocess : %.2f s",
                     _stage_times["preprocess"] / event_count)
            log.info("    locate     : %.2f s",
                     _stage_times["locate"]     / event_count)
            if ENABLE_TT_CACHE:
                tt_total = _tt_hits + _tt_misses
                hit_rate = (100.0 * _tt_hits / tt_total) if tt_total else 0.0
                log.info(
                    "  Traveltime cache : %d hits / %d misses (%.1f%% hit rate)",
                    _tt_hits, _tt_misses, hit_rate,
                )
    log.info("─" * 70)

    return processed_rows


def _checkpoint_path(outfile: str, event_count: int) -> str:
    p = Path(outfile)
    return str(p.parent / f"{p.stem}_checkpoint_{event_count}.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global TT_CACHE_DIR, ENABLE_TT_CACHE

    parser = argparse.ArgumentParser(
        description="Locate events from master catalog (producer-consumer pipeline)"
    )
    parser.add_argument("--catalog",
                        default="catalog_output/master_catalog.csv")
    parser.add_argument("--outfile",
                        default="catalog_output/located_events.csv")
    parser.add_argument("--nmax", type=int, default=None,
                        help="Limit to first N events after slicing (testing)")

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume_after_event_id", default=None,
        help="Skip all events up to and including this event_id (preferred)"
    )
    resume_group.add_argument(
        "--start_idx", type=int, default=0,
        help="Resume from this raw row index (legacy)"
    )

    parser.add_argument("--download_workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--fdsn_client", default=FDSN_CLIENT)
    parser.add_argument("--tt_cache_dir", default=str(TT_CACHE_DIR),
                        help="Directory for cached traveltime .npz files "
                             "(only used when --enable_tt_cache is set)")
    parser.add_argument("--enable_tt_cache", action="store_true",
                        help="Enable the (station_set, grid) traveltime cache. "
                             "OFF by default — see OPT-1 note in header. "
                             "Only worth enabling on long runs where station "
                             "sets repeat heavily.")
    parser.add_argument("--no_prewarm_coords", action="store_true",
                        help="Skip the bbox-wide coords pre-warm at startup")
    args = parser.parse_args()

    # Honor user override of TT cache dir + enable flag
    TT_CACHE_DIR    = Path(args.tt_cache_dir)
    ENABLE_TT_CACHE = bool(args.enable_tt_cache)

    log.info("Loading catalog: %s", args.catalog)
    catalog = pd.read_csv(args.catalog)
    log.info("Total events in catalog: %d", len(catalog))

    catalog = (
        catalog[catalog["most_common_class"].isin(["su", "px"])]
        .copy()
        .reset_index(drop=True)
    )
    log.info("Filtered to su/px only: %d events remaining", len(catalog))

    if args.resume_after_event_id is not None:
        if "event_id" not in catalog.columns:
            parser.error("--resume_after_event_id requires an 'event_id' column.")
        match = catalog.index[catalog["event_id"] == args.resume_after_event_id]
        if len(match) == 0:
            log.warning(
                "event_id '%s' not found — starting from the beginning.",
                args.resume_after_event_id,
            )
        else:
            resume_row = int(match[-1]) + 1
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

    if args.nmax is not None:
        catalog = catalog.head(args.nmax).copy()
        log.info("Limiting to %d events (--nmax)", args.nmax)

    log.info(
        "Pipeline config: download_workers=%d  prefetch=%d  "
        "station_download_workers=%d  checkpoint_every=%d",
        args.download_workers, args.prefetch,
        STATION_DOWNLOAD_WORKERS, CHECKPOINT_EVERY,
    )
    if ENABLE_TT_CACHE:
        log.info(
            "Grid config (tt_cache enabled): "
            "lat=[%.2f,%.2f]/%d  lon=[%.2f,%.2f]/%d  dep=[%.1f,%.1f]/%d  hash=%s",
            LOC_GRID["lats"][0], LOC_GRID["lats"][-1], len(LOC_GRID["lats"]),
            LOC_GRID["lons"][0], LOC_GRID["lons"][-1], len(LOC_GRID["lons"]),
            LOC_GRID["deps"][0], LOC_GRID["deps"][-1], len(LOC_GRID["deps"]),
            _GRID_HASH,
        )
        log.info("Traveltime cache dir: %s", TT_CACHE_DIR)
    else:
        log.info("Traveltime cache: DISABLED (XCOR builds match old script)")

    fdsn_client = Client(args.fdsn_client)

    # OPT-2: pre-warm the coordinates cache before processing starts.
    if not args.no_prewarm_coords and len(catalog):
        years = sorted({pd.Timestamp(s).year for s in catalog["rounded_start"]})
        prewarm_coords_cache(fdsn_client, years)

    processed_rows = run_pipeline(
        catalog, fdsn_client,
        download_workers=args.download_workers,
        prefetch=args.prefetch,
        outfile=args.outfile,
    )

    save_final_output(processed_rows, args.outfile)


if __name__ == "__main__":
    main()