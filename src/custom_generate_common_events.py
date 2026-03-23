from obspy.clients.fdsn import Client
import obspy
import matplotlib.pyplot as plt
import re
import pandas as pd
from glob import glob
import argparse
from datetime import datetime


from datetime import datetime, timedelta


## running example - python custom_generate_common_events.py  --start 2025-12-10T00:00:00  --end   2025-12-10T23:59:59


# -------------------- User Inputs --------------------
parser = argparse.ArgumentParser(description="Generate common events from daily detection CSVs.")
parser.add_argument("--start", type=str, required=True, help="Start time in UTC, e.g., '2025-12-10 00:00'")
parser.add_argument("--end", type=str, required=True, help="End time in UTC, e.g., '2025-12-10 12:00'")
args = parser.parse_args()

user_start = datetime.strptime(args.start, "%Y-%m-%dT%H:%M:%S")
user_end   = datetime.strptime(args.end,   "%Y-%m-%dT%H:%M:%S")


# -------------------- Find matching CSVs --------------------
#all_files = glob.glob("logs/*_to_*_events.csv")


file_start = user_start.strftime("%Y%m%d_%H%M")
file_end = user_end.strftime("%Y%m%d_%H%M")



matched_files = glob(f'../logs/*{file_start}*{file_end}*')

if not matched_files:
    raise FileNotFoundError("No event files found overlapping the requested time range")


print(f"Found {len(matched_files)} event files")





# Load each into a DataFrame
dfs = [pd.read_csv(f) for f in matched_files]

# Check how many files loaded
print(f"Loaded {len(dfs)} event files")


# Combine all station events into one DataFrame
df_all = pd.concat(dfs, ignore_index=True)
print(df_all.head())


# Convert start_time to datetime if it's not already
df_all["start_time"] = pd.to_datetime(df_all["start_time"])



# --- Time-tolerance association (cluster by gaps) ---
GAP_SECONDS = 20  # try 15–30s depending on detector timing jitter

# Ensure sorted by time
df_all = df_all.sort_values("start_time").reset_index(drop=True)

# Compute time gaps between consecutive detections
dt = df_all["start_time"].diff().dt.total_seconds().fillna(1e9)

# New cluster whenever the gap is larger than tolerance
df_all["cluster_id"] = (dt > GAP_SECONDS).cumsum()

# (Optional but recommended) If a station has multiple detections in the same cluster,
# keep the strongest one so it doesn't pollute station lists / stats.
df_all = (df_all.sort_values("max_prob", ascending=False)
                .drop_duplicates(subset=["cluster_id", "station"], keep="first"))

# Group by cluster_id instead of rounded_start
grouped = df_all.groupby("cluster_id").agg(
    rounded_start=("start_time", "min"),  # keep a representative event time (earliest)
    num_stations=("station", "nunique"),
    stations=("station", lambda x: sorted(set(x))),
    all_classes=("class", lambda x: list(x)),
    most_common_class=("class", lambda x: x.mode()[0] if not x.mode().empty else "unknown"),
    mean_auc=("auc", "mean"),
    mean_max=("max_prob", "mean"),
    mean_prob=("mean_prob", "mean"),
).reset_index()




# Set threshold
N = 4

# Filter the grouped DataFrame
common_events = grouped[grouped["num_stations"] >= N].copy()

# View or save
print(common_events)



# Save output
output_file = f"../logs/common_{args.start}_to_{args.end}_events.csv"
common_events.to_csv(output_file, index=False)
print(f"Saved common events to {output_file}")