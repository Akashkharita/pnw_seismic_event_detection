"""
run_detection_sequential.py
---------------------------
Runs daily_detection_dynamic.py sequentially day by day.
Use this for small date ranges or testing.
For the full 2010-2025 run use submit_detection_slurm.py instead.

Usage:
    python run_detection_sequential.py
    python run_detection_sequential.py --start 2020-01-01 --end 2020-01-31
"""

from datetime import datetime, timedelta
import subprocess
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=str, default="2010-01-01")
parser.add_argument("--end",   type=str, default="2026-01-01")
args = parser.parse_args()

start   = datetime.strptime(args.start, "%Y-%m-%d")
end     = datetime.strptime(args.end,   "%Y-%m-%d")
current = start
total   = (end - start).days + 1
done    = 0

print(f"Running {total} days from {args.start} to {args.end}")

while current <= end:
    start_time = current.strftime("%Y-%m-%dT00:00:00")
    end_time   = current.strftime("%Y-%m-%dT23:59:59")

    print(f"\n[{done+1}/{total}] {start_time}")

    result = subprocess.run([
        "python", "daily_detection_dynamic.py",
        "--start", start_time,
        "--end",   end_time,
        "--save_station_list",
    ])

    if result.returncode != 0:
        print(f"  WARNING: non-zero exit code {result.returncode} for {start_time}")

    current += timedelta(days=1)
    done    += 1

print(f"\nDone. {done} days processed.")