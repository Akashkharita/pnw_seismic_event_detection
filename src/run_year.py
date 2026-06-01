from datetime import datetime, timedelta
import subprocess

start = datetime(2010, 1, 1)
end = datetime(2026, 1, 1)

current = start

while current <= end:
    
    start_time = current.strftime("%Y-%m-%dT00:00:00")
    end_time = current.strftime("%Y-%m-%dT23:59:59")

    print("Running:", start_time)

    subprocess.run([
        "python", "daily_detection_dynamic.py",
        "--start", start_time,
        "--end", end_time,
        #"--stations_json", "mt_rainier_stations.json"
    ])

    current += timedelta(days=1)