#!/bin/bash
START_DATE="2020-12-31"
END_DATE="2025-12-31"
current_date=$START_DATE

while [[ "$current_date" != "$(date -I -d "$END_DATE + 1 day")" ]]; do
    yyyy=$(date -d "$current_date" +%Y)
    mm=$(date -d "$current_date" +%m)
    dd=$(date -d "$current_date" +%d)
    
    start_ts="${yyyy}-${mm}-${dd}T00:00:00"
    end_ts="${yyyy}-${mm}-${dd}T23:59:59"
    dir_date="${yyyy}${mm}${dd}_0000_${yyyy}${mm}${dd}_2359"
    input_dir="../logs/mt_rainier_detections/${dir_date}"
    output_dir="../logs/mt_rainier_common_detections_v3"

    echo "Processing $current_date..."
    python gen_com_events_v3.py \
        --start "$start_ts" \
        --end "$end_ts" \
        --input_dir "$input_dir" \
        --output_dir "$output_dir" \
        --min_stations 3 \
        --gap_seconds 20

    current_date=$(date -I -d "$current_date + 1 day")
done