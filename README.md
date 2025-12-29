
## Using the trained **QuakeXNet** model with **SeisBench**

This section explains how to (1) install SeisBench, (2) register the custom `QuakeXNet` model class, and (3) place the trained weights where SeisBench expects them.

### 1) Install SeisBench

If you don’t already have SeisBench installed, install it into your environment:

```bash
pip install seisbench
````

---

### 2) Copy the model definition into SeisBench

SeisBench discovers models from its internal `seisbench/models/` package. Copy your custom model file there:

```bash
cp src/quakexnet.py <PATH_TO_SITE_PACKAGES>/seisbench/models/quakexnet.py
```

> The target folder is the installed package directory:
> `seisbench/seisbench/models/`

To quickly locate the correct `models/` directory:

```bash
python -c "import seisbench, os; print(os.path.join(os.path.dirname(seisbench.__file__), 'models'))"
```

---

### 3) Register `QuakeXNet` inside `seisbench.models`

Open:

```
<PATH_TO_SITE_PACKAGES>/seisbench/models/__init__.py
```

Add this import:

```python
from .quakexnet import QuakeXNet
```

This makes the model available as:

```python
import seisbench.models as sbm
sbm.QuakeXNet
```

---

### 4) Add trained weights to the SeisBench cache

SeisBench caches model files under `~/.seisbench/`. Create the expected directory:

```bash
mkdir -p ~/.seisbench/models/v3/quakexnet
```

Copy the trained weights file:

```bash
cp src/base.pt.v3 ~/.seisbench/models/v3/quakexnet/base.pt.v3
```

---

### 5) Create the minimal metadata file

In the same cache directory, create `base.json.v3` with an empty JSON object:

```bash
echo '{}' > ~/.seisbench/models/v3/quakexnet/base.json.v3
```

Final directory layout:

```text
~/.seisbench/models/v3/quakexnet/
├── base.pt.v3
└── base.json.v3
```

---

### 6) Import and use the model

Once the model is registered and the weights are in place:

```python
import seisbench.models as sbm

model = sbm.QuakeXNet()
```

---

# Seismic Event Detection Pipeline

This directory contains code to run a trained **QuakeXNet** model on continuous seismic data, detect events at **individual stations**, and then combine detections to identify **network-level common events** observed across multiple stations.

---

## Directory Overview

### 1) `src/custom_daily_detection.py`

**Purpose:** Run QuakeXNet on continuous waveform data for each station and log per-station event detections.

#### Workflow

1. **Load model & station list**

   * Loads the pre-trained **QuakeXNet** model.
   * Loads stations from `stations.json`.
   * Uses a **user-defined** start/end time window.

2. **Download continuous waveform data**

   * Downloads waveform data from **IRIS** using **ObsPy** for each station.

3. **Run model inference (sliding window)**

   * Window length: **100 s**
   * Stride: **10 s**
   * Produces per-window class probabilities for:

     * `eq` (earthquake)
     * `px` (explosion/phase)
     * `su` (surface event)

4. **Smooth probability curves**

   * Applies a **5-sample moving average** (≈ 50 s) to reduce short spikes and noise fluctuations.

5. **Detect events from smoothed probabilities**

   * **Start condition:** smoothed probability ≥ **0.15**
   * **End condition:** smoothed probability < **0.15**
   * Event is recorded only if **max probability ≥ 0.5** (default).
   * For each detected event, the script records:

     * `mean_prob`, `max_prob`, `auc`
     * start/end indices
     * start/end UTC timestamps

#### Output

* One CSV **per station** with detections and metrics. Example:

| station | network | class | auc  | mean_prob | max_prob | start_index | end_index | start_time           | end_time             |
| ------- | ------- | ----- | ---- | --------- | -------- | ----------- | --------- | -------------------- | -------------------- |
| PARA    | CC      | eq    | 3.37 | 0.35      | 0.54     | 5429        | 5438      | 2025-12-13T14:44:22Z | 2025-12-13T14:45:52Z |
| PARA    | CC      | eq    | 7.02 | 0.60      | 0.96     | 5561        | 5572      | 2025-12-13T15:06:22Z | 2025-12-13T15:08:12Z |

---

### 2) `src/custom_generate_common_events.py`

**Purpose:** Aggregate per-station detections and identify **network-level common events** detected across multiple stations.

#### Workflow

1. **Merge per-station detection CSVs**

   * Loads all station CSVs for the chosen day/time range and concatenates them.

2. **Time-align detections**

   * Rounds event start times to the nearest **10 seconds** to align slightly offset detections.
   * Example: `12:10:43 → 12:10:40`, `12:10:46 → 12:10:50`

3. **Group and compute aggregated metrics**

   * Groups detections by the rounded start time and computes:

     * `num_stations`: number of **unique stations** that detected something in that window
     * `stations`: list of stations in the group
     * `most_common_class`: most frequent class across stations (ties broken by first)
     * `mean_auc`, `mean_max`, `mean_prob`: averages across **all detections in the group**, regardless of class

4. **Filter to common events**

   * Keeps only events detected by at least **N stations** (default: **4**).

#### Output

* One CSV **per day** with network-level common events. Example:

| rounded_start             | num_stations | stations                        | most_common_class | mean_auc | mean_max | mean_prob |
| ------------------------- | ------------ | ------------------------------- | ----------------- | -------- | -------- | --------- |
| 2025-08-03 20:03:30+00:00 | 4            | ['RCM', 'RER', 'STAR', 'OBSR']  | su                | 4.72     | 0.73     | 0.42      |
| 2025-08-03 23:28:40+00:00 | 4            | ['RER', 'STAR', 'PANH', 'MILD'] | px                | 3.60     | 0.61     | 0.35      |

---

## Notes on Metrics

* **num_stations**: Count of unique stations with a detection in the rounded time window (any class).
* **stations**: List of stations contributing detections to the grouped event.
* **most_common_class**: Most frequent class among station detections (ties broken by first).
* **mean_auc / mean_max / mean_prob**: Mean values computed across all detections in the group (not restricted to the most common class).

---

## Summary

1. `custom_daily_detection.py` runs QuakeXNet per station, smooths probabilities, and logs event windows + confidence metrics.
2. `custom_generate_common_events.py` time-aligns and aggregates detections across stations to produce a network-level catalog.
3. Outputs include per-station CSVs and daily network-level CSVs with both timing and confidence statistics.

```

