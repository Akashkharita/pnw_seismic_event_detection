## Using the trained **QuakeXNet** model with **SeisBench**

### 1) Install SeisBench

Install SeisBench in your environment (example via pip):

```bash
pip install seisbench
```

---

### 2) Add the custom model file to SeisBench

Copy your model definition into SeisBench’s models package:

```bash
cp src/quakexnet.py <PATH_TO_SITE_PACKAGES>/seisbench/models/quakexnet.py
```

> You should copy `src/quakexnet.py` into:
> `seisbench/seisbench/models/` (i.e., the installed package directory)

---

### 3) Register the model in `seisbench.models`

Open the following file:

```
<PATH_TO_SITE_PACKAGES>/seisbench/models/__init__.py
```

Add this line:

```python
from .quakexnet import QuakeXNet
```

---

### 4) Add the trained weights to the SeisBench cache directory

Create the target directory (if it doesn’t exist):

```bash
mkdir -p ~/.seisbench/models/v3/quakexnet
```

Copy the trained weights:

```bash
cp src/base.pt.v3 ~/.seisbench/models/v3/quakexnet/base.pt.v3
```

---

### 5) Create the minimal model metadata file

In the same directory (`~/.seisbench/models/v3/quakexnet`), create `base.json.v3`:

```bash
echo '{}' > ~/.seisbench/models/v3/quakexnet/base.json.v3
```

Your directory should look like:

```text
~/.seisbench/models/v3/quakexnet/
├── base.pt.v3
└── base.json.v3
```

---

### 6) Import and use the model

Once the model is registered, you should be able to import it like:

```python
import seisbench.models as sbm

model = sbm.QuakeXNet()
```




# Seismic Event Detection Pipeline

This directory contains code to automatically run a trained **QuakeXNet** deep learning model on seismic stations over any desired time range, detect events at individual stations, and identify **commonly detected network-level events** across multiple stations.

---

## Directory Overview

### 1. `src/custom_daily_detection.py`

**Purpose:** Detect events at individual stations for a given time window.

#### Workflow

1. **Load model & stations**
   - Loads a pre-trained **QuakeXNet** model.
   - Loads a list of seismic stations from `stations.json`.
   - Defines a **user-specified** time window.

2. **Download waveform data**
   - Fetches continuous waveform data from **IRIS** for each station using **ObsPy**.

3. **Run model inference**
   - Sliding window: **100 s length**, **10 s stride**.
   - Produces **class probabilities** for each window:  
     `eq` (earthquake), `px` (explosion/phase), `su` (surface event), sampled every **10 s**.

4. **Smooth probabilities**
   - Applies a **5-sample moving average** (~50 s) to reduce isolated spikes and short noise fluctuations.

5. **Detect events**
   - Event **starts** when smoothed probability ≥ **0.15**.
   - Event **ends** when smoothed probability < **0.15**.
   - Event is only logged if **max probability ≥ 0.5** (default).
   - For each event, metrics are recorded:
     **mean probability**, **max probability**, **area under curve (AUC)**,
     start/end indices, and corresponding UTC timestamps.

#### Output

- One CSV **per station**, containing detected events and metrics. Example:

| station | network | class | auc  | mean_prob | max_prob | start_index | end_index | start_time           | end_time             |
| ------- | ------- | ----- | ---- | --------- | -------- | ----------- | --------- | -------------------- | -------------------- |
| PARA    | CC      | eq    | 3.37 | 0.35      | 0.54     | 5429        | 5438      | 2025-12-13T14:44:22Z | 2025-12-13T14:45:52Z |
| PARA    | CC      | eq    | 7.02 | 0.60      | 0.96     | 5561        | 5572      | 2025-12-13T15:06:22Z | 2025-12-13T15:08:12Z |

---

### 2. `src/custom_generate_common_events.py`

**Purpose:** Combine detections from multiple stations to find **network-level common events**.

#### Workflow

1. **Merge station CSVs**
   - Loads all per-station CSVs for the selected day and concatenates them.

2. **Round start times**
   - Aligns events by rounding to the nearest **10 s**.
   - Example: `12:10:43 → 12:10:40`, `12:10:46 → 12:10:50`.
   - Ensures slightly offset detections from different stations are grouped as the same event.

3. **Group & aggregate**
   - Groups events by **rounded start time**.
   - Computes:
     - `num_stations`: number of **unique stations** that detected the event (any class).
     - `stations`: list of unique stations in the group.
     - `most_common_class`: most frequent class across stations (if tied, picks the first).
     - `mean_auc`, `mean_max`, `mean_prob`: mean of these metrics **across all stations, regardless of class**.

4. **Filter common events**
   - Keeps events detected by at least **N stations** (default = **4**).

#### Output

- One CSV **per day** listing **network-level events**, with aggregated metrics. Example:

| rounded_start             | num_stations | stations                                | most_common_class | mean_auc | mean_max | mean_prob |
| ------------------------- | ------------ | --------------------------------------- | ----------------- | -------- | -------- | --------- |
| 2025-08-03 20:03:30+00:00 | 4            | ['RCM', 'RER', 'STAR', 'OBSR']          | su                | 4.72     | 0.73     | 0.42      |
| 2025-08-03 23:28:40+00:00 | 4            | ['RER', 'STAR', 'STAR', 'PANH', 'MILD'] | px                | 3.60     | 0.61     | 0.35      |

---

## Notes on Metrics

- **num_stations**: Number of unique stations that detected an event in the rounded time window (any class).
- **stations**: List of unique stations that detected the event.
- **most_common_class**: Class detected most frequently across stations in the window. If tied, the first class in the list is chosen.
- **mean_auc, mean_max, mean_prob**: Average metrics across **all stations in the time window**, regardless of class.

---

## Summary

This pipeline:

1. Detects events **per station** with QuakeXNet.
2. Smooths and thresholds probabilities to avoid spurious detections.
3. Aggregates detections **across stations** to create a **network-level catalog of common events**.
4. Produces **per-station and daily network CSVs**, containing both temporal and confidence metrics.
