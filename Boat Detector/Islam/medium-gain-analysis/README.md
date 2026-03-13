# Boat Detection Pipeline — MAR FUTURA
**Passive Acoustic Monitoring | Chilean Coastal Sites | Nov–Dec 2025**

Automated vessel detection from hydrophone recordings across 5 Chilean coastal monitoring sites. The pipeline extracts spectral features from continuous hydrophone recordings, applies a rule-based detection algorithm, and produces a clean vessel event log with timestamps, duration, and acoustic characteristics.

---

## Sites

| Site | Hydrophone | Type | Recording Period | Files |
|---|---|---|---|---|
| Las Cruces | 26 | Marine Protected Area | Nov 12 – Dec 8 | 18,492 |
| Matanzas | 32 | Natural low-impact coast | Nov 12 – Dec 8 | 17,702 |
| San Antonio | 38 | Commercial port | Nov 13 – Dec 2 | 15,610 |
| Ventanas | 36 | Industrial port | Nov 7 – Nov 28 | 15,181 |
| Zapallar | 34 | Recreational coast | Nov 7 – Nov 28 | 15,159 |

**Total: 82,144 WAV files — 1,369 hours of continuous recording**

All recordings: 60 seconds, 16 kHz, 16-bit mono. Timestamps are in UTC — UTC-3 correction applied in the pipeline for Chile local time (CLST, November–December).

---

## How to Run

Run the three scripts in order. Each saves its output for the next step.

### Prerequisites

```r
install.packages(c("tuneR", "signal", "parallel"))
```

### Step 1 — Extract Features
```r
source("R/01_extract_features.R")
# Input:  WAV folders per site (named exactly as the site names above)
# Output: boat_features_all.csv
```

Processes all WAV files in parallel. Extracts 6 spectral features per file focused on the 100–600 Hz vessel frequency band. Runtime is approximately 8–10 minutes for the full dataset on a multi-core machine.

### Step 2 — Apply Detection Rule
```r
source("R/02_boat_detection.R")
# Input:  boat_features_all.csv
# Output: outputs/boat_detections_FINAL.csv
```

Applies the rule-based detector. Nighttime only (22:00–06:00 Chile local time). Site-specific thresholds, infrastructure tone exclusion, and biological exclusion filter for Matanzas.

### Step 3 — Report Events
```r
source("R/03_report_events.R")
# Input:  boat_features_all.csv (with boat_detected column from Step 2)
# Output: outputs/vessel_events_FINAL.csv
```

Reconstructs vessel events from consecutive detection files. Each event = an unbroken run of flagged 1-minute files.

---

## Results

**52 nighttime vessel events detected across 1,369 hours**

| Site | Events | Notes |
|---|---|---|
| San Antonio 38 | 22 | Consistent commercial shipping, activity across all nights |
| Zapallar 34 | 13 | Mostly pre-dawn (05:00–06:00), likely artisanal fishing |
| Matanzas 32 | 15 | After biological exclusion |
| Las Cruces 26 | 1 | Very low — expected for a marine protected area |
| Ventanas 36 | 1 | Heavy industrial noise floor limits detection |

---

## Detection Algorithm

### Feature Extraction
Each WAV file is processed using a Welch periodogram (1024-point FFT, 50% overlap, Hanning window) on the first 10 seconds of audio. Six features are extracted from the 100–600 Hz boat frequency band:

| Feature | Description |
|---|---|
| `boat_tonality` | Signal-to-noise ratio above median noise floor (dB) |
| `peak_freq` | Dominant frequency in band (Hz) |
| `n_harmonic_peaks` | Number of distinct spectral peaks ≥20 Hz apart |
| `spectral_entropy` | Uniformity of energy distribution across band |
| `band_energy_ratio` | Boat band energy relative to full spectrum |
| `rms` | Overall signal level |

### Detection Rule
A file is flagged as a vessel detection if all of the following conditions are met:

```
boat_tonality    > threshold (25 dB for ports / 30 dB for coastal sites)
peak_freq        in 100–600 Hz
n_harmonic_peaks ≥ 1
peak_freq        NOT in 520–595 Hz   (excludes fixed infrastructure tone)
NOT (peak_freq < 200 Hz AND n_harmonic_peaks < 2)
hour_chile       in 22:00–06:00      (nighttime only, UTC-3)
```

For Matanzas only, an additional exclusion is applied:
```
peak_freq NOT in 100–135 Hz   (biological sound exclusion)
```

### Cluster Filter
Detections at port sites (San Antonio, Ventanas) require a minimum of 2 consecutive flagged files to form a valid event, reducing isolated false positives. Coastal sites allow single-file events.

### Site-Specific Thresholds

| Site | Tonality Threshold | Min Cluster |
|---|---|---|
| Las Cruces 26 | 30 dB | 1 file |
| Matanzas 32 | 30 dB | 1 file |
| San Antonio 38 | 25 dB | 2 files |
| Ventanas 36 | 25 dB | 2 files |
| Zapallar 34 | 30 dB | 1 file |

---

## Output Files

| File | Description |
|---|---|
| `outputs/vessel_events_FINAL.csv` | One row per vessel event — date, time (Chile + UTC), duration, max tonality, peak frequency |
| `outputs/boat_detections_FINAL.csv` | One row per flagged file with all acoustic features |

---

## Notes

- All timestamps in output files are **Chile local time (UTC-3)**, corresponding to Chile Summer Time (CLST) during November–December
- The 594 Hz infrastructure tone present at Las Cruces, Ventanas, and Zapallar is excluded by the 520–595 Hz band filter
- Ventanas is the most challenging site due to the heavy industrial noise floor — the low detection count reflects this limitation rather than absence of vessel traffic
- Detection focuses on **nighttime hours only (22:00–06:00 Chile local)** per project requirements
