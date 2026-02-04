# MAR_FUTURA Code Repository

This repository contains analysis and detection workflows used in the MAR FUTURA project.

The codebase is organized by *project category*:

- **Boat Detector** (AGILE / Islam's way)
- **NDSI** (soundscape metrics in Python and R)
- **Shrimp Detector** (snapping shrimp click-rate analysis)

Many workflows can be run:

- **Locally** using your Mac + Google Drive File Stream or local disks.
- **In Google Colab** typically with Google Drive mounted.

---

## Repository layout

- `Boat Detector/`
  - `Agile/`
    - `CreateModel.ipynb` (train a classifier)
    - `RunModel.ipynb` (embed a folder and run inference; main “batch” notebook)
    - `RunSingleFile.py` (run inference on one WAV)
    - `agile_classifier_v2.pt` (trained classifier file)
  - `Islam/` (placeholder / in-progress)

- `NDSI/`
  - `SiteAnalysis_Python.ipynb` (single-site analysis in Python)
  - `NDSI_Sites_Comparison_Python.ipynb` (multi-site comparison in Python)
  - `R files/` (not needed now really)
    - `SiteAnalysis_R.ipynb` (single-site analysis in R)
    - `NDSI_Sites_Comparisson_R.ipynb` (multi-site comparison in R)
    - `Plots.R` (plot helpers)

- `ShrimpDetector/`
  - `Detector/`
    - `ClickRateAnalysis.ipynb` (single-site click-rate analysis)
    - `ClickRate_Sites_Comparison.ipynb` (multi-site click-rate comparison)
    - Additional Python scripts (prefixed with `# `) These were written by Margherita
  - `Figures/` (plot scripts + outputs)

---

# Running in Colab vs Local

## Colab (general pattern)

Most notebooks that support Colab will follow this pattern:

1. Open the notebook in Colab.
2. Mount Google Drive (or a Shared Drive) so Colab can read the WAVs and write results.
3. Install required packages (the notebook may do this for you).
4. Run the notebook cells in order.

**Important:** Colab storage (`/content`) is ephemeral. If a workflow stages embeddings to `/content`, it must copy results back to Drive to persist them.

## Local (general pattern)

Typical local setup:

1. Create/activate a Python environment.
2. Install dependencies.
3. Set local paths in the notebook/script configuration section.
4. Run.

This repo does **not** currently include a pinned `requirements.txt`, so install packages as needed per workflow.

---

# Category: Boat Detector

Location: `Boat Detector/Agile/`

This folder contains an audio embedding + classifier workflow built on `perch_hoplite` (AGILE).

## What it does

- Splits WAV audio into fixed windows (e.g., 5 seconds).
- Computes embeddings for each window and stores them in a local database folder.
- Loads a trained linear classifier.
- Writes:
  - A **window-level CSV** (scores/logits per window)
  - A second **window-level CSV with all logits** (debug / full outputs)
  - A **file-level CSV** derived from rules like “minimum consecutive windows over threshold”

## Main entrypoints

### 1) Batch inference on a folder (Notebook)

- **Notebook:** `Boat Detector/Agile/RunModel.ipynb`

This is the main notebook to embed a folder of WAV files and run inference.

#### Local run

1. Open `RunModel.ipynb` locally.
2. Run the cells in this order:

- **Imports**
- **Configuration (LOCAL ONLY)**
  - Set:
    - `input_audio_dir` (folder containing WAVs)
    - `db_path` (folder to store embeddings DB)
    - `output_csv_filepath`
    - `classifier_path`
- **Configuration (COMMON - ALWAYS RUN)**
- **Embed folder, load classifier, and run inference**

**Notes on paths**

- The repository currently uses Google Drive File Stream paths like:
  - `/Users/Rodrigo/Library/CloudStorage/GoogleDrive-royanedel@marfutura.org/Unidades compartidas/Hydrophones/...`

If you change machines or Drive accounts, update the paths in the configuration cells.

#### Colab run

If you use Colab:

1. Run the **Colab-only** config cell(s) (mount Drive, define Drive paths, optional rsync).
2. Run the **Common** config cell.
3. Run the embed/inference cell.

Colab workflows often include a sync step back to Drive so the DB persists beyond the session.

### 2) Single WAV inference (Python script)

- **Script:** `Boat Detector/Agile/RunSingleFile.py`

This runs the same embedding + classifier pipeline but only for one WAV.


---

# Category: NDSI

Location: `NDSI/`

These workflows compute NDSI and related soundscape metrics from WAV files.

## Python notebooks

### 1) Single-site analysis

- **Notebook:** `NDSI/SiteAnalysis_Python.ipynb`

Typical steps:

- Set `site_dir` to the folder containing WAVs.
- Set `output_dir` where CSV/plots are written.
- Run the notebook to produce metrics and plots.

### 2) Multi-site comparison

- **Notebook:** `NDSI/NDSI_Sites_Comparison_Python.ipynb`

Typical steps:

- Define a dictionary of site names → folder paths.
- Run to compute metrics across sites and generate comparison plots.

## R notebooks

Location: `NDSI/R files/`

- `SiteAnalysis_R.ipynb`
- `NDSI_Sites_Comparisson_R.ipynb`

These are R equivalents of the Python workflows.

**Note:** R notebook execution depends on your local R + Jupyter/R kernel setup. Dont need them for now.

---

# Category: Shrimp Detector

Location: `ShrimpDetector/Detector/`

These notebooks/scripts estimate snapping shrimp activity using click detection and click-rate metrics.

## Main notebooks

### 1) Single-site click rate

- **Notebook:** `ShrimpDetector/Detector/ClickRateAnalysis.ipynb`

Typical steps:

- Set `site_dir` (WAV folder)
- Set `output_dir`
- Run to compute click-rate per file/time and generate plots.

### 2) Multi-site click rate comparison

- **Notebook:** `ShrimpDetector/Detector/ClickRate_Sites_Comparison.ipynb`

Typical steps:

- Define multiple site folders.
- Run to compute metrics across sites and generate comparison plots.

---

# How to contribute / update

Recommended workflow:

- Create a feature branch.
- Make changes.
- Commit.
- Merge into `main` after asking Rod.

