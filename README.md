# EEG analysis for BCI research

Python and Jupyter materials for EEG processing, motor imagery, and brain-computer-interface research.

## Contents

- `EEG_from_ground_up/`: six lessons on fundamentals, spectral structure, cleaning and quality control, time-frequency analysis, event-related activity and uncertainty, and decoding. Original Python helpers, locked dependency versions, saved public-dataset outputs, and HTML previews are included.
- `Synaptrix Motor Imagery/`: the original motor-imagery analysis script and notebooks, with participant-derived notebook outputs cleared in this repository copy.
- `motor_imagery_analysis.ipynb`: the separate root walkthrough version, also with outputs cleared.
- `electrode_montage.png`: the supplied montage illustration.

## Data and privacy

The 17 local Synaptrix participant EEG CSV recordings are not included. Their saved notebook plots and text outputs are also excluded from this repository copy. Original local recordings and original notebook files were not modified. Notebook code and explanations remain unchanged.

To run the Synaptrix examples, place separately authorized recordings in `Synaptrix Motor Imagery/data/` on your own machine. That path is ignored by Git. Do not add participant recordings or participant-derived outputs to a commit without reviewing the intended data-sharing scope. Git does not automatically remove notebook outputs on subsequent saves.

Downloaded public PhysioNet EDF files, Python environments, caches, notebook checkpoints, and duplicate ZIP deliveries are also excluded. The course retains its original public-data provenance records and downloader. From `EEG_from_ground_up/`, `python download_data.py` downloads the course inputs; course loaders can also fetch individual runs. Network access is needed if those data are absent.

## Running

For the existing installation on the original Mac, select the **Python (EEG ground up)** Jupyter kernel for the course notebooks, as described in `EEG_from_ground_up/README.md`. A clone on another machine needs a compatible environment; the original dependency lock is `EEG_from_ground_up/requirements-lock.txt`. Virtual environments are not portable and are not included.

For the Synaptrix walkthrough, use a Python kernel with NumPy, pandas, SciPy, and Matplotlib. Jupyter/IPython supplies notebook execution; interactive plots additionally use `ipympl`. See the setup cells for static versus interactive plotting.

## Import and scientific scope

Imported on 2026-11-06. Original Python code, course notebooks, public-dataset course results, and documentation were copied without modification. Three participant-analysis notebook copies have cleared outputs and execution counts; their cell sources are unchanged. The notebooks were not rerun during this import. Syntax and notebook-schema checks do not constitute new scientific validation.

The Synaptrix walkthrough visualizes task windows, filtered traces, and FFT magnitude. It does not train or evaluate a classifier or control a device. The course's saved decoding demonstrations and their limitations are documented in `EEG_from_ground_up/VALIDATION.md`; they do not establish clinical or real-time BCI performance.
