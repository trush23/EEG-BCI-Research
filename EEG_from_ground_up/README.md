# EEG from the ground up

Six runnable Python/Jupyter lessons, built around the PhysioNet EEG Motor Movement/Imagery Dataset. Start with Notebook 01. The Synaptrix files are not inputs to this course and are unchanged.

## Start here — using Anaconda/JupyterLab on this Mac

Everything needed is already installed. You do **not** need to install packages, recreate an environment, or run terminal commands.

1. Open JupyterLab from Anaconda as usual, then open `01_EEG_fundamentals.ipynb`.
2. Click the kernel name at the upper right. In your screenshot it says **Python [conda env:base]**. Alternatively, use **Kernel → Change Kernel**.
3. Choose **Python (EEG ground up)** and confirm the selection.
4. Use **Run → Run All Cells**. Read the notebook from the top.
5. Choose the same **Python (EEG ground up)** kernel for the other five notebooks.

If that kernel is not listed, refresh the JupyterLab browser page. If it still does not appear, stop and relaunch JupyterLab from Anaconda Navigator.

**What is a kernel?** It is the Python process that executes a notebook. Anaconda can launch JupyterLab while a notebook uses a different Python environment. The base environment does not contain all the EEG packages; the registered EEG environment does.

**The error you saw:** `PackageNotFoundError: ... moabb` occurred because the notebook used the base kernel. The EEG environment already contains MOABB and the other course packages. The version-reporting helper now also handles missing optional packages without raising this error.

The notebook files now request the EEG kernel by default. A notebook that was already open may keep its previous kernel until you change it manually. Your data and saved analyses are unchanged.

## Learning sequence

| Notebook | Main question | Tools and outputs |
|---|---|---|
| [01 — Fundamentals](01_EEG_fundamentals.ipynb) | What is measured, and how are trials identified? | MNE; traces, electrode map, referencing, event inventory |
| [02 — Spectral structure](02_Spectral_structure.ipynb) | How does power vary with frequency and condition? | Welch and multitaper PSD; alpha power; scalp maps |
| [03 — Cleaning](03_Cleaning_and_quality_control.ipynb) | What is changed or discarded by preprocessing? | MNE filtering/reference/ICA, EEGLAB concepts, AutoReject; QC and retention |
| [04 — Time–frequency](04_Time_frequency_analysis.ipynb) | How does oscillatory power change around a cue? | STFT versus Morlet; resolution, padding, baseline normalization |
| [05 — Event-related activity](05_Event_related_activity_and_uncertainty.ipynb) | What are trial-level effects and the correct uncertainty units? | Cue-locked averages, mu-band changes, participant bootstrap, exact sign flips |
| [06 — Decoding](06_Decoding_MOABB_pyRiemann_Braindecode.ipynb) | Which methods classify an unseen run? | MOABB access; MNE CSP; pyRiemann MDM/tangent space; Braindecode EEGNet |

Each notebook contains mathematical explanations, executable steps, checks, exercises, and links to primary documentation. Each runs independently; shared code in `course_utils.py` only handles I/O, provenance, versions, and plotting style. Scientific calculations remain visible in notebook cells.

## Scope and data

- First participant for foundations; participants 1–3 for the small multi-participant examples.
- Baseline runs 1/2: eyes open/closed. Imagery runs 4/8/12: left/right hand imagery. Other run types have different event meanings.
- MNE's EEGBCI loader downloads EDF files; MOABB is restricted to the three hand-imagery runs. Its inputs are first cached through MNE from PhysioNet; unused feet runs are excluded.
- Data are downloaded into `data/`. The first run needs internet access; subsequent runs reuse cached data.
- `outputs/data_manifest.json` identifies MNE-loaded input files with source URLs and local SHA-256 hashes. `outputs/download_inventory.json` records all locally cached EDFs, including any unused downloads.
- No synthetic data substitutes for missing recordings. Failed downloads stop execution.
- These are teaching-scale analyses, not evidence of clinical benefit, general intention recognition, or real-time control.

## Advanced: recreating the environment on another machine

A separate `.venv` environment is provided locally; the existing Anaconda installation was not modified. The full tested package snapshot is `requirements-lock.txt`. It was resolved on macOS ARM with Python 3.12; other platforms may need compatible package versions, especially PyTorch.

To reproduce in a new checkout, from this folder:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python download_data.py
.venv/bin/python run_notebooks.py
```

Select `.venv/bin/python` as the notebook interpreter in your editor. If your Jupyter application needs a named kernel, register one explicitly:

```bash
.venv/bin/python -m ipykernel install --user --name eeg-ground-up --display-name "Python (EEG ground up)"
```

The execution helper selects this environment's Python directly; kernel registration is not necessary for it. To execute one lesson:

```bash
.venv/bin/python run_notebooks.py 03
```

Notebook execution saves figures and tables inline and exports readable HTML to `previews/`. Use a fresh kernel and Run All. Open notebooks from this folder or its project parent. Files in `outputs/` are regenerated; preserve a copy before changing parameters if you need to compare experiments.

## Scientific choices

- Cleaning is not universal: 1–40 Hz in the oscillatory lessons; the implications for slow ERP analysis are stated. No automatic ICA exclusions or unconditional notch filtering.
- Notebook 03's fitted ICA/AutoReject demonstrations are descriptive and are not imported into the decoding experiment.
- PSD, wavelet power, and voltage have different units. Every conversion is explicit.
- Time–frequency comparisons use padded epochs, explicit baseline intervals, and edge exclusion.
- Notebook 05 preserves participant/run/trial identity and aggregates within participant before population summaries. N=3 is too small for dependable population inference; bootstrap and sign-flip calculations are educational.
- Notebook 06 fixes the last run per participant as test data for every model. Run IDs and split membership are saved. No test-set tuning or overlapping crops are used.
- Scaling, CSP, and tangent-space references are fitted on training trials. Fixed offline filters operate within source runs.
- EEGNet is a fixed 20-epoch, single-seed CPU demonstration, not a tuned deep-learning benchmark. A low score must not be hidden by switching splits or seeds.
- Held-out-run performance is not cross-day, cross-participant, or asynchronous performance. Those require separate protocols.

## How the resources contribute

| Resource | Contribution to this course |
|---|---|
| [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/) | Primary public recordings and experimental protocol |
| [MNE-Python](https://mne.tools/stable/auto_tutorials/index.html) | Data structures, electrode coordinates, referencing, filtering, ICA, spectral/time-frequency analysis, CSP |
| [EEGLAB](https://eeglab.org/tutorials/ConceptsGuide/ICA_background.html) | Conceptual background for interpreting ICA; MATLAB is not required |
| [AutoReject](https://autoreject.github.io/stable/index.html) | Explicit automated epoch/channel repair demonstration |
| [MOABB](https://moabb.neurotechx.com/docs/auto_examples/tutorials/index.html) | Harmonized loading, task definition, metadata, and roadmap to cross-session/cross-subject evaluation |
| [pyRiemann](https://pyriemann.readthedocs.io/en/latest/) | Shrinkage covariance, manifold-distance classifier, tangent-space features |
| [Braindecode](https://braindecode.org/stable/) | EEGNet architecture, trained with a visible PyTorch loop |
| [OpenNeuro](https://docs.openneuro.org/user_guide.html) | Extension path for EEG-BIDS data and acquisition/event metadata; no second dataset is downloaded automatically |
| [BNCI](https://bnci-horizon-2020.eu/database/data-sets) | Extension path to documented multi-session benchmarks; no claim that different datasets are interchangeable |

The selection emphasizes complementary strengths rather than claiming an objective ranking of portals. Tutorial prose and code are original; external resources are cited as methodological and API references. Cite PhysioNet's dataset and underlying BCI2000 publication, and relevant software/method papers when publishing derived research. Dataset licenses and software licenses remain their respective authors' terms.

## Validation

See `outputs/execution_status.json` for actual execution status and `VALIDATION.md` for the completed review. A successful run is an implementation check, not proof that a scientific conclusion generalizes.
