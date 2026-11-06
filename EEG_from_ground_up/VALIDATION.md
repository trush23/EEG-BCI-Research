# Validation record — 16 October 2026

All six notebooks executed top-to-bottom using the project's isolated Python 3.12 environment. Their saved outputs include 18 rendered figures. The package environment passed `pip check`; exact versions are in `requirements-lock.txt`.

## Checks performed

- Notebook schemas, execution counts, and absence of cell errors.
- Input EDF identities and SHA-256 values against the saved local download inventory.
- Physical units, sample spacing, event-label conventions, and finite arrays.
- Average-reference zero-sum property and preservation of original recordings.
- PSD integration versus time-domain variance (Notebook 02 median ratio approximately 0.957).
- Independent verification of one MNE Morlet-power calculation against direct SciPy convolution; relative discrepancy below 1e-14.
- ICA with no exclusions reconstructs the input to floating-point precision.
- Participant/run/trial identifiers are unique in the event-related analysis.
- All 135 decoding trials have explicit split assignments: EDF runs 4 and 8 train, run 12 test, within each of three participants. Every model uses the same 45 held-out trials.
- Recomputed all 18 model/participant balanced-accuracy and AUC values directly from saved predictions.
- Inspected exported figures, including the final QC heatmap, time-frequency maps, uncertainty plots, learning curves, and model comparison. Refined crowded channel and ICA layouts.

## Findings that must remain visible

- The example 150 µV rejection rule retains zero of 15 trials. This is explicitly reported; it is not presented as a usable cleaned dataset. AutoReject retains 13 of 15 in its descriptive fit.
- ICA was inspected with no components removed; the notebook does not claim to have identified or eliminated ocular artifacts.
- Notebook 05 uses three participants and 135 trials. Its group uncertainty demonstration is deliberately limited by the small participant count.
- EEGNet is a fixed 20-epoch, single-seed demonstration and performs near chance in balanced accuracy. No test-set tuning was performed to improve the displayed result.
- The result cannot establish cross-day, cross-participant, clinical, or asynchronous control performance.

## Presentation limitation

Local HTML files were exported, but the browser's security policy blocked `file://` access. Full HTML/notebook page-layout inspection in a browser was therefore not completed. Individual rendered figures were inspected directly, and notebook structure/output presence was checked programmatically. To complete the page-layout check, open each `.ipynb` in Jupyter or its corresponding HTML file in your own browser.

## Non-fatal messages

The environment may report missing optional `ipywidgets` progress displays. Static figures and calculations still execute. MNE warns that concatenating Epochs drops attached annotations; Notebook 05 retains the event labels and participant/run/trial metadata explicitly, and the original EDF annotations remain unchanged. These messages are retained rather than silently suppressed.

Run `python run_notebooks.py` from the activated course environment to reproduce execution and HTML exports. See `outputs/execution_status.json` for per-notebook completion records. File manifests identify the downloaded inputs; they do not independently authenticate upstream scientific labels.

## Anaconda/Jupyter startup repair

Registered `Python (EEG ground up)` as the user-visible Jupyter kernel and verified that Anaconda discovers it with the intended project Python executable. Executed the first setup cell successfully through that registered kernel. Notebook metadata now requests it by default. The version-reporting helper handles unavailable package metadata without failing. The README starts with click-by-click instructions for the existing Anaconda installation. Scientific code and input data are unchanged.
