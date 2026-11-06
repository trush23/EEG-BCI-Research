"""Shared I/O and provenance only; scientific calculations live in the notebooks."""
from pathlib import Path
import os, json, hashlib, importlib.metadata
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUTPUT = ROOT / 'outputs'
for directory in (DATA, OUTPUT, ROOT / '.cache'):
    directory.mkdir(exist_ok=True)
os.environ.setdefault('MNE_DATA', str(DATA))
os.environ.setdefault('MNE_DATASETS_EEGBCI_PATH', str(DATA))
os.environ.setdefault('MNE_DONTWRITE_HOME', 'true')
os.environ['MOABB_DOWNLOAD_PROVIDER'] = 'upstream'
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.cache'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache'))
import numpy as np
import pandas as pd
import mne
mne.set_log_level('ERROR')
SEED = 20261016

def versions():
    names = ['numpy','scipy','pandas','matplotlib','mne','moabb','pyriemann','braindecode','torch','scikit-learn','autoreject']
    installed = {}
    for package in names:
        try:
            installed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed[package] = 'Not installed in this kernel'
    return pd.Series(installed, name='version')

def load_run(subject=1, run=4):
    """Download/cache one unmodified PhysioNet EDF; never substitute synthetic data."""
    files = mne.datasets.eegbci.load_data(subject, [run], path=DATA, update_path=False, verbose=False)
    raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage('standard_1005')
    path = Path(files[0])
    entry = dict(subject=int(subject), run=int(run), file=str(path.relative_to(ROOT)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), sampling_hz=float(raw.info['sfreq']),
        channels=len(raw.ch_names), samples=int(raw.n_times),
        source=f'https://physionet.org/files/eegmmidb/1.0.0/S{subject:03d}/S{subject:03d}R{run:02d}.edf')
    manifest = OUTPUT / 'data_manifest.json'
    records = json.loads(manifest.read_text()) if manifest.exists() else {}
    records[f'S{subject:03d}R{run:02d}'] = entry
    manifest.write_text(json.dumps(records, indent=2))
    return raw

def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.figsize':(9,4), 'figure.dpi':110, 'font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.2,
        'axes.prop_cycle':plt.cycler(color=['#246a9c','#c27524','#6d537b','#667c39'])})

def save_table(frame, name):
    frame.to_csv(OUTPUT / name, index=False)
    return frame
