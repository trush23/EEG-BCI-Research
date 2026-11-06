from pathlib import Path
from typing import Dict, List, Optional, Sequence
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import pandas as pd
from matplotlib.widgets import Button
from scipy.signal import butter, filtfilt

def collect_task_windows(
    data_dir: str,
    window_size_s: float,
    stride_s: float,
    snip_s: float = 0.5,
    tasks: Optional[Sequence[str]] = None,
    sample_rate: float = 250.0,
) -> Dict[str, List[np.ndarray]]:
    """
    Build a task->windows dictionary from stopwatch CSV files.

    Args:
        data_dir: Folder that holds the raw_no_tongue CSV recordings.
        window_size_s: Length of each window in seconds.
        stride_s: Step size between successive windows in seconds.
        snip_s: Length to snip before each phase in seconds.
        tasks: Tasks to include (defaults to left/right/legs/none).
        sample_rate: Sampling frequency used when converting seconds->samples.

    Returns:
        Dict where keys are task names and values are lists of windows shaped
        (num_channels, window_size_samples).
    """

    if window_size_s <= 0:
        raise ValueError("window_size_s must be > 0")
    if stride_s <= 0:
        raise ValueError("stride_s must be > 0")
    if snip_s <= 0:
        raise ValueError("snip_s must be > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    selected_tasks = (
        list(tasks)
        if tasks is not None
        else ["left_hand", "right_hand", "legs", "none"]
    )
    windows_by_task: Dict[str, List[np.ndarray]] = {task: [] for task in selected_tasks}

    window_samples = int(window_size_s * sample_rate)
    stride_samples = int(stride_s * sample_rate)
    snip_length = int(snip_s * sample_rate)

    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window_size_s and stride_s must translate to >=1 sample")

    csv_paths = sorted(Path(data_dir).glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df = df.drop(['ch9', 'quality'], axis=1, errors='ignore')

        rename_map = {ch: name for ch, name in electrode_montage.items() if ch in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)

        if "phase" not in df or "task" not in df or "trial" not in df:
            continue

        channel_cols = [name for name in electrode_montage.values() if name in df.columns]

        source_id = csv_path.stem
        df["trial_source"] = df["trial"].apply(lambda t: f"{source_id}_{t}")

        imagery_df = df[df["phase"] == "IMAGERY"]
        relax_df = df[df["phase"] == "RELAX"]

        for task in selected_tasks:
            if task == "none":
                task_df = relax_df.copy()
            else:
                task_df = imagery_df[imagery_df["task"] == task].copy()
            if task_df.empty:
                continue

            for _, trial_df in task_df.groupby("trial_source"):
                trial_data = trial_df[channel_cols].to_numpy(dtype=float)
                num_samples, _ = trial_data.shape
                if num_samples - snip_length < window_samples:
                    continue

                for start in range(snip_length, num_samples - window_samples + 1, stride_samples):
                    window = trial_data[start : start + window_samples].T

                    if np.isnan(window).any():
                        continue

                    windows_by_task[task].append(window)

    return windows_by_task


def bandpass(window, order=4, bandpass_lims=(8, 30), sample_rate=250):
    low_lim = bandpass_lims[0]
    high_lim = bandpass_lims[1]

    nyquist = 0.5 * sample_rate
    band = [low_lim / nyquist, high_lim / nyquist]
    b, a = butter(N=order, Wn=band, btype="band")
    return filtfilt(b, a, window, axis=1)


def plot_task_windows(
    windows_by_task: Dict[str, List[np.ndarray]],
    task: str,
    bandpass_lims: Tuple[float] = (8, 30),
    sample_rate: float = 250.0,
    channel_labels: Optional[Sequence[str]] = None,
) -> None:
    """
    Interactively visualize task windows with time-series and FFT plots.

    Args:
        windows_by_task: Dict mapping task names to window arrays.
        task: Task name to visualize.
        bandpass_lims: lower and upper thresholds for bandpassing a window of shape (num_channels, num_samples)
        sample_rate: Sampling frequency for both time axis and FFT computation.
        channel_labels: Optional custom labels for channels (defaults to electrode order).
    """

    if not windows_by_task:
        raise ValueError("windows_by_task cannot be empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    if task not in windows_by_task:
        raise KeyError(f"Task '{task}' not found in windows_by_task")

    task_windows = windows_by_task[task]
    if not task_windows:
        raise ValueError(f"No windows available for task '{task}'")

    num_channels, num_samples = task_windows[0].shape
    labels = list(channel_labels) if channel_labels is not None else None
    if labels is None:
        if "electrode_montage" in globals():
            labels = list(electrode_montage.values())
        else:
            labels = [f"ch{i+1}" for i in range(num_channels)]
    if len(labels) != num_channels:
        raise ValueError("channel_labels length must match number of channels per window")

    def get_filtered_window(index: int) -> np.ndarray:
        window = task_windows[index]
        if window.shape != (num_channels, num_samples):
            raise ValueError("All windows must share the same shape")
        return bandpass(window, bandpass_lims=bandpass_lims)

    fig, (ax_time, ax_fft) = plt.subplots(2, 1, figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2, hspace=0.35)

    counter_text = fig.text(0.5, 0.05, "", ha="center", va="center", fontsize=12)
    state = {"index": 0}

    def update_plots() -> None:
        idx = state["index"]
        filtered = get_filtered_window(idx)
        time_axis = np.arange(num_samples) / sample_rate
        fft_freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)
        fft_vals = np.abs(np.fft.rfft(filtered, axis=1))

        ax_time.clear()
        ax_fft.clear()

        for ch_idx, label in enumerate(labels):
            ax_time.plot(time_axis, filtered[ch_idx], label=label)
            ax_fft.plot(fft_freqs, fft_vals[ch_idx], label=label)

        ax_time.set_title(f"Task: {task} – Time Series")
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Amplitude")

        ax_fft.set_title("FFT Magnitude")
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("|FFT|")
        ax_fft.set_xlim(0.0, 45.0)

        ax_time.legend(loc="upper right", ncol=2)
        ax_fft.legend(loc="upper right", ncol=2)

        counter_text.set_text(f"Window {idx + 1} / {len(task_windows)}")
        fig.canvas.draw_idle()

    def go_next(_event) -> None:
        if state["index"] < len(task_windows) - 1:
            state["index"] += 1
            update_plots()

    def go_prev(_event) -> None:
        if state["index"] > 0:
            state["index"] -= 1
            update_plots()

    ax_prev = plt.axes([0.3, 0.02, 0.1, 0.05])
    ax_next = plt.axes([0.6, 0.02, 0.1, 0.05])
    btn_prev = Button(ax_prev, "Back")
    btn_next = Button(ax_next, "Next")
    btn_prev.on_clicked(go_prev)
    btn_next.on_clicked(go_next)

    update_plots()
    plt.show()


if __name__ == "__main__":
    default_dir = Path(__file__).resolve().parent / "data"

    electrode_montage = {
        "ch1": "CP3",
        "ch2": "CP4",
        "ch3": "C3",
        "ch4": "C4",
        "ch5": "FCz",
        "ch6": "FC4",
        "ch7": "FC3",
        "ch8": "Cz",
    }

    windows_by_task = collect_task_windows(
        data_dir=str(default_dir),
        window_size_s=2.0,
        stride_s=1.0,
    )

    total = sum(len(v) for v in windows_by_task.values())
    print(f"Collected {total} windows from {default_dir}")
    for task, task_windows in windows_by_task.items():
        shape = task_windows[0].shape if task_windows else "n/a"
        print(f"- {task}: {len(task_windows)} windows; window shape: {shape}")

    plot_task_windows(windows_by_task, "right_hand")