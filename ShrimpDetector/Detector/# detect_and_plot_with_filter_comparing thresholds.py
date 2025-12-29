# detect_and_plot_click_counts_barplot.py
# Margherita Silvestri
# 08-10-25
# Snapping shrimp click detector: grid search for envelope/derivative thresholds,
# outputs only a barplot of number of detected clicks per combination.

import os
import glob
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import hilbert, butter, filtfilt
import matplotlib.pyplot as plt

def safe_read_wav(path):
    """Read a WAV file, return (sr, data) or (None, None) on error."""
    try:
        return wav.read(path)
    except Exception as e:
        print(f"Skipping {os.path.basename(path)}: {e}")
        return None, None

def highpass_filter(sig, sr, cutoff=2000, order=4):
    """Zero-phase Butterworth high-pass filter."""
    nyq = 0.5 * sr
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='high')
    return filtfilt(b, a, sig)

def count_clicks(wav_path, env_thresh_factor, der_thresh_factor, t_min=0.04):
    """Return number of detected clicks for given thresholds."""
    sr, data = safe_read_wav(wav_path)
    if data is None:
        return 0
    if data.ndim > 1:
        data = data[:,0]
    data = data.astype(np.float32)
    data /= np.max(np.abs(data)) + 1e-10
    sig = data - np.mean(data)
    sig = highpass_filter(sig, sr)
    env  = np.abs(hilbert(sig))
    denv = np.diff(env, prepend=env[0])
    e_th = env.mean() + env.std() * env_thresh_factor
    d_th = denv.mean() + denv.std() * der_thresh_factor
    min_samp = int(t_min * sr)
    clicks = []
    last_idx = -min_samp
    i = 0
    while i < len(denv):
        if (denv[i] > d_th and env[i] > e_th and (i - last_idx) >= min_samp):
            clicks.append(i/sr)
            last_idx = i
            i += min_samp
        else:
            i += 1
    return len(clicks)

if __name__ == "__main__":
    # Settings
    input_folder = r"C:\Users\margh\Desktop\RECORDINGSMARFUTURA\RECORDINGS\LASCRUCES"
    output_dir   = os.path.join(input_folder, "click_count_barplots")
    os.makedirs(output_dir, exist_ok=True)
    env_factors = [2.0, 2.5, 3.0]
    der_factors = [2.0, 2.5, 3.0]
    t_min = 0.04

    wav_files = glob.glob(os.path.join(input_folder, "*.wav"))

    for wav_path in wav_files:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        bar_labels = []
        click_numbers = []
        for e in env_factors:
            for d in der_factors:
                n_clicks = count_clicks(wav_path, e, d, t_min)
                label = f"env={e},der={d}"
                bar_labels.append(label)
                click_numbers.append(n_clicks)
                print(f"{base}: {label} -> {n_clicks} clicks")

        # Plot barplot for this file
        plt.figure(figsize=(10,4))
        plt.bar(bar_labels, click_numbers, color='steelblue')
        plt.ylabel("Detected clicks")
        plt.title(f"Detected clicks per threshold for {base}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base}_click_count_barplot.png"), dpi=200)
        plt.close()
