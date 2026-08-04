# -*- coding: utf-8 -*-
"""
extract_features.py
===================

Feature extraction script for the FFF printer wear monitoring dataset
(Data in Brief companion to Fu et al., 2026, Progress in Additive Manufacturing).

For each of the 15 monitored wear states, this script:

    1. Loads the raw three-channel vibration time-series (LVM format,
       channels: Head, Motor, Frame).
    2. Computes 27 features per channel following the categorisation of
       Downey, A.R.J., "Machine Learning for Engineering Problem Solving",
       Chapter 3 - Machine Learning Workflows:

           * 11 statistical features
           *  8 time-series / signal-shape features
           *  8 frequency-domain features

       => 27 features x 3 sensors = 81 feature columns per wear state.

    3. Writes two output files:

           features.csv              - raw extracted features
           features_normalized.csv   - each feature divided by its
                                       wear-state-1 value (for plotting)

Reference (Chapter 3):
    https://github.com/austindowney/Machine-Learning-for-Engineering-Problem-Solving

Author:  Yanzhou Fu  (Florida Atlantic University)
Project: ARTS Laboratory, University of South Carolina
"""

import os
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks


# ============================================================
# Configuration
# ============================================================

# Directory containing the raw LVM files (Sample_*.lvm)
DATA_DIR = r"./Ender5_Vibration_Data"

# Output directory for the feature CSVs
OUTPUT_DIR = r"./output"

# LVM file format
SKIP_HEADER_LINES = 22
CHANNELS = ["Head", "Motor", "Frame"]

# Frequency-band edges (Hz) for band-power features.
# These match the bands used in the companion PAM manuscript.
# Adjust if Chapter 3 defines different bands.
BAND_LOW  = (0.0,   100.0)   # low band  -> band_power_0_100
BAND_HIGH = (100.0, 500.0)   # high band -> band_power_100_500

# ------------------------------------------------------------
# Wear-state mapping: (wear_state_index, lvm_filename, cumulative_hours)
#
# Samples 12-19 in the raw acquisition (approx 850-1250 h) are excluded
# from this dataset because the printer was operated with a 0.6 mm
# nozzle during that interval before reverting to 0.4 mm.  Only data
# under the fixed 0.4 mm configuration is included here.  See
# maintenance_log.csv for the full event timeline.
#
# Edit the filename column to match your local naming convention.
# ------------------------------------------------------------
WEAR_STATE_MAPPING = [
    ( 1, "Sample_1.lvm",   180),
    ( 2, "Sample_2.lvm",   240),
    ( 3, "Sample_3.lvm",   320),
    ( 4, "Sample_4.lvm",   390),
    ( 5, "Sample_5.lvm",   440),
    ( 6, "Sample_6.lvm",   480),
    ( 7, "Sample_7.lvm",   570),
    ( 8, "Sample_8.lvm",   630),
    ( 9, "Sample_9.lvm",   710),
    (10, "Sample_10.lvm",  760),
    (11, "Sample_11.lvm",  805),
    (12, "Sample_20.lvm", 1340),  # post-nozzle-revert; re-indexed
    (13, "Sample_21.lvm", 1405),
    (14, "Sample_22.lvm", 1470),
    (15, "Sample_23.lvm", 1525),
]


# ============================================================
# Data loading
# ============================================================

def load_lvm(filepath):
    """
    Load a 4-column LVM file (Time, Head, Motor, Frame).

    Returns a pandas DataFrame with columns ["Time", "Head", "Motor", "Frame"].
    """
    with open(filepath, "r") as f:
        lines = f.readlines()[SKIP_HEADER_LINES:]

    rows = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) < 4:
            continue
        try:
            rows.append([float(p) for p in parts[:4]])
        except ValueError:
            continue

    return pd.DataFrame(rows, columns=["Time", "Head", "Motor", "Frame"])


# ============================================================
# Feature group 1 -- Statistical features (11)
# ============================================================

def compute_statistical_features(x):
    """
    11 statistical features describing the amplitude distribution
    of a time-domain vibration signal.

    Returns dict with keys:
        mean, std, var, min, max, range, median, iqr, mad,
        skewness, kurtosis
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    mean    = np.mean(x)
    std     = np.std(x, ddof=1)              # sample std
    var     = np.var(x, ddof=1)              # sample variance
    xmin    = np.min(x)
    xmax    = np.max(x)
    xrange  = xmax - xmin
    median  = np.median(x)
    iqr_val = np.percentile(x, 75) - np.percentile(x, 25)
    mad     = np.mean(np.abs(x - mean))      # mean absolute deviation

    # higher-order moments (unbiased)
    if std > 0:
        skewness = np.sum((x - mean) ** 3) / (n * std ** 3)
        kurt     = np.sum((x - mean) ** 4) / (n * std ** 4) - 3.0   # excess
    else:
        skewness = 0.0
        kurt     = 0.0

    return {
        "mean":     mean,
        "std":      std,
        "var":      var,
        "min":      xmin,
        "max":      xmax,
        "range":    xrange,
        "median":   median,
        "iqr":      iqr_val,
        "mad":      mad,
        "skewness": skewness,
        "kurtosis": kurt,
    }


# ============================================================
# Feature group 2 -- Time-series / signal-shape features (8)
# ============================================================

def compute_time_series_features(x):
    """
    8 time-series / signal-shape features capturing waveform
    geometry (RMS, peakiness, crest, impulsiveness, etc.).

    Returns dict with keys:
        rms, peak, peak_to_peak, abs_mean,
        crest_factor, shape_factor, impulse_factor, clearance_factor
    """
    x = np.asarray(x, dtype=float)
    eps = 1e-12  # guard against division by zero on flat signals

    rms          = np.sqrt(np.mean(x ** 2))
    peak         = np.max(np.abs(x))
    peak_to_peak = np.max(x) - np.min(x)
    abs_mean     = np.mean(np.abs(x))

    # Square root amplitude (used in clearance factor)
    sra          = np.mean(np.sqrt(np.abs(x))) ** 2

    crest_factor     = peak / (rms      + eps)   # peak / RMS
    shape_factor     = rms  / (abs_mean + eps)   # RMS / |mean|
    impulse_factor   = peak / (abs_mean + eps)   # peak / |mean|
    clearance_factor = peak / (sra      + eps)   # peak / SRA

    return {
        "rms":              rms,
        "peak":             peak,
        "peak_to_peak":     peak_to_peak,
        "abs_mean":         abs_mean,
        "crest_factor":     crest_factor,
        "shape_factor":     shape_factor,
        "impulse_factor":   impulse_factor,
        "clearance_factor": clearance_factor,
    }


# ============================================================
# Feature group 3 -- Frequency-domain features (8)
# ============================================================

def compute_frequency_features(x, fs):
    """
    8 frequency-domain features computed from the one-sided
    amplitude spectrum of the signal.

    Parameters
    ----------
    x  : 1-D ndarray  -- time-domain signal
    fs : float        -- sampling frequency in Hz

    Returns dict with keys:
        peak_freq, peak_ampl,
        spectral_centroid, spectral_spread,
        spectral_skewness, spectral_kurtosis,
        band_power_0_100, band_power_100_500
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    # One-sided amplitude spectrum
    yf    = rfft(x)
    freqs = rfftfreq(n, d=1.0 / fs)
    amp   = (2.0 / n) * np.abs(yf)
    amp[0] = amp[0] / 2.0   # DC component is not doubled

    eps = 1e-12

    # ---- Peak detection ---------------------------------------------------
    # Use find_peaks with a half-max prominence to ignore minor ripples,
    # then fall back to argmax if no peak found.
    peaks, _ = find_peaks(amp, height=np.max(amp) * 0.5)
    if peaks.size > 0:
        peak_idx = peaks[np.argmax(amp[peaks])]
    else:
        peak_idx = int(np.argmax(amp))

    peak_freq = float(freqs[peak_idx])
    peak_ampl = float(amp[peak_idx])

    # ---- Spectral moments (amplitude-weighted) ---------------------------
    total_amp = np.sum(amp) + eps
    centroid  = np.sum(freqs * amp) / total_amp
    spread    = np.sqrt(np.sum(((freqs - centroid) ** 2) * amp) / total_amp)

    if spread > 0:
        spec_skew = np.sum(((freqs - centroid) ** 3) * amp) / (total_amp * spread ** 3)
        spec_kurt = np.sum(((freqs - centroid) ** 4) * amp) / (total_amp * spread ** 4) - 3.0
    else:
        spec_skew = 0.0
        spec_kurt = 0.0

    # ---- Band powers (sum of amplitudes in band) -------------------------
    # NOTE: this matches the convention used in the companion PAM paper
    # (sum of amplitudes, not the integral of |X(f)|^2).  Documented in
    # the DiB Data Description section.
    mask_low  = (freqs >= BAND_LOW[0])  & (freqs <= BAND_LOW[1])
    mask_high = (freqs >  BAND_HIGH[0]) & (freqs <= BAND_HIGH[1])
    band_low_power  = np.sum(amp[mask_low])
    band_high_power = np.sum(amp[mask_high])

    return {
        "peak_freq":          peak_freq,
        "peak_ampl":          peak_ampl,
        "spectral_centroid":  centroid,
        "spectral_spread":    spread,
        "spectral_skewness":  spec_skew,
        "spectral_kurtosis":  spec_kurt,
        "band_power_0_100":   band_low_power,
        "band_power_100_500": band_high_power,
    }


# ============================================================
# Per-wear-state feature extraction
# ============================================================

def extract_features_for_file(filepath):
    """
    Run all three feature groups for all three channels of one LVM file.

    Returns a flat dict with 81 entries plus the inferred sampling rate.
    """
    df = load_lvm(filepath)

    # Sampling frequency from time vector
    dt = df["Time"].iloc[1] - df["Time"].iloc[0]
    fs = 1.0 / dt

    row = {}
    for ch in CHANNELS:
        sig = df[ch].to_numpy()

        feats = {}
        feats.update(compute_statistical_features(sig))
        feats.update(compute_time_series_features(sig))
        feats.update(compute_frequency_features(sig, fs))

        for name, val in feats.items():
            row[f"{ch}_{name}"] = val

    return row, fs


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []
    fs_recorded = None

    for wear_state, fname, hours in WEAR_STATE_MAPPING:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(path):
            print(f"[warn] missing file for wear state {wear_state}: {path}")
            continue

        feats, fs = extract_features_for_file(path)
        if fs_recorded is None:
            fs_recorded = fs

        feats = {"wear_state": wear_state,
                 "cumulative_hours": hours,
                 **feats}
        all_rows.append(feats)

        print(f"[ok] wear state {wear_state:2d}  ({hours:4d} h)  -> {fname}")

    if not all_rows:
        raise RuntimeError("No wear-state files were loaded. "
                           "Check DATA_DIR and WEAR_STATE_MAPPING.")

    # ----- Raw features ---------------------------------------------------
    features_df = pd.DataFrame(all_rows)
    raw_path = os.path.join(OUTPUT_DIR, "features.csv")
    features_df.to_csv(raw_path, index=False)

    # ----- Normalized features (for figure plotting) ---------------------
    # Each feature column is divided by its value at wear state 1.
    # Metadata columns are passed through unchanged.
    meta_cols    = ["wear_state", "cumulative_hours"]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    baseline = features_df.loc[features_df["wear_state"] == 1, feature_cols].iloc[0]
    baseline_safe = baseline.replace(0, np.nan)   # avoid divide-by-zero

    normalized_df = features_df.copy()
    normalized_df[feature_cols] = features_df[feature_cols].divide(baseline_safe)

    norm_path = os.path.join(OUTPUT_DIR, "features_normalized.csv")
    normalized_df.to_csv(norm_path, index=False)

    # ----- Summary --------------------------------------------------------
    print()
    print("Extraction complete.")
    print(f"  wear states processed : {len(all_rows)}")
    print(f"  feature columns       : {len(feature_cols)}  "
          f"(expected 81 = 27 x 3 sensors)")
    print(f"  sampling rate (Hz)    : {fs_recorded:.1f}")
    print(f"  raw features          : {raw_path}")
    print(f"  normalized features   : {norm_path}")


if __name__ == "__main__":
    main()
