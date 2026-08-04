"""
plot_feature_traces.py
----------------------
Generate normalized feature-trace figures for the FFF wear-monitoring dataset,
reading the same output produced by extract_features.py.

Revisions per A. Downey:
  * each feature category (statistical / time-series / frequency-domain) is its
    OWN figure, per sensor  -> 3 sensors x 3 categories = 9 figures
  * the category name is embedded in the y-axis label
  * wide-and-short aspect ratio so the publisher does not compress it
  * dual x-axis: sample number on top row, cumulative printing time (h) beneath

Input : features_normalized.csv  (rows = 15 samples; columns = {sensor}_{feature}
        over sensors Head/Motor/Frame, plus a cumulative-hours column).
        Feature values are already wear-state-1 normalized; each trace is then
        min-max scaled here only to stack it within its band.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ----------------------------------------------------------------------
# publication style (swap in usetex=True locally to match the cas-sc body)
# ----------------------------------------------------------------------
rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 5,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 500,
    "savefig.bbox": "tight",
})

# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
INPUT_FILE   = "features_normalized.csv"
X_TIME_COLUMN = "cumulative_hours"          # <-- confirm exact header
OUTDIR       = "figures"

# sensor display label -> column prefix used by extract_features.py (CHANNELS)
SENSOR_PREFIXES = {
    "frame":          "Frame",
    "head": "Head",
    "motor":  "Motor",
}

SAMPLE_TICK_STRIDE = 1     # label every Nth sample on the top row
TIME_TICK_STRIDE   = 2     # label every Nth sample's hours on the bottom row
BAND_SCALE         = 0.75  # vertical fraction of a band each trace fills

# ----------------------------------------------------------------------
# feature catalog: (feature_key, display_label), in bottom-to-top plot order.
# feature_key is the suffix after "{sensor}_" in the CSV header.
# Frequency keys taken from extract_features.py; the statistical / time-series
# suffixes are best-guess -- verify against the real header (see resolve()).
# ----------------------------------------------------------------------
FEATURE_GROUPS = {
    "statistical": [
        ("mean",         "mean"),
        ("std",          "std. dev."),
        ("var",          "variance"),
        ("min",          "minimum"),
        ("max",          "maximum"),
        ("range",        "range"),
        ("median",       "median"),
        ("iqr",          "IQR"),
        ("mad",          "mean abs. dev."),
        ("skewness",     "skewness"),
        ("kurtosis",     "kurtosis"),
    ],
    "time-series": [
        ("rms",              "RMS"),
        ("peak",             "peak"),
        ("peak_to_peak",     "peak-to-peak"),
        ("abs_mean",         "absolute mean"),
        ("crest_factor",     "crest factor"),
        ("shape_factor",     "shape factor"),
        ("impulse_factor",   "impulse factor"),
        ("clearance_factor", "clearance factor"),
    ],
    "frequency-domain": [
        ("peak_freq",          "peak freq."),
        ("peak_ampl",          "peak ampl."),
        ("spectral_centroid",  "spectral centroid"),
        ("spectral_spread",    "spectral spread"),
        ("spectral_skewness",  "spectral skewness"),
        ("spectral_kurtosis",  "spectral kurtosis"),
        ("band_power_0_100",   "band 0\u2013100 Hz"),
        ("band_power_100_500", "band 100\u2013500 Hz"),
    ],
}


def resolve(df, prefix, key):
    """Case-insensitive lookup of '{prefix}_{key}' in df.columns."""
    want = f"{prefix}_{key}".lower()
    lut = {c.lower(): c for c in df.columns}
    if want in lut:
        return lut[want]
    raise KeyError(
        f"column '{prefix}_{key}' not found. Available for {prefix}: "
        + ", ".join(c for c in df.columns if c.lower().startswith(prefix.lower() + "_"))
    )


def plot_group(df, sensor, group, outdir=OUTDIR):
    """Render one figure: one feature category for one sensor, dual x-axis."""
    prefix = SENSOR_PREFIXES[sensor]
    feats = FEATURE_GROUPS[group]
    n = len(feats)

    samples = np.arange(1, len(df) + 1)
    hours = df[X_TIME_COLUMN].to_numpy(dtype=float)

    width, height = 7.2, 0.27 * n + 1.35   # wide-and-short + room for 2nd axis
    fig, ax = plt.subplots(figsize=(width, height))

    ticks = []
    for i, (key, _) in enumerate(feats):
        y = df[resolve(df, prefix, key)].to_numpy(dtype=float)
        span = np.nanmax(y) - np.nanmin(y)
        yn = (y - np.nanmin(y)) / span if span > 0 else np.zeros_like(y)
        ax.plot(samples, yn * BAND_SCALE + i, marker="o", markersize=3, linewidth=1.0)
        ticks.append(i + BAND_SCALE / 2)

    ax.set_yticks(ticks)
    ax.set_yticklabels([lbl for _, lbl in feats])
    ax.set_ylim(-0.35, n - 1 + BAND_SCALE + 0.35)
    ax.set_ylabel(f"normalized {group}\nfeatures for {sensor}")
    #f"normalized {group}\nfeatures"
    #ax.set_ylabel(f"normalized {sensor} feature traces")
    ax.grid(True, alpha=0.4, linewidth=0.5)
    ax.margins(x=0.03)

    ax.set_xticks(samples[::SAMPLE_TICK_STRIDE])
    ax.set_xlabel("sample number", labelpad=4)
    ax.tick_params(axis="x", pad=2)
    
    secax = ax.secondary_xaxis("bottom")
    secax.spines["bottom"].set_position(("outward", 42))
    
    idx = samples[::TIME_TICK_STRIDE]
    secax.set_xticks(idx)
    secax.set_xticklabels([f"{hours[i - 1]:.0f}" for i in idx])
    secax.set_xlabel("cumulative printing time (h)", labelpad=4)
    secax.tick_params(axis="x", length=3, pad=2)

    os.makedirs(outdir, exist_ok=True)
    stem = f"feature_traces_{sensor.replace(' ', '_')}_{group.replace('-', '_')}"
    fig.savefig(os.path.join(outdir, stem + ".pdf"))
    fig.savefig(os.path.join(outdir, stem + ".png"))
    plt.close(fig)
    return stem


def main():
    df = pd.read_csv(INPUT_FILE)
    for sensor in SENSOR_PREFIXES:
        for group in FEATURE_GROUPS:
            plot_group(df, sensor, group)


if __name__ == "__main__":
    main()