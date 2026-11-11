"""
Empirical swimming behaviour of the plankton, built from the tracked trajectories.

The turning kernel stores, for each bin of "deviation of the heading from upward", all
measured (turning angle, speed) pairs between two consecutive video frames. Simulated
plankton draw a pair from the bin that matches their own current heading, so the
simulated swimming reproduces the measured turning and speed statistics.

This module only defines functions; it is imported by coupled_simulation.py and
demo_swimming_behavior.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DELTA_T_MEAS = 0.1    # frame interval of the tracking data [s] (10 fps)
TARGET_ANGLE = np.pi / 2  # upward, the direction the plankton are expected to swim to
DATA_DIR = Path(__file__).parent / "data"  # data/<measurement>/traj_<condition>_scaled.pkl


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def build_turning_kernel(df, target_angle, n_bins=8, min_bin_samples=200):
    """
    Build the turning kernel from measured trajectories.

    Parameters
        df: DataFrame with columns tank, particle, t, speed, orientation (as saved by
            Trajectory Analysis/load_and_save.py)
        target_angle: heading [rad] the deviation is measured from
        n_bins: number of deviation bins between -pi and pi
        min_bin_samples: a bin with fewer samples uses the pooled samples of all bins

    Returns a dict with
        bin_edges: deviation bin edges [rad]
        kernel:    list of (turning_angles, speeds) arrays, one entry per bin
    """
    df = df.dropna(subset=["speed", "orientation"]).sort_values(["tank", "particle", "t"])
    g = df.groupby(["tank", "particle"])

    dphi = g["orientation"].diff().apply(wrap_angle)  # turning angle between consecutive frames
    consecutive = g["t"].diff() == 1                   # skip pairs that span a gap in the track
    valid = dphi.notna() & consecutive

    # Deviation from the target angle in the frame BEFORE the turn. shift(1) runs over the
    # whole table, but only rows where `valid` is True (never the first row of a track) are used.
    dev_before = wrap_angle(target_angle - df["orientation"]).shift(1)

    dphi_v = dphi[valid].to_numpy()
    speed_v = df["speed"][valid].to_numpy()
    dev_v = dev_before[valid].to_numpy()

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.clip(np.digitize(dev_v, bin_edges) - 1, 0, n_bins - 1)

    kernel = []
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() < min_bin_samples:
            kernel.append((dphi_v, speed_v))
        else:
            kernel.append((dphi_v[sel], speed_v[sel]))

    return {"bin_edges": bin_edges, "kernel": kernel}


def sample_turning_kernel(turning_kernel, phi, target_angle):
    """
    Draw one (turning angle, speed) pair for every heading in phi, from the kernel bin
    that matches that heading's current deviation from target_angle.
    """
    bin_edges = turning_kernel["bin_edges"]
    kernel = turning_kernel["kernel"]
    n_bins = len(kernel)

    dev = wrap_angle(target_angle - phi)
    bin_idx = np.clip(np.digitize(dev, bin_edges) - 1, 0, n_bins - 1)

    dphi_sample = np.zeros_like(phi)
    speed_sample = np.zeros_like(phi)
    for b in range(n_bins):
        sel = bin_idx == b
        n_sel = int(sel.sum())
        if n_sel == 0:
            continue
        dphi_pool, speed_pool = kernel[b]
        draw = np.random.randint(0, len(dphi_pool), size=n_sel)
        dphi_sample[sel] = dphi_pool[draw]
        speed_sample[sel] = speed_pool[draw]

    return dphi_sample, speed_sample


def load_behavior_parameters(measurement, condition):
    """
    Build the turning kernel and the pool of initial headings for one species and light condition.

    Parameters
        measurement: e.g. "Artemia_0805", a folder in data/
        condition:   "still", "breeze" or "stormy"

    Returns a dict with
        turning_kernel:   see build_turning_kernel
        initial_headings: all measured headings [rad], to draw starting headings from
    """
    path_to_traj = DATA_DIR / measurement / f"traj_{condition}_scaled.pkl"
    df_traj = pd.read_pickle(path_to_traj)
    df_traj = df_traj.dropna(subset=["speed", "orientation"])

    turning_kernel = build_turning_kernel(df_traj, TARGET_ANGLE)

    print(f"turning kernel: {len(turning_kernel['kernel'])} bins, "
          f"{[len(pool[0]) for pool in turning_kernel['kernel']]} samples/bin")

    return {
        "turning_kernel": turning_kernel,
        "initial_headings": df_traj["orientation"].to_numpy(),
    }
