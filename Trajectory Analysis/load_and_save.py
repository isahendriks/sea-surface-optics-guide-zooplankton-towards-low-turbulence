"""
Load, clean and save the tracked trajectories, and test the effect of the light regime
on the vertical swimming velocity and the swimming direction.

For every species, light regime ("still" = calm, "breeze" = moderate, "stormy" = rough,
"nothing" = no light, control) and tank, the tracks are
  1. scaled from pixels to mm (y axis pointing up),
  2. converted to per-frame velocities dx, dy [mm/s],
  3. cut to the analysis time window (not for the control),
  4. cleaned of tracking artifacts (steps < ARTIFACT_THRESHOLD in x or y),
  5. gap filled by linear interpolation,
  6. given an orientation (swimming direction) and speed,
  7. cleaned of stationary tracks (net displacement < MIN_MOVEMENT) and
  8. of short tracks (< min_frames frames).
The cleaned tables are saved as traj_<regime>_scaled.pkl (used by plot_full_per_species.py
and the kinematic simulations). Per tank, the mean dy and the speed-weighted circular
variance V of the swimming direction are computed; regimes are compared with a one-way
ANOVA and Tukey's HSD, using the tanks as replicates.

Input:  <TRACKING_DATA_DIR>/<species>/tracking_results/<regime><tank>.xml
Output: <TRACKING_DATA_DIR>/<species>/trajectories_scaled/traj_<regime>_scaled.pkl
"""
#%% Imports and settings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp

from post_processing_functions import (count_stationary_traj, fill_gaps_linear, load_xml, slice_time_window,
                                       speed_weighted_circular_variance, tukey_posthoc)

# Folder with one subfolder per species (download the tracking data set and point to it here)
TRACKING_DATA_DIR = Path(__file__).resolve().parent / "data"

FRAME_RATE = 10               # [frames/s]
MIN_MOVEMENT = 0.3            # tracks with a smaller net displacement are stationary [mm]
ARTIFACT_THRESHOLD = 0.0001   # steps smaller than this in x or y are tracking artifacts [mm]
REGIMES = ['still', 'breeze', 'stormy', 'nothing']  # calm, moderate, rough, control (no light)

# Recording settings per species
#   h_cropped_pxl: height of the cropped video that was tracked [pxl] (to flip the y axis)
#   w_raw_pxl, w_raw_mm: width of the raw video [pxl] and the field of view it covers [mm]
#   binning: binning of the tracked video relative to the raw video
#   time_start, time_end: analysis time window [frames] (time_start skips the acclimation)
#   min_frames: shortest track that is kept [frames]
SPECIES = {
    'Artemia_0805': dict(h_cropped_pxl=1280, w_raw_pxl=853, w_raw_mm=70.6, binning=1,
                         time_start=150, time_end=10000, min_frames=30, tanks=['1', '2', '3']),
    'BarnacleLarvae_2102': dict(h_cropped_pxl=630, w_raw_pxl=3648, w_raw_mm=110, binning=4,
                                time_start=0, time_end=10000, min_frames=30, tanks=['1', '2', '3']),
    'Cladocerans_2605': dict(h_cropped_pxl=533, w_raw_pxl=2256, w_raw_mm=90.6, binning=4,
                             time_start=0, time_end=10000, min_frames=30, tanks=['1', '2', '3', '4', '5']),
}

#%% Functions

def process_tank(df, regime, tank, settings, scale):
    """Clean the tracks of one tank, as loaded by load_xml (steps 1-8 above)."""
    # 1. Scale to mm and flip the y axis so it points up
    df['x'] = df['x'].astype(float) * scale
    df['y'] = (settings['h_cropped_pxl'] - df['y'].astype(float) - 1) * scale

    # 2. Velocity per frame [mm/s]; drops the first detection of every track
    df = df.sort_values(['particle', 't']).reset_index(drop=True)
    df['dt'] = df.groupby('particle')['t'].diff()
    df['dx'] = df.groupby('particle')['x'].diff() / df['dt'] * FRAME_RATE
    df['dy'] = df.groupby('particle')['y'].diff() / df['dt'] * FRAME_RATE
    df = df.dropna(subset=['dx', 'dy', 'dt']).reset_index(drop=True)

    # 3. Analysis time window (the control keeps the full recording)
    if regime != 'nothing':
        df = slice_time_window(df, settings['time_start'], settings['time_end'])
    df = df.reset_index(drop=True)

    # 4. Remove artifacts: steps with |x or y change| below the threshold since the previous
    #    detection of the track (this also drops the first remaining detection of every track)
    x_diff = df.groupby('particle', sort=False)['x'].diff().abs()
    y_diff = df.groupby('particle', sort=False)['y'].diff().abs()
    df = df[(x_diff >= ARTIFACT_THRESHOLD) & (y_diff >= ARTIFACT_THRESHOLD)]

    # 5. Fill gaps (inserted frames get a position but no velocity)
    df = fill_gaps_linear(df)

    # 6. Swimming direction and speed
    df['orientation'] = np.arctan2(df['dy'], df['dx'])  # [rad]
    df['orientation_degrees'] = np.degrees(df['orientation'])
    df['speed'] = np.sqrt(df['dx'] ** 2 + df['dy'] ** 2)  # [mm/s]

    # 7. Remove stationary tracks
    stat_ids = count_stationary_traj(df, threshold=MIN_MOVEMENT)
    if stat_ids:
        df = df[~df['particle'].isin(stat_ids)].reset_index(drop=True)

    df['tank'] = int(tank)

    # 8. Remove short tracks
    df['nSpots'] = df.groupby('particle')['t'].transform('count')
    df = df[df['nSpots'] >= settings['min_frames']].reset_index(drop=True)
    return df


def analyze_species(measurement):
    settings = SPECIES[measurement]
    tanks = settings['tanks']
    scale = settings['w_raw_mm'] / settings['w_raw_pxl'] * settings['binning']  # [mm/pxl]

    path_results = TRACKING_DATA_DIR / measurement / 'tracking_results'
    path_save = TRACKING_DATA_DIR / measurement / 'trajectories_scaled'
    path_save.mkdir(parents=True, exist_ok=True)

    ### Load and clean all tanks, and save the cleaned tables
    dfs = []
    organisms_per_tank = []
    for regime in REGIMES:
        data_regime = []
        for tank in tanks:
            print(f"\rProcessing {regime} {tank} ...", end='\r', flush=True)
            df_tank = load_xml(path_results / f"{regime}{tank}.xml")

            # Rough estimate of the number of animals in the tank: detections at t = 0, times 2
            organisms_per_tank.append(len(df_tank[df_tank['t'] == 0]) * 2)

            data_regime.append(process_tank(df_tank, regime, tank, settings, scale))
        dfs.append(pd.concat(data_regime, ignore_index=True))

    for regime, df in zip(REGIMES, dfs):
        df.to_pickle(path_save / f'traj_{regime}_scaled.pkl')

    ### Common analysis window: end at the shortest recording of the light regimes
    analysis_time_start = settings['time_start']
    analysis_time_end = settings['time_end']
    for df in dfs[:-1]:
        for tank in tanks:
            end_time = df[df['tank'] == int(tank)]['t'].max()
            if end_time < analysis_time_end:
                analysis_time_end = end_time

    dfs_window = [slice_time_window(df, analysis_time_start, analysis_time_end) for df in dfs[:-1]]
    dfs_window.append(slice_time_window(dfs[-1], 0, analysis_time_end))  # control from t = 0

    ### Per tank: mean dy and speed-weighted circular variance V
    tank_means = [[np.mean(df['dy'][df['tank'] == int(tank)]) for tank in tanks] for df in dfs_window]
    tank_V = [[speed_weighted_circular_variance(df[df['tank'] == int(tank)]) for tank in tanks] for df in dfs_window]

    n_replicates = len(tanks)
    means = [np.mean(m) for m in tank_means]
    SE = [np.std(m, ddof=1) / np.sqrt(n_replicates) for m in tank_means]
    mean_V = [np.mean(v) for v in tank_V]
    SE_V = [np.std(v, ddof=1) / np.sqrt(n_replicates) for v in tank_V]

    ### One-way ANOVA and Tukey's HSD over the three light regimes (not the control)
    _, p_value = sp.stats.f_oneway(*tank_means[:3])
    _, p_value_V = sp.stats.f_oneway(*tank_V[:3])
    letters = tukey_posthoc(REGIMES[:3], tank_means[:3])
    letters_V = tukey_posthoc(REGIMES[:3], tank_V[:3])

    print(f"\n=== Statistical results for {measurement} ===")
    print(f"Analysis window: {analysis_time_start} to {analysis_time_end} frames")
    print(f"Artifact threshold: {ARTIFACT_THRESHOLD:.4f} mm / {ARTIFACT_THRESHOLD/scale:.4f} pxls")
    print(f"Minimum movement to consider non-stationary: {MIN_MOVEMENT:.2f} mm / {MIN_MOVEMENT/scale:.2f} pxls")
    print(f"Minimum trajectory length: {settings['min_frames']} frames / {settings['min_frames']/FRAME_RATE:.1f} seconds")

    print(f"\n=== ANOVA results for {measurement} ===")
    print(f"P-value dy: {p_value:.4f}")
    print(f"P-value V: {p_value_V:.4f}")

    print("\n=== Mean dy and speed weighted circular variance V (per frame), mean ± SE over the tanks ===")
    for i, regime in enumerate(REGIMES):
        # The control has no letter (not part of the Tukey test)
        print(f"{regime.capitalize()}: Mean dy = {means[i]:.2f} ± {SE[i]:.2f} mm/s  [{letters.get(regime, '-')}], "
              f"V = {mean_V[i]:.3f} ± {SE_V[i]:.3f}  [{letters_V.get(regime, '-')}]")

    print(f"mean number of organisms per tank: {np.mean(organisms_per_tank):.1f} ± {np.std(organisms_per_tank, ddof=1):.1f}")

#%% Run all species
if __name__ == "__main__":
    for measurement in SPECIES:
        analyze_species(measurement)
