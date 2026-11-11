"""Functions for loading and cleaning the tracking data and for the statistics, used by load_and_save.py."""
from itertools import combinations
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def load_xml(path):
    """
    Load a tracking result (.xml, TrackMate "particle/detection" format) as a DataFrame
    with one row per detection: particle (track id), nSpots, t [frame], x, y [pxl].
    """
    root = ET.parse(path).getroot()
    rows = []
    for particle_id, particle in enumerate(root.findall('particle')):
        for detection in particle.findall('detection'):
            rows.append({
                'particle': particle_id,
                'nSpots': int(particle.attrib['nSpots']),
                't': int(detection.attrib['t']),
                'x': float(detection.attrib['x']),
                'y': float(detection.attrib['y']),
            })
    return pd.DataFrame(rows)


def count_stationary_traj(df, threshold=0.0):
    """Ids of the tracks whose distance between first and last position is below threshold."""
    stat_ids = []
    for particle_id, group in df.groupby('particle'):
        distance = np.sqrt((group.iloc[-1]['x'] - group.iloc[0]['x']) ** 2 + (group.iloc[-1]['y'] - group.iloc[0]['y']) ** 2)
        if distance < threshold:
            stat_ids.append(particle_id)
    return stat_ids


def fill_gaps_linear(df):
    """
    Fill missing frames within every track with linearly interpolated positions.

    Only particle, t, x and y are filled in for the inserted rows; their other columns
    (such as dx, dy) stay NaN. Returns the table sorted by particle and t, with a
    recalculated dt column.
    """
    df = df.sort_values(by=['particle', 't']).reset_index(drop=True)
    df['dt'] = df.groupby('particle', sort=False)['t'].diff()

    new_rows = []
    for idx in df[df['dt'] > 1].index:
        t_start, x_start, y_start = df.loc[idx - 1, 't'], df.loc[idx - 1, 'x'], df.loc[idx - 1, 'y']
        t_end, x_end, y_end = df.loc[idx, 't'], df.loc[idx, 'x'], df.loc[idx, 'y']
        gap_length = int(t_end - t_start - 1)
        for i in range(1, gap_length + 1):
            fraction = i / (gap_length + 1)
            new_rows.append({'particle': df.loc[idx, 'particle'], 't': t_start + i,
                             'x': x_start + (x_end - x_start) * fraction,
                             'y': y_start + (y_end - y_start) * fraction})

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = df.sort_values(by=['particle', 't']).reset_index(drop=True)

    df = df.drop(columns=['dt'])
    df['dt'] = df.groupby('particle', sort=False)['t'].diff()
    return df


def slice_time_window(df, t_start=0, t_end=None):
    """Copy of df with t_start <= t <= t_end."""
    if t_end is None:
        return df[df['t'] >= t_start].copy()
    return df[(df['t'] >= t_start) & (df['t'] <= t_end)].copy()


def speed_weighted_circular_variance(df):
    """
    Circular variance of the swimming direction, every frame weighted by its speed:
    V = 1 - |sum(speed * exp(i * orientation))| / sum(speed).
    0 = all frames in the same direction, 1 = directions uniformly spread.
    """
    df = df.dropna(subset=['orientation', 'speed'])
    angles = df['orientation'].values
    weights = df['speed'].values
    R = np.abs(np.sum(weights * np.exp(1j * angles))) / np.sum(weights)
    return 1 - R


def compact_letter_display(group_names, reject_lookup, group_means):
    """
    Letters for a post-hoc test: groups that share a letter do not differ significantly.
    Uses the smallest number of letters (exhaustive search), groups ordered by mean.

    reject_lookup: {frozenset((a, b)): True if a and b differ significantly}
    """
    ordered_names = sorted(group_names, key=lambda name: group_means[name], reverse=True)
    name_to_index = {name: idx for idx, name in enumerate(ordered_names)}
    nonsig_pairs = {
        tuple(sorted((name_to_index[a], name_to_index[b])))
        for a, b in combinations(ordered_names, 2)
        if not reject_lookup[frozenset((a, b))]
    }

    # Every group and every non-significant pair must be covered by a letter
    universe = set(range(len(ordered_names))) | nonsig_pairs
    candidate_subsets = []
    for subset_size in range(1, len(ordered_names) + 1):
        for subset in combinations(range(len(ordered_names)), subset_size):
            if all(tuple(sorted(pair)) in nonsig_pairs for pair in combinations(subset, 2)):
                cover = set(subset)
                cover.update({tuple(sorted(pair)) for pair in combinations(subset, 2)})
                candidate_subsets.append((subset, cover))
    candidate_subsets.sort(key=lambda item: (-len(item[1]), len(item[0]), item[0]))

    best_solution = None

    def search(covered, chosen):
        nonlocal best_solution
        if best_solution is not None and len(chosen) >= len(best_solution):
            return
        if covered == universe:
            best_solution = list(chosen)
            return
        target = next(iter(universe - covered))
        for subset, cover in candidate_subsets:
            if target in cover:
                search(covered | cover, chosen + [subset])

    search(set(), [])

    if best_solution is None:
        return {name: "a" for name in group_names}

    letters = {name: "" for name in ordered_names}
    for letter_idx, subset in enumerate(best_solution):
        for idx in subset:
            letters[ordered_names[idx]] += chr(ord('a') + letter_idx)
    return letters


def tukey_posthoc(group_names, group_values):
    """Tukey's HSD (alpha = 0.05) between the groups; returns the compact letter display."""
    combined_values = np.concatenate([np.asarray(values) for values in group_values])
    combined_labels = np.concatenate([np.repeat(name, len(values)) for name, values in zip(group_names, group_values)])
    tukey = pairwise_tukeyhsd(combined_values, combined_labels, alpha=0.05)

    reject_lookup = {}
    for group_a, group_b, meandiff, p_adj, lower, upper, reject in tukey.summary().data[1:]:
        reject_lookup[frozenset((group_a, group_b))] = bool(reject)

    group_means = {name: float(np.mean(values)) for name, values in zip(group_names, group_values)}
    return compact_letter_display(group_names, reject_lookup, group_means)
