from collections.abc import Sequence
from itertools import combinations
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tqdm import tqdm


def load_xml(path: str) -> pd.DataFrame:
    """    
    Loads .xml file and saves it as dataframe incl orientations and speed
    """

    tree = ET.parse(path)
    root = tree.getroot()

    data = []

    for particle_id, particle in enumerate(root.findall('particle')):
        for detection in particle.findall('detection'):

            # Combine particle and detection attributes
            entry = {
                'particle': particle_id,
                'nSpots': int(particle.attrib['nSpots']),
                't': int(detection.attrib['t']),
                'x': float(detection.attrib['x']),
                'y': float(detection.attrib['y']),
            }
            data.append(entry)

    df = pd.DataFrame(data)

    return df

def count_stationary_traj(df: pd.DataFrame, threshold: float = 0.0) -> list[int]:
    """ 
    Calculates total distance moved for each trajectory based on starting and ending position
    and returns ids of trajectories that moved less than the treshold (considered stationary)
    """
    
    stat_ids = []
    for particle_id, group in df.groupby('particle'):
        x_start = group.iloc[0]['x']
        y_start = group.iloc[0]['y']
        x_end = group.iloc[-1]['x']
        y_end = group.iloc[-1]['y']

        distance = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)

        if distance < threshold:
            stat_ids.append(particle_id)
    
    return stat_ids

def fill_gaps_linear(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Fill time gaps by linearly interpolating positions between consecutive detections.
    
    Identifies gaps where dt > 1 for each particle, then inserts missing time points with
    linearly interpolated x and y positions. Returns the gap-filled dataframe with updated dt.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: particle, t, x, y
    verbose : bool, optional
        If True, prints number of gaps before and after filling. Default is True.
    
    Returns
    -------
    pd.DataFrame
        Sorted dataframe with filled gaps and recalculated dt column.
    """
    # Identify all gaps first (before modifying dataframe)
    df = df.sort_values(by=['particle', 't']).reset_index(drop=True)
    df['dt'] = df.groupby('particle', sort=False)['t'].diff()
    
    gap_indices = df[df['dt'] > 1].index
    if verbose:
        print(f'\nnumber of gaps before filling: {len(gap_indices)}')
    
    # Find gaps and collect gap info
    gap_data = []
    for idx in gap_indices:
        particle_id = df.loc[idx, 'particle']
        t_start = df.loc[idx - 1, 't']
        x_start = df.loc[idx - 1, 'x']
        y_start = df.loc[idx - 1, 'y']
        t_end = df.loc[idx, 't']
        x_end = df.loc[idx, 'x']
        y_end = df.loc[idx, 'y']
        
        # Generate missing frames
        gap_length = int(t_end - t_start - 1)
        for i in range(1, gap_length + 1):
            t_new = t_start + i
            x_new = x_start + (x_end - x_start) * (i / (gap_length + 1))
            y_new = y_start + (y_end - y_start) * (i / (gap_length + 1))
            gap_data.append({'particle': particle_id, 't': t_new, 'x': x_new, 'y': y_new})
            
    # Add all new entries at once
    if gap_data:
        df = pd.concat([df, pd.DataFrame(gap_data)], ignore_index=True)
        df = df.sort_values(by=['particle', 't']).reset_index(drop=True)
    
    # Recalculate dt and verify gaps are filled
    df = df.drop(columns=['dt'])
    df['dt'] = df.groupby('particle', sort=False)['t'].diff()
    gap_indices_after = df[df['dt'] > 1].index
    if verbose:
        print(f'number of gaps after filling: {len(gap_indices_after)}')
    
    return df


def mean_dy_per_particle(df_treatment: pd.DataFrame, tank_id: int) -> float:
    df_tank = df_treatment[df_treatment['tank'] == tank_id]
    df_particle = df_tank.groupby('particle', as_index=False).agg(
        dy=('dy', 'mean'),
        nSpots=('dy', 'size'),
    )
    return np.average(df_particle['dy'], weights=df_particle['nSpots'])


def compact_letter_display(
    group_names: Sequence[str],
    reject_lookup: dict[frozenset[str], bool],
    group_means: dict[str, float],
) -> dict[str, str]:
    ordered_names = sorted(group_names, key=lambda name: group_means[name], reverse=True)
    name_to_index = {name: idx for idx, name in enumerate(ordered_names)}
    nonsig_pairs = {
        tuple(sorted((name_to_index[a], name_to_index[b])))
        for a, b in combinations(ordered_names, 2)
        if not reject_lookup[frozenset((a, b))]
    }

    universe = set(range(len(ordered_names))) | nonsig_pairs
    candidate_subsets = []

    for subset_size in range(1, len(ordered_names) + 1):
        for subset in combinations(range(len(ordered_names)), subset_size):
            if all(tuple(sorted(pair)) in nonsig_pairs for pair in combinations(subset, 2)):
                cover: set[object] = set(subset)
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

    letters = {}
    for idx, name in enumerate(ordered_names):
        letters[name] = ""

    if best_solution is None:
        return {name: "a" for name in group_names}

    alphabet = [chr(ord('a') + i) for i in range(26)]
    for letter_idx, subset in enumerate(best_solution):
        letter = alphabet[letter_idx]
        for idx in subset:
            letters[ordered_names[idx]] += letter

    return letters


def tukey_posthoc(
    group_names: Sequence[str],
    group_values: Sequence[np.ndarray],
) -> tuple[dict[str, float], dict[str, str], pd.DataFrame]:
    combined_values = np.concatenate([np.asarray(values) for values in group_values])
    combined_labels = np.concatenate([
        np.repeat(name, len(values))
        for name, values in zip(group_names, group_values)
    ])

    tukey = pairwise_tukeyhsd(combined_values, combined_labels, alpha=0.05)

    reject_lookup: dict[frozenset[str], bool] = {}
    pvalue_lookup: dict[frozenset[str], float] = {}
    for row in tukey.summary().data[1:]:
        group_a, group_b, meandiff, p_adj, lower, upper, reject = row
        key = frozenset((group_a, group_b))
        reject_lookup[key] = bool(reject)
        pvalue_lookup[key] = float(p_adj)

    group_means = {
        name: float(np.mean(values))
        for name, values in zip(group_names, group_values)
    }
    letters = compact_letter_display(group_names, reject_lookup, group_means)

    summary_rows: list[dict[str, object]] = []
    for row in tukey.summary().data[1:]:
        group_a, group_b, meandiff, p_adj, lower, upper, reject = row
        summary_rows.append({
            'group1': group_a,
            'group2': group_b,
            'meandiff': meandiff,
            'p_adj': p_adj,
            'lower': lower,
            'upper': upper,
            'reject': reject,
        })

    return group_means, letters, pd.DataFrame(summary_rows)

def slice_time_window(
    df: pd.DataFrame,
    t_start: int | float = 0,
    t_end: int | float | None = None,
) -> pd.DataFrame:
    """Return a copy of df restricted to the requested time window."""
    if t_end is None:
        return df[df['t'] >= t_start].copy()
    return df[(df['t'] >= t_start) & (df['t'] <= t_end)].copy()

def calc_orientation_and_speed(df: pd.DataFrame) -> pd.DataFrame:

    """
    Calculates the orientation and the speed of the particles and adds it to the dataframe
    Notice the - sign before dy, this is because the origin of the image is at the top left cornet so the y axis is inverted
    """
    df.sort_values(by=['particle', 't'])

    df['dx'] = df.groupby('particle')['x'].diff()
    df['dy'] = -df.groupby('particle')['y'].diff()

    df['orientation'] = np.arctan2(df['dy'], df['dx']) # in radians
    df['orientation'] = df.groupby('particle')['orientation'].bfill() # fill the NaN values with the previous value

    # Add average orientation
    df['orientation_degrees'] = np.degrees(df['orientation'])

    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2) # in pixels per frame
    df['speed'] = df.groupby('particle')['speed'].bfill()

    df['speed_change'] = df.groupby('particle')['speed'].transform(lambda x: x.max() - x.min())
    df['orientation_change'] = df.groupby('particle')['orientation'].transform(lambda x: x.max() - x.min())

    df.sort_values(by=['particle', 't'], inplace=True)

    return df

def load_frames(path: str, total_frames: int | None = None, verbose: bool = True) -> np.ndarray:

    """
    Function to load the video frames, supporting various compressions including PNG-compressed AVIs.

    Input:
    path = path to video file
    total_frames = number of frames to read (None = read all)
    verbose = print loading information

    output: 
    frames = numpy array with all frames in video 
    """

    frames: list[np.ndarray] = []
    errors: list[str] = []
    
    # Method 1: Try imageio without specifying plugin (auto-detect)
    try:
        import imageio.v3 as iio
        if verbose:
            print(f"Trying imageio (auto-detect): {path}")
        
        vid = iio.imopen(path, 'r')
        
        frame_count = 0
        if total_frames is None:
            for frame in tqdm(vid, desc="Loading frames"):
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frames.append(frame)
                frame_count += 1
        else:
            for i, frame in enumerate(tqdm(vid, total=total_frames, desc="Loading frames")):
                if i >= total_frames:
                    break
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frames.append(frame)
                frame_count += 1
        
        vid.close()
        
        if frame_count > 0:
            if verbose:
                print(f"✓ Successfully loaded {frame_count} frames with imageio")
            return np.array(frames)
            
    except Exception as e:
        errors.append(f"imageio auto-detect: {type(e).__name__}: {e}")
        if verbose:
            print(f"✗ imageio auto-detect failed: {type(e).__name__}: {e}")
    
    # Method 2: Try PyAV (best for unusual codecs)
    try:
        import av
        if verbose:
            print(f"Trying PyAV: {path}")
        
        container = av.open(path)
        stream = container.streams.video[0]
        
        frame_count = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                if total_frames is not None and frame_count >= total_frames:
                    break
                    
                img = frame.to_ndarray(format='gray')
                frames.append(img)
                frame_count += 1
                
                if verbose and frame_count % 100 == 0:
                    print(f"  Loaded {frame_count} frames...", end='\r')
            
            if total_frames is not None and frame_count >= total_frames:
                break
        
        container.close()
        
        if frame_count > 0:
            if verbose:
                print(f"✓ Successfully loaded {frame_count} frames with PyAV")
            return np.array(frames)
            
    except ImportError:
        errors.append("PyAV not installed (pip install av)")
        if verbose:
            print(f"✗ PyAV not available (install with: pip install av)")
    except Exception as e:
        errors.append(f"PyAV: {type(e).__name__}: {e}")
        if verbose:
            print(f"✗ PyAV failed: {type(e).__name__}: {e}")
    
    # Method 3: Try OpenCV with different settings
    try:
        if verbose:
            print(f"Trying OpenCV: {path}")
        
        cap = cv2.VideoCapture(path)
        
        if not cap.isOpened():
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file")
        
        if total_frames is None:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if verbose:
                print(f"  OpenCV reports {total_frames} frames")
        
        frame_count = 0
        for i in tqdm(range(total_frames if total_frames > 0 else 10000), desc="Loading frames"):
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        if frame_count > 0:
            if verbose:
                print(f"✓ Successfully loaded {frame_count} frames with OpenCV")
            return np.array(frames)
        else:
            raise RuntimeError(f"OpenCV opened file but read 0 frames (codec issue?)")
            
    except Exception as e:
        errors.append(f"OpenCV: {type(e).__name__}: {e}")
        if verbose:
            print(f"✗ OpenCV failed: {type(e).__name__}: {e}")
    
    # All methods failed
    error_summary = "\n  - ".join(errors)
    raise RuntimeError(
        f"Failed to load video from: {path}\n"
        f"All methods failed:\n  - {error_summary}\n"
        f"For PNG-compressed AVIs, try: pip install av"
    )