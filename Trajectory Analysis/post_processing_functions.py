import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np 
import cv2
from tqdm import tqdm
from scipy import stats


def load_xml(path):
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

def count_stationary_traj(df, threshold=0.0):
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

def fill_gaps_linear(df, verbose=True):
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

def slice_time_window(df, t_start=0, t_end=None):
    """Return a copy of df restricted to the requested time window."""
    if t_end is None:
        return df[df['t'] >= t_start].copy()
    return df[(df['t'] >= t_start) & (df['t'] <= t_end)].copy()

def calc_orientation_and_speed(df):

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

def load_frames(path, total_frames=None, verbose=True):

    """
    Function to load the video frames, supporting various compressions including PNG-compressed AVIs.

    Input:
    path = path to video file
    total_frames = number of frames to read (None = read all)
    verbose = print loading information

    output: 
    frames = numpy array with all frames in video 
    """

    frames = []
    errors = []
    
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

    return df_av, df_av_grouped

# Unused helpers (commented out)
# def fill_time_gaps(df, position_cols=("x", "y")):
#     """Add missing frames per particle and linearly interpolate positions.
#
#     Expected columns: particle, t, and position columns (default: x, y).
#     Returns a copy with inserted rows for missing integer t values between the
#     first and last observation of each particle. Position columns are linearly
#     interpolated; other columns are forward/back filled within the particle.
#     A fresh dt column is added (frame difference per particle).
#     """
#     required = {"particle", "t"}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(f"Missing required columns: {missing}")
#
#     if df.empty:
#         return df.copy()
#
#     df_sorted = df.sort_values(["particle", "t"]).copy()
#     if "dt" in df_sorted.columns:
#         df_sorted = df_sorted.drop(columns=["dt"])
#
#     filled_parts = []
#     for pid, grp in df_sorted.groupby("particle", sort=False):
#         grp = grp.sort_values("t")
#         full_t = pd.RangeIndex(grp["t"].min(), grp["t"].max() + 1, step=1)
#
#         expanded = grp.set_index("t").reindex(full_t)
#         expanded["particle"] = pid
#
#         non_pos_cols = [c for c in expanded.columns if c not in position_cols]
#         expanded[non_pos_cols] = expanded[non_pos_cols].ffill().bfill()
#
#         for col in position_cols:
#             if col in expanded.columns:
#                 expanded[col] = expanded[col].interpolate()
#
#         expanded = expanded.reset_index().rename(columns={"index": "t"})
#         filled_parts.append(expanded)
#
#     out = pd.concat(filled_parts, ignore_index=True)
#     out = out.sort_values(["particle", "t"]).reset_index(drop=True)
#     out["dt"] = out.groupby("particle", sort=False)["t"].diff()
#
#     return out
#
# def get_scale(raw_frame, cropped_frame, w_raw_mm, binning):
#     """Caclculates the mm/pxl scale taking into account the binning and original video width"""
#     _, w_raw_pxl = raw_frame.shape # get dimensions raw video [pxl]
#     _, w_cropped_pxl = cropped_frame.shape # get dimension cropped video [pxl]
#
#     w_cropped_mm = (w_raw_mm*w_cropped_pxl)/(w_raw_pxl/binning) # Calculate width of cropped video [mm]
#
#     scale = w_cropped_mm/w_cropped_pxl
#
#     return scale
#
# def compute_average_vector(df):
#     """
#     Computes the average vector from a DataFrame containing 'orientation' and 'magnitude'.
#
#     Parameters:
#     df (pd.DataFrame): DataFrame with columns 'orientation' (angles in radians) and 'magnitude'.
#
#     Returns:
#     tuple: (average_magnitude, average_angle) where:
#         - average_magnitude (float): Magnitude of the average vector.
#         - average_angle (float): Angle of the average vector in radians.
#     """
#
#     # Compute weighted sum of dx and dy
#     avg_dx = np.sum(df['dx']) / len(df) # np.sum(df['speed'])  * df['speed']
#     avg_dy = np.sum(df['dy']) / len(df) # np.sum(df['speed'])  * df['speed']
#     
#     # Compute magnitude
#     magnitude = np.sqrt(avg_dx**2 + avg_dy**2)
#     
#     # Compute orientation in degrees
#     orientation = np.arctan2(avg_dy, avg_dx)
#     
#     return magnitude, orientation
#
#
# def load_3tanks(measurement, treatment, path_to_tracking_results):
#         
#     dataframes=[]
#     for tank in ['1', '2', '3']:
#         path_xml = path_to_tracking_results + measurement + "\\tracking_results\\" + treatment + tank + ".xml"    
#         
#         df_tank = load_xml(path_xml)
#         
#         calc_orientation_and_speed(df_tank)
#
#         df_tank['tank'] = tank  # Add tank number to the dataframe
#         dataframes.append(df_tank)
#
#     df = pd.concat(dataframes)
#
#     # remove shorter tracks
#     # df = df[df['nSpots'] > 50].copy()
#     # df.sort_values(['particle', 't']) 
#
#     return df
#     
# def calc_av_vecs(path_to_tracking_results, time_start, time_end):
#     """
#     Calculates average vector for each treatment in the following steps:
#
#     1. Load data from three tanks and treatments and calculate average vector for each tank
#     2. Subtract background (nothing) from other treatments
#     3. Calculate average vector for each treatment group (still, breeze, stormy) from the three tanks
#
#     Returns df with average vectors for each treatment and replicates, and df_av_grouped with mean for each treatment
#     """
#
#     # Define treatments and tanks
#     treatments = ['still', 'breeze', 'stormy']#, 'nothing']
#     tanks = [1, 2, 3]
#     average_vectors = []
#
#     # Calculate average vector for each treatment separately
#     for treatment in treatments:
#         for tank in tanks:
#             path_xml = path_to_tracking_results + treatment + str(tank) + ".xml"
#             df = load_xml(path_xml)
#             # print(max(df['t']))
#             df = df[(df['t'] > time_start) & (df['t'] < time_end)]
#
#             calc_orientation_and_speed(df)      
#             m_av, or_av = compute_average_vector(df)
#
#             dx = m_av * np.cos(or_av) 
#             dy = m_av * np.sin(or_av)
#             
#             # Append the results to the list
#             average_vectors.append({
#                 'treatment': treatment,
#                 'tank': tank,
#                 'm_av': m_av,
#                 'or_av': np.degrees(or_av),
#                 
#                 'dx': dx,
#                 'dy': dy
#             })
#
#     df_av = pd.DataFrame(average_vectors)
#     
#     # Calculate mean for each column    
#     df_av_grouped = df_av.groupby('treatment').agg({
#         'dx': 'mean',
#         'dy': 'mean',
#         'm_av': 'mean',
#         'or_av': 'mean'
#     }).reset_index().rename(columns={'m_av': 'm', 'or_av': 'or'})
#     
#     return df_av, df_av_grouped