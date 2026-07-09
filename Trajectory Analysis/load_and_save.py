import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import scipy as sp
from post_processing_functions import (
    count_stationary_traj,
    fill_gaps_linear,
    load_frames,
    load_xml,
    slice_time_window,
    tukey_posthoc,
)


measurement = 'Cladocerans_2605'

path_to_tracking_data = "R:\\LU24A1047-PLS\\TrackingData\\"
path_to_tracking_results = os.path.join(path_to_tracking_data, measurement, 'tracking_results')
path_to_save_scaledtraj = os.path.join(path_to_tracking_data, measurement, 'trajectories_scaled')
os.makedirs(path_to_save_scaledtraj, exist_ok=True)

if measurement == 'Artemia_0805':
    path_raw_avi = os.path.join(path_to_tracking_data, "Artemia_0805\\resized_videos\\still1.avi")
    path_cropped_avi = os.path.join(path_to_tracking_data, "Artemia_0805\\cropped_videos\\still1.avi")
    h_cropped_pxl, w_cropped_pxl = 1280, 853 # height, width of cropped video [pxl]
    w_raw_pxl, h_raw_pxl = 853, 1280 # width, height of raw video [pxl]
    w_raw_mm = 70.6 # width of raw video [mm]
    binning = 1 # binning factor of the camera used to record the raw video
    time_start = 150 # start time of the analysis window [frames]
    time_end =  10000 # end time of the analysis window [frames]
    min_frames = 30 # minimum number of frames for a trajectory to be included in the analysis
    tanks = ['1', '2', '3'] # number of tanks per treatment

elif measurement == 'BarnacleLarvae_2102':
    path_raw_avi = os.path.join(path_to_tracking_data, os.path.join(measurement, "raw_videos", "still1.avi"))
    path_cropped_avi = os.path.join(path_to_tracking_data, os.path.join(measurement, "cropped_videos", "still1.avi"))
    cropped_frame = load_frames(path_cropped_avi,1)[0]
    h_cropped_pxl, w_cropped_pxl = 630, 930 # height, width of cropped video [pxl]
    w_raw_pxl, h_raw_pxl = 3648, 5472 # width, height of raw video [pxl]
    w_raw_mm = 110 # width of raw video [mm]
    binning = 4 # binning factor of the camera used to record the raw video
    time_start = 0 # start time of the analysis window [frames]
    time_end =  10000 # end time of the analysis window [frames]
    min_frames = 30 # minimum number of frames for a trajectory to be included in the analysis
    tanks = ['1', '2', '3'] # number of tanks per treatment

elif measurement == 'Cladocerans_2605':
    path_raw_avi = os.path.join(path_to_tracking_data, "Cladocerans_2605\\resized_videos\\still1.avi")
    path_cropped_avi = os.path.join(path_to_tracking_data, "Cladocerans_2605\\cropped_videos\\still1.avi")
    h_cropped_pxl, w_cropped_pxl = 533, 549 # height, width of cropped video [pxl]
    h_raw_pxl, w_raw_pxl = 2256, 2256 # height, width of raw video [pxl]
    w_raw_mm = 90.6 # width of raw video [mm]
    binning = 4 # binning factor of the camera used to record the raw video
    time_start = 0 # start time of the analysis window [frames]
    time_end =  10000 # end time of the analysis window [frames]
    min_frames = 30 # minimum number of frames for a trajectory to be included in the analysis
    tanks = ['1', '2', '3', '4', '5']   # number of tanks per treatment
    
else:
    raise ValueError("Measurement not recognized.")

# Calculate scale factor to convert pixel coordinates to mm
scale = w_raw_mm / w_raw_pxl
scale = scale * binning

# Define parameters for trajectory analysis
min_movement = 0.3
artifact_treshold = 0.0001

frame_rate = 10

analysis_time_start = time_start
analysis_time_end = time_end

# define treatments
treatments = ['still', 'breeze', 'stormy', 'nothing']

# Load and visualize tracking results for a random selection of frames to validate the tracking results
treatment = 'still1'
path_xml = os.path.join(path_to_tracking_results, treatment + ".xml")
path_avi = os.path.join(path_to_tracking_data, measurement, "cropped_videos", treatment + ".avi")

df = load_xml(path_xml)
frames = load_frames(path_avi)

num_frames = len(frames)
indices = np.random.randint(0, num_frames, size=10)

w,h = frames[1].shape

for i in indices:
    print(['figure ' + str(i) + '/' + str(len(frames))])
    frame = frames[i]
 
    df_current_frame = df[df['t'] == i]
    x_pos = df_current_frame['x'].values
    y_pos = df_current_frame['y'].values

    plt.figure()
    plt.imshow(frame, cmap='gray')
    plt.scatter(x_pos, y_pos, s=25, edgecolors='r', facecolors='none')
    plt.show()
    plt.close()

# Load and process tracking results for all treatments and tanks
dfs=[None]*len(treatments)
organisms_per_tank = []

for i, treatment in enumerate(treatments):
    data_perframe = []

    for tank in tanks:
        print(f"\rProcessing {treatment} {tank} ...", end='\r', flush=True)
        
        path_xml = os.path.join(path_to_tracking_results, treatment + tank + ".xml")  # Define path to the XML file for the current treatment and tank
        df_tank = load_xml(path_xml)  # Load the tracking data from the XML file into a DataFrame
        
        # Count the number of organisms in the tank at time t=0 to make estimate of the number of organisms per tank
        organisms_tank = len(df_tank[df_tank['t'] == 0])*2
        organisms_per_tank.append(organisms_tank)
        
        # 1. Scale the full data and flip y-axis to match the orientation of the video
        df_tank['x'] = df_tank['x'].astype(float) * scale
        df_tank['y'] = (h_cropped_pxl - df_tank['y'].astype(float) - 1) * scale    

        # 2. Compute dx/dy on full trajectory
        df_tank = df_tank.sort_values(['particle', 't']).reset_index(drop=True)
        df_tank['dt'] = df_tank.groupby('particle')['t'].diff()
        df_tank['dx'] = df_tank.groupby('particle')['x'].diff() / df_tank['dt'] * frame_rate
        df_tank['dy'] = df_tank.groupby('particle')['y'].diff() / df_tank['dt'] * frame_rate

        # Drop rows with NaN values in dx, dy, dt (first entry of each trajectory and any trajectories with only one spot)
        df_tank = df_tank.dropna(subset=['dx', 'dy', 'dt']).reset_index(drop=True)
        
        # 3. Filter time window
        if treatment != 'nothing':  # For control treatment, keep full time window for analysis
            df_tank = slice_time_window(df_tank, time_start, time_end)
        
        df_tank = df_tank.reset_index(drop=True)
        
        # 4. remove artifacts and lineraly interpolate gaps in trajectories
        x_diff = df_tank.groupby('particle', sort=False)['x'].diff().abs()
        y_diff = df_tank.groupby('particle', sort=False)['y'].diff().abs()
        df_tank = df_tank[(x_diff >= artifact_treshold) & (y_diff >= artifact_treshold)]
        df_tank = fill_gaps_linear(df_tank, max_gap=5, frame_rate=frame_rate)
 
        # 6. Calculate orientation and speed
        df_tank['orientation'] = np.arctan2(df_tank['dy'], df_tank['dx']) #[radians]
        df_tank['orientation_degrees'] = np.degrees(df_tank['orientation']) #[degrees]
        df_tank['speed'] = np.sqrt(df_tank['dx'] ** 2 + df_tank['dy'] ** 2) #[mm/s]  
        
        # 7. Remove stationary trajectories (displacement < min_movement) 
        stat_ids = count_stationary_traj(df_tank, threshold=min_movement)
        if stat_ids:
            df_tank = df_tank[~df_tank['particle'].isin(stat_ids)].reset_index(drop=True)

        df_tank['tank'] = int(tank)  # Add tank number as integer column

        # 8. Remove short trajectories (less than min_frames)
        df_tank['nSpots'] = df_tank.groupby('particle')['t'].transform('count')
        df_tank = df_tank[df_tank['nSpots'] >= min_frames].reset_index(drop=True)

        data_perframe.append(df_tank) # Dataframe after processing with all trajectories per frame
        
    dfs[i] = pd.concat(data_perframe, ignore_index=True)


# Save scaled and processed data into pickle files for plotting scripts    
for i, df in enumerate(dfs):
    df.to_pickle(os.path.join(path_to_save_scaledtraj, f'traj_{treatments[i]}_scaled.pkl'))

# Find the minimum end time across all treatments (excluding control) to ensure consistent analysis window
min_end_time = float('inf')
for df in dfs[:-1]:  # Exclude control treatment which has full time window
    for tank in tanks:
        end_time = df[df['tank'] == int(tank)]['t'].max()
        if end_time < min_end_time:
            min_end_time = end_time

# Adjust analysis_time_end to the minimum end time across treatments
if analysis_time_end > min_end_time:
    analysis_time_end = min_end_time

# Slice the dataframes for each treatment to the analysis time window
df_still_perframe = slice_time_window(dfs[0], analysis_time_start, analysis_time_end)
df_breeze_perframe = slice_time_window(dfs[1], analysis_time_start, analysis_time_end)
df_stormy_perframe = slice_time_window(dfs[2], analysis_time_start, analysis_time_end)
df_control_perframe = slice_time_window(dfs[3], 0, analysis_time_end)

# Calculate mean dy for each tank and treatment
still_means_perframe = [np.mean(df_still_perframe['dy'][df_still_perframe['tank'] == int(tank)]) for tank in tanks]
breeze_means_perframe = [np.mean(df_breeze_perframe['dy'][df_breeze_perframe['tank'] == int(tank)]) for tank in tanks]
stormy_means_perframe = [np.mean(df_stormy_perframe['dy'][df_stormy_perframe['tank'] == int(tank)]) for tank in tanks]
control_means_perframe = [np.mean(df_control_perframe['dy'][df_control_perframe['tank'] == int(tank)]) for tank in tanks]    

# Calculate means and 95% CI for each treatment
means_perframe = [np.mean(still_means_perframe), np.mean(breeze_means_perframe), np.mean(stormy_means_perframe), np.mean(control_means_perframe)]
STD_perframe = [np.std(still_means_perframe, ddof=1), np.std(breeze_means_perframe, ddof=1), np.std(stormy_means_perframe, ddof=1), np.std(control_means_perframe, ddof=1)]
n_replicates = len(tanks)
t_critical = sp.stats.t.ppf(0.975, df=n_replicates - 1)
CI95_perframe = [STD_perframe[i] / np.sqrt(n_replicates) * t_critical for i in range(len(treatments))]

# Calculate V for each tank and treatment
def V(angles: np.ndarray) -> float:
    angles = angles[~np.isnan(angles)]
    R = np.sqrt(np.mean(np.sin(angles))**2 + np.mean(np.cos(angles))**2)
    return 1 - R

still_V_perframe = [V(df_still_perframe['orientation'][df_still_perframe['tank'] == int(tank)].values) for tank in tanks]
breeze_V_perframe = [V(df_breeze_perframe['orientation'][df_breeze_perframe['tank'] == int(tank)].values) for tank in tanks]
stormy_V_perframe = [V(df_stormy_perframe['orientation'][df_stormy_perframe['tank'] == int(tank)].values) for tank in tanks]
control_V_perframe = [V(df_control_perframe['orientation'][df_control_perframe['tank'] == int(tank)].values) for tank in tanks]

mean_V_perframe = [np.mean(still_V_perframe), np.mean(breeze_V_perframe), np.mean(stormy_V_perframe), np.mean(control_V_perframe)]
STD_V_perframe = [np.std(still_V_perframe, ddof=1), np.std(breeze_V_perframe, ddof=1), np.std(stormy_V_perframe, ddof=1), np.std(control_V_perframe, ddof=1)]
CI95_V_perframe = [STD_V_perframe[i] / np.sqrt(n_replicates) * t_critical for i in range(len(treatments))]

f_stat_perframe, p_value_perframe = sp.stats.f_oneway(still_means_perframe, breeze_means_perframe, stormy_means_perframe)
f_stat_V_perframe, p_value_V_perframe = sp.stats.f_oneway(still_V_perframe, breeze_V_perframe, stormy_V_perframe)

print(f"\n=== Statistical results for {measurement} ===")
print(f"Analysis window: {analysis_time_start} to {analysis_time_end} frames")
print(f"Artifact threshold: {artifact_treshold:.4f} mm / {artifact_treshold/scale:.4f} pxls")
print(f"Minimum movement to consider non-stationary: {min_movement:.2f} mm / {min_movement/scale:.2f} pxls")
print(f"Minimum trajectory length: {min_frames} frames / {min_frames/frame_rate:.1f} seconds")

print(f"\n=== ANOVA results for {measurement} ===")
print(f"P-value dy: {p_value_perframe:.4f}")
print(f"P-value V: {p_value_V_perframe:.4f}")

group_means_perframe, letters_perframe, tukey_table_perframe = tukey_posthoc(
    treatments,
    [still_means_perframe, breeze_means_perframe, stormy_means_perframe, control_means_perframe],
)

group_means_V_perframe, letters_V_perframe, tukey_table_V_perframe = tukey_posthoc(
    treatments,
    [still_V_perframe, breeze_V_perframe, stormy_V_perframe, control_V_perframe],
)

print("\n=== Mean dy (per frame)===")
for i, treatment in enumerate(treatments):
    print(f"{treatment.capitalize()}: Mean dy = {means_perframe[i]:.2f} ± {STD_perframe[i]:.2f} mm/s  [{letters_perframe[treatment]}], V = {mean_V_perframe[i]:.2f} ± {STD_V_perframe[i]:.2f}")

print(f"mean number of organisms per tank: {np.mean(organisms_per_tank):.1f} ± {np.std(organisms_per_tank, ddof=1):.1f}")
    